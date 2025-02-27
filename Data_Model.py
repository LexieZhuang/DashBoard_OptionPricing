import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


class YahooFinanceData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None

    def download_data(self, start_date=None, end_date=None):
        try:
            # download data from yahoo finance
            self.data = yf.download(
                self.ticker, start=start_date, end=end_date)
            # try:
            #     self.data.index = self.data.index.tz_convert(None)
            # except TypeError:
            #     pass
            print("Data downloaded successfully!")
        except Exception as e:
            print("Failed to download data:", e)

    def get_data(self):
        if self.data is not None:
            return self.data
        else:
            print("Please download data first!")

    def get_close_prices(self):
        if self.data is not None:
            return self.data.iloc[:,0]
        else:
            print("Please download data first!")

    def get_date(self):
        if self.data is not None:
            return self.data.index
        else:
            print("Please download data first!")


class HestonUKF:
    def __init__(self, dt):
        self.dt = dt
        self.n_dim_state = 6  # Dimension of state vector: S, v, mu, kappa, theta, sigma
        self.points = MerweScaledSigmaPoints(
            n=self.n_dim_state, alpha=0.1, beta=2., kappa=1.)
        self.ukf = UnscentedKalmanFilter(dim_x=self.n_dim_state, dim_z=1, fx=self._heston_model,
                                         hx=self._observation_model, dt=self.dt, points=self.points)
        self.estimates = []  # Store parameter estimates
        self.estimated_log_prices = []  # Store estimated log prices
        self.parameters = None  # Estimated parameters

    def fit(self, prices):
        prices = prices.copy()
        self.ukf.dt = self.dt
        self._initialize_filter(prices)

        for price in prices[1:]:
            if price <= 0:
                continue
            log_price = np.log(price)
            self.ukf.predict()
            self.ukf.update(np.array([log_price]))
            self.estimates.append(self.ukf.x.copy())
            self.estimated_log_prices.append(self.ukf.x[0])

        model_parameters = self.estimates[-1]
        # S0, v0, mu, kappa, theta, sigma = model_parameters
        v0 = model_parameters[1]      # Initial volatility
        theta = model_parameters[4]   # Long-term volatility
        kappa = model_parameters[3]   # Mean reversion speed
        sigma = model_parameters[5]   # Volatility of volatility
        # Long-term mean return under risk-neutral measure
        mu = model_parameters[2]
        S0 = model_parameters[0]      # Initial asset's log price

        self.parameters = {
            'v0': v0,
            'theta': theta,
            'kappa': kappa,
            'sigma': sigma,
            'mu': mu,
            'S0': S0
        }

    def update_parameters(self, para_dict):
        # some times the input might be string...
        for k in para_dict.keys():
            para_dict[k] = float(para_dict[k])
        self.parameters = para_dict

    def get_parameters(self):
        if self.parameters is None:
            raise RuntimeError(
                "Please fit the model first,then we can get the parameters.")

        return self.parameters

    def monte_carlo_simulate(self, dt, M, N):
        '''
        dt : time gap btw 2 time steps (/year)
        M : # of paths
        N : # of steps
        '''
        if self.parameters is None:
            raise RuntimeError(
                "Please fit the model before running Monte Carlo simulation.")

        np.random.seed(42)

        model_parameters = self.parameters
        # S0, v0, mu, kappa, theta, sigma = model_parameters
        v0 = model_parameters['v0']      # Initial volatility
        theta = model_parameters['theta']   # Long-term volatility
        kappa = model_parameters['kappa']   # Mean reversion speed
        sigma = model_parameters['sigma']   # Volatility of volatility
        # Long-term mean return under risk-neutral measure
        mu = model_parameters['mu']
        S0 = model_parameters['S0']      # Initial asset's log price

        dW1 = np.random.normal(scale=np.sqrt(dt), size=(M, N))
        dW2 = np.random.normal(scale=np.sqrt(dt), size=(M, N))

        logS = np.zeros((M, N + 1))
        v = np.zeros((M, N + 1))
        logS[:, 0] = S0
        v[:, 0] = v0
        epsilon = 1e-6

        for t in range(1, N + 1):
            v[:, t] = v[:, t-1] + kappa * (theta - np.maximum(
                v[:, t-1], epsilon)) * dt + sigma * np.sqrt(np.maximum(v[:, t-1], epsilon)) * dW1[:, t-1]
            v[:, t] = np.maximum(v[:, t], epsilon)
            # print(v[:, t-1])
            logS[:, t] = logS[:, t-1] + \
                (mu - 0.5 * v[:, t-1]) * dt + np.sqrt(np.maximum(v[:, t-1], epsilon)) * dW2[:, t-1]

        return logS

    def _heston_model(self, x, dt):
        S, v, mu, kappa, theta, sigma = x
        v = np.clip(v, 0.01, 1.0)
        mu = np.clip(mu, -0.2, 0.2)
        kappa = np.clip(kappa, 0.1, 1.5)
        theta = np.clip(theta, 0.01, 0.5)
        sigma = np.clip(sigma, 0.01, 0.6)
        dw1 = np.random.normal(scale=np.sqrt(dt))
        dw2 = np.random.normal(scale=np.sqrt(dt))
        dv = kappa * (theta - v) * dt + sigma * np.sqrt(v) * dw2
        dS = mu * S * dt + np.sqrt(v) * S * dw1
        return np.array([S + dS, v + dv, mu, kappa, theta, sigma])

    def _observation_model(self, x):
        return np.array([x[0]])  # Assume only observe log price

    def _initialize_filter(self, prices):
        S0 = np.log(prices.iloc[0])
        v0 = 0.2  # Initial volatility assumption
        mu0 = 0.05  # Initial mu assumption
        kappa0 = 0.3  # Initial kappa assumption
        theta0 = 0.2  # Initial theta assumption
        sigma0 = 0.1  # Initial sigma assumption
        self.ukf.x = np.array([S0, v0, mu0, kappa0, theta0, sigma0])
        self.ukf.P *= 0.1
        self.ukf.R = np.array([[0.1]])  # Observation noise
        self.ukf.Q = np.eye(self.n_dim_state) * 0.01  # Process noise
