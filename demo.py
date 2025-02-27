import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from datetime import date
from Tools import Warnings, Graphs
from Data_Model import YahooFinanceData, HestonUKF

WN = Warnings()
GX = Graphs()

app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.MINTY], suppress_callback_exceptions=True)

# ============================================== ticker searcher bar
ticker_searcher_div = html.Div([
    # ticker searcher
    html.P("Please input the stock ticker you want to model"),
    dbc.InputGroup([
        dbc.Input(id="ticker", placeholder="type the ticker"),
        dbc.Button("Search", id='input-ticker-button', n_clicks=0),
    ])
])

alert_tikcer_not_found_holder = html.Div(
    id='alert_tikcer_not_found_holder', children=[])
alert_tikcer_not_found = dbc.Alert(
    "Can't find the data of this ticker. Please input a correct one.",
    id="alert_tikcer_not_found",
    dismissable=True,
    color="warning"
)

# ============================================== historical price figure card
historical_price_plot_card = dbc.Card([
    dbc.CardHeader("Historical price info"),
    dbc.CardBody([
        dbc.Spinner(dcc.Graph(id='historical-price-fig',
                    figure=GX.get_empty_ts_fig()))
    ])
])

# ============================================ model estimation
heston_card = dbc.Card([
    dbc.CardHeader("Estimation"),
    dbc.CardBody([
        html.P("We use Heston model to model the price:"),
        dcc.Markdown(
            r'''
            $$dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S$$
            
            $$dv_t = \kappa ( \theta - v_t ) \, dt + \sigma \sqrt{v_t} \, dW_t^v$$
            ''',
            mathjax=True,
        ),
        dbc.Spinner(
            dbc.Table([
                html.Thead([
                    html.Tr([html.Th("parameters"),
                             html.Th("estimated value")])
                ]),

        html.Tbody([
            html.Tr([
                dcc.Markdown(r'$\hat{\mu}$', mathjax=True),
                html.Td(id="mu_hat")
            ]),
            html.Tr([
                dcc.Markdown(r'$\hat{\theta}$', mathjax=True),
                html.Td(id="theta_hat")
            ]),
            html.Tr([
                dcc.Markdown(r'$\hat{\kappa}$', mathjax=True),
                html.Td(id="kappa_hat")
            ]),
            html.Tr([
                dcc.Markdown(r'$\hat{\sigma}$', mathjax=True),
                html.Td(id="sigma_hat")
            ])
        ])
                    ], className="table-primary text-center")
                )
            ])
        ], style={"width": "28rem"})

estimate_div = html.Div([

    html.P("Select the date range for estimating the model"),
    dbc.InputGroup([
        dbc.InputGroupText("from"),
        dcc.DatePickerSingle(id='estimation-start', placeholder='mm/dd/yyyy'),
        dbc.InputGroupText("to"),
        dcc.DatePickerSingle(id='estimation-end', placeholder='mm/dd/yyyy'),
        dbc.Button("Estimate", id='estimate-button', n_clicks=0),
        WN.generate_no_ticker_pop("estimate-button"),
        WN.generate_no_date_pop("estimate-button"),
        WN.generate_date_order_error("estimate-button"),
        WN.generate_short_window_pop("estimate-button")
    ]),

    html.Br(),
    heston_card
])

# ============================================== Monte Carlo simulation
monte_carlo_card = dbc.Card([
    dbc.CardHeader("Options for Monte Carlo Simulation"),
    dbc.CardBody([
        dbc.InputGroup([
            dbc.InputGroupText("Monte Carlo start date"),
            dcc.DatePickerSingle(id='Monte-start', placeholder='mm/dd/yyyy'),
            dbc.InputGroupText("Predict historical price"),
            dcc.DatePickerSingle(id='Monte-end',
                                 className='border border-2',
                                 placeholder='mm/dd/yyyy',
                                 max_date_allowed=date.today()),
        ], className='mb-3'),
        html.P("The frequency of simulation"),
        html.P("(length of each time step = 1/freq (/day))"),
        dbc.InputGroup([
            dcc.Input(id='sim-freq-per-day', type="number",
                      className='border border-2'),
            dbc.InputGroupText("per day")
        ], className='mb-3'),
        WN.generate_no_int_pop('sim-freq-per-day'),
        html.P("How many paths do you want?"),
        dbc.InputGroup([
            dbc.InputGroupText("Number of paths"),
            dcc.Input(id='sim-n-paths', type="number",
                      className='border border-2'),
        ], className='mb-3'),
        WN.generate_no_int_pop('sim-n-paths'),
        html.P("Confidential interval"),
        # dbc.RadioItems(id='confidence-interval-radio',
        #                options=[
        #                    {'label': '90%', 'value': 90},
        #                    {'label': '95%', 'value': 95},
        #                    {'label': '99%', 'value': 99}
        #                ],
        #                value=90,  # default value
        #                inline=True,
        #                className='mb-3'),
        dbc.Button('Run Simulation', id='simulate-button',
                   n_clicks=0, className="d-grid gap-2 col-6 mx-auto"),
        WN.generate_no_ticker_pop('simulate-button'),
        # 'no-params-pop-simulate-button'
        WN.generate_no_params_pop('simulate-button'),
    ])
], style={"width": "28rem"})

simulation_plot_card = dbc.Card([
    dbc.CardHeader("Simulation paths"),
    dbc.CardBody([
        dbc.RadioItems(
            id="confidence-interval-radio",
            options=[
               {'label': '90%', 'value': 90},
               {'label': '95%', 'value': 95},
               {'label': '99%', 'value': 99}

            ],
            value=90,  # default value
            inline=True,
            className='mb-3'
        ),

        dbc.Spinner(dcc.Graph(id='simulation-paths-fig'))
    ])
])

# ======================================================== Sensitivity Analysis
para_change_card = dbc.Card([
    dbc.CardHeader(
        "Customize your model parameters to predict the future price"),
    dbc.CardBody([
        # dbc.InputGroup([
        #     dbc.InputGroupText("from"),
        #     dcc.DatePickerSingle(id='estimation-start_1',
        #                          placeholder='mm/dd/yyyy'),
        #     dbc.InputGroupText("to"),
        #     dcc.DatePickerSingle(id='estimation-end_1',
        #                          placeholder='mm/dd/yyyy'),
        #     WN.generate_no_date_pop("update-button"),
        #     WN.generate_date_order_error("update-button"),
        #     WN.generate_short_window_pop("update-button")
        # ]),
        # html.Br(),
        dbc.InputGroup([
            dbc.InputGroupText("The predict date"),
            dcc.DatePickerSingle(
                id='prediction-end', className='border border-2', placeholder='mm/dd/yyyy'),
        ], className='mb-3'),
        WN.generate_no_date_pop(
            "prediction-end", content='The simulation end should be in the future.'),

        html.P("You can enter the changed value of parameters:"),

        dbc.Table([
            html.Thead([
                html.Tr([html.Th("Parameters"), html.Th(
                    "Estimation"), html.Th("Custom Value")])
            ]),
            html.Tbody([
                html.Tr([
                    dcc.Markdown(r'$\hat{\mu}$', mathjax=True),
                    html.Td(id="mu_hat_heston"),
                    html.Td(dcc.Input(id="mu_hat_input", type="number",
                            placeholder="Enter new value", className='border border-2'))
                ]),
                html.Tr([
                    dcc.Markdown(r'$\hat{\\theta}$', mathjax=True),
                    html.Td(id="theta_hat_heston"),
                    html.Td(dcc.Input(id="theta_hat_input", type="number",
                            placeholder="Enter new value", className='border border-2'))
                ]),
                html.Tr([
                    dcc.Markdown(r'$\hat{\\kappa}$', mathjax=True),
                    html.Td(id="kappa_hat_heston"),
                    html.Td(dcc.Input(id="kappa_hat_input", type="number",
                            placeholder="Enter new value", className='border border-2'))
                ]),
                html.Tr([
                    dcc.Markdown(r'$\hat{\\sigma}$', mathjax=True),
                    html.Td(id="sigma_hat_heston"),
                    html.Td(dcc.Input(id="sigma_hat_input", type="number",
                            placeholder="Enter new value", className='border border-2'))
                ])
            ])
        ], className="table-primary text-center"),
        dbc.Button('Update Simulation', id='update-button',
                   n_clicks=0, className="d-grid gap-2 col-6 mx-auto"),
        WN.generate_no_params_pop('update-button'),
        WN.generate_lack_action_pop('update-button'),
        html.Br(),
        dcc.Markdown(
            r'$\hat{\mu}$: rate of return of the asset', mathjax=True),
        dcc.Markdown(r'$\hat{\\theta}$: long-term variance', mathjax=True),
        dcc.Markdown(r'$\hat{\\kappa}$: rate of mean reversion', mathjax=True),
        dcc.Markdown(r'$\hat{\\sigma}$: volatility of volatility', mathjax=True)


    ])
], style={"width": "28rem"})


statistical_card = dbc.Card([
    dbc.CardHeader("Moments"),
    dbc.CardBody([
        dbc.Spinner(
            dbc.Table([
                html.Thead([
                    html.Tr([html.Th("Expectation"), html.Th("Variance"),
                             html.Th("Skewness"), html.Th("Kurtosis")])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(id="mom1"),
                        html.Td(id="mom2"),
                        html.Td(id="mom3"),
                        html.Td(id="mom4")
                    ]),
                ])
            ], className="table-primary text-center")
        ),

        dbc.RadioItems(
            id="cdf-or-pdf-or-boxplot",
            options=[
                {"label": "CDF", "value": "CDF"},
                {"label": "PDF", "value": "PDF"},
                {"label": "Boxplot", "value": "Boxplot"}

            ],
            value="CDF",
            inline=True,
        ),

        dbc.Spinner(dcc.Graph(id='price-distribution'))
    ])
])


######################################################################
app.layout = dbc.Container([
    dcc.Store(id='model-parameters'),
    dcc.Store(id='last-date-known'),
    dcc.Store(id='distribution-data'),
    dcc.Store(id='historical_data_simulation'),
    dbc.Row(
        [
            html.H1("STOCK ECONOMIC SCENARIOS GENERATOR",
                    className="text-center")
        ],
    ),
    html.Br(),
    dbc.Row(
        [
            html.H2("Parameter Estimation")
        ],
    ),
    dbc.Row(
        [
            dbc.Col(ticker_searcher_div, width={'size': 11}),

        ],
    ),
    dbc.Row(
        [
            dbc.Col(alert_tikcer_not_found_holder, width={'size': 11}),
        ]
    ),
    html.Br(),
    dbc.Row(
        [
            dbc.Col([estimate_div,
                     html.Br(),
                     monte_carlo_card], width={'size': 5, 'offset': 0.5}),
            dbc.Col([html.Br(), historical_price_plot_card, html.Br(),
                    simulation_plot_card], width={'size': 6, 'offset': 0.5}),
        ],
    ),
    html.Br(),
    dbc.Row(
        [
            html.H2("Statistical Analysis")
        ],
    ),
    html.Br(),
    dbc.Row(
        [
            dbc.Col([para_change_card], width={'size': 5, 'offset': 0.5}),
            dbc.Col([statistical_card], width={'size': 6, 'offset': 0.5}),
        ],
    ),
])


@app.callback(
    [Output('historical-price-fig', 'figure'),
     Output('estimation-start', 'min_date_allowed'),
     Output('estimation-end', 'max_date_allowed'),
     Output('alert_tikcer_not_found_holder', 'children'),],
    Input('input-ticker-button', 'n_clicks'),
    [State('ticker', 'value')]
)
def update_historical_info(n_clicks, ticker):
    if n_clicks > 0:
        ticker = ticker.replace(" ", "")
        FinanceData = YahooFinanceData(ticker)
        FinanceData.download_data()
        close_price = FinanceData.get_close_prices()
        available_dates = FinanceData.get_date()

        print(len(close_price))
        if len(close_price) == 0:
            return GX.get_empty_ts_fig(), None, None, alert_tikcer_not_found

        close_price = go.Scatter(
            x=available_dates, y=close_price, mode='lines', line_color='#A9A9A9')
        fig = GX.get_empty_ts_fig()
        fig.add_trace(close_price)

        min_date = available_dates.min()
        max_date = available_dates.max()

        return fig, min_date, max_date, dash.no_update
    return dash.no_update, None, None, dash.no_update


@app.callback(
    [Output('model-parameters', 'data'),
     Output('last-date-known', 'data'),
     Output('mu_hat', 'children'),
     Output('theta_hat', 'children'),
     Output('kappa_hat', 'children'),
     Output('sigma_hat', 'children'),
     Output('mu_hat_heston', 'children'),
     Output('theta_hat_heston', 'children'),
     Output('kappa_hat_heston', 'children'),
     Output('sigma_hat_heston', 'children'),
     Output('mu_hat_input', 'value'),
     Output('theta_hat_input', 'value'),
     Output('kappa_hat_input', 'value'),
     Output('sigma_hat_input', 'value'),
     Output('no-ticker-pop-estimate-button', 'is_open'),
     Output('no-date-pop-estimate-button', 'is_open'),
     Output('wrong-date-order-pop-estimate-button', 'is_open'),
     Output('short-window-pop-estimate-button', 'is_open'),

     ],

    Input('estimate-button', 'n_clicks'),

    [State('ticker', 'value'),
     State('estimation-start', 'date'),
     State('estimation-end', 'date')]
)
def show_estimation_result(n_clicks, ticker, estimation_start, estimation_end):
    if n_clicks > 0:
        if ticker is None:
            return [None]*14 + [True, False, False, False]

        FinanceData = YahooFinanceData(ticker)
        FinanceData.download_data()
        close_price = FinanceData.get_close_prices()
        all_historical_dates = FinanceData.get_date()

        if (estimation_start is None) or (estimation_end is None):
            return [None]*14 + [False, True, False, False]

        try:
            estimation_start = pd.to_datetime(estimation_start)
        except ValueError:
            return [None]*14 + [False, True, False, False]

        try:
            estimation_end = pd.to_datetime(estimation_end)
        except ValueError:
            return [None]*14 + [False, True, False, False]

        if estimation_start < all_historical_dates.min() or estimation_end > all_historical_dates.max():
            return [None]*14 + [False, True, False, False]

        if estimation_start >= estimation_end:
            return [None]*14 + [False, False, True, False]

        estimated_data = close_price[(close_price.index > estimation_start) & (
            close_price.index < estimation_end)]

        heston_ukf = HestonUKF(dt=1.0)
        try:
            heston_ukf.fit(estimated_data)
        except IndexError:
            return [None]*14 + [False, False, False, True]
        model_parameters = heston_ukf.get_parameters()

        return [model_parameters, all_historical_dates[-1]] + ["{:.4f}".format(model_parameters[parameter]) for parameter in ['mu', 'theta', 'kappa', 'sigma']]*3 + [False]*4

    return [None]*14 + [False]*4


@app.callback(
    [Output('historical_data_simulation', 'data'),
     Output('no-ticker-pop-simulate-button', 'is_open'),
        # Output('no-date-pop-simulation-end', 'is_open'),
        Output('no-params-pop-simulate-button', 'is_open'),
        Output('no-int-pop-sim-freq-per-day', 'is_open'),
        Output('no-int-pop-sim-n-paths', 'is_open'),
     ],
    Input('simulate-button', 'n_clicks'),
    [State('ticker', 'value'),
     State('Monte-start', 'date'),
     State('model-parameters', 'data'),
     State('Monte-end', 'date'),
     State('sim-freq-per-day', 'value'),
     State('sim-n-paths', 'value')
     ]
)
def update_simulation_paths_figure(n_clicks, ticker, estimation_start, model_parameters, sim_end, n_per_day, n_paths):
    if n_clicks > 0:

        if ticker is None:
            # return dash.no_update, True, False, False, False, False
            return None, True, False, False, False

        # if sim_start is None:
        #     # return dash.no_update, False, False, True, False, False
        #     return None, False, True, False, False

        # sim_start = pd.to_datetime(sim_start)

# warning::: There are 2 warnings temporary commented!!
        try:
            sim_end = pd.to_datetime(sim_end)
        except ValueError:
            # return dash.no_update, False, True, False, False, False
            return None, False, False, False, False

        # if sim_end <= estimation_start:
        #     return dash.no_update, False, True, False, False, False

        estimation_start = pd.to_datetime(estimation_start)
        if model_parameters is None:
            # return dash.no_update, False, False, True, False, False
            return None, False, True, False, False

        if not isinstance(n_per_day, int):
            # return dash.no_update, False, False, False, True, False
            return None, False, False, True, False

        if not isinstance(n_paths, int):
            # return dash.no_update, False, False, False, False, True
            return None, False, False, False, True

        estimation_start = pd.to_datetime(estimation_start)
        FinanceData = YahooFinanceData(ticker)
        FinanceData.download_data()
        close_price = FinanceData.get_close_prices()
        start_pos = close_price.index.searchsorted(
            estimation_start, side='left')
        end_pos = close_price.index.searchsorted(sim_end, side='right')
        if start_pos == 0:
            start_pos = close_price.index[0]
        else:
            start_pos = close_price.index[start_pos]
        if end_pos == len(close_price.index):
            end_pos = len(close_price.index) - 1
        else:
            end_pos = close_price.index[end_pos]

        start_pos = pd.to_datetime(start_pos)
        end_pos = pd.to_datetime(end_pos)
        heston_ukf = HestonUKF(dt=1.0)
        model_parameters['S0'] = np.log(close_price.loc[start_pos])
        heston_ukf.update_parameters(model_parameters)

        sim_dates = pd.date_range(start_pos, end_pos)
        T = ((end_pos - start_pos).days + 1) / 365
        N = n_per_day * (len(sim_dates) - 1)

        dt = T / N

        logS = heston_ukf.monte_carlo_simulate(dt, n_paths, N)
        logS_used = np.empty((logS.shape[0], sim_dates.shape[0]))
        for i in range(logS_used.shape[1]):
            logS_used[:, i] = logS[:, n_per_day * i]

        his_price = close_price[(close_price.index >= start_pos) & (
            close_price.index <= end_pos)]
        S_used = np.exp(logS_used)
        traces = []

        traces.append(go.Scatter(x=his_price.index, y=his_price, mode='lines',
                      name='Historical Price', line=dict(color='black')))

        data = {'S_used': S_used, 'traces': traces, 'sim_dates': sim_dates}
        # lower_bound = np.percentile(S_used, (100-confidence)/2, axis=0)
        # upper_bound = np.percentile(
        #     S_used, 100-(100-confidence)/2, axis=0)

        # traces = []

        # traces.append(go.Scatter(x=sim_dates, y=lower_bound,
        #               mode='lines', name=f'{(100-confidence)/2}th Percentile', line=dict(color='red')))

        # traces.append(go.Scatter(x=sim_dates, y=upper_bound, mode='lines',
        #               name=f'{100-(100-confidence)/2}th Percentile', line=dict(color='blue')))

        # traces.append(go.Scatter(x=his_price.index, y=his_price, mode='lines',
        #               name='Historical Price', line=dict(color='black')))

        # # Add the individual paths if needed
        # for i in range(min(20, S_used.shape[0])):
        #     trace = go.Scatter(
        #         x=sim_dates, y=S_used[i], mode='lines', opacity=0.3, showlegend=False)
        #     traces.append(trace)

        # layout = go.Layout(
        #     title='Monte Carlo Simulation with 90% Confidence Interval',
        #     xaxis=dict(title='Date'),
        #     yaxis=dict(title='Price'),
        #     paper_bgcolor='white',
        #     plot_bgcolor='white',
        # )
        # fig = go.Figure(data=traces, layout=layout)

    #     return fig, False, False, False, False, False
    # return dash.no_update, False, False, False, False, False
        return data, False, False, False, False
    return [None] + [False]*4


@app.callback(
    [Output('simulation-paths-fig', 'figure')],

    [Input('historical_data_simulation', 'data'),
     Input('confidence-interval-radio', 'value')]
)
def update_his_distribution_figure(data, confidence):
    if data is None:
        return dash.no_update
    S_used = data['S_used']
    traces = data['traces']
    sim_dates = data['sim_dates']

    lower_bound = np.percentile(S_used, (100-confidence)/2, axis=0)
    upper_bound = np.percentile(
        S_used, 100-(100-confidence)/2, axis=0)

    sim_dates = pd.to_datetime(sim_dates)
    sim_dates = [date.strftime('%Y-%m-%d') for date in sim_dates]

    traces.append(go.Scatter(x=sim_dates, y=lower_bound,
                             mode='lines', name=f'{(100-confidence)/2}th Percentile', line=dict(color='red')))

    traces.append(go.Scatter(x=sim_dates, y=upper_bound, mode='lines',
                             name=f'{100-(100-confidence)/2}th Percentile', line=dict(color='blue')))

    for i in range(min(20, len(S_used))):
        trace = go.Scatter(
            x=sim_dates, y=S_used[i], mode='lines', opacity=0.3, showlegend=False)
        traces.append(trace)

    layout = go.Layout(
        title='Monte Carlo Simulation with 90% Confidence Interval',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig = go.Figure(data=traces, layout=layout)
    return [fig]


@ app.callback(

    [Output('distribution-data', 'data'),
     Output('mom1', 'children'),
     Output('mom2', 'children'),
     Output('mom3', 'children'),
     Output('mom4', 'children'),
     Output('no-params-pop-update-button', 'is_open'),
     Output('lack-action-pop-update-button', 'is_open')],

    # Input('cdf-or-pdf-or-boxplot','value')
    [Input('update-button', 'n_clicks')],

    [State('simulate-button', 'n_clicks'),
     State('model-parameters', 'data'),
     State('last-date-known', 'data'),
     State('ticker', 'value'),
     # State('estimation-start_1', 'date'),
     # State('estimation-end_1', 'date'),
     State('mu_hat_input', 'value'),
     State('theta_hat_input', 'value'),
     State('kappa_hat_input', 'value'),
     State('sigma_hat_input', 'value'),
     State('prediction-end', 'date'),
     State('sim-freq-per-day', 'value'),
     State('sim-n-paths', 'value')
     ]


)
# def update_parameters_paths_figure(n_clicks, n_simulate_clicks, model_parameters, sim_start, ticker, estimation_start, estimation_end, mu_hat_input, theta_hat_input, kappa_hat_input, sigma_hat_input, sim_end, n_per_day, n_paths):
def update_parameters_paths_figure(n_clicks, n_simulate_clicks, model_parameters, sim_start, ticker, mu_hat_input, theta_hat_input, kappa_hat_input, sigma_hat_input, sim_end, n_per_day, n_paths):
    if n_clicks > 0:
        if n_simulate_clicks == 0:
            return [None]*5 + [False, True]
        # repeat the fit model
        sim_start = pd.to_datetime(sim_start)
        sim_end = pd.to_datetime(sim_end)
        heston_ukf = HestonUKF(dt=1.0)

        if model_parameters is None:
            return [None]*5 + [True, False]

        heston_ukf.update_parameters(model_parameters)
        model_parameters_or = heston_ukf.get_parameters()
        update_model_parameters = {
            'v0': model_parameters_or['v0'],
            'theta': theta_hat_input,
            'kappa': kappa_hat_input,
            'sigma': sigma_hat_input,
            'mu': mu_hat_input,
            'S0': model_parameters_or['S0']
        }

        heston_ukf.update_parameters(update_model_parameters)

        # monte carlo
        sim_dates = pd.date_range(sim_start, sim_end)
        T = (sim_end - sim_start).days / 365
        N = n_per_day * (len(sim_dates) - 1)

        dt = T / N

        logS = heston_ukf.monte_carlo_simulate(dt, n_paths, N)
        logS_used = np.empty((logS.shape[0], sim_dates.shape[0]))
        for i in range(logS_used.shape[1]):
            logS_used[:, i] = logS[:, n_per_day * i]

        S_used = np.exp(logS_used)
        final_prices = S_used[:, -1]

        # calculate the mom value
        mean_price = "{:.4f}".format(np.mean(final_prices))
        variance_price = "{:.4f}".format(np.var(final_prices))
        skewness_price = "{:.4f}".format(stats.skew(final_prices))
        kurtosis_price = "{:.4f}".format(stats.kurtosis(final_prices))

        # prepare the data for future plotting
        x = np.linspace(final_prices.min(), final_prices.max(), 1000)
        pdf_y = stats.gaussian_kde(final_prices)(x)
        cdf_y = np.cumsum(pdf_y)
        cdf_y = cdf_y / cdf_y[-1]

        data = {'x': x, 'pdf_y': pdf_y, 'cdf_y': cdf_y}

        return data, mean_price, variance_price, skewness_price, kurtosis_price, False, False

    return [None]*5 + [False]*2


@ app.callback(
    Output('price-distribution', 'figure'),

    [Input('distribution-data', 'data'),
     Input('cdf-or-pdf-or-boxplot', 'value')]
)
def update_distribution_figure(data, type):

    if data is None:
        return dash.no_update

    x = data['x']

    fig = go.Figure()

    if type == 'PDF':
        y = data['pdf_y']
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                      name='PDF', opacity=0.3, showlegend=False))
        fig.update_layout(
            # title='Probability Density Function',
            xaxis_title='Stock Price',
            yaxis_title='Probability Density',
            # margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
            # title_font_size=15
        )
        return fig

    if type == 'CDF':
        y = data['cdf_y']
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                      name='PDF', opacity=0.3, showlegend=False))
        fig.update_layout(
            # title='Probability Density Function',
            xaxis_title='Stock Price',
            yaxis_title='Cumulative Probability',
            # margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
            # title_font_size=15
        )
        return fig

    if type == 'Boxplot':
        fig.add_trace(go.Box(y=x, name='Boxplot'))
        return fig

    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True,port=5500)
# app.run_server(

#     debug=True,
#     port=5500
# )
