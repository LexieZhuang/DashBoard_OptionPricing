# Data Visualization DashBoard for Option Pricing

## Overview
This project is a web-based application that models stock prices using the Heston model with an Unscented Kalman Filter (UKF). It enables users to estimate model parameters, run Monte Carlo simulations, and perform sensitivity analysis. This is the final project for NYU FRE-6411 Data Visualization.

## Features
- **Stock Price Data Retrieval**: Fetch historical stock price data from Yahoo Finance.
- **Heston Model Estimation**: Estimate model parameters using an Unscented Kalman Filter.
- **Monte Carlo Simulation**: Generate future stock price scenarios.
- **Statistical Analysis**: Compute key statistical measures such as expectation, variance, skewness, and kurtosis.
- **Interactive Dashboard**: A Dash web application with interactive visualizations and parameter customization.

## Project Structure
```
├── Data_Model.py      # Handles data fetching and model estimation (Yahoo Finance, Heston UKF)
├── Tools.py          # Contains helper classes for warnings and graphical utilities
├── demo.py           # Main entry point for the Dash web application
└── README.md         # Project documentation
```

## Installation
### Prerequisites
Ensure you have Python 3 installed, then install the required dependencies:
```bash
pip install dash dash-bootstrap-components pandas numpy yfinance plotly filterpy scipy
```

## Usage
### Running the Application
Start the Dash web application by executing:
```bash
python demo.py
```
Then, navigate to `http://127.0.0.1:5500/` in your web browser.

### Components
- **Stock Search**: Enter a stock ticker to fetch historical data.
- **Heston Model Estimation**: Select a date range and estimate parameters.
- **Monte Carlo Simulation**: Simulate future stock prices based on estimated parameters.
- **Sensitivity Analysis**: Modify model parameters and observe changes in predictions.
- **Statistical Analysis**: View summary statistics and probability distributions of simulated prices.


