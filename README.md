# Data Visualization DashBoard for Option Pricing

## Overview
This project is a web-based application that models stock prices using the Heston model with an Unscented Kalman Filter (UKF). It enables users to estimate model parameters, run Monte Carlo simulations, and perform sensitivity analysis. This is the final project for NYU FRE-6411 Data Visualization.
## Example:
![Dashboard_ScreenShot](https://github.com/user-attachments/assets/405d5e22-6297-43d5-9ace-8c8594ee7bed)

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


## Usage
### Running the Application
Start the Dash web application by executing:
```bash
python demo.py
```
Then, navigate to `http://127.0.0.1:5500/` in your web browser.

