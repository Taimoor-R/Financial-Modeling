import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from lstm_model import LSTMStockPredictor
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables
ticker = 'NVDA'  # Example ticker for Nvidia
start_date = '2000-01-01'
train_end_date = '2023-12-31'
prediction_start_date = '2024-01-01'
prediction_end_date = '2024-12-31'
lstm_predictor = LSTMStockPredictor(look_back=25)
scaler = None

# Fetch stock data till 2023 for training
def fetch_training_data(ticker, start_date, train_end_date):
    stock_data = yf.download(ticker, start=start_date, end=train_end_date)
    stock_data = stock_data.dropna()
    return stock_data

# Fetch stock data for 2024 for prediction and actual comparison
def fetch_2024_data(ticker, prediction_start_date, prediction_end_date):
    stock_data_2024 = yf.download(ticker, start=prediction_start_date, end=prediction_end_date)
    stock_data_2024 = stock_data_2024.dropna()
    return stock_data_2024

# Generate training data graph (till 2023)
def generate_training_data_graph(stock_data_till_2023):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data_till_2023.index, stock_data_till_2023['Close'], label="Actual (Till 2023)", color='blue')
    plt.title(f"Stock Price of {ticker} (2000-2023)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_data_graph.png")
    plt.close()

# Generate predictions vs actual for 2024 graph
def generate_predictions_vs_actual_graph(actual_data_2024, predicted_data_2024):
    plt.figure(figsize=(14, 7))
    plt.plot(actual_data_2024.index, actual_data_2024['Close'], label="Actual (2024)", color='green')
    plt.plot(predicted_data_2024.index, predicted_data_2024['Close'], label="Predicted (2024)", linestyle="--", color='red')
    plt.title(f"Stock Price Predictions vs Actual for {ticker} (2024)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("predictions_vs_actual_2024.png")
    plt.close()

# Main function to train the model and predict for 2024
def train_and_predict(ticker):
    # Fetch and plot training data (till 2023)
    print("Fetching training data...")
    stock_data_till_2023 = fetch_training_data(ticker, start_date, train_end_date)
    print("Generating training data graph...")
    generate_training_data_graph(stock_data_till_2023)
    
    # Simulate training process (replace with real LSTM training and predictions)
    print("Simulating predictions for 2024...")
    actual_stock_data_2024 = fetch_2024_data(ticker, prediction_start_date, prediction_end_date)
    
    # Mock predictions for 2024
    predicted_stock_data_2024 = actual_stock_data_2024.copy()
    predicted_stock_data_2024["Close"] = actual_stock_data_2024["Close"] * 1.02  # Simulated prediction

    print("Generating predictions vs actual graph...")
    generate_predictions_vs_actual_graph(actual_stock_data_2024, predicted_stock_data_2024)
    
    return stock_data_till_2023, actual_stock_data_2024, predicted_stock_data_2024

# Dash layout
app.layout = html.Div([
    html.H1(f'Stock Analysis for {ticker}'),
    dcc.Graph(id='training-graph'),
    dcc.Graph(id='prediction-graph'),
    dcc.Interval(
        id='interval-component',
        interval=300000,  # Refresh interval in milliseconds
        n_intervals=0
    )
])

# Update both graphs on refresh
@app.callback(
    [Output('training-graph', 'figure'), Output('prediction-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    print("Fetching and training on stock data...")
    
    stock_data_till_2023, actual_stock_data_2024, predicted_stock_data_2024 = train_and_predict(ticker)
    
    # Create training graph
    training_trace = go.Scatter(x=stock_data_till_2023.index, y=stock_data_till_2023['Close'], mode='lines', name='Actual (Till 2023)', line=dict(color='blue'))

    # Create predictions vs actual graph
    actual_trace = go.Scatter(x=actual_stock_data_2024.index, y=actual_stock_data_2024['Close'], mode='lines', name='Actual (2024)', line=dict(color='green'))
    predicted_trace = go.Scatter(x=predicted_stock_data_2024.index, y=predicted_stock_data_2024['Close'], mode='lines', name='Predicted (2024)', line=dict(dash='dash', color='red'))

    # Layouts
    layout_training = go.Layout(title=f"Stock Price of {ticker} (2000-2023)", xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    layout_predictions = go.Layout(title=f"Predictions vs Actual for {ticker} (2024)", xaxis=dict(title='Date'), yaxis=dict(title='Price'))

    training_figure = {'data': [training_trace], 'layout': layout_training}
    prediction_figure = {'data': [actual_trace, predicted_trace], 'layout': layout_predictions}

    return training_figure, prediction_figure

if __name__ == '__main__':
    app.run_server(debug=True)
