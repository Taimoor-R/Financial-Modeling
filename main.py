import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from lstm_model import LSTMStockPredictor
from reinforcement_learning import train_reinforcement_learning_model

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables
ticker = 'NVDA'
start_date = '2000-01-01'
end_date = '2023-12-31'
lstm_predictor = LSTMStockPredictor(look_back=25)
scaler = None

# Fetch historical stock data for a single ticker
def fetch_one_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.dropna()  # Drop rows with missing values
    stock_data.reset_index(inplace=True)
    return stock_data

# Add technical indicators to the stock data
def add_technical_indicators(stock_data):
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
    stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
    
    delta = stock_data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    mid = stock_data['Close'].rolling(window=20).mean()
    std_dev = stock_data['Close'].rolling(window=20).std()
    stock_data['BB_Upper'] = mid + (std_dev * 2)
    stock_data['BB_Lower'] = mid - (std_dev * 2)

    stock_data['Momentum'] = stock_data['Close'].diff(4)

    stock_data = stock_data.dropna()
    return stock_data

# Train model on one stock
def train_model_on_one_stock():
    stock_data = fetch_one_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)
    if 'Date' in stock_data.columns:
        stock_data.set_index('Date', inplace=True)
    lstm_predictor.train(stock_data[['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum']])

# Define the app layout and graph update logic
app.layout = html.Div([
    html.H1(f'Stock Analysis for {ticker}'),
    dcc.Graph(id='stock-graph'),
    dcc.Interval(id='interval-component', interval=300000, n_intervals=0)
])

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    global lstm_predictor, scaler

    stock_data = fetch_one_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)

    if scaler is None:
        train_model_on_one_stock()
        scaler = lstm_predictor.scaler

    predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)

    stock_trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual')
    prediction_trace = go.Scatter(x=predictions_df.index, y=predictions_df['Predicted'], mode='lines', name='Predicted', line=dict(dash='dash'))

    layout = go.Layout(title=f'Stock Prices and Predictions for {ticker}', xaxis=dict(title='Date'), yaxis=dict(title='Value'))

    return {'data': [stock_trace, prediction_trace], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
