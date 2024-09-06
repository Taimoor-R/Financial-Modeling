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
ticker = 'NVDA'  # Example ticker for Nvidia
start_date = '2000-01-01'
end_date = '2023-12-31'
lstm_predictor = LSTMStockPredictor(look_back=25)
scaler = None
rl_model = None

# Fetch historical stock data for a single ticker
def fetch_one_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.dropna()  # Drop rows with missing values
    stock_data.reset_index(inplace=True)  # Reset index to make Date a column
    print(f"Fetched stock data for {ticker} with shape: {stock_data.shape}")
    return stock_data

# Train model on one stock
def train_model_on_one_stock():
    print("Training LSTM on one stock...")
    stock_data = fetch_one_stock_data(ticker, start_date, end_date)
    if 'Date' in stock_data.columns:
        stock_data.set_index('Date', inplace=True)
    lstm_predictor.train(stock_data[['Close', 'Volume']])
    print("LSTM model trained on one stock.")

# Generate predictions using LSTM model
def generate_predictions(stock_data, predictor, scaler):
    print(f"Generating predictions for stock data with shape: {stock_data.shape}")
    if 'Close' not in stock_data.columns:
        print(f"Error: 'Close' column not found in stock_data.")
        return pd.DataFrame()

    predictions = []
    for i in range(len(stock_data) - predictor.look_back):
        features = stock_data.iloc[i:i + predictor.look_back][['Close', 'Volume']]
        prediction = predictor.predict(features, scaler)
        predictions.append(prediction)
    print(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[predictor.look_back:])

# Train reinforcement learning model using LSTM predictions
def train_reinforcement_learning(stock_data):
    print("Training reinforcement learning model...")
    global lstm_predictor, scaler
    predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)
    train_reinforcement_learning_model(stock_data, lstm_predictor, scaler)
    print("Reinforcement learning model training completed.")

# Get stock data for a specific stock
def get_stock_data():
    print(f"Fetching stock data for {ticker}")
    stock_data = fetch_one_stock_data(ticker, start_date, end_date)
    print(f"Fetched stock data for {ticker}")
    return stock_data

# Define the app layout
app.layout = html.Div([
    html.H1(f'Stock Analysis for {ticker}'),
    dcc.Graph(id='stock-graph'),
    dcc.Interval(
        id='interval-component',
        interval=300000,  # Refresh interval in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    global lstm_predictor, scaler

    try:
        print(f"Update graph called with interval: {n}")
        stock_data = get_stock_data()

        if scaler is None:
            print("Scaler not found, training model on one stock")
            train_model_on_one_stock()
            scaler = lstm_predictor.train(stock_data)  # Update scaler for one stock

        predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)

        # Create traces for the graph
        stock_trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual')
        prediction_trace = go.Scatter(x=predictions_df.index, y=predictions_df['Predicted'], mode='lines', name='Predicted', line=dict(dash='dash'))

        layout = go.Layout(title=f'Stock Prices and Predictions for {ticker}', xaxis=dict(title='Date'), yaxis=dict(title='Value'))

        print("Graph updated successfully.")
        return {'data': [stock_trace, prediction_trace], 'layout': layout}
    except Exception as e:
        print(f"Error in update function: {e}")
        return {'data': [], 'layout': go.Layout(title='Error')}

if __name__ == '__main__':
    app.run_server(debug=True)
