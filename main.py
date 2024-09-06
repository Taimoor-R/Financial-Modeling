import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from lstm_model import LSTMStockPredictor
from reinforcement_learning import train_reinforcement_learning_model, StockTradingEnv

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables
ticker = 'NVDA'  # Example ticker for Nvidia
start_date = '2000-01-01'
end_date = '2023-12-31'
top_50_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'BABA', 'JPM', 'V',
                  'NFLX', 'DIS', 'PYPL', 'BAC', 'ADBE', 'INTC', 'CSCO', 'AMD', 'PEP', 'KO',
                  'XOM', 'PFE', 'MRNA', 'NKE', 'WMT', 'HD', 'PG', 'UNH', 'CRM', 'QCOM',
                  'T', 'CVX', 'COST', 'SBUX', 'SPGI', 'IBM', 'LLY', 'GE', 'MO', 'MDT',
                  'HON', 'TXN', 'BA', 'MMM', 'LOW', 'CAT', 'F', 'GM', 'BMY']
lstm_predictor = LSTMStockPredictor(look_back=5)
scaler = None
rl_model = None

# Fetch historical stock data for top 50 tickers
def fetch_top_50_stock_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        print(f"Fetching data for: {ticker}")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if not stock_data.empty:
            stock_data['Ticker'] = ticker  # Add a ticker column
            stock_data.reset_index(inplace=True)  # Reset the index to make Date a column
            print(f"Columns for {ticker}: {stock_data.columns}")  # Debugging statement
            all_data.append(stock_data[['Date', 'Close', 'Volume', 'Ticker']])
        else:
            print(f"Warning: Data not found for {ticker}")

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.dropna()  # Drop rows with missing values

    print(f"Combined data columns: {combined_data.columns}")
    
    return combined_data

# Train model on top 50 stock data
def train_model_on_top_50_stocks():
    print("Starting to fetch and train model on top 50 stocks...")
    stock_data = fetch_top_50_stock_data(top_50_tickers, start_date, end_date)
    combined_data = pd.concat(stock_data, ignore_index=True)

    print(f"Combined data after concatenation columns: {combined_data.columns}")

    # Set Date as the index if it's available
    if 'Date' in combined_data.columns:
        combined_data.set_index('Date', inplace=True)
    else:
        print("Error: 'Date' column is missing after concatenation")

    print("Training LSTM model...")
    lstm_predictor.train(combined_data[['Close', 'Volume']])
    print("LSTM model training completed.")

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
    stock_data = fetch_top_50_stock_data([ticker], start_date, end_date)

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
            print("Scaler not found, training model on top 50 stocks")
            train_model_on_top_50_stocks()  # Train model on top 50 stocks (2000-2023 data)
            scaler = lstm_predictor.train(stock_data)  # Update if needed

        predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)

        # After generating LSTM predictions, train reinforcement learning model
        train_reinforcement_learning(stock_data)

        # Create traces for the graph
        stock_trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual')
        prediction_trace = go.Scatter(x=predictions_df.index, y=predictions_df['Predicted'], mode='lines', name='Predicted', line=dict(dash='dash'))

        layout = go.Layout(title=f'Stock Prices and Indicators for {ticker}', xaxis=dict(title='Date'), yaxis=dict(title='Value'))

        print("Graph updated successfully.")
        return {'data': [stock_trace, prediction_trace], 'layout': layout}
    except Exception as e:
        print(f"Error in update function: {e}")
        return {'data': [], 'layout': go.Layout(title='Error')}

if __name__ == '__main__':
    app.run_server(debug=True)
