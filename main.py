import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from lstm_model import LSTMStockPredictor
# from reinforcement_learning import train_reinforcement_learning_model  # Commented for testing

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
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.dropna()  # Drop rows with missing values
    stock_data.reset_index(inplace=True)
    print(f"Fetched stock data for {ticker} with shape: {stock_data.shape}")
    return stock_data

# Add technical indicators to the stock data
def add_technical_indicators(stock_data):
    print(f"Adding technical indicators to stock data with shape: {stock_data.shape}")
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
    print(f"Technical indicators added. Final shape: {stock_data.shape}")
    return stock_data

# Train model on one stock
def train_model_on_one_stock():
    print("Starting to train LSTM model on one stock...")
    stock_data = fetch_one_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)
    
    print(f"Stock data with technical indicators shape: {stock_data.shape}")
    if 'Date' in stock_data.columns:
        stock_data.set_index('Date', inplace=True)
        print(f"'Date' column set as index. Data now has shape: {stock_data.shape}")
        
    lstm_predictor.train(stock_data[['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum']])
    print("LSTM model successfully trained on one stock.")

# Generate predictions using LSTM model
def generate_predictions(stock_data, predictor, scaler):
    print(f"Generating predictions for stock data with shape: {stock_data.shape}")
    if 'Close' not in stock_data.columns:
        print(f"Error: 'Close' column not found in stock_data.")
        return pd.DataFrame()

    predictions = []
    for i in range(len(stock_data) - predictor.look_back):
        features = stock_data.iloc[i:i + predictor.look_back][['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum']]
        prediction = predictor.predict(features, scaler)
        predictions.append(prediction)
    
    print(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[predictor.look_back:])

# Get stock data for a specific stock
def get_stock_data():
    print(f"Fetching stock data for {ticker}")
    stock_data = fetch_one_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)
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
            scaler = lstm_predictor.scaler  # Get the scaler from the LSTM predictor
        
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
    print("Starting Dash app...")
    app.run_server(debug=True)
