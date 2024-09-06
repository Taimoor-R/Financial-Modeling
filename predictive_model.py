import pandas as pd
from data_collector import fetch_stock_data, fetch_current_news
from lstm_model import LSTMStockPredictor
from reinforcement_learning import train_reinforcement_learning_model, load_reinforcement_learning_model
from visualization import plot_stock_data, plot_live_updates
import time

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Fetch historical data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Calculate indicators
    stock_data['SMA'] = calculate_sma(stock_data, 50)
    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data['MACD'], stock_data['MACD_Signal'] = calculate_macd(stock_data)
    stock_data['Upper_Band'], stock_data['Lower_Band'] = calculate_bollinger_bands(stock_data)

    # Initialize and train LSTM model
    lstm_predictor = LSTMStockPredictor(look_back=5)
    scaler = lstm_predictor.train(stock_data)

    # Train RL model
    train_reinforcement_learning_model(stock_data)

    # Load the trained RL model
    rl_model = load_reinforcement_learning_model("stock_trading_model")

    # Display initial stock data and predictions
    predictions = []
    for i in range(len(stock_data) - lstm_predictor.look_back):
        features = stock_data.iloc[i:i + lstm_predictor.look_back]
        prediction = lstm_predictor.predict(features, scaler)
        predictions.append(prediction)
    
    predictions_df = pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[lstm_predictor.look_back:])
    plot_stock_data(stock_data, predictions_df)

    # Real-time updates
    while True:
        # Fetch new data and news
        new_stock_data = fetch_stock_data(ticker, end_date, pd.Timestamp.now().strftime('%Y-%m-%d'))
        news = fetch_current_news(ticker)
        
       
