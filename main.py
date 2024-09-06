import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from data_collector import fetch_stock_data, fetch_current_news
from lstm_model import LSTMStockPredictor

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

def fetch_top_50_stock_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        print(f"Fetching data for: {ticker}")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Debugging print statements to inspect the DataFrame
        print(f"Columns for {ticker}: {stock_data.columns}")
        
        if not stock_data.empty and 'Close' in stock_data.columns:
            print(f"Appending 'Close' column for {ticker}")
            all_data.append(stock_data[['Close']])  # No renaming, just append the 'Close' column
        else:
            print(f"Warning: 'Close' column not found for {ticker}")
    
    combined_data = pd.concat(all_data, axis=1)  # Concatenate all data along columns
    combined_data = combined_data.dropna()  # Drop rows with missing values
    print("Combined data columns:", combined_data.columns)
    return combined_data

def calculate_indicators(data):
    print(f"Calculating indicators for {ticker}")
    
    # Calculate SMA
    data['SMA'] = data['Close'].rolling(window=20).mean()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Bollinger_High'] = data['SMA_20'] + 2 * data['Close'].rolling(window=20).std()
    data['Bollinger_Low'] = data['SMA_20'] - 2 * data['Close'].rolling(window=20).std()
    
    return data

def analyze_sentiment(news):
    if not news:
        return 0  # Return neutral sentiment if no news
    sentiments = [TextBlob(item['headline']).sentiment.polarity for item in news]
    return np.mean(sentiments)  # Average sentiment score for simplicity

def generate_predictions(stock_data, predictor, scaler):
    print(f"Generating predictions for {ticker} using column: 'Close'")
    
    if 'Close' not in stock_data.columns:
        print(f"Error: 'Close' column not found in stock_data.")
        return pd.DataFrame()

    predictions = []
    for i in range(len(stock_data) - predictor.look_back):
        features = stock_data.iloc[i:i + predictor.look_back][['Close']]
        prediction = predictor.predict(features, scaler)
        predictions.append(prediction)
    return pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[predictor.look_back:])

def train_model_on_top_50_stocks():
    # Fetch data for all top 50 tickers
    combined_data = fetch_top_50_stock_data(top_50_tickers, start_date, end_date)
    
    # Train the LSTM model on the combined data (only using 'Close' prices)
    lstm_predictor.train(combined_data)

def get_stock_and_news_data():
    stock_data = fetch_top_50_stock_data([ticker], start_date, end_date)
    
    # Calculate indicators for the single stock
    stock_data = calculate_indicators(stock_data)

    # Use fetch_current_news from data_collector to get news for sentiment analysis
    news = fetch_current_news(ticker)
    news_sentiment = analyze_sentiment(news)
    
    return stock_data, news_sentiment

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
        stock_data, news_sentiment = get_stock_and_news_data()

        if scaler is None:
            train_model_on_top_50_stocks()  # Train model on top 50 stocks (2000-2023 data)
            scaler = lstm_predictor.train(stock_data)  # Update if needed

        predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)

        # Create traces for the graph
        print(f"Plotting data for {ticker}, using column: 'Close'")
        
        if 'Close' not in stock_data.columns:
            print(f"Error: 'Close' column not found in stock_data.")
            return {'data': [], 'layout': go.Layout(title='Error')}
        
        stock_trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual')
        prediction_trace = go.Scatter(x=predictions_df.index, y=predictions_df['Predicted'], mode='lines', name='Predicted', line=dict(dash='dash'))
        sma_trace = go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='SMA', line=dict(color='orange'))
        macd_trace = go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='green'))
        macd_signal_trace = go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='red'))
        bollinger_high_trace = go.Scatter(x=stock_data.index, y=stock_data['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(dash='dash', color='grey'))
        bollinger_low_trace = go.Scatter(x=stock_data.index, y=stock_data['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(dash='dash', color='grey'))

        layout = go.Layout(title=f'Stock Prices and Indicators for {ticker}', xaxis=dict(title='Date'), yaxis=dict(title='Value'))

        return {'data': [stock_trace, prediction_trace, sma_trace, macd_trace, macd_signal_trace, bollinger_high_trace, bollinger_low_trace], 'layout': layout}
    except Exception as e:
        print(f"Error in update function: {e}")
        return {'data': [], 'layout': go.Layout(title='Error')}

if __name__ == '__main__':
    app.run_server(debug=True)
