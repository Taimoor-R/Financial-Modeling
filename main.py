import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from data_collector import fetch_current_news
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

        # Ensure Date is part of the data (usually Date is the index from yfinance)
        if not stock_data.empty:
            stock_data['Ticker'] = ticker  # Add a ticker column
            stock_data.reset_index(inplace=True)  # Reset the index to make Date a column
            print(f"Columns for {ticker}: {stock_data.columns}")  # Debugging statement
            all_data.append(stock_data[['Date', 'Close', 'Volume', 'Ticker']])
        else:
            print(f"Warning: Data not found for {ticker}")

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.dropna()  # Drop rows with missing values

    # Debugging: Check if 'Date' is in the columns
    print(f"Combined data columns: {combined_data.columns}")
    
    return combined_data

# Fetch sentiment and ensure Date is handled correctly
def fetch_and_analyze_news(ticker, start_date, end_date):
    news_data = fetch_current_news(ticker, start_date, end_date)
    if news_data:
        news_sentiments = [TextBlob(item['headline']).sentiment.polarity for item in news_data]
        avg_sentiment = np.mean(news_sentiments)  # Average sentiment score
    else:
        avg_sentiment = 0  # Neutral sentiment if no news
    return avg_sentiment

# Train model on top 50 stock data including sentiment
def train_model_on_top_50_stocks():
    stock_data = fetch_top_50_stock_data(top_50_tickers, start_date, end_date)
    combined_data = []
    
    for ticker in top_50_tickers:
        print(f"Fetching sentiment data for {ticker}")
        avg_sentiment = fetch_and_analyze_news(ticker, start_date, end_date)
        stock_data_ticker = stock_data[stock_data['Ticker'] == ticker].copy()
        stock_data_ticker['Sentiment'] = avg_sentiment  # Add sentiment as a feature
        
        # Debugging: Check if 'Date' is present in the ticker data
        print(f"Ticker data columns: {stock_data_ticker.columns}")
        
        combined_data.append(stock_data_ticker)

    combined_data = pd.concat(combined_data, ignore_index=True)

    # Debugging: Check if 'Date' is present in the combined data
    print(f"Combined data after concatenation columns: {combined_data.columns}")

    # Set Date as the index if it's available
    if 'Date' in combined_data.columns:
        combined_data.set_index('Date', inplace=True)
    else:
        print("Error: 'Date' column is missing after concatenation")

    # Train LSTM model with stock data and sentiment
    lstm_predictor.train(combined_data[['Close', 'Volume', 'Sentiment']])

# Generate predictions using LSTM model
def generate_predictions(stock_data, predictor, scaler):
    if 'Close' not in stock_data.columns:
        print(f"Error: 'Close' column not found in stock_data.")
        return pd.DataFrame()

    predictions = []
    for i in range(len(stock_data) - predictor.look_back):
        features = stock_data.iloc[i:i + predictor.look_back][['Close', 'Volume', 'Sentiment']]
        prediction = predictor.predict(features, scaler)
        predictions.append(prediction)
    return pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[predictor.look_back:])

# Train reinforcement learning model using LSTM predictions
def train_reinforcement_learning(stock_data, sentiment_scores):
    global lstm_predictor, scaler
    predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)
    train_reinforcement_learning_model(stock_data, sentiment_scores, lstm_predictor, scaler)

# Get stock data and sentiment for a specific stock
def get_stock_and_news_data():
    stock_data = fetch_top_50_stock_data([ticker], start_date, end_date)

    # Fetch sentiment data for the ticker
    avg_sentiment = fetch_and_analyze_news(ticker, start_date, end_date)

    # Add sentiment as a feature
    stock_data['Sentiment'] = avg_sentiment

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
        stock_data = get_stock_and_news_data()

        if scaler is None:
            train_model_on_top_50_stocks()  # Train model on top 50 stocks (2000-2023 data)
            scaler = lstm_predictor.train(stock_data)  # Update if needed

        predictions_df = generate_predictions(stock_data, lstm_predictor, scaler)

        # After generating LSTM predictions, train reinforcement learning model
        sentiment_scores = stock_data['Sentiment'].values
        train_reinforcement_learning(stock_data, sentiment_scores)

        # Create traces for the graph
        stock_trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual')
        prediction_trace = go.Scatter(x=predictions_df.index, y=predictions_df['Predicted'], mode='lines', name='Predicted', line=dict(dash='dash'))

        layout = go.Layout(title=f'Stock Prices and Indicators for {ticker}', xaxis=dict(title='Date'), yaxis=dict(title='Value'))

        return {'data': [stock_trace, prediction_trace], 'layout': layout}
    except Exception as e:
        print(f"Error in update function: {e}")
        return {'data': [], 'layout': go.Layout(title='Error')}

if __name__ == '__main__':
    app.run_server(debug=True)