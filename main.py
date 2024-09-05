import pandas as pd
from data_collector import fetch_stock_data, fetch_current_news
from lstm_model import LSTMStockPredictor
from reinforcement_learning import train_reinforcement_learning_model, load_reinforcement_learning_model, StockTradingEnv
from visualization import plot_stock_data, plot_live_updates
import time

def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Fetch historical data
    print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(f"Fetched historical data: {stock_data.head()}")

    # Initialize and train LSTM model
    print("Initializing and training LSTM model...")
    lstm_predictor = LSTMStockPredictor(look_back=5)
    scaler = lstm_predictor.train(stock_data)
    print("LSTM model trained.")

    # Train RL model
    print("Training Reinforcement Learning model...")
    try:
        train_reinforcement_learning_model(stock_data)
    except Exception as e:
        print(f"Error training RL model: {e}")
    
    # Load the trained RL model
    print("Loading the trained RL model...")
    try:
        rl_model = load_reinforcement_learning_model("stock_trading_model")
    except Exception as e:
        print(f"Error loading RL model: {e}")
        return
    
    # Display initial stock data and predictions
    print("Generating initial predictions...")
    predictions = []
    for i in range(len(stock_data) - lstm_predictor.look_back):
        features = stock_data.iloc[i:i + lstm_predictor.look_back]
        prediction = lstm_predictor.predict(features, scaler)
        predictions.append(prediction)
    
    predictions_df = pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[lstm_predictor.look_back:])
    plot_stock_data(stock_data, predictions_df)
    print("Initial stock data and predictions plotted.")

    # Real-time updates
    print("Starting real-time updates...")
    while True:
        print("Fetching new data and news...")
        new_stock_data = fetch_stock_data(ticker, end_date, pd.Timestamp.now().strftime('%Y-%m-%d'))
        news = fetch_current_news(ticker)

        print("Latest news:")
        for item in news:
            print(f"Headline: {item['headline']}")
            print(f"Link: {item['link']}")
        
        # Predict future values
        print("Generating future predictions...")
        predictions = []
        for i in range(len(new_stock_data) - lstm_predictor.look_back):
            features = new_stock_data.iloc[i:i + lstm_predictor.look_back]
            prediction = lstm_predictor.predict(features, scaler)
            predictions.append(prediction)
        
        # Create DataFrame for predictions
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'], index=new_stock_data.index[lstm_predictor.look_back:])
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        # Plot updated data
        plot_stock_data(pd.concat([stock_data, new_stock_data]), predictions_df)
        print("Updated data plotted.")

        # Plot live updates
        plot_live_updates(new_stock_data)
        print("Live updates plotted.")

        # Use RL model for trading decision
        print("Making trading decision using RL model...")
        try:
            env = StockTradingEnv(new_stock_data)
            obs = env.reset()
            action, _ = rl_model.predict(obs)
            print(f"Recommended action: {['Buy', 'Hold', 'Sell'][action]}")
        except Exception as e:
            print(f"Error using RL model: {e}")

        # Wait for 5 minutes
        print("Waiting for 5 minutes before next update...")
        time.sleep(300)

if __name__ == "__main__":
    main()
