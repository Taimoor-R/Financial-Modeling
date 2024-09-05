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

    print("Fetching historical data...")
    # Fetch historical data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(f"Fetched historical data for {ticker} from {start_date} to {end_date}")

    print("Initializing and training LSTM model...")
    # Initialize and train LSTM model
    lstm_predictor = LSTMStockPredictor(look_back=5)
    scaler = lstm_predictor.train(stock_data)
    print("LSTM model trained.")

    # Train RL model
    print("Training Reinforcement Learning model...")
    train_reinforcement_learning_model(stock_data)
    print("Reinforcement Learning model trained.")

    # Load the trained RL model
    print("Loading Reinforcement Learning model...")
    rl_model = load_reinforcement_learning_model("stock_trading_model")
    print("Reinforcement Learning model loaded.")

    # Display initial stock data and predictions
    print("Generating initial predictions...")
    predictions = []
    for i in range(len(stock_data) - lstm_predictor.look_back):
        features = stock_data.iloc[i:i + lstm_predictor.look_back]
        prediction = lstm_predictor.predict(features, scaler)
        predictions.append(prediction)
    
    predictions_df = pd.DataFrame(predictions, columns=['Predicted'], index=stock_data.index[lstm_predictor.look_back:])
    print("Initial predictions generated.")
    
    plot_stock_data(stock_data, predictions_df)

    # Real-time updates
    while True:
        print("Fetching new stock data and news...")
        # Fetch new data and news
        new_stock_data = fetch_stock_data(ticker, end_date, pd.Timestamp.now().strftime('%Y-%m-%d'))
        news = fetch_current_news(ticker)
        
        print("Latest news:")
        for item in news:
            print(f"Headline: {item['headline']}")
            print(f"Link: {item['link']}")
        
        # Predict future values
        print("Predicting future values...")
        predictions = []
        for i in range(len(new_stock_data) - lstm_predictor.look_back):
            features = new_stock_data.iloc[i:i + lstm_predictor.look_back]
            prediction = lstm_predictor.predict(features, scaler)
            predictions.append(prediction)
        
        # Create DataFrame for predictions
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'], index=new_stock_data.index[lstm_predictor.look_back:])
        print("Future values predicted.")
        
        # Update end_date
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        # Plot updated data
        print("Plotting updated stock data...")
        plot_stock_data(pd.concat([stock_data, new_stock_data]), predictions_df)
        print("Updated stock data plotted.")

        # Plot live updates
        print("Plotting live updates...")
        plot_live_updates(new_stock_data)
        print("Live updates plotted.")

        # Use RL model for trading decision
        print("Getting RL model action recommendation...")
        env = StockTradingEnv(new_stock_data)
        obs = env.reset()
        action, _ = rl_model.predict(obs)
        print(f"Recommended action: {['Buy', 'Hold', 'Sell'][action]}")

        # Wait for 5 minutes
        print("Waiting for 5 minutes before the next update...")
        time.sleep(300)

if __name__ == "__main__":
    main()
