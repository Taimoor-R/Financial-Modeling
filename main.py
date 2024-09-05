import pandas as pd
from data_collector import fetch_stock_data, fetch_current_news
from lstm_model import LSTMStockPredictor
from reinforcement_learning import train_reinforcement_learning_model, load_reinforcement_learning_model
from visualization import plot_stock_data, plot_live_updates
import time

def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Fetch historical data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

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
        
        print("Latest news:")
        for item in news:
            print(f"Headline: {item['headline']}")
            print(f"Link: {item['link']}")
        
        # Predict future values
        predictions = []
        for i in range(len(new_stock_data) - lstm_predictor.look_back):
            features = new_stock_data.iloc[i:i + lstm_predictor.look_back]
            prediction = lstm_predictor.predict(features, scaler)
            predictions.append(prediction)
        
        # Create DataFrame for predictions
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'], index=new_stock_data.index[lstm_predictor.look_back:])

        # Update end_date
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        # Plot updated data
        plot_stock_data(pd.concat([stock_data, new_stock_data]), predictions_df)

        # Plot live updates
        plot_live_updates(new_stock_data)

        # Use RL model for trading decision
        env = StockTradingEnv(new_stock_data)
        obs = env.reset()
        action, _ = rl_model.predict(obs)
        print(f"Recommended action: {['Buy', 'Hold', 'Sell'][action]}")

        # Wait for 5 minutes
        time.sleep(300)

if __name__ == "__main__":
    main()
