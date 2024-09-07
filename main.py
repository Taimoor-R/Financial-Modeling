from data_collector import get_stock_data
from lstm_model import build_hybrid_model, train_hybrid_model, evaluate_hybrid_model

def main(ticker='AAPL'):
    print(f"Starting data collection for ticker: {ticker}")
    
    # 1. Data Collection for a single stock
    stock_data = get_stock_data([ticker], '2000-01-01', '2022-12-31')  # Data up to 2022 for training
    dates = stock_data[ticker]['Date'].values  # Extract dates for visualization
    print("Collected historical stock data (up to 2022)")

    # 2. Build and train the hybrid model (1D CNN + LSTM)
    print("Building hybrid model")
    hybrid_model = build_hybrid_model()

    print("Training hybrid model")
    trained_model, scaler = train_hybrid_model(hybrid_model, stock_data[ticker], dates)

    # 3. Evaluate the model on unseen test data (e.g., stock data from 2022 onwards)
    print("Evaluating hybrid model on recent data")
    stock_data_recent = get_stock_data([ticker], '2022-01-01', '2023-12-31')
    recent_dates = stock_data_recent[ticker]['Date'].values
    evaluate_hybrid_model(trained_model, stock_data_recent[ticker], scaler, recent_dates)

if __name__ == "__main__":
    print("Starting main program")
    main(ticker='AAPL')  # Example: Apple
    print("Main program finished")
