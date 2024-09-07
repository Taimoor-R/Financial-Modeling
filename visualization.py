import matplotlib.pyplot as plt

def plot_predictions(stock_data, lstm_predictions, rl_predictions):
    """
    Plot actual stock prices, LSTM predictions, and reinforcement learning fine-tuned predictions.
    """
    print("Starting plot generation for stock data")
    plt.figure(figsize=(10, 6))

    # Plot each stock's data
    for ticker, data in stock_data.items():
        print(f"Plotting data for {ticker}")
        actual_prices = data['Close']
        
        # Get the LSTM predictions for the current ticker
        lstm_pred = lstm_predictions[ticker][:len(actual_prices)]  # Ensure predictions are the same length as actual data
        
        # Get the RL predictions for the current ticker
        rl_pred = [p[0] for p in rl_predictions[ticker][:len(actual_prices)]]  # Fine-tuned predictions from RL
        
        # Plot actual prices
        plt.plot(data['Date'], actual_prices, label=f"{ticker} Actual Prices")

        # Plot LSTM predictions
        plt.plot(data['Date'][-len(lstm_pred):], lstm_pred, label=f"{ticker} LSTM Predictions")

        # Plot RL predictions
        plt.plot(data['Date'][-len(rl_pred):], rl_pred, label=f"{ticker} RL Predictions")

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction: LSTM & Reinforcement Learning')
    plt.show()
    print("Plot generation complete")
