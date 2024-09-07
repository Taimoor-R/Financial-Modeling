import matplotlib.pyplot as plt

def plot_stock_predictions(actual_prices, predicted_prices, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="black", label="Actual Stock Price")
    plt.plot(predicted_prices, color="green", label="Predicted Stock Price")
    plt.title(f"{ticker} Stock Price Prediction vs Actual")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
