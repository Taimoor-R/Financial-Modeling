import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_data(stock_data, predictions_df):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Price')
    plt.plot(predictions_df.index, predictions_df['Predicted'], label='Predicted Price', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price and Predictions')
    plt.legend()
    plt.show()

def plot_live_updates(stock_data):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Live Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Live Stock Price Updates')
    plt.legend()
    plt.show()
