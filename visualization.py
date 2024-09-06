import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_data(data, predictions=None):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Actual')
    if predictions is not None:
        plt.plot(predictions.index, predictions['Predicted'], label='Predicted', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Prices')
    plt.legend()
    plt.show()

def plot_live_updates(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Live Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Live Stock Data')
    plt.legend()
    plt.show()
