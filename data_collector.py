import pandas as pd
import yfinance as yf

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data for multiple tickers from Yahoo Finance for the given date range.
    """
    stock_data = {}
    for ticker in tickers:
        stock_df = yf.download(ticker, start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)  # Reset index to keep the date as a column
        stock_data[ticker] = stock_df
    
    return stock_data

if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'GOOG', 'MSFT']
    start = '2000-01-01'
    end = '2022-12-31'
    stock_data = get_stock_data(tickers, start, end)
    for ticker, data in stock_data.items():
        print(f"\nData for {ticker}:")
        print(data.head())
