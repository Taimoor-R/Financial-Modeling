import pandas as pd
from data_collector import fetch_stock_data, fetch_current_news
import time

def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Fetch historical data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

if __name__ == "__main__":
    main()