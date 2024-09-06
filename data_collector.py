import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json

# Fetch historical stock data for the ticker
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.dropna()  # Drop rows with NaN values
    return stock_data

# Fetch relevant news for a ticker and a date range
def fetch_current_news(ticker, start_date, end_date):
    # News from MarketWatch
    news_marketwatch = fetch_news_marketwatch(ticker, start_date, end_date)
    
    # News from Yahoo Finance
    news_yahoo = fetch_news_yahoo(ticker, start_date, end_date)
    
    # News from Google News (which has a better date range query)
    news_google = fetch_news_google(ticker, start_date, end_date)
    
    # Combine news from all sources
    all_news = news_marketwatch + news_yahoo + news_google
    return all_news

# Scrape news from MarketWatch (older news may require pagination)
def fetch_news_marketwatch(ticker, start_date, end_date):
    url = f'https://www.marketwatch.com/investing/stock/{ticker}/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('div', class_='article__content')

    # Scrape the headlines and links
    news = []
    for item in news_items:
        headline = item.find('a').text
        link = item.find('a')['href']
        
        # You may need to add additional logic here to filter news based on the date
        # as MarketWatch doesn't offer a date filter in its URL.
        
        news.append({'headline': headline, 'link': link})
    
    return news

# Scrape news from Yahoo Finance (older news may require pagination)
def fetch_news_yahoo(ticker, start_date, end_date):
    url = f'https://finance.yahoo.com/quote/{ticker}/news?p={ticker}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('li', class_='js-stream-content')
    
    news = []
    for item in news_items:
        headline = item.find('h3')
        link = item.find('a')
        if headline and link:
            news.append({'headline': headline.text, 'link': 'https://finance.yahoo.com' + link['href']})
    
    return news

# Use Google News to fetch articles within a specific date range
def fetch_news_google(ticker, start_date, end_date):
    # Google News query with a specific date range
    # Use the date range for better results (e.g., 2000-01-01 to 2023-12-31)
    url = f'https://news.google.com/search?q={ticker}+before:{end_date}+after:{start_date}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('article')

    news = []
    for item in news_items:
        headline = item.find('h3')
        link = item.find('a')
        if headline and link:
            news.append({'headline': headline.text, 'link': 'https://news.google.com' + link['href']})
    
    return news
