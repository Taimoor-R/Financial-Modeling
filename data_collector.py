import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_current_news(ticker):
    # News from MarketWatch
    news_marketwatch = fetch_news_marketwatch(ticker)
    
    # News from Yahoo Finance
    news_yahoo = fetch_news_yahoo(ticker)
    
    # Combine news from both sources
    all_news = news_marketwatch + news_yahoo
    return all_news

def fetch_news_marketwatch(ticker):
    url = f'https://www.marketwatch.com/investing/stock/{ticker}/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('div', class_='article__content')
    news = [{'headline': item.find('a').text, 'link': item.find('a')['href']} for item in news_items]
    return news

def fetch_news_yahoo(ticker):
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

def fetch_news_google(ticker):
    url = f'https://news.google.com/search?q={ticker}'
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
