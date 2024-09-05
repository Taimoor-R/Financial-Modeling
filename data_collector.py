import yfinance as yf
import requests
from bs4 import BeautifulSoup

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_current_news(ticker):
    url = f'https://www.marketwatch.com/investing/stock/{ticker}/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('div', class_='article__content')
    news = [{'headline': item.find('a').text, 'link': item.find('a')['href']} for item in news_items]
    return news
