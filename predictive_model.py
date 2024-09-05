import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, data):
        data = data.copy()
        data['Target'] = data['Close'].shift(-1)  # Predict next day's closing price
        data = data.dropna()
        X = data[['Open', 'High', 'Low', 'Volume']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def predict(self, features):
        return self.model.predict([features])[0]

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)
