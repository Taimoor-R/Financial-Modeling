import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

class LSTMStockPredictor:
    def __init__(self, look_back=5):
        self.look_back = look_back
        self.model = None

    def preprocess_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:i + self.look_back])
            y.append(scaled_data[i + self.look_back])
        
        return np.array(X), np.array(y), scaler

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, data):
        X, y, scaler = self.preprocess_data(data)
        self.build_model()
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        return scaler

    def predict(self, data, scaler):
        last_sequence = data[['Close']].values[-self.look_back:].reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = self.model.predict(last_sequence)
        return scaler.inverse_transform(prediction)[0, 0]

    def save_model(self, filename):
        if self.model is not None:
            self.model.save(filename)
        else:
            raise ValueError("Model is not trained. Please train the model before saving.")

    def load_model(self, filename):
        self.model = load_model(filename)
