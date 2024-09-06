import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input

class LSTMStockPredictor:
    def __init__(self, look_back=25, dropout_rate=0.2, units=50):
        self.look_back = look_back
        self.dropout_rate = dropout_rate
        self.units = units
        self.model = None

    def calculate_technical_indicators(self, data):
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['RSI'] = self.calculate_rsi(data)
        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data)
        data['BB_Mid'], data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data)
        data['Momentum'] = data['Close'].diff(4)
        data = data.dropna()
        return data

    def calculate_rsi(self, data, window=14):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data):
        macd = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        return macd, macd_signal

    def calculate_bollinger_bands(self, data, window=20):
        mid = data['Close'].rolling(window=window).mean()
        std_dev = data['Close'].rolling(window=window).std()
        upper = mid + (std_dev * 2)
        lower = mid - (std_dev * 2)
        return mid, upper, lower

    def preprocess_data(self, data):
        data = self.calculate_technical_indicators(data)
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:i + self.look_back])
            y.append(scaled_data[i + self.look_back][0])

        return np.array(X), np.array(y), scaler

    def build_model(self):
        model = Sequential([
            Input(shape=(self.look_back, 12)),
            Bidirectional(LSTM(units=self.units, return_sequences=True)),
            Dropout(self.dropout_rate),
            LSTM(units=self.units),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, data):
        X, y, scaler = self.preprocess_data(data)
        self.build_model()

        checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        self.model.fit(X, y, epochs=3, batch_size=16, validation_split=0.2, callbacks=[checkpoint, early_stopping], verbose=1)

        self.load_model('best_model.keras')
        return scaler

    def predict(self, data, scaler):
        last_sequence = data[['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum']].values[-self.look_back:]
        last_sequence = scaler.transform(last_sequence)
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = self.model.predict(last_sequence)
        return scaler.inverse_transform(prediction)[0, 0]

    def save_model(self, filename):
        if self.model:
            self.model.save(filename)
        else:
            raise ValueError("Model is not trained. Please train the model before saving.")

    def load_model(self, filename):
        self.model = load_model(filename)
