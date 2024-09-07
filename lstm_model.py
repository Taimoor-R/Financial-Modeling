import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class LSTMStockPredictor:
    def __init__(self, look_back=5, dropout_rate=0.2, units=50):
        """
        Initialize the LSTM stock predictor with enhanced architecture.

        Args:
            look_back (int): Number of previous time steps to use for prediction.
            dropout_rate (float): Dropout rate for regularization.
            units (int): Number of units in LSTM layers.
        """
        self.look_back = look_back
        self.dropout_rate = dropout_rate
        self.units = units
        self.model = None

    def preprocess_data(self, data):
        """
        Preprocess stock data for LSTM model, including scaling.

        Args:
            data (pd.DataFrame): Stock data with at least a 'Close' column.

        Returns:
            X (np.ndarray): Input features for the model.
            y (np.ndarray): Target values for the model.
            scaler (MinMaxScaler): Fitted scaler for transforming data.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']].values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:i + self.look_back])
            y.append(scaled_data[i + self.look_back])

        return np.array(X), np.array(y), scaler

    def build_model(self):
        """
        Build and compile the enhanced LSTM model.
        """
        model = Sequential([
            Bidirectional(LSTM(units=self.units, return_sequences=True, input_shape=(self.look_back, 1))),
            Dropout(self.dropout_rate),
            LSTM(units=self.units),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, data):
        """
        Train the LSTM model on the provided data with callbacks.

        Args:
            data (pd.DataFrame): Stock data with at least a 'Close' column.

        Returns:
            scaler (MinMaxScaler): Fitted scaler for transforming data.
        """
        X, y, scaler = self.preprocess_data(data)
        self.build_model()

        # Define callbacks
        checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        self.model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, 
                       callbacks=[checkpoint, early_stopping], verbose=1)

        # Load the best model
        self.load_model('best_model.keras')

        return scaler

    def predict(self, data, scaler):
        """
        Predict the next stock price using the trained model.

        Args:
            data (pd.DataFrame): Stock data with at least a 'Close' column.
            scaler (MinMaxScaler): Scaler used to preprocess the data.

        Returns:
            float: Predicted stock price.
        """
        last_sequence = data[['Close']].values[-self.look_back:].reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = self.model.predict(last_sequence)
        return scaler.inverse_transform(prediction)[0, 0]

    def save_model(self, filename):
        """
        Save the trained model to a file.

        Args:
            filename (str): Path to the file where the model should be saved.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model:
            self.model.save(filename)
        else:
            raise ValueError("Model is not trained. Please train the model before saving.")

    def load_model(self, filename):
        """
        Load a trained model from a file.

        Args:
            filename (str): Path to the file from which the model should be loaded.
        """
        self.model = load_model(filename)
