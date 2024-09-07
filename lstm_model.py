import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def build_hybrid_model():
    """
    Build a hybrid model that combines 1D CNN and LSTM for stock price prediction.
    """
    print("Building hybrid 1D CNN + LSTM model structure")
    
    model = Sequential()
    
    # 1D CNN layer to extract features
    model.add(Input(shape=(25, 1)))  # Input layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM layers to capture sequential dependencies
    model.add(LSTM(units=128, return_sequences=True))  # Increased LSTM units
    model.add(Dropout(0.2))  # Dropout to reduce overfitting
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))

    # Dense layer for output
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Hybrid model built successfully")
    return model

def preprocess_data(stock_data, validation_split=0.2):
    """
    Preprocess stock data by scaling and reshaping it for the hybrid model. Split into training and validation sets.
    """
    print("Preprocessing stock data for hybrid model")
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_scaled = scaler.fit_transform(stock_data)

    X, y = [], []
    for i in range(25, len(stock_scaled)):
        X.append(stock_scaled[i-25:i, 0])
        y.append(stock_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Split into training and validation sets
    split_idx = int(len(X) * (1 - validation_split))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    return X_train, y_train, X_val, y_val, scaler

def train_hybrid_model(model, stock_data, dates):
    """
    Train the hybrid model on a single stock and validate performance.
    """
    print("Starting hybrid model training")

    X_train, y_train, X_val, y_val, scaler = preprocess_data(stock_data['Close'].values.reshape(-1, 1))

    # Reshape the data to 3D [samples, time steps, features] for 1D CNN + LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Predict on validation set
    y_val_pred = model.predict(X_val)
    
    # Inverse transform the predictions and actual values
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    y_val_pred = scaler.inverse_transform(y_val_pred)
    
    # Calculate validation metrics (MSE, MAE)
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f"Validation MSE: {mse}")
    print(f"Validation MAE: {mae}")

    # Plot predictions against actual values with dates
    plot_predictions(dates[len(dates) - len(y_val):], y_val, y_val_pred)

    return model, scaler

def plot_predictions(dates, y_actual, y_pred):
    """
    Plot the actual and predicted stock prices with dates.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_actual, label="Actual Prices", color='blue')
    plt.plot(dates, y_pred, label="Predicted Prices", color='red')
    plt.title('Stock Price Prediction: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def evaluate_hybrid_model(model, stock_data, scaler, dates):
    """
    Evaluate the hybrid model on unseen stock data and return predictions.
    """
    print("Evaluating hybrid model on test data")
    
    X_test, y_test, _, _, _ = preprocess_data(stock_data['Close'].values.reshape(-1, 1), validation_split=0)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Inverse transform the predictions and actual values
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred)

    # Calculate error metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test MAE: {mae}")

    # Plot actual vs predicted values with dates
    plot_predictions(dates, y_test, y_pred)

    return y_test, y_pred, mse, mae
