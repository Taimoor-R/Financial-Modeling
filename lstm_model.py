import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def build_hybrid_blstm_model(look_back=50, learning_rate=0.001):
    """
    Build an optimized hybrid model that combines multiple CNN layers and Bidirectional LSTM for stock price prediction.
    """
    print("Building optimized hybrid CNN + Bidirectional LSTM model structure")
    
    model = Sequential()

    # 1D CNN layers to extract features
    model.add(Input(shape=(look_back, 1)))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())  # Batch normalization to stabilize and speed up training
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    
    # Bidirectional LSTM layers to capture long-term dependencies
    model.add(Bidirectional(LSTM(units=256, return_sequences=True)))  # BLSTM with 256 units
    model.add(Dropout(0.4))  # Higher dropout to prevent overfitting
    model.add(Bidirectional(LSTM(units=256)))  # Another BLSTM layer
    model.add(Dropout(0.4))

    # Output layer for the predicted stock price
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error', learning_rate=learning_rate)
    
    print("Optimized hybrid CNN + BLSTM model built successfully")
    return model

def preprocess_data(stock_data, look_back=50, validation_split=0.2):
    """
    Preprocess stock data by scaling and reshaping it for the hybrid model. Split into training and validation sets.
    """
    print(f"Preprocessing stock data with look-back window of {look_back}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_scaled = scaler.fit_transform(stock_data)

    X, y = [], []
    for i in range(look_back, len(stock_scaled)):
        X.append(stock_scaled[i-look_back:i, 0])
        y.append(stock_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Split into training and validation sets
    split_idx = int(len(X) * (1 - validation_split))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    return X_train, y_train, X_val, y_val, scaler

def train_hybrid_blstm_model(model, stock_data, dates, look_back=50):
    """
    Train the optimized hybrid CNN + BLSTM model on a single stock and validate performance.
    """
    print("Starting hybrid CNN + BLSTM model training")

    X_train, y_train, X_val, y_val, scaler = preprocess_data(stock_data['Close'].values.reshape(-1, 1), look_back=look_back)

    # Reshape the data to 3D [samples, time steps, features] for CNN + BLSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    # Early stopping to prevent overfitting and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), 
              callbacks=[early_stopping, lr_scheduler])
    
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

def evaluate_hybrid_blstm_model(model, stock_data, scaler, dates, look_back=50):
    """
    Evaluate the hybrid CNN + BLSTM model on unseen stock data and return predictions.
    """
    print("Evaluating hybrid CNN + BLSTM model on test data")
    
    X_test, y_test, _, _, _ = preprocess_data(stock_data['Close'].values.reshape(-1, 1), look_back=look_back, validation_split=0)
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
