import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Input, concatenate, RepeatVector, Lambda
from keras.models import Model
from keras.losses import Huber
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# List of stocks to train on
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Add more stocks as needed

# Step 1: Download Stock Data for Multiple Stocks and Find Common Dates
def download_and_align_stocks(stocks, start="2000-01-01", end="2022-12-31"):
    stock_data = {}
    common_dates = None

    # Download data for each stock
    for stock in stocks:
        df = yf.download(stock, start=start, end=end)
        df['Ticker'] = stock  # Add stock identifier
        stock_data[stock] = df['Close']  # Store Close price

        # Find the intersection of dates across all stocks
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    # Align all stocks on the common dates
    common_dates = sorted(list(common_dates))
    aligned_data = pd.DataFrame(index=common_dates)

    for stock in stocks:
        aligned_data[stock] = stock_data[stock].loc[common_dates]

    return aligned_data

# Step 2: Add Technical Indicators (SMA, RSI, etc.)
def add_technical_indicators(df, stock):
    df[f'SMA_50_{stock}'] = ta.sma(df[stock], length=50)
    df[f'SMA_200_{stock}'] = ta.sma(df[stock], length=200)
    df[f'RSI_{stock}'] = ta.rsi(df[stock], length=14)
    return df

# Step 3: Create Sequences for LSTM with Stock Embeddings
def create_sequences_with_embedding(df, sequence_length, stock_to_id):
    x_data, stock_ids, y_data = [], [], []
    
    for stock in stocks:
        stock_data = df[stock].dropna().values
        stock_id = stock_to_id[stock]
        
        for i in range(sequence_length, len(stock_data)):
            x_data.append(stock_data[i-sequence_length:i])  # Price sequence
            stock_ids.append(stock_id)  # Stock ID
            y_data.append(stock_data[i])  # Next price (target)
    
    x_data = np.array(x_data)
    stock_ids = np.array(stock_ids)
    y_data = np.array(y_data)
    
    return x_data, stock_ids, y_data

# Step 4: Build LSTM with Stock Embedding Layer (Using Huber Loss and Custom Learning Rate)
def build_model_with_embedding(sequence_length, stock_count, embedding_dim=10):
    # Input for stock price sequences
    price_input = Input(shape=(sequence_length, 1))
    
    # Input for stock IDs
    stock_id_input = Input(shape=(1,))
    
    # Embedding layer for stock IDs
    stock_embedding = Embedding(input_dim=stock_count, output_dim=embedding_dim)(stock_id_input)
    
    # Remove the extra dimension using Lambda layer to handle KerasTensor
    stock_embedding = Lambda(lambda x: tf.squeeze(x, axis=1))(stock_embedding)
    
    # Repeat embedding for each time step
    stock_embedding = RepeatVector(sequence_length)(stock_embedding)
    
    # Concatenate stock price and embedding
    merged = concatenate([price_input, stock_embedding])
    
    # LSTM layers
    lstm_output = LSTM(128, return_sequences=False)(merged)
    output = Dense(1)(lstm_output)  # Predict next price
    
    # Compile the model with Huber loss and custom learning rate (0.0001)
    model = Model(inputs=[price_input, stock_id_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(delta=1))  # Set learning rate to 0.0001
    
    return model

# Step 5: Test the model on unseen data
def test_model(model, scaler, test_stock_data, sequence_length, stock_id, actual_prices):
    # Ensure the test data is in the right format (NumPy array)
    test_stock_data = test_stock_data.reshape(-1, 1)  # Reshape for scaler

    # Scale test data using the same scaler from training
    scaled_test_data = scaler.transform(test_stock_data)
    
    # Create sequences for testing
    x_test = []
    for i in range(sequence_length, len(scaled_test_data)):
        x_test.append(scaled_test_data[i-sequence_length:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], sequence_length, 1))
    
    # Create stock ID array for testing
    stock_ids_test = np.array([stock_id] * len(x_test))
    
    # Predict future prices
    predicted_prices_scaled = model.predict([x_test, stock_ids_test])
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
    
    # Calculate MSE (Mean Squared Error)
    mse = mean_squared_error(actual_prices[-len(predicted_prices):], predicted_prices)
    print(f"Test MSE: {mse}")

    # Plot the predictions vs actual prices
    plt.figure(figsize=(14, 8))
    plt.plot(actual_prices[-len(predicted_prices):], label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='green')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Main Workflow
def main():
    # Download multiple stock data and align dates based on common dates
    df = download_and_align_stocks(stocks)
    
    # Assign unique stock IDs
    stock_to_id = {stock: idx for idx, stock in enumerate(stocks)}

    # Add technical indicators for each stock
    for stock in stocks:
        df = add_technical_indicators(df, stock)

    # Sequence length
    sequence_length = 50
    
    # Create sequences with stock embeddings for training
    x_data, stock_ids_data, y_data = create_sequences_with_embedding(df, sequence_length, stock_to_id)
    
    # Reshape x_data for LSTM input
    x_data = x_data.reshape((x_data.shape[0], sequence_length, 1))
    
    # Scale the data
    scaler = MinMaxScaler()
    x_data_scaled = scaler.fit_transform(x_data.reshape(-1, 1)).reshape(x_data.shape)
    
    # Build and train the LSTM model
    model = build_model_with_embedding(sequence_length, stock_count=len(stocks))
    
    # Implement ReduceLROnPlateau to reduce learning rate when the model performance plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)

    # Train the model with ReduceLROnPlateau callback
    model.fit([x_data_scaled, stock_ids_data], y_data, epochs=200, batch_size=16, validation_split=0.3, callbacks=[reduce_lr])

    # Step 6: Test the model on new data (e.g., Microsoft (MSFT) data from 2023-2024)
    test_stock = 'MSFT'
    test_stock_id = stock_to_id[test_stock]
    
    # Download test data for 2023-2024
    test_df = yf.download(test_stock, start="2023-01-01", end="2024-12-31")
    actual_prices = test_df['Close'].values.reshape(-1, 1)
    
    # Test the model on Microsoft stock data
    test_model(model, scaler, actual_prices, sequence_length, test_stock_id, actual_prices)

if __name__ == "__main__":
    main()
