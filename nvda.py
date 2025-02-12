import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)



st.title('NVIDIA Stock Price Prediction App')
st.subheader('Upload stock data to get the predicted price')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        st.error("Error: CSV file must contain a 'Close' column.")
    else:
        df = df[['Close']]  # Keep only the 'Close' price column

        # Initialize and fit MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df)

        # Ensure we have at least 50 data points
        if len(df_scaled) < 70:
            st.error("Error: The dataset must have at least 70 closing prices.")
        else:
            # Prepare last 50 days of data for prediction
            data_70 = df_scaled[-70:].reshape(1, 70, 1)

            # Predict stock price
            prediction = loaded_model.predict(data_70)

            # Convert prediction back to original scale
            prediction_actual = scaler.inverse_transform(prediction)

            # Display the predicted price
            st.success(f'The predicted stock price for the next day is: ${prediction_actual[0][0]:.2f}')

    # Check if 'Date' column exists for further processing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        # Extract closing prices
        data = df[['Close']].values

        # Normalize data using MinMaxScaler
        data_scaled = scaler.transform(data)

        # Function to create sequences (sliding window)
        def create_sequences(data, time_steps=70):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:i + time_steps])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)

        # Create sequences for LSTM
        time_steps = 70
        X, y = create_sequences(data_scaled, time_steps)

        # Split data into training and testing sets (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        st.write(f"Training Data Size: {X_train.shape}, Testing Data Size: {X_test.shape}")