import numpy as np
import pandas as pd
import time
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import requests
import json
from textblob import TextBlob
from binance.client import Client
from datetime import datetime, timedelta
import os

# Binance API Credentials (Replace with your own)
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
client = Client(API_KEY, API_SECRET)
MODEL_PATH = "btc_price_model.h5"


# Function to fetch historical BTC data (for training)
def fetch_historical_btc_data():
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2016", "31 Jan, 2025")
    btc_data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_asset_volume', 'number_of_trades',
                                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                             'ignore'])
    btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='ms')
    btc_data = btc_data[['timestamp', 'close']]
    btc_data.set_index('timestamp', inplace=True)
    btc_data = btc_data.astype(float)
    return btc_data


# Function to fetch live BTC data (last 60 timestamps)
def fetch_live_btc_data():
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_4HOUR, "10 days ago UTC")
    btc_data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_asset_volume', 'number_of_trades',
                                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                             'ignore'])
    btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='ms')
    btc_data = btc_data[['timestamp', 'close']]
    btc_data.set_index('timestamp', inplace=True)
    btc_data = btc_data.astype(float)
    return btc_data.tail(60)


# Function to fetch live sentiment data
def fetch_live_sentiment_data():
    api_key = '194deffb4b18400eb5f247d702bb93e7'  # Replace with your API key
    url = f'https://newsapi.org/v2/everything?q=bitcoin&apiKey={api_key}'

    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data:", response.status_code)
        return None  # Handle API errors gracefully

    data = json.loads(response.text)

    if 'articles' not in data or not data['articles']:
        print("No articles found.")
        return None  # Handle missing articles

    sentiment_scores = []
    for article in data['articles']:
        title = article.get('title', '')  # Use an empty string if title is missing
        description = article.get('description', '')  # Use an empty string if description is missing

        text = f"{title}. {description}"
        sentiment = TextBlob(text).sentiment.polarity  # Returns value between -1 (negative) and +1 (positive)
        sentiment_scores.append(sentiment)

    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0  # Avoid error on empty list
    return avg_sentiment


# Function to preprocess data
def preprocess_data(btc_data, sentiment_data, scaler):
    scaled_btc_data = scaler.transform(btc_data)
    combined_data = np.column_stack((scaled_btc_data, np.full(len(btc_data), sentiment_data)))
    X_live = np.array([combined_data])
    return X_live


# Function to build and train the model
def build_and_train_model():
    btc_data = fetch_historical_btc_data()
    sentiment_data = np.zeros(len(btc_data))  # Assuming neutral sentiment (0.0) for historical data
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_data_scaled = scaler.fit_transform(btc_data)
    combined_data = np.column_stack((btc_data_scaled, sentiment_data))
    X_train, y_train = [], []
    for i in range(60, len(combined_data)):
        X_train.append(combined_data[i - 60:i, :])
        y_train.append(combined_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=20, batch_size=32)
    model.save(MODEL_PATH)
    return model, scaler


# Load or train model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_data = fetch_historical_btc_data()
    scaler.fit(btc_data)
else:
    model, scaler = build_and_train_model()

# Live Prediction Loop
while True:
    print("\nFetching latest BTC data and making predictions...")
    btc_data_live = fetch_live_btc_data()
    sentiment_live = fetch_live_sentiment_data()
    X_live = preprocess_data(btc_data_live, sentiment_live, scaler)

    future_prices = []
    for _ in range(3):
        predicted_price = model.predict(X_live)
        predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))[0][0]
        future_prices.append(predicted_price)
        new_entry = np.array([[predicted_price, sentiment_live]])
        X_live = np.append(X_live[:, 1:, :], new_entry.reshape(1, 1, 2), axis=1)

    print("Predicted BTC prices for the next 12 hours:")
    for i, price in enumerate(future_prices):
        future_time = datetime.now() + timedelta(hours=(i + 1) * 4)
        print(f"{future_time.strftime('%Y-%m-%d %H:%M:%S')} - ${price:.2f}")

    print("\nWaiting for the next update...\n")
    time.sleep(4 * 60 * 60)
