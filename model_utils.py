import os
import numpy as np
np.float = np.float64  # TensorFlow 2.19
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Вимкнути логування TF
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def prepare_lstm_data(data, seq_length=60):
    # Етап 1: Нормалізація зі збереженням контексту
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    # Етап 2: Формування часових вікон з ковзанням
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data_scaled[i-seq_length:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test, scaler

def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Обучаємо модель LSTM...")
    for epoch in tqdm(range(10)):
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

    return model

def predict_multiple_days(model, X_test, n_days, scaler):
    predictions = []
    last_sequence = X_test[-1]
    for _ in range(n_days):
        forecast = model.predict(last_sequence.reshape(1, last_sequence.shape[0], 1))
        forecast_price = scaler.inverse_transform(forecast)[0][0]
        predictions.append(forecast_price)
        last_sequence = np.append(last_sequence[1:], forecast_price)
    return predictions

def predict_trend(model, X_test, scaler):
    forecast = model.predict(X_test[-1].reshape(1, X_test.shape[1], 1))
    forecast_price = scaler.inverse_transform(forecast)[0][0]
    last_close = X_test[-1][-1]
    trend = "Рост" if forecast_price > last_close else "Спадіння"
    return forecast_price, trend

def save_model(model, symbol):
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{symbol}_model.h5')
    print(f"Модель збережена як models/{symbol}_model.h5")

def load_saved_model(symbol):
    model_path = f'models/{symbol}_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

