import yfinance as yf
import pandas as pd
import numpy as np
import talib
import os
import pickle

def get_data(symbol, start_date, end_date, cache_dir='data_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{symbol}_{start_date}_{end_date}.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if not data.empty:
        with open(cache_file, 'wb') as f:
            pickle.dump(data[['Close', 'High', 'Low']], f)
    
    return data[['Close', 'High', 'Low']] if not data.empty else None

def compute_indicators(data):
    
    try:
        close_prices = data['Close'].values.astype('float64')
        high_prices = data['High'].values.astype('float64')
        low_prices = data['Low'].values.astype('float64')

        if len(close_prices) < 50:
            print(" Недостатньо данних для розрахунку індикаторів.")
            return None
        
        close_prices = close_prices.flatten()
        high_prices = high_prices.flatten()
        low_prices = low_prices.flatten()

        data['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        data['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
        data['RSI'] = talib.RSI(close_prices, timeperiod=14)
        
        macd, macdsignal, _ = talib.MACD(close_prices)
        data['MACD'] = macd
        data['MACD_signal'] = macdsignal
        
        upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=20)
        data['BB_upper'] = upperband
        data['BB_middle'] = middleband
        data['BB_lower'] = lowerband
        
        data['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
        data['EMA_26'] = talib.EMA(close_prices, timeperiod=26)
        
        data['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        return data.dropna()
    except Exception as e:
        print(f" Помилка розрахунку індикаторів: {e}")
        return None