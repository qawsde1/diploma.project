import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
import numpy as np
import pandas as pd
from data_utils import get_data, compute_indicators
from model_utils import prepare_lstm_data, train_lstm_model, save_model, load_saved_model
from viz_utils import plot_data

def main():
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    while True:
        symbol = input("\nВведіть тикер (наприклад AAPL, BTC-USD) або 'q' для виходу: ").strip().upper()
        
        if symbol == 'Q':
            print("\nЗавершення роботи...")
            break
            
        try:
            # Завантаження даних
            print(f"\n Завантаження даних для {symbol}...")
            data = get_data(symbol, start_date, end_date)
            if data is None:
                continue
                
            # Розрахунок індикаторів
            print(" Розрахунок технічних індикаторів...")
            data = compute_indicators(data)
            if data is None:
                continue
            
            # Відображення даних
            print("\n Останні значення:")
            print(data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'ATR']].tail())
            
            # Візуалізація
            plot_data(data)
            
            # Робота з моделлю
            model = load_saved_model(symbol)
            if model:
                print("\nЗнайдено збережену модель для цього тикеру!")
            else:
                print("\nМодель не знайдена, потрібно навчання.")
                
            if input("Виконати навчання моделі? (y/n): ").lower() == 'y':
                X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(data)
                model = train_lstm_model(X_train, y_train)
                save_model(model, symbol)
                
        except Exception as e:
            print(f"\n Помилка: {e}")
            continue


if __name__ == "__main__":
    main()