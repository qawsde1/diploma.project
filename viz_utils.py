import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

def plot_data(data, forecast=None, forecast_7_days=None):
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 1]})
    
    # Основний графік (ліва частина)
    ax1.plot(data['Close'], label='Ціна (Close)', color='blue', linewidth=2)
    ax1.plot(data['SMA_50'], label='SMA 50', color='red', linestyle='--', alpha=0.8)
    ax1.plot(data['SMA_200'], label='SMA 200', color='green', linestyle='--', alpha=0.8)
    
    if forecast is not None:
        ax1.scatter(data.index[-1], forecast, color='purple', s=100, 
                   label=f'Прогноз: {float(forecast):.2f}', zorder=5)
    
    if forecast_7_days is not None:
        ax1.plot(data.index[-7:], forecast_7_days, color='orange', marker='o', 
                linestyle='-', linewidth=2, label='Прогноз на 7 днів')
    
    ax1.set_title('Графік цін та технічних індикаторів', fontsize=14, pad=20)
    ax1.set_xlabel('Дата', fontsize=12)
    ax1.set_ylabel('Ціна', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    
    # Аналітична панель (права частина)
    ax2.axis('off')
    
    # Отримання останніх значень (з вашого оригінального коду)
    last_close = data['Close'].values[-1]
    last_sma_50 = data['SMA_50'].values[-1]
    last_sma_200 = data['SMA_200'].values[-1]
    last_rsi = data['RSI'].values[-1]
    last_ema_12 = data['EMA_12'].values[-1]
    last_ema_26 = data['EMA_26'].values[-1]
    last_atr = data['ATR'].values[-1]
    last_macd = data['MACD'].values[-1]
    last_macd_signal = data['MACD_signal'].values[-1]
    last_bb_upper = data['BB_upper'].values[-1]
    last_bb_liddle = data['BB_middle'].values[-1]
    last_bb_lower = data['BB_lower'].values[-1]

    # Конвертація numpy значень у float
    def safe_float(x):
        return float(x.item()) if isinstance(x, np.ndarray) else float(x)
    
    last_close = safe_float(last_close)
    last_sma_50 = safe_float(last_sma_50)
    last_sma_200 = safe_float(last_sma_200)
    last_rsi = safe_float(last_rsi)
    last_ema_12 = safe_float(last_ema_12)
    last_ema_26 = safe_float(last_ema_26)
    last_atr = safe_float(last_atr)
    last_macd = safe_float(last_macd)
    last_macd_signal = safe_float(last_macd_signal)
    last_bb_upper = safe_float(last_bb_upper)
    last_bb_liddle = safe_float(last_bb_liddle)
    last_bb_lower = safe_float(last_bb_lower)

    # Форматування тексту
    analysis_text = "Аналітична панель:\n\n"
    analysis_text += "Основні показники:\n"
    analysis_text += f"• Ціна закриття: {last_close:.2f}\n"
    analysis_text += f"• SMA 50: {last_sma_50:.2f}\n"
    analysis_text += f"• SMA 200: {last_sma_200:.2f}\n"
    analysis_text += f"• RSI: {last_rsi:.2f}\n"
    analysis_text += f"• ATR: {last_atr:.2f}\n\n"
    
    analysis_text += "MACD:\n"
    analysis_text += f"• MACD: {last_macd:.2f}\n"
    analysis_text += f"• Signal: {last_macd_signal:.2f}\n\n"
    
    analysis_text += "Bollinger Bands:\n"
    analysis_text += f"• Верхня: {last_bb_upper:.2f}\n"
    analysis_text += f"• Середня: {last_bb_liddle:.2f}\n"
    analysis_text += f"• Нижня: {last_bb_lower:.2f}\n"

    ax2.text(0.05, 0.95, analysis_text, ha='left', va='top', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.show()