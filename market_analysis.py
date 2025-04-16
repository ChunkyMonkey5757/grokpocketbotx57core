# market_analysis.py
import pandas as pd
import numpy as np

def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate the Average True Range (ATR) to measure volatility."""
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1] if not atr.empty else 0.0

def determine_trade_duration(atr: float, min_duration: int = 1, max_duration: int = 5) -> int:
    """Determine trade duration (1-5 minutes) based on ATR volatility."""
    if atr == 0:
        return min_duration  # Default to minimum if ATR calculation fails
    # Normalize ATR to map to 1-5 minutes
    # Lower ATR -> shorter duration (less volatility), higher ATR -> longer duration
    normalized_atr = min(max(atr / 1000, 0), 1)  # Adjust divisor based on typical ATR values for your asset
    duration = int(min_duration + (max_duration - min_duration) * normalized_atr)
    return max(min_duration, min(max_duration, duration))