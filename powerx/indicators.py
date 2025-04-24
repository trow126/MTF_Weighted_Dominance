"""
Technical indicators module for PowerX strategy
This module contains functions to calculate various technical indicators
used in the PowerX strategy with Monte Carlo position sizing.
"""

import numpy as np
import pandas as pd
import talib


def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    return talib.RSI(data['close'].values, timeperiod=period)


def calculate_stochastic(data, k_period=14, smooth_period=3):
    """Calculate Stochastic oscillator"""
    stoch_k, stoch_d = talib.STOCH(
        data['high'].values,
        data['low'].values,
        data['close'].values,
        fastk_period=k_period,
        slowk_period=smooth_period,
        slowk_matype=0,
        slowd_period=smooth_period,
        slowd_matype=0
    )
    return stoch_k


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD indicator"""
    macd, signal, hist = talib.MACD(
        data['close'].values,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )
    return macd, signal, hist


def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    return talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)


def calculate_supertrend(data, multiplier=3, period=10):
    """
    Calculate SuperTrend indicator
    
    Returns:
    - supertrend_values: The SuperTrend line values
    - supertrend_direction: Direction of SuperTrend (1 for downtrend, -1 for uptrend)
    """
    atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
    
    # Calculate basic upper and lower bands
    hl2 = (data['high'] + data['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize SuperTrend values
    supertrend = np.zeros_like(data['close'].values)
    direction = np.zeros_like(data['close'].values)
    
    # First value is always in downtrend
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    # Calculate SuperTrend values
    for i in range(1, len(data)):
        # If current close price crosses above upper band, trend changes to uptrend
        if data['close'][i] > upper_band[i-1]:
            direction[i] = -1
        # If current close price crosses below lower band, trend changes to downtrend
        elif data['close'][i] < lower_band[i-1]:
            direction[i] = 1
        # Otherwise, trend remains the same as previous
        else:
            direction[i] = direction[i-1]
            
            if direction[i] < 0 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
                
            if direction[i] > 0 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        # Set SuperTrend value based on direction
        if direction[i] > 0:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = lower_band[i]
    
    return supertrend, direction
