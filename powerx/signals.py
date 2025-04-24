"""
Signal generation module for PowerX strategy
This module contains functions for generating trading signals based on 
the indicator values and strategy conditions.
"""

import numpy as np
import pandas as pd


def add_signal_conditions(df, higher_tf_df=None):
    """
    Add signal condition columns to the dataframe based on indicator values.
    
    Args:
        df (pd.DataFrame): Dataframe with indicator values
        higher_tf_df (pd.DataFrame): Higher timeframe dataframe with indicator values
        
    Returns:
        pd.DataFrame: Dataframe with added signal condition columns
    """
    # Basic bar conditions
    df['green_bar_condition'] = (df['rsi'] > 50) & (df['stoch'] > 50) & (df['macd_hist'] > 0)
    df['red_bar_condition'] = (df['rsi'] < 50) & (df['stoch'] < 50) & (df['macd_hist'] < 0)
    df['black_bar_condition'] = ~((df['green_bar_condition'] == True) | (df['red_bar_condition'] == True))
    
    # Trend conditions
    df['is_trend_up'] = df['supertrend_direction'] < 0
    df['is_trend_down'] = df['supertrend_direction'] > 0
    
    # Higher timeframe conditions
    if higher_tf_df is not None:
        # 修正: リサンプリングの代わりに手動で高い時間足のデータをマッピング
        df['higher_tf_rsi'] = np.nan
        
        # 各時間足のデータに対応する高い時間足のデータを見つける
        for idx in df.index:
            # 現在の時間より前の高い時間足のデータを取得
            mask = higher_tf_df.index <= idx
            if mask.any():
                # 最も近い高い時間足のデータを使用
                closest_idx = higher_tf_df.index[mask][-1]
                df.loc[idx, 'higher_tf_rsi'] = higher_tf_df.loc[closest_idx, 'rsi']
        
        # 残りのNaN値を埋める
        df['higher_tf_rsi'] = df['higher_tf_rsi'].fillna(method='ffill')
    else:
        # 高い時間足のデータがない場合、自身のRSIを使用（テスト用）
        df['higher_tf_rsi'] = df['rsi']
    
    # Higher timeframe conditions
    df['higher_tf_condition_long'] = df['higher_tf_rsi'] > 50
    df['higher_tf_condition_short'] = df['higher_tf_rsi'] < 50
    
    return df


def generate_entry_signals(df, allow_longs=True, allow_shorts=True):
    """
    Generate entry signals based on conditions.
    
    Args:
        df (pd.DataFrame): Dataframe with condition columns
        allow_longs (bool): Whether to allow long entries
        allow_shorts (bool): Whether to allow short entries
        
    Returns:
        pd.DataFrame: Dataframe with added entry signal columns
    """
    # Shift condition to compare with previous bar
    df['green_bar_condition_prev'] = df['green_bar_condition'].shift(1).fillna(False)
    df['red_bar_condition_prev'] = df['red_bar_condition'].shift(1).fillna(False)
    
    # Long entry condition
    if allow_longs:
        df['long_entry'] = (
            (df['green_bar_condition'] == True) & 
            (df['green_bar_condition_prev'] == False) & 
            (df['is_trend_up'] == True) & 
            (df['higher_tf_condition_long'] == True)
        )
    else:
        df['long_entry'] = False
    
    # Short entry condition
    if allow_shorts:
        df['short_entry'] = (
            (df['red_bar_condition'] == True) & 
            (df['red_bar_condition_prev'] == False) & 
            (df['is_trend_down'] == True) & 
            (df['higher_tf_condition_short'] == True)
        )
    else:
        df['short_entry'] = False
    
    return df


def generate_exit_signals(df):
    """
    Generate exit signals based on conditions.
    
    Args:
        df (pd.DataFrame): Dataframe with condition columns
        
    Returns:
        pd.DataFrame: Dataframe with added exit signal columns
    """
    # Long exit signal
    df['long_exit'] = (df['black_bar_condition'] == True) | (df['red_bar_condition'] == True)
    
    # Short exit signal
    df['short_exit'] = (df['black_bar_condition'] == True) | (df['green_bar_condition'] == True)
    
    return df


def calculate_sl_tp(df, sl_multiplier=1.5, tp_multiplier=3.0):
    """
    Calculate stop loss and take profit levels based on ATR.
    
    Args:
        df (pd.DataFrame): Dataframe with price and ATR values
        sl_multiplier (float): Stop loss multiplier for ATR
        tp_multiplier (float): Take profit multiplier for ATR
        
    Returns:
        pd.DataFrame: Dataframe with added SL/TP columns
    """
    # Long SL/TP
    df['long_entry_price'] = df['high'] + df['tick_size']
    df['long_sl'] = df['long_entry_price'] - (sl_multiplier * df['atr'])
    df['long_tp'] = df['long_entry_price'] + (tp_multiplier * df['atr'])
    
    # Short SL/TP
    df['short_entry_price'] = df['low'] - df['tick_size']
    df['short_sl'] = df['short_entry_price'] + (sl_multiplier * df['atr'])
    df['short_tp'] = df['short_entry_price'] - (tp_multiplier * df['atr'])
    
    return df
