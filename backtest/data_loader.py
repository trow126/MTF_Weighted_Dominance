"""
Data loader for backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple

from brokers.broker_base import BrokerBase


class DataLoader:
    """Data loader for backtesting"""
    
    def __init__(self, broker: Optional[BrokerBase] = None):
        """
        Initialize data loader
        
        Args:
            broker (BrokerBase): Broker instance for loading data (optional)
        """
        self.broker = broker
    
    def load_csv_data(self, csv_file: str, higher_tf_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data from CSV file(s)
        
        Args:
            csv_file (str): Path to main timeframe CSV file
            higher_tf_file (str): Path to higher timeframe CSV file (optional)
            
        Returns:
            tuple: (main_df, higher_tf_df) DataFrames with price data
        """
        # Load main timeframe data
        df = pd.read_csv(csv_file, parse_dates=['time'])
        df.set_index('time', inplace=True)
        
        # Load higher timeframe data if provided
        higher_tf_df = None
        if higher_tf_file and higher_tf_file != "null":
            higher_tf_df = pd.read_csv(higher_tf_file, parse_dates=['time'])
            higher_tf_df.set_index('time', inplace=True)
        
        return df, higher_tf_df
    
    def load_mt5_data(self, symbol: str, timeframe: str, start_date: datetime,
                     end_date: Optional[datetime] = None, 
                     higher_timeframe: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data from MT5 broker
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Main timeframe
            start_date (datetime): Start date
            end_date (datetime): End date (optional)
            higher_timeframe (str): Higher timeframe (optional)
            
        Returns:
            tuple: (main_df, higher_tf_df) DataFrames with price data
        """
        if not self.broker:
            raise ValueError("Broker not provided for MT5 data loading")
        
        # Load main timeframe data
        df = self.broker.get_historical_data(symbol, timeframe, start_date, end_date)
        
        # Load higher timeframe data if provided
        higher_tf_df = None
        if higher_timeframe:
            higher_tf_df = self.broker.get_historical_data(symbol, higher_timeframe, start_date, end_date)
        
        return df, higher_tf_df
    
    def generate_synthetic_data(self, num_bars: int = 1000, trend_strength: float = 0.6,
                               volatility: float = 1.0, timeframe: str = 'H1',
                               higher_timeframe: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Generate synthetic price data for backtesting
        
        Args:
            num_bars (int): Number of bars to generate
            trend_strength (float): Strength of the trend (0.0-1.0)
            volatility (float): Volatility factor
            timeframe (str): Main timeframe
            higher_timeframe (str): Higher timeframe (optional)
            
        Returns:
            tuple: (main_df, higher_tf_df) DataFrames with synthetic price data
        """
        # Map timeframe to minutes
        timeframe_to_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
        }
        
        minutes = timeframe_to_minutes.get(timeframe, 60)
        
        # Generate timestamps
        end_date = datetime.now()
        timestamps = [end_date - timedelta(minutes=minutes * i) for i in range(num_bars)]
        timestamps.reverse()
        
        # Generate synthetic price data with trend and noise
        close = np.zeros(num_bars)
        close[0] = 1.0  # Start price
        
        # Create trend component
        trend = np.linspace(-1, 1, num_bars)
        trend *= trend_strength
        
        # Add random walk component
        for i in range(1, num_bars):
            close[i] = close[i-1] + (0.001 * trend[i]) + (np.random.randn() * 0.001 * volatility)
        
        # Calculate OHLC from close
        high = close + (np.random.rand(num_bars) * 0.001 * volatility)
        low = close - (np.random.rand(num_bars) * 0.001 * volatility)
        open_price = close - (np.random.rand(num_bars) * 0.0005 * volatility) + (np.random.rand(num_bars) * 0.0005 * volatility)
        
        # Create main timeframe DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.rand(num_bars) * 1000,
            'symbol': 'SYNTH',
            'tick_size': 0.00001
        }, index=timestamps)
        
        # Generate higher timeframe data if requested
        higher_tf_df = None
        if higher_timeframe:
            higher_minutes = timeframe_to_minutes.get(higher_timeframe, 240)
            ratio = higher_minutes // minutes
            
            if ratio > 1:
                # Downsample the data
                num_higher_bars = num_bars // ratio
                higher_timestamps = timestamps[::ratio][:num_higher_bars]
                
                higher_open = open_price[::ratio][:num_higher_bars]
                higher_close = close[::ratio][:num_higher_bars]
                
                # Calculate proper high/low for each higher timeframe candle
                higher_high = np.zeros(num_higher_bars)
                higher_low = np.zeros(num_higher_bars)
                
                for i in range(num_higher_bars):
                    start_idx = i * ratio
                    end_idx = min((i + 1) * ratio, num_bars)
                    higher_high[i] = np.max(high[start_idx:end_idx])
                    higher_low[i] = np.min(low[start_idx:end_idx])
                
                higher_tf_df = pd.DataFrame({
                    'open': higher_open,
                    'high': higher_high,
                    'low': higher_low,
                    'close': higher_close,
                    'volume': np.random.rand(num_higher_bars) * 1000,
                    'symbol': 'SYNTH',
                    'tick_size': 0.00001
                }, index=higher_timestamps)
        
        return df, higher_tf_df
