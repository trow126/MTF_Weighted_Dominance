"""
Data loader for backtest engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List, Tuple, Union, Optional

from powerx.mt5_handler import MT5Handler


class DataLoader:
    """
    Data loader for backtest engine
    """
    
    def __init__(self, mt5_handler: Optional[MT5Handler] = None):
        """
        Initialize data loader
        
        Args:
            mt5_handler (MT5Handler, optional): MT5 handler for data retrieval
        """
        self.mt5_handler = mt5_handler
    
    def load_mt5_data(self,
                     symbol: str,
                     timeframe: str,
                     start_date: datetime,
                     end_date: Optional[datetime] = None,
                     higher_timeframe: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data from MT5
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe as string ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1')
            start_date (datetime): Start date
            end_date (datetime, optional): End date (defaults to now)
            higher_timeframe (str, optional): Higher timeframe to load
            
        Returns:
            tuple: (data, higher_tf_data) Tuple with main and higher timeframe data
        """
        if self.mt5_handler is None:
            self.mt5_handler = MT5Handler()
        
        # Get main timeframe data
        data = self.mt5_handler.get_historical_data(
            symbol, timeframe, start_date, end_date, include_partial=False
        )
        
        # Get higher timeframe data if requested
        higher_tf_data = None
        if higher_timeframe:
            higher_tf_data = self.mt5_handler.get_historical_data(
                symbol, higher_timeframe, start_date, end_date, include_partial=False
            )
        
        return data, higher_tf_data
    
    def load_csv_data(self,
                     main_file: str,
                     higher_tf_file: Optional[str] = None,
                     datetime_format: str = '%Y-%m-%d %H:%M:%S') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data from CSV files
        
        Args:
            main_file (str): Path to main timeframe CSV file
            higher_tf_file (str, optional): Path to higher timeframe CSV file
            datetime_format (str): Datetime format for parsing index
            
        Returns:
            tuple: (data, higher_tf_data) Tuple with main and higher timeframe data
        """
        # Load main timeframe data
        data = pd.read_csv(main_file)
        
        # Convert 'time' column to datetime and set as index
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'], format=datetime_format)
            data.set_index('time', inplace=True)
        elif 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], format=datetime_format)
            data.set_index('date', inplace=True)
        
        # Rename columns if needed
        column_mapping = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'bid': 'open',
            'bidh': 'high',
            'bidl': 'low',
            'bidc': 'close',
            'ask': 'open',
            'askh': 'high',
            'askl': 'low',
            'askc': 'close'
        }
        
        data = data.rename(columns={col: column_mapping[col] for col in data.columns if col in column_mapping})
        
        # Load higher timeframe data if provided
        higher_tf_data = None
        if higher_tf_file:
            higher_tf_data = pd.read_csv(higher_tf_file)
            
            # Convert 'time' column to datetime and set as index
            if 'time' in higher_tf_data.columns:
                higher_tf_data['time'] = pd.to_datetime(higher_tf_data['time'], format=datetime_format)
                higher_tf_data.set_index('time', inplace=True)
            elif 'date' in higher_tf_data.columns:
                higher_tf_data['date'] = pd.to_datetime(higher_tf_data['date'], format=datetime_format)
                higher_tf_data.set_index('date', inplace=True)
            
            # Rename columns if needed
            higher_tf_data = higher_tf_data.rename(columns={col: column_mapping[col] for col in higher_tf_data.columns if col in column_mapping})
        
        return data, higher_tf_data
    
    def resample_to_higher_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to higher timeframe
        
        Args:
            data (pd.DataFrame): Data to resample
            timeframe (str): Target timeframe ('5min', '15min', '30min', '1H', '4H', '1D', '1W', '1M')
            
        Returns:
            pd.DataFrame: Resampled data
        """
        # Convert timeframe string to pandas format
        timeframe_map = {
            'M1': '1min',
            'M5': '5min',
            'M15': '15min',
            'M30': '30min',
            'H1': '1H',
            'H4': '4H',
            'D1': '1D',
            'W1': '1W',
            'MN1': '1M'
        }
        
        resample_timeframe = timeframe_map.get(timeframe, timeframe)
        
        # Ensure data index has a frequency
        if data.index.freq is None:
            # Try to infer frequency
            data = data.asfreq(pd.infer_freq(data.index))
            
            # If frequency cannot be inferred, use the timeframe parameter
            if data.index.freq is None:
                # Create a new index with the specified frequency
                new_index = pd.date_range(start=data.index[0], end=data.index[-1], freq=resample_timeframe)
                data = data.reindex(new_index, method='ffill')
        
        # Resample OHLCV data
        resampled = data.resample(resample_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled
    
    def generate_synthetic_data(self,
                              num_bars: int = 1000,
                              trend_strength: float = 0.6,
                              volatility: float = 1.0,
                              gap_probability: float = 0.05,
                              timeframe: str = 'D1',
                              higher_timeframe: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Generate synthetic price data for backtesting
        
        Args:
            num_bars (int): Number of bars to generate
            trend_strength (float): Strength of the trend (0-1)
            volatility (float): Volatility multiplier
            gap_probability (float): Probability of price gap
            timeframe (str): Timeframe as string ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1')
            higher_timeframe (str, optional): Higher timeframe
            
        Returns:
            tuple: (data, higher_tf_data) Tuple with main and higher timeframe data
        """
        # Map timeframe string to timedelta
        timeframe_map = {
            'M1': timedelta(minutes=1),
            'M5': timedelta(minutes=5),
            'M15': timedelta(minutes=15),
            'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1),
            'H4': timedelta(hours=4),
            'D1': timedelta(days=1),
            'W1': timedelta(weeks=1),
            'MN1': timedelta(days=30)
        }
        
        # Map timeframe string to pandas frequency
        pandas_freq_map = {
            'M1': 'T',
            'M5': '5T',
            'M15': '15T',
            'M30': '30T',
            'H1': 'H',
            'H4': '4H',
            'D1': 'D',
            'W1': 'W',
            'MN1': 'M'
        }
        
        # Generate timestamps
        end_date = datetime.now()
        freq = pandas_freq_map.get(timeframe, 'D')
        
        if timeframe in timeframe_map:
            start_date = end_date - timeframe_map[timeframe] * num_bars
            # 修正: periods と freq を同時に指定せず、start と end と freq を指定
            timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
            # 必要な数のバーだけ取得
            if len(timestamps) > num_bars:
                timestamps = timestamps[-num_bars:]
        else:
            # Default to daily
            start_date = end_date - timedelta(days=num_bars)
            timestamps = pd.date_range(start=start_date, end=end_date, freq='D')
            if len(timestamps) > num_bars:
                timestamps = timestamps[-num_bars:]
        
        # 実際に生成されたバー数に合わせる
        num_bars = len(timestamps)
        
        # Generate price data with trend, noise, and cycles
        prices = np.zeros(num_bars)
        
        # Initial price
        prices[0] = 100.0
        
        # Add trend component
        trend = np.linspace(0, trend_strength * 20, num_bars)
        
        # Add cyclical components of different frequencies
        cycles = (
            np.sin(np.linspace(0, 4 * np.pi, num_bars)) * 5 +  # Medium cycle
            np.sin(np.linspace(0, 20 * np.pi, num_bars)) * 2 +  # Short cycle
            np.sin(np.linspace(0, 2 * np.pi, num_bars)) * 10    # Long cycle
        )
        
        # Add random walk component
        random_walk = np.cumsum(np.random.normal(0, volatility, num_bars))
        
        # Combine components
        prices = prices[0] + trend + cycles + random_walk
        
        # Ensure prices are positive
        prices = np.maximum(prices, 1.0)
        
        # Generate OHLC data from close prices
        data = pd.DataFrame(index=timestamps)
        data['close'] = prices
        
        # Add random price gaps
        gaps = np.random.random(num_bars) < gap_probability
        gap_sizes = np.random.normal(0, volatility * 2, num_bars)
        gap_prices = np.zeros(num_bars)
        
        for i in range(1, num_bars):
            if gaps[i]:
                gap_prices[i] = gap_prices[i-1] + gap_sizes[i]
            else:
                gap_prices[i] = gap_prices[i-1]
        
        data['close'] = data['close'] + gap_prices
        
        # Generate open, high, low from close
        data['open'] = data['close'].shift(1)
        data.loc[data.index[0], 'open'] = data['close'].iloc[0] * 0.999
        
        # Random intrabar range
        high_offsets = np.abs(np.random.normal(0, volatility, num_bars))
        low_offsets = -np.abs(np.random.normal(0, volatility, num_bars))
        
        data['high'] = data['close'] + high_offsets
        data['low'] = data['close'] + low_offsets
        
        # Ensure high is maximum and low is minimum
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        # Add volume
        base_volume = np.random.normal(1000, 200, num_bars)
        trend_volume = np.abs(data['close'].diff()) * 1000
        data['volume'] = np.maximum(base_volume + trend_volume.fillna(0), 100)
        
        # Add tick_size
        data['tick_size'] = 0.01
        
        # Create higher timeframe data if requested
        higher_tf_data = None
        if higher_timeframe:
            # Use resampling for higher timeframe data
            if higher_timeframe in pandas_freq_map:
                # Determine higher timeframe periods
                higher_freq = pandas_freq_map.get(higher_timeframe)
                higher_data = data.resample(higher_freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'tick_size': 'first'
                })
                higher_tf_data = higher_data.dropna()
            else:
                # Generate a smaller dataset for higher timeframe
                higher_bars = num_bars // 4  # Arbitrary ratio
                higher_tf_data, _ = self.generate_synthetic_data(
                    num_bars=higher_bars,
                    trend_strength=trend_strength,
                    volatility=volatility * 1.5,  # Higher volatility on higher timeframe
                    gap_probability=gap_probability * 2,  # More gaps on higher timeframe
                    timeframe=higher_timeframe
                )
        
        return data, higher_tf_data
