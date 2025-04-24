"""
Data provider abstract interface for trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime


class DataProvider(ABC):
    """Abstract data provider class"""
    
    @abstractmethod
    def get_data(self, symbol: str, timeframe: str, 
                 start_date: datetime, end_date: Optional[datetime] = None,
                 include_partial: bool = True) -> pd.DataFrame:
        """
        Get data for a symbol
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe as string
            start_date (datetime): Start date
            end_date (datetime): End date (optional)
            include_partial (bool): Whether to include partial candle
        
        Returns:
            pd.DataFrame: DataFrame with data
        """
        pass
    
    @abstractmethod
    def get_multi_timeframe_data(self, symbol: str, 
                                timeframe: str, higher_timeframe: str,
                                start_date: datetime, 
                                end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get data for multiple timeframes
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Primary timeframe
            higher_timeframe (str): Higher timeframe
            start_date (datetime): Start date
            end_date (datetime): End date (optional)
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple of (primary_data, higher_tf_data)
        """
        pass
