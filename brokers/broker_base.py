"""
Abstract base broker interface for trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime


class BrokerBase(ABC):
    """Abstract base broker class"""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                            start_date: datetime, end_date: Optional[datetime] = None,
                            include_partial: bool = True) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe as string ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1')
            start_date (datetime): Start date
            end_date (datetime): End date (optional, defaults to now)
            include_partial (bool): Whether to include partial (current) candle
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            dict: Account info dictionary
        """
        pass
    
    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions
        
        Args:
            symbol (str): Symbol to filter positions (optional)
            
        Returns:
            list: List of open positions
        """
        pass
    
    @abstractmethod
    def open_position(self, symbol: str, order_type: str, volume: float,
                     price: Optional[float] = None, sl: Optional[float] = None, 
                     tp: Optional[float] = None, comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Open a new position
        
        Args:
            symbol (str): Symbol name
            order_type (str): Order type ('BUY' or 'SELL')
            volume (float): Trade volume in lots
            price (float): Price for pending orders (optional)
            sl (float): Stop loss price (optional)
            tp (float): Take profit price (optional)
            comment (str): Order comment (optional)
            
        Returns:
            dict: Order result dictionary
        """
        pass
    
    @abstractmethod
    def close_position(self, ticket: int) -> bool:
        """
        Close an open position by ticket
        
        Args:
            ticket (int): Position ticket
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def modify_position(self, ticket: int, sl: Optional[float] = None,
                       tp: Optional[float] = None) -> bool:
        """
        Modify an open position (SL/TP)
        
        Args:
            ticket (int): Position ticket
            sl (float): New stop loss price (optional)
            tp (float): New take profit price (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
