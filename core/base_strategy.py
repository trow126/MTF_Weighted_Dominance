"""
Base strategy abstract class for trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from brokers.broker_base import BrokerBase
from core.position_sizer import PositionSizer


class BaseStrategy(ABC):
    """Abstract base strategy class"""
    
    def __init__(self, broker: BrokerBase, position_sizer: PositionSizer,
                symbol: str, timeframe: str, params: Dict[str, Any] = None):
        """
        Initialize strategy
        
        Args:
            broker (BrokerBase): Broker instance
            position_sizer (PositionSizer): Position sizer instance
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            params (dict): Strategy parameters
        """
        self.broker = broker
        self.position_sizer = position_sizer
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}
        self.is_running = False
        self.current_position = None
        self.last_processed_time = None
    
    @abstractmethod
    def prepare_data(self, bars_count: int = 500) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare data for analysis
        
        Args:
            bars_count (int): Number of historical bars to retrieve
            
        Returns:
            tuple: (df, higher_tf_df) DataFrames with indicator values
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, 
                        higher_tf_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            higher_tf_data (pd.DataFrame): DataFrame with higher timeframe data
            
        Returns:
            pd.DataFrame: DataFrame with signal columns
        """
        pass
    
    @abstractmethod
    def check_for_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for trading signals in the latest bar
        
        Args:
            data (pd.DataFrame): DataFrame with signal columns
            
        Returns:
            dict: Signal information
        """
        pass
    
    @abstractmethod
    def process_signals(self, signals: Dict[str, Any]) -> bool:
        """
        Process trading signals
        
        Args:
            signals (dict): Signal information
            
        Returns:
            bool: True if action was taken, False otherwise
        """
        pass
    
    def run(self, interval_seconds: int = 5) -> None:
        """
        Run the strategy in a loop
        
        Args:
            interval_seconds (int): Sleep interval between checks
        """
        self.is_running = True
        print(f"Starting strategy for {self.symbol} at {datetime.now()}")
        
        try:
            while self.is_running:
                # Prepare data
                data, higher_tf_data = self.prepare_data()
                
                if data is not None:
                    # Check if we have a new bar
                    latest_time = data.index[-1]
                    if self.last_processed_time is None or latest_time > self.last_processed_time:
                        # Generate signals
                        data = self.generate_signals(data, higher_tf_data)
                        
                        # Check signals
                        signals = self.check_for_signals(data)
                        
                        # Process signals
                        action_taken = self.process_signals(signals)
                        
                        # Update last processed time
                        self.last_processed_time = latest_time
                        
                        # Print status
                        if not action_taken:
                            print(f"No action taken at {datetime.now()}")
                
                # Sleep for interval
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("Strategy stopped by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.is_running = False
            self.broker.disconnect()
            print("Strategy stopped")
    
    def stop(self) -> None:
        """Stop the strategy"""
        self.is_running = False
