"""
Abstract base signal generator interface
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional


class SignalGenerator(ABC):
    """Abstract base signal generator class"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize signal generator
        
        Args:
            params (dict): Signal parameters
        """
        self.params = params or {}
    
    @abstractmethod
    def generate(self, data: pd.DataFrame, 
                higher_tf_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate signals
        
        Args:
            data (pd.DataFrame): Price data with indicators
            higher_tf_data (pd.DataFrame): Higher timeframe data (optional)
            
        Returns:
            pd.DataFrame: DataFrame with added signal columns
        """
        pass
