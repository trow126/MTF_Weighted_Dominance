"""
Abstract base indicator interface for technical indicators
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional


class Indicator(ABC):
    """Abstract base indicator class"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize indicator
        
        Args:
            params (dict): Indicator parameters
        """
        self.params = params or {}
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Any:
        """
        Calculate indicator
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            Any: Indicator value(s)
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters
        
        Returns:
            dict: Parameters dictionary
        """
        return self.params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set parameters
        
        Args:
            params (dict): Parameters dictionary
        """
        self.params.update(params)
