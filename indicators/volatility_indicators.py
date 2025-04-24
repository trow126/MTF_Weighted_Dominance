"""
Volatility indicators for technical analysis
"""

import pandas as pd
import numpy as np
import talib
from typing import Any, Dict, List, Tuple, Optional

from indicators.indicator_base import Indicator


class ATR(Indicator):
    """Average True Range indicator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize ATR indicator
        
        Args:
            params (dict): Parameters (period)
        """
        super().__init__(params)
        self.period = self.params.get('period', 14)
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate ATR
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            np.ndarray: ATR values
        """
        return talib.ATR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=self.period
        )


class SuperTrend(Indicator):
    """SuperTrend indicator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize SuperTrend indicator
        
        Args:
            params (dict): Parameters (multiplier, period)
        """
        super().__init__(params)
        self.multiplier = self.params.get('multiplier', 3.0)
        self.period = self.params.get('period', 10)
    
    def calculate(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SuperTrend
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            tuple: (supertrend_values, supertrend_direction)
        """
        # Calculate ATR
        atr = talib.ATR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=self.period
        )
        
        # Calculate basic upper and lower bands
        hl2 = (data['high'] + data['low']) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)
        
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
