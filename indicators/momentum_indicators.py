"""
Momentum indicators for technical analysis
"""

import pandas as pd
import numpy as np
import talib
from typing import Any, Dict, List, Tuple, Optional

from indicators.indicator_base import Indicator


class RSI(Indicator):
    """Relative Strength Index indicator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize RSI indicator
        
        Args:
            params (dict): Parameters (period)
        """
        super().__init__(params)
        self.period = self.params.get('period', 14)
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate RSI
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            np.ndarray: RSI values
        """
        return talib.RSI(data['close'].values, timeperiod=self.period)


class Stochastic(Indicator):
    """Stochastic oscillator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize Stochastic indicator
        
        Args:
            params (dict): Parameters (k_period, smooth_period)
        """
        super().__init__(params)
        self.k_period = self.params.get('k_period', 14)
        self.smooth_period = self.params.get('smooth_period', 3)
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Stochastic oscillator
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            np.ndarray: Stochastic %K values
        """
        stoch_k, stoch_d = talib.STOCH(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            fastk_period=self.k_period,
            slowk_period=self.smooth_period,
            slowk_matype=0,
            slowd_period=self.smooth_period,
            slowd_matype=0
        )
        return stoch_k


class MACD(Indicator):
    """Moving Average Convergence Divergence indicator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize MACD indicator
        
        Args:
            params (dict): Parameters (fast_period, slow_period, signal_period)
        """
        super().__init__(params)
        self.fast_period = self.params.get('fast_period', 12)
        self.slow_period = self.params.get('slow_period', 26)
        self.signal_period = self.params.get('signal_period', 9)
    
    def calculate(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            tuple: (macd, signal, histogram)
        """
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        return macd, signal, hist
