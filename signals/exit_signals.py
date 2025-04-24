"""
Exit signal generators for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from signals.signal_base import SignalGenerator


class PowerXExitSignal(SignalGenerator):
    """PowerX exit signal generator"""
    
    def generate(self, data: pd.DataFrame, 
                higher_tf_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate exit signals
        
        Args:
            data (pd.DataFrame): Price data with indicators
            higher_tf_data (pd.DataFrame): Higher timeframe data (optional)
            
        Returns:
            pd.DataFrame: DataFrame with added exit signal columns
        """
        # Long exit signal
        data['long_exit'] = (data['black_bar_condition'] == True) | (data['red_bar_condition'] == True)
        
        # Short exit signal
        data['short_exit'] = (data['black_bar_condition'] == True) | (data['green_bar_condition'] == True)
        
        return data


class SLTPCalculator(SignalGenerator):
    """Stop loss and take profit calculator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize SL/TP calculator
        
        Args:
            params (dict): Parameters (sl_multiplier, tp_multiplier)
        """
        super().__init__(params)
        self.sl_multiplier = self.params.get('sl_multiplier', 1.5)
        self.tp_multiplier = self.params.get('tp_multiplier', 3.0)
    
    def generate(self, data: pd.DataFrame, 
                higher_tf_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate stop loss and take profit levels
        
        Args:
            data (pd.DataFrame): Price data with indicators
            higher_tf_data (pd.DataFrame): Higher timeframe data (optional)
            
        Returns:
            pd.DataFrame: DataFrame with added SL/TP columns
        """
        # Long SL/TP
        data['long_entry_price'] = data['high'] + data['tick_size']
        data['long_sl'] = data['long_entry_price'] - (self.sl_multiplier * data['atr'])
        data['long_tp'] = data['long_entry_price'] + (self.tp_multiplier * data['atr'])
        
        # Short SL/TP
        data['short_entry_price'] = data['low'] - data['tick_size']
        data['short_sl'] = data['short_entry_price'] + (self.sl_multiplier * data['atr'])
        data['short_tp'] = data['short_entry_price'] - (self.tp_multiplier * data['atr'])
        
        return data
