"""
Entry signal generators for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from signals.signal_base import SignalGenerator


class PowerXEntrySignal(SignalGenerator):
    """PowerX entry signal generator"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize PowerX entry signal generator
        
        Args:
            params (dict): Signal parameters (allow_longs, allow_shorts)
        """
        super().__init__(params)
        self.allow_longs = self.params.get('allow_longs', True)
        self.allow_shorts = self.params.get('allow_shorts', True)
    
    def generate(self, data: pd.DataFrame, 
                higher_tf_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate entry signals
        
        Args:
            data (pd.DataFrame): Price data with indicators
            higher_tf_data (pd.DataFrame): Higher timeframe data (optional)
            
        Returns:
            pd.DataFrame: DataFrame with added entry signal columns
        """
        # Shift condition to compare with previous bar
        data['green_bar_condition_prev'] = data['green_bar_condition'].shift(1).fillna(False)
        data['red_bar_condition_prev'] = data['red_bar_condition'].shift(1).fillna(False)
        
        # Long entry condition
        if self.allow_longs:
            data['long_entry'] = (
                (data['green_bar_condition'] == True) & 
                (data['green_bar_condition_prev'] == False) & 
                (data['is_trend_up'] == True) & 
                (data['higher_tf_condition_long'] == True)
            )
        else:
            data['long_entry'] = False
        
        # Short entry condition
        if self.allow_shorts:
            data['short_entry'] = (
                (data['red_bar_condition'] == True) & 
                (data['red_bar_condition_prev'] == False) & 
                (data['is_trend_down'] == True) & 
                (data['higher_tf_condition_short'] == True)
            )
        else:
            data['short_entry'] = False
        
        return data
