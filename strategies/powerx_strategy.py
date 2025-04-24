"""
PowerX strategy implementation with Basic Monte Carlo position sizing
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from core.base_strategy import BaseStrategy
from brokers.broker_base import BrokerBase
from core.position_sizer import PositionSizer
from indicators.momentum_indicators import RSI, Stochastic, MACD
from indicators.volatility_indicators import ATR, SuperTrend
from signals.entry_signals import PowerXEntrySignal
from signals.exit_signals import PowerXExitSignal, SLTPCalculator


class PowerXStrategy(BaseStrategy):
    """PowerX strategy with Basic Monte Carlo position sizing"""
    
    def __init__(self, broker: BrokerBase, position_sizer: PositionSizer,
                symbol: str, timeframe: str, higher_timeframe: str,
                params: Dict[str, Any] = None):
        """
        Initialize PowerX strategy
        
        Args:
            broker (BrokerBase): Broker instance
            position_sizer (PositionSizer): Position sizer instance
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            higher_timeframe (str): Higher timeframe for confirmation
            params (dict): Strategy parameters
        """
        super().__init__(broker, position_sizer, symbol, timeframe, params)
        self.higher_timeframe = higher_timeframe
        
        # Create indicators
        self.rsi = RSI({"period": params.get("rsi_period", 7)})
        self.stoch = Stochastic({
            "k_period": params.get("stoch_k_period", 14),
            "smooth_period": params.get("stoch_smooth_period", 3)
        })
        self.macd = MACD({
            "fast_period": params.get("macd_fast_period", 12),
            "slow_period": params.get("macd_slow_period", 26),
            "signal_period": params.get("macd_signal_period", 9)
        })
        self.atr = ATR({"period": params.get("atr_period", 14)})
        self.supertrend = SuperTrend({
            "multiplier": params.get("supertrend_multiplier", 4.0),
            "period": params.get("supertrend_period", 10)
        })
        
        # Create signal generators
        self.entry_signal = PowerXEntrySignal({
            "allow_longs": params.get("allow_longs", True),
            "allow_shorts": params.get("allow_shorts", True)
        })
        self.exit_signal = PowerXExitSignal()
        self.sltp_calculator = SLTPCalculator({
            "sl_multiplier": params.get("sl_multiplier", 1.5),
            "tp_multiplier": params.get("tp_multiplier", 3.0)
        })
    
    def prepare_data(self, bars_count: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for analysis
        
        Args:
            bars_count (int): Number of historical bars to retrieve
            
        Returns:
            tuple: (df, higher_tf_df) DataFrames with price data
        """
        # Calculate end date
        end_date = datetime.now()
        
        # For timeframe conversion
        timeframe_to_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
        }
        
        # Calculate start date based on bars_count and timeframe
        minutes = timeframe_to_minutes.get(self.timeframe, 15)
        start_date = end_date - timedelta(minutes=minutes * bars_count)
        
        # Calculate higher timeframe start date (need more history for proper calculations)
        higher_minutes = timeframe_to_minutes.get(self.higher_timeframe, 60)
        higher_bars_needed = max(bars_count // (higher_minutes // minutes), 100)
        higher_start_date = end_date - timedelta(minutes=higher_minutes * higher_bars_needed)
        
        # Get data from broker
        df = self.broker.get_historical_data(
            self.symbol, self.timeframe, start_date, end_date
        )
        
        higher_tf_df = self.broker.get_historical_data(
            self.symbol, self.higher_timeframe, higher_start_date, end_date
        )
        
        if df is None or higher_tf_df is None:
            print("Failed to retrieve data")
            return None, None
        
        return df, higher_tf_df
    
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
        # Calculate indicators for main timeframe
        data['rsi'] = self.rsi.calculate(data)
        data['stoch'] = self.stoch.calculate(data)
        macd_result = self.macd.calculate(data)
        data['macd'] = macd_result[0]
        data['macd_signal'] = macd_result[1]
        data['macd_hist'] = macd_result[2]
        data['atr'] = self.atr.calculate(data)
        supertrend_result = self.supertrend.calculate(data)
        data['supertrend'] = supertrend_result[0]
        data['supertrend_direction'] = supertrend_result[1]
        
        # Calculate indicators for higher timeframe
        if higher_tf_data is not None:
            higher_tf_data['rsi'] = self.rsi.calculate(higher_tf_data)
            higher_tf_data['stoch'] = self.stoch.calculate(higher_tf_data)
            
            # Map higher timeframe data to main timeframe
            data['higher_tf_rsi'] = np.nan
            
            # Find corresponding higher timeframe data for each main timeframe bar
            for idx in data.index:
                # Get higher timeframe data up to this time
                mask = higher_tf_data.index <= idx
                if mask.any():
                    # Use the latest higher timeframe data
                    closest_idx = higher_tf_data.index[mask][-1]
                    data.loc[idx, 'higher_tf_rsi'] = higher_tf_data.loc[closest_idx, 'rsi']
            
            # Fill remaining NaN values
            data['higher_tf_rsi'] = data['higher_tf_rsi'].fillna(method='ffill')
        else:
            # If no higher timeframe data, use own RSI
            data['higher_tf_rsi'] = data['rsi']
        
        # Add signal conditions
        data = self._add_signal_conditions(data)
        
        # Generate entry signals
        data = self.entry_signal.generate(data)
        
        # Generate exit signals
        data = self.exit_signal.generate(data)
        
        # Calculate SL/TP
        data = self.sltp_calculator.generate(data)
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def _add_signal_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal conditions to the dataframe
        
        Args:
            data (pd.DataFrame): Price data with indicators
            
        Returns:
            pd.DataFrame: DataFrame with condition columns
        """
        # Basic bar conditions
        data['green_bar_condition'] = (data['rsi'] > 50) & (data['stoch'] > 50) & (data['macd_hist'] > 0)
        data['red_bar_condition'] = (data['rsi'] < 50) & (data['stoch'] < 50) & (data['macd_hist'] < 0)
        data['black_bar_condition'] = ~((data['green_bar_condition'] == True) | (data['red_bar_condition'] == True))
        
        # Trend conditions
        data['is_trend_up'] = data['supertrend_direction'] < 0
        data['is_trend_down'] = data['supertrend_direction'] > 0
        
        # Higher timeframe conditions
        data['higher_tf_condition_long'] = data['higher_tf_rsi'] > 50
        data['higher_tf_condition_short'] = data['higher_tf_rsi'] < 50
        
        return data
    
    def check_for_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for trading signals in the latest bar
        
        Args:
            data (pd.DataFrame): DataFrame with signal columns
            
        Returns:
            dict: Signal information
        """
        # Get the latest bar
        latest_bar = data.iloc[-1]
        
        # Check for signals
        signals = {
            'long_entry': bool(latest_bar['long_entry']),
            'short_entry': bool(latest_bar['short_entry']),
            'long_exit': bool(latest_bar['long_exit']),
            'short_exit': bool(latest_bar['short_exit']),
            'bar_time': latest_bar.name,
            'long_entry_price': latest_bar['long_entry_price'],
            'long_sl': latest_bar['long_sl'],
            'long_tp': latest_bar['long_tp'],
            'short_entry_price': latest_bar['short_entry_price'],
            'short_sl': latest_bar['short_sl'],
            'short_tp': latest_bar['short_tp']
        }
        
        return signals
    
    def process_signals(self, signals: Dict[str, Any]) -> bool:
        """
        Process trading signals
        
        Args:
            signals (dict): Signal information
            
        Returns:
            bool: True if action was taken, False otherwise
        """
        # Get current positions
        positions = self.broker.get_positions(self.symbol)
        
        # Check if we have an open position for this symbol
        current_position = None
        if positions:
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    current_position = pos
                    break
        
        # Store current position
        self.current_position = current_position
        
        # Check for exit signals first
        if current_position:
            if (current_position['type'] == 'BUY' and signals['long_exit']) or \
               (current_position['type'] == 'SELL' and signals['short_exit']):
                # Close position
                success = self.broker.close_position(current_position['ticket'])
                if success:
                    print(f"Closed {current_position['type']} position at {datetime.now()}")
                    
                    # Update position sizer based on trade result
                    is_win = current_position['profit'] > 0
                    self.position_sizer.update(is_win)
                    
                    return True
        
        # Check for entry signals if no position is open
        if not current_position:
            if signals['long_entry'] and self.params.get('allow_longs', True):
                # Calculate position size
                qty = self.position_sizer.calculate_position_size()
                
                # Open long position
                result = self.broker.open_position(
                    self.symbol, 'BUY', qty,
                    sl=signals['long_sl'],
                    tp=signals['long_tp'],
                    comment='PowerX Long Entry'
                )
                
                if result:
                    print(f"Opened LONG position at {datetime.now()}, size: {qty}")
                    return True
            
            elif signals['short_entry'] and self.params.get('allow_shorts', True):
                # Calculate position size
                qty = self.position_sizer.calculate_position_size()
                
                # Open short position
                result = self.broker.open_position(
                    self.symbol, 'SELL', qty,
                    sl=signals['short_sl'],
                    tp=signals['short_tp'],
                    comment='PowerX Short Entry'
                )
                
                if result:
                    print(f"Opened SHORT position at {datetime.now()}, size: {qty}")
                    return True
        
        return False
