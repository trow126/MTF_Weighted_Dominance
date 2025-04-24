"""
PowerX Strategy with Basic Monte Carlo
Main strategy implementation
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from powerx.mt5_handler import MT5Handler
from powerx.indicators import (
    calculate_rsi, calculate_stochastic, calculate_macd, 
    calculate_atr, calculate_supertrend
)
from powerx.signals import (
    add_signal_conditions, generate_entry_signals,
    generate_exit_signals, calculate_sl_tp
)
from powerx.monte_carlo import MonteCarloPositionSizer


class PowerXStrategy:
    """
    PowerX Strategy with Basic Monte Carlo implementation
    """
    
    def __init__(self, 
                 symbol, 
                 timeframe='M15', 
                 higher_timeframe='H1',
                 rsi_period=7,
                 stoch_k_period=14,
                 stoch_smooth_period=3,
                 macd_fast_period=12,
                 macd_slow_period=26,
                 macd_signal_period=9,
                 atr_period=14,
                 supertrend_multiplier=4.0,
                 supertrend_period=10,
                 sl_multiplier=1.5,
                 tp_multiplier=3.0,
                 allow_longs=True,
                 allow_shorts=True,
                 mt5_login=None,
                 mt5_password=None,
                 mt5_server=None):
        """
        Initialize the PowerX strategy
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            higher_timeframe (str): Higher timeframe for confirmation
            rsi_period (int): RSI period
            stoch_k_period (int): Stochastic %K period
            stoch_smooth_period (int): Stochastic smoothing period
            macd_fast_period (int): MACD fast period
            macd_slow_period (int): MACD slow period
            macd_signal_period (int): MACD signal period
            atr_period (int): ATR period
            supertrend_multiplier (float): SuperTrend multiplier
            supertrend_period (int): SuperTrend period
            sl_multiplier (float): Stop loss ATR multiplier
            tp_multiplier (float): Take profit ATR multiplier
            allow_longs (bool): Allow long entries
            allow_shorts (bool): Allow short entries
            mt5_login (int): MT5 account login
            mt5_password (str): MT5 account password
            mt5_server (str): MT5 server name
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.higher_timeframe = higher_timeframe
        
        # Indicator parameters
        self.rsi_period = rsi_period
        self.stoch_k_period = stoch_k_period
        self.stoch_smooth_period = stoch_smooth_period
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.atr_period = atr_period
        self.supertrend_multiplier = supertrend_multiplier
        self.supertrend_period = supertrend_period
        
        # SL/TP parameters
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        
        # Trading parameters
        self.allow_longs = allow_longs
        self.allow_shorts = allow_shorts
        
        # MT5 connection
        self.mt5 = MT5Handler(mt5_login, mt5_password, mt5_server)
        
        # Monte Carlo position sizer
        self.position_sizer = MonteCarloPositionSizer()
        
        # State variables
        self.is_running = False
        self.current_position = None
        self.last_processed_time = None
    
    def prepare_data(self, bars_count=500):
        """
        Prepare data for analysis
        
        Args:
            bars_count (int): Number of historical bars to retrieve
            
        Returns:
            tuple: (df, higher_tf_df) DataFrames with indicator values
        """
        # Calculate start date
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
        
        # Get data from MT5
        df = self.mt5.get_historical_data(
            self.symbol, self.timeframe, start_date, end_date
        )
        
        higher_tf_df = self.mt5.get_historical_data(
            self.symbol, self.higher_timeframe, higher_start_date, end_date
        )
        
        if df is None or higher_tf_df is None:
            print("Failed to retrieve data")
            return None, None
        
        # Calculate indicators for main timeframe
        df['rsi'] = calculate_rsi(df, self.rsi_period)
        df['stoch'] = calculate_stochastic(df, self.stoch_k_period, self.stoch_smooth_period)
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(
            df, self.macd_fast_period, self.macd_slow_period, self.macd_signal_period
        )
        df['atr'] = calculate_atr(df, self.atr_period)
        df['supertrend'], df['supertrend_direction'] = calculate_supertrend(
            df, self.supertrend_multiplier, self.supertrend_period
        )
        
        # Calculate indicators for higher timeframe
        higher_tf_df['rsi'] = calculate_rsi(higher_tf_df, self.rsi_period)
        higher_tf_df['stoch'] = calculate_stochastic(
            higher_tf_df, self.stoch_k_period, self.stoch_smooth_period
        )
        
        # Add signal conditions
        df = add_signal_conditions(df, higher_tf_df)
        df = generate_entry_signals(df, self.allow_longs, self.allow_shorts)
        df = generate_exit_signals(df)
        df = calculate_sl_tp(df, self.sl_multiplier, self.tp_multiplier)
        
        # Drop NaN values
        df = df.dropna()
        
        return df, higher_tf_df
    
    def check_for_signals(self, df):
        """
        Check for trading signals in the latest bar
        
        Args:
            df (pd.DataFrame): DataFrame with signal columns
            
        Returns:
            dict: Signal information
        """
        # Get the latest bar
        latest_bar = df.iloc[-1]
        
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
    
    def process_signals(self, signals):
        """
        Process trading signals
        
        Args:
            signals (dict): Signal information
            
        Returns:
            bool: True if action was taken, False otherwise
        """
        # Get current positions
        positions = self.mt5.get_positions(self.symbol)
        
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
                success = self.mt5.close_position(current_position['ticket'])
                if success:
                    print(f"Closed {current_position['type']} position at {datetime.now()}")
                    
                    # Update Monte Carlo sequence based on trade result
                    is_win = current_position['profit'] > 0
                    self.position_sizer.update_sequence(is_win)
                    
                    return True
        
        # Check for entry signals if no position is open
        if not current_position:
            if signals['long_entry'] and self.allow_longs:
                # Calculate position size
                qty = self.position_sizer.calculate_bet_amount()
                
                # Open long position
                result = self.mt5.open_position(
                    self.symbol, 'BUY', qty,
                    sl=signals['long_sl'],
                    tp=signals['long_tp'],
                    comment='PowerX Long Entry'
                )
                
                if result:
                    print(f"Opened LONG position at {datetime.now()}, size: {qty}")
                    return True
            
            elif signals['short_entry'] and self.allow_shorts:
                # Calculate position size
                qty = self.position_sizer.calculate_bet_amount()
                
                # Open short position
                result = self.mt5.open_position(
                    self.symbol, 'SELL', qty,
                    sl=signals['short_sl'],
                    tp=signals['short_tp'],
                    comment='PowerX Short Entry'
                )
                
                if result:
                    print(f"Opened SHORT position at {datetime.now()}, size: {qty}")
                    return True
        
        return False
    
    def run(self, interval_seconds=5):
        """
        Run the strategy in a loop
        
        Args:
            interval_seconds (int): Sleep interval between checks
        """
        self.is_running = True
        print(f"Starting PowerX strategy for {self.symbol} at {datetime.now()}")
        print(f"Monte Carlo sequence: {self.position_sizer.get_current_sequence()}")
        
        try:
            while self.is_running:
                # Prepare data
                df, _ = self.prepare_data()
                
                if df is not None:
                    # Check if we have a new bar
                    latest_time = df.index[-1]
                    if self.last_processed_time is None or latest_time > self.last_processed_time:
                        # Get signals
                        signals = self.check_for_signals(df)
                        
                        # Process signals
                        action_taken = self.process_signals(signals)
                        
                        # Update last processed time
                        self.last_processed_time = latest_time
                        
                        # Print status
                        if not action_taken:
                            print(f"No action taken at {datetime.now()}")
                            print(f"Current Monte Carlo sequence: {self.position_sizer.get_current_sequence()}")
                
                # Sleep for interval
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("Strategy stopped by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.is_running = False
            self.mt5.disconnect()
            print("Strategy stopped")
    
    def stop(self):
        """Stop the strategy"""
        self.is_running = False
