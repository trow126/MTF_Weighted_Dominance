"""
Backtest engine for trading strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from indicators.momentum_indicators import RSI, Stochastic, MACD
from indicators.volatility_indicators import ATR, SuperTrend
from signals.entry_signals import PowerXEntrySignal
from signals.exit_signals import PowerXExitSignal, SLTPCalculator


class BacktestEngine:
    """Backtest engine for trading strategies"""
    
    def __init__(self, data: pd.DataFrame, higher_tf_data: Optional[pd.DataFrame] = None,
                initial_capital: float = 10000.0, commission: float = 0.0, 
                slippage: float = 0.0, **params):
        """
        Initialize backtest engine
        
        Args:
            data (pd.DataFrame): Main timeframe data
            higher_tf_data (pd.DataFrame): Higher timeframe data (optional)
            initial_capital (float): Initial capital
            commission (float): Commission per trade
            slippage (float): Slippage in pips
            **params: Strategy parameters
        """
        self.data = data.copy()
        self.higher_tf_data = higher_tf_data.copy() if higher_tf_data is not None else None
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.params = params
        
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
        
        # Results storage
        self.equity_curve = None
        self.trades = None
        self.stats = None
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Calculate indicators and signals for backtest data
        
        Returns:
            pd.DataFrame: DataFrame with indicators and signals
        """
        # Make a copy of the data
        df = self.data.copy()
        
        # Calculate indicators for main timeframe
        df['rsi'] = self.rsi.calculate(df)
        df['stoch'] = self.stoch.calculate(df)
        macd_result = self.macd.calculate(df)
        df['macd'] = macd_result[0]
        df['macd_signal'] = macd_result[1]
        df['macd_hist'] = macd_result[2]
        df['atr'] = self.atr.calculate(df)
        supertrend_result = self.supertrend.calculate(df)
        df['supertrend'] = supertrend_result[0]
        df['supertrend_direction'] = supertrend_result[1]
        
        # Calculate indicators for higher timeframe if available
        if self.higher_tf_data is not None:
            higher_tf_df = self.higher_tf_data.copy()
            higher_tf_df['rsi'] = self.rsi.calculate(higher_tf_df)
            higher_tf_df['stoch'] = self.stoch.calculate(higher_tf_df)
            
            # Map higher timeframe data to main timeframe
            df['higher_tf_rsi'] = np.nan
            
            # Find corresponding higher timeframe data for each main timeframe bar
            for idx in df.index:
                # Get higher timeframe data up to this time
                mask = higher_tf_df.index <= idx
                if mask.any():
                    # Use the latest higher timeframe data
                    closest_idx = higher_tf_df.index[mask][-1]
                    df.loc[idx, 'higher_tf_rsi'] = higher_tf_df.loc[closest_idx, 'rsi']
            
            # Fill remaining NaN values
            df['higher_tf_rsi'] = df['higher_tf_rsi'].fillna(method='ffill')
        else:
            # If no higher timeframe data, use own RSI
            df['higher_tf_rsi'] = df['rsi']
        
        # Add signal conditions
        df = self._add_signal_conditions(df)
        
        # Generate entry signals
        df = self.entry_signal.generate(df)
        
        # Generate exit signals
        df = self.exit_signal.generate(df)
        
        # Calculate SL/TP
        df = self.sltp_calculator.generate(df)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
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
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run backtest on the data
        
        Returns:
            dict: Backtest results
        """
        # Prepare data with indicators and signals
        df = self.prepare_data()
        
        # Initialize results
        equity = [self.initial_capital]
        trades = []
        positions = []
        current_position = None
        
        # Loop through each bar
        for i in range(1, len(df)):
            prev_bar = df.iloc[i-1]
            current_bar = df.iloc[i]
            
            # Update equity with open positions
            if current_position:
                # Calculate current position value
                if current_position['type'] == 'BUY':
                    profit = (current_bar['open'] - current_position['entry_price']) * current_position['size']
                else:  # SELL
                    profit = (current_position['entry_price'] - current_bar['open']) * current_position['size']
                
                # Add to equity
                equity.append(equity[-1] + profit)
            else:
                # No position, equity stays the same
                equity.append(equity[-1])
            
            # Check for exit signals first
            if current_position:
                # Check for SL/TP hits
                if current_position['type'] == 'BUY':
                    sl_hit = current_bar['low'] <= current_position['sl']
                    tp_hit = current_bar['high'] >= current_position['tp']
                else:  # SELL
                    sl_hit = current_bar['high'] >= current_position['sl']
                    tp_hit = current_bar['low'] <= current_position['tp']
                
                # Check for exit signal
                exit_signal = (current_position['type'] == 'BUY' and current_bar['long_exit']) or \
                             (current_position['type'] == 'SELL' and current_bar['short_exit'])
                
                # Close position if any exit condition is met
                if sl_hit or tp_hit or exit_signal:
                    exit_price = None
                    exit_reason = None
                    
                    if sl_hit:
                        exit_price = current_position['sl']
                        exit_reason = 'SL'
                    elif tp_hit:
                        exit_price = current_position['tp']
                        exit_reason = 'TP'
                    else:
                        # Exit signal - use open of next bar as exit price
                        if i + 1 < len(df):
                            exit_price = df.iloc[i+1]['open']
                        else:
                            exit_price = current_bar['close']
                        exit_reason = 'Signal'
                    
                    # Apply slippage
                    if current_position['type'] == 'BUY':
                        exit_price -= self.slippage * df.iloc[0]['tick_size']
                    else:  # SELL
                        exit_price += self.slippage * df.iloc[0]['tick_size']
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'BUY':
                        profit = (exit_price - current_position['entry_price']) * current_position['size']
                    else:  # SELL
                        profit = (current_position['entry_price'] - exit_price) * current_position['size']
                    
                    # Subtract commission
                    profit -= self.commission
                    
                    # Update equity
                    equity[-1] = equity[-2] + profit
                    
                    # Record trade
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_bar.name,
                        'type': current_position['type'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'size': current_position['size'],
                        'profit': profit,
                        'profit_pct': profit / equity[-2] * 100,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # Close position
                    current_position = None
            
            # Check for entry signals if no position is open
            if not current_position:
                # Long entry
                if prev_bar['long_entry']:
                    # Calculate position size (fixed for now)
                    size = 1.0
                    
                    # Get entry price (open of next bar)
                    entry_price = current_bar['open']
                    
                    # Apply slippage
                    entry_price += self.slippage * df.iloc[0]['tick_size']
                    
                    # Open position
                    current_position = {
                        'entry_time': current_bar.name,
                        'type': 'BUY',
                        'entry_price': entry_price,
                        'size': size,
                        'sl': prev_bar['long_sl'],
                        'tp': prev_bar['long_tp']
                    }
                
                # Short entry
                elif prev_bar['short_entry']:
                    # Calculate position size (fixed for now)
                    size = 1.0
                    
                    # Get entry price (open of next bar)
                    entry_price = current_bar['open']
                    
                    # Apply slippage
                    entry_price -= self.slippage * df.iloc[0]['tick_size']
                    
                    # Open position
                    current_position = {
                        'entry_time': current_bar.name,
                        'type': 'SELL',
                        'entry_price': entry_price,
                        'size': size,
                        'sl': prev_bar['short_sl'],
                        'tp': prev_bar['short_tp']
                    }
            
            # Record position state
            positions.append(current_position.copy() if current_position else None)
        
        # Close any open position at the end of the backtest
        if current_position:
            exit_price = df.iloc[-1]['close']
            
            # Apply slippage
            if current_position['type'] == 'BUY':
                exit_price -= self.slippage * df.iloc[0]['tick_size']
            else:  # SELL
                exit_price += self.slippage * df.iloc[0]['tick_size']
            
            # Calculate profit/loss
            if current_position['type'] == 'BUY':
                profit = (exit_price - current_position['entry_price']) * current_position['size']
            else:  # SELL
                profit = (current_position['entry_price'] - exit_price) * current_position['size']
            
            # Subtract commission
            profit -= self.commission
            
            # Update final equity
            equity[-1] += profit
            
            # Record trade
            trade = {
                'entry_time': current_position['entry_time'],
                'exit_time': df.index[-1],
                'type': current_position['type'],
                'entry_price': current_position['entry_price'],
                'exit_price': exit_price,
                'size': current_position['size'],
                'profit': profit,
                'profit_pct': profit / equity[-2] * 100,
                'exit_reason': 'End of Test'
            }
            trades.append(trade)
        
        # Create equity curve
        equity_df = pd.DataFrame({
            'equity': equity
        }, index=df.index)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate statistics
        stats = self._calculate_statistics(equity_df, trades_df)
        
        # Store results
        self.equity_curve = equity_df
        self.trades = trades_df
        self.stats = stats
        
        # Return results
        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'stats': stats
        }
    
    def _calculate_statistics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate backtest statistics
        
        Args:
            equity_df (pd.DataFrame): Equity curve
            trades_df (pd.DataFrame): Trades information
            
        Returns:
            dict: Statistics dictionary
        """
        # Initialize statistics
        stats = {}
        
        # Basic statistics
        stats['initial_capital'] = self.initial_capital
        stats['final_equity'] = equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital
        stats['total_return'] = ((stats['final_equity'] / self.initial_capital) - 1) * 100
        
        # Trade statistics
        if not trades_df.empty:
            stats['total_trades'] = len(trades_df)
            stats['winning_trades'] = len(trades_df[trades_df['profit'] > 0])
            stats['losing_trades'] = len(trades_df[trades_df['profit'] <= 0])
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            
            stats['gross_profit'] = trades_df[trades_df['profit'] > 0]['profit'].sum()
            stats['gross_loss'] = trades_df[trades_df['profit'] <= 0]['profit'].sum()
            stats['profit_factor'] = abs(stats['gross_profit'] / stats['gross_loss']) if stats['gross_loss'] != 0 else float('inf')
            
            stats['avg_profit_per_trade'] = trades_df['profit'].mean()
            stats['avg_profit_per_winning_trade'] = trades_df[trades_df['profit'] > 0]['profit'].mean() if stats['winning_trades'] > 0 else 0
            stats['avg_loss_per_losing_trade'] = trades_df[trades_df['profit'] <= 0]['profit'].mean() if stats['losing_trades'] > 0 else 0
            
            # Calculate win/lose streaks
            trades_df['is_win'] = trades_df['profit'] > 0
            win_streak = 0
            lose_streak = 0
            current_win_streak = 0
            current_lose_streak = 0
            
            for win in trades_df['is_win']:
                if win:
                    current_win_streak += 1
                    current_lose_streak = 0
                else:
                    current_lose_streak += 1
                    current_win_streak = 0
                
                win_streak = max(win_streak, current_win_streak)
                lose_streak = max(lose_streak, current_lose_streak)
            
            stats['longest_win_streak'] = win_streak
            stats['longest_lose_streak'] = lose_streak
            
            # Calculate average trade duration
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
                stats['avg_trade_duration'] = trades_df['duration'].mean()
            else:
                stats['avg_trade_duration'] = None
        else:
            # No trades
            stats['total_trades'] = 0
            stats['winning_trades'] = 0
            stats['losing_trades'] = 0
            stats['win_rate'] = 0
            stats['gross_profit'] = 0
            stats['gross_loss'] = 0
            stats['profit_factor'] = 0
            stats['avg_profit_per_trade'] = 0
            stats['avg_profit_per_winning_trade'] = 0
            stats['avg_loss_per_losing_trade'] = 0
            stats['longest_win_streak'] = 0
            stats['longest_lose_streak'] = 0
            stats['avg_trade_duration'] = None
        
        # Calculate drawdown statistics
        if not equity_df.empty:
            # Calculate drawdown
            equity_df['previous_peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['previous_peak']) / equity_df['previous_peak'] * 100
            
            stats['max_drawdown'] = abs(equity_df['drawdown'].min())
            
            # Calculate drawdown duration
            equity_df['is_in_drawdown'] = equity_df['equity'] < equity_df['previous_peak']
            
            # Find the longest stretch of drawdown
            in_drawdown = False
            current_drawdown_start = None
            drawdown_periods = []
            
            for i, row in equity_df.iterrows():
                if row['is_in_drawdown'] and not in_drawdown:
                    in_drawdown = True
                    current_drawdown_start = i
                elif not row['is_in_drawdown'] and in_drawdown:
                    in_drawdown = False
                    drawdown_periods.append((current_drawdown_start, i))
                    current_drawdown_start = None
            
            if in_drawdown:
                drawdown_periods.append((current_drawdown_start, equity_df.index[-1]))
            
            if drawdown_periods:
                drawdown_durations = [(end - start).days for start, end in drawdown_periods]
                stats['max_drawdown_duration'] = max(drawdown_durations) if drawdown_durations else 0
            else:
                stats['max_drawdown_duration'] = 0
            
            # Calculate annualized return and Sharpe ratio
            if len(equity_df) > 1:
                days = (equity_df.index[-1] - equity_df.index[0]).days
                if days > 0:
                    stats['annualized_return'] = stats['total_return'] * (365 / days)
                    
                    # Daily returns
                    equity_df['daily_return'] = equity_df['equity'].pct_change()
                    avg_daily_return = equity_df['daily_return'].mean()
                    std_daily_return = equity_df['daily_return'].std()
                    
                    # Sharpe ratio (assuming 0% risk-free rate)
                    stats['sharpe_ratio'] = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
                else:
                    stats['annualized_return'] = 0
                    stats['sharpe_ratio'] = 0
            else:
                stats['annualized_return'] = 0
                stats['sharpe_ratio'] = 0
        else:
            stats['max_drawdown'] = 0
            stats['max_drawdown_duration'] = 0
            stats['annualized_return'] = 0
            stats['sharpe_ratio'] = 0
        
        return stats
    
    def monte_carlo_analysis(self, num_simulations: int = 1000, 
                            confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis on backtest results
        
        Args:
            num_simulations (int): Number of simulations to run
            confidence_level (float): Confidence level for statistics
            
        Returns:
            dict: Monte Carlo analysis results
        """
        if self.trades is None or len(self.trades) == 0:
            raise ValueError("No trades available for Monte Carlo analysis")
        
        # Extract trade results as percentage returns
        trade_results_pct = self.trades['profit_pct'].values
        
        # Initialize simulation results
        final_equity = np.zeros(num_simulations)
        max_drawdown = np.zeros(num_simulations)
        
        # Run simulations
        for i in range(num_simulations):
            # Shuffle trade results
            shuffled_results = np.random.choice(trade_results_pct, size=len(trade_results_pct), replace=False)
            
            # Calculate equity curve
            equity = np.zeros(len(shuffled_results) + 1)
            equity[0] = 100.0  # Start at 100%
            
            for j, r in enumerate(shuffled_results):
                equity[j+1] = equity[j] * (1 + r/100)
            
            # Calculate drawdown
            previous_peak = np.maximum.accumulate(equity)
            drawdown = (equity - previous_peak) / previous_peak * 100
            
            # Store results
            final_equity[i] = equity[-1]
            max_drawdown[i] = abs(drawdown.min())
        
        # Calculate statistics
        alpha = (1 - confidence_level) / 2
        
        final_equity_stats = {
            'mean': np.mean(final_equity),
            'median': np.median(final_equity),
            'std': np.std(final_equity),
            'min': np.min(final_equity),
            'max': np.max(final_equity),
            'lower': np.percentile(final_equity, alpha * 100),
            'upper': np.percentile(final_equity, (1 - alpha) * 100)
        }
        
        max_drawdown_stats = {
            'mean': np.mean(max_drawdown),
            'median': np.median(max_drawdown),
            'std': np.std(max_drawdown),
            'min': np.min(max_drawdown),
            'max': np.max(max_drawdown),
            'lower': np.percentile(max_drawdown, alpha * 100),
            'upper': np.percentile(max_drawdown, (1 - alpha) * 100)
        }
        
        # Calculate success probability and expected return
        success_probability = np.mean(final_equity > 100.0) * 100
        expected_return = np.mean(final_equity) - 100.0
        
        # Return results
        return {
            'simulations': num_simulations,
            'confidence_level': confidence_level,
            'final_equity': final_equity_stats,
            'max_drawdown': max_drawdown_stats,
            'success_probability': success_probability,
            'expected_return': expected_return,
            'risk': max_drawdown_stats['mean']
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot backtest results
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        
        # Create figure and subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        equity_plot = self.equity_curve['equity'].plot(ax=axes[0], linewidth=2)
        equity_plot.set_title('Equity Curve')
        equity_plot.set_ylabel('Equity')
        equity_plot.grid(True)
        
        # Add trades to equity curve
        if self.trades is not None and len(self.trades) > 0:
            for _, trade in self.trades.iterrows():
                if trade['profit'] > 0:
                    color = 'green'
                else:
                    color = 'red'
                
                axes[0].plot([trade['exit_time']], [self.equity_curve.loc[trade['exit_time'], 'equity']], 
                         marker='o', markersize=5, color=color)
        
        # Plot drawdown
        if 'drawdown' in self.equity_curve.columns:
            drawdown_plot = self.equity_curve['drawdown'].plot(ax=axes[1], color='red', linewidth=1)
            drawdown_plot.set_title('Drawdown')
            drawdown_plot.set_ylabel('Drawdown %')
            drawdown_plot.grid(True)
            
            # Add line at 0%
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Plot trade results
        if self.trades is not None and len(self.trades) > 0:
            # Calculate cumulative profit
            self.trades['cumulative_profit'] = self.trades['profit'].cumsum()
            
            # Plot cumulative profit
            profit_plot = self.trades['cumulative_profit'].plot(ax=axes[2], linewidth=2)
            profit_plot.set_title('Cumulative Profit')
            profit_plot.set_ylabel('Profit')
            profit_plot.set_xlabel('Trade Number')
            profit_plot.grid(True)
            
            # Add line at 0
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
