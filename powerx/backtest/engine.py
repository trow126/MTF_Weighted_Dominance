"""
Backtesting engine for PowerX strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

from powerx.indicators import (
    calculate_rsi, calculate_stochastic, calculate_macd, 
    calculate_atr, calculate_supertrend
)
from powerx.signals import (
    add_signal_conditions, generate_entry_signals,
    generate_exit_signals, calculate_sl_tp
)
from powerx.monte_carlo import MonteCarloPositionSizer


class BacktestEngine:
    """
    Backtesting engine for PowerX strategy with Basic Monte Carlo position sizing
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 higher_tf_data: Optional[pd.DataFrame] = None,
                 initial_capital: float = 10000.0,
                 commission: float = 0.0,
                 slippage: float = 0.0,
                 rsi_period: int = 7,
                 stoch_k_period: int = 14,
                 stoch_smooth_period: int = 3,
                 macd_fast_period: int = 12,
                 macd_slow_period: int = 26,
                 macd_signal_period: int = 9,
                 atr_period: int = 14,
                 supertrend_multiplier: float = 4.0,
                 supertrend_period: int = 10,
                 sl_multiplier: float = 1.5,
                 tp_multiplier: float = 3.0,
                 allow_longs: bool = True,
                 allow_shorts: bool = True):
        """
        Initialize the backtest engine
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            higher_tf_data (pd.DataFrame): Higher timeframe price data (optional)
            initial_capital (float): Initial capital for backtesting
            commission (float): Commission per trade (in percentage)
            slippage (float): Slippage per trade (in percentage)
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
        """
        self.data = data.copy()
        self.higher_tf_data = higher_tf_data.copy() if higher_tf_data is not None else None
        
        # Backtesting parameters
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Strategy parameters
        self.rsi_period = rsi_period
        self.stoch_k_period = stoch_k_period
        self.stoch_smooth_period = stoch_smooth_period
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.atr_period = atr_period
        self.supertrend_multiplier = supertrend_multiplier
        self.supertrend_period = supertrend_period
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.allow_longs = allow_longs
        self.allow_shorts = allow_shorts
        
        # Position sizer
        self.position_sizer = MonteCarloPositionSizer()
        
        # Results
        self.equity_curve = None
        self.trades = []
        self.stats = {}
    
    def prepare_data_for_backtest(self) -> pd.DataFrame:
        """
        Prepare data for backtesting by calculating indicators and signals
        
        Returns:
            pd.DataFrame: Processed data with indicators and signals
        """
        # Add tick_size if not present (for backtesting)
        if 'tick_size' not in self.data.columns:
            # Use a default tick size of 0.00001 for currencies
            self.data['tick_size'] = 0.00001
        
        # Calculate indicators
        self.data['rsi'] = calculate_rsi(self.data, self.rsi_period)
        self.data['stoch'] = calculate_stochastic(
            self.data, self.stoch_k_period, self.stoch_smooth_period
        )
        self.data['macd'], self.data['macd_signal'], self.data['macd_hist'] = calculate_macd(
            self.data, self.macd_fast_period, self.macd_slow_period, self.macd_signal_period
        )
        self.data['atr'] = calculate_atr(self.data, self.atr_period)
        self.data['supertrend'], self.data['supertrend_direction'] = calculate_supertrend(
            self.data, self.supertrend_multiplier, self.supertrend_period
        )
        
        # Calculate higher timeframe indicators if provided
        if self.higher_tf_data is not None:
            self.higher_tf_data['rsi'] = calculate_rsi(self.higher_tf_data, self.rsi_period)
            self.higher_tf_data['stoch'] = calculate_stochastic(
                self.higher_tf_data, self.stoch_k_period, self.stoch_smooth_period
            )
        
        # Add signal conditions
        self.data = add_signal_conditions(self.data, self.higher_tf_data)
        self.data = generate_entry_signals(self.data, self.allow_longs, self.allow_shorts)
        self.data = generate_exit_signals(self.data)
        self.data = calculate_sl_tp(self.data, self.sl_multiplier, self.tp_multiplier)
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        return self.data
    
    def run_backtest(self) -> Dict:
        """
        Run backtest on the prepared data
        
        Returns:
            dict: Backtest results and statistics
        """
        # Prepare data
        self.prepare_data_for_backtest()
        
        # Initialize variables
        equity = self.initial_capital
        position = None
        entry_bar = None
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        equity_curve = [equity]
        trades = []
        
        # Loop through each bar
        for i in range(1, len(self.data)):
            current_bar = self.data.iloc[i]
            previous_bar = self.data.iloc[i-1]
            
            # Handle open positions
            if position is not None:
                # Check if SL or TP was hit
                if position == 'long':
                    if current_bar['low'] <= stop_loss:
                        # SL hit
                        exit_price = stop_loss
                        pnl = (exit_price - entry_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
                        equity += pnl
                        trades.append({
                            'entry_time': entry_bar.name,
                            'exit_time': current_bar.name,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'exit_type': 'stop_loss'
                        })
                        position = None
                        
                        # Update Monte Carlo sequence
                        self.position_sizer.update_sequence(False)
                    
                    elif current_bar['high'] >= take_profit:
                        # TP hit
                        exit_price = take_profit
                        pnl = (exit_price - entry_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
                        equity += pnl
                        trades.append({
                            'entry_time': entry_bar.name,
                            'exit_time': current_bar.name,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'exit_type': 'take_profit'
                        })
                        position = None
                        
                        # Update Monte Carlo sequence
                        self.position_sizer.update_sequence(True)
                    
                    elif current_bar['long_exit']:
                        # Signal exit
                        exit_price = current_bar['close']
                        pnl = (exit_price - entry_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
                        equity += pnl
                        trades.append({
                            'entry_time': entry_bar.name,
                            'exit_time': current_bar.name,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'exit_type': 'signal'
                        })
                        position = None
                        
                        # Update Monte Carlo sequence
                        is_win = pnl > 0
                        self.position_sizer.update_sequence(is_win)
                
                elif position == 'short':
                    if current_bar['high'] >= stop_loss:
                        # SL hit
                        exit_price = stop_loss
                        pnl = (entry_price - exit_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
                        equity += pnl
                        trades.append({
                            'entry_time': entry_bar.name,
                            'exit_time': current_bar.name,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'exit_type': 'stop_loss'
                        })
                        position = None
                        
                        # Update Monte Carlo sequence
                        self.position_sizer.update_sequence(False)
                    
                    elif current_bar['low'] <= take_profit:
                        # TP hit
                        exit_price = take_profit
                        pnl = (entry_price - exit_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
                        equity += pnl
                        trades.append({
                            'entry_time': entry_bar.name,
                            'exit_time': current_bar.name,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'exit_type': 'take_profit'
                        })
                        position = None
                        
                        # Update Monte Carlo sequence
                        self.position_sizer.update_sequence(True)
                    
                    elif current_bar['short_exit']:
                        # Signal exit
                        exit_price = current_bar['close']
                        pnl = (entry_price - exit_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
                        equity += pnl
                        trades.append({
                            'entry_time': entry_bar.name,
                            'exit_time': current_bar.name,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'exit_type': 'signal'
                        })
                        position = None
                        
                        # Update Monte Carlo sequence
                        is_win = pnl > 0
                        self.position_sizer.update_sequence(is_win)
            
            # Enter new positions if no current position
            if position is None:
                # Long entry
                if previous_bar['long_entry']:
                    # Calculate position size
                    bet_amount = self.position_sizer.calculate_bet_amount()
                    position_size = bet_amount
                    
                    position = 'long'
                    entry_bar = previous_bar
                    entry_price = previous_bar['long_entry_price']
                    stop_loss = previous_bar['long_sl']
                    take_profit = previous_bar['long_tp']
                
                # Short entry
                elif previous_bar['short_entry']:
                    # Calculate position size
                    bet_amount = self.position_sizer.calculate_bet_amount()
                    position_size = bet_amount
                    
                    position = 'short'
                    entry_bar = previous_bar
                    entry_price = previous_bar['short_entry_price']
                    stop_loss = previous_bar['short_sl']
                    take_profit = previous_bar['short_tp']
            
            # Update equity curve
            if position == 'long':
                unrealized_pnl = (current_bar['close'] - entry_price) * position_size
                equity_curve.append(equity + unrealized_pnl)
            elif position == 'short':
                unrealized_pnl = (entry_price - current_bar['close']) * position_size
                equity_curve.append(equity + unrealized_pnl)
            else:
                equity_curve.append(equity)
        
        # Close any open position at the end
        if position is not None:
            last_bar = self.data.iloc[-1]
            exit_price = last_bar['close']
            
            if position == 'long':
                pnl = (exit_price - entry_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
            else:  # short
                pnl = (entry_price - exit_price) * position_size - (entry_price + exit_price) * position_size * self.commission / 100
            
            equity += pnl
            trades.append({
                'entry_time': entry_bar.name,
                'exit_time': last_bar.name,
                'position': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'pnl': pnl,
                'exit_type': 'end_of_test'
            })
            
            # Update final equity
            equity_curve[-1] = equity
        
        # Convert to DataFrames
        self.equity_curve = pd.Series(equity_curve, index=self.data.index)
        self.trades = pd.DataFrame(trades)
        
        # Calculate statistics
        self.calculate_stats()
        
        return {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'stats': self.stats
        }
    
    def calculate_stats(self) -> Dict:
        """
        Calculate performance statistics from backtest results
        
        Returns:
            dict: Performance statistics
        """
        if len(self.trades) == 0:
            self.stats = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_duration': 0,
                'avg_trade_duration': timedelta(0),
                'avg_profit_per_trade': 0,
                'avg_profit_per_winning_trade': 0,
                'avg_loss_per_losing_trade': 0,
                'longest_win_streak': 0,
                'longest_lose_streak': 0
            }
            return self.stats
        
        # Basic statistics
        self.trades['duration'] = self.trades['exit_time'] - self.trades['entry_time']
        self.trades['is_win'] = self.trades['pnl'] > 0
        
        total_trades = len(self.trades)
        winning_trades = self.trades[self.trades['is_win']].shape[0]
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = self.trades[self.trades['is_win']]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(self.trades[~self.trades['is_win']]['pnl'].sum()) if losing_trades > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns and drawdowns
        total_return = (self.equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate daily returns
        daily_returns = self.equity_curve.pct_change().dropna()
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Annualized return
        days = (self.data.index[-1] - self.data.index[0]).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        # Maximum drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve / peak - 1) * 100
        max_drawdown = abs(drawdown.min())
        
        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_start = is_drawdown[is_drawdown].index
        drawdown_end = is_drawdown[~is_drawdown].index
        
        if len(drawdown_start) > 0 and len(drawdown_end) > 0:
            drawdown_periods = pd.DataFrame({
                'start': drawdown_start,
                'end': drawdown_end
            })
            drawdown_periods['duration'] = drawdown_periods['end'] - drawdown_periods['start']
            max_drawdown_duration = drawdown_periods['duration'].max().days
        else:
            max_drawdown_duration = 0
        
        # Average trade statistics
        avg_trade_duration = self.trades['duration'].mean() if total_trades > 0 else timedelta(0)
        avg_profit_per_trade = self.trades['pnl'].mean() if total_trades > 0 else 0
        avg_profit_per_winning_trade = self.trades[self.trades['is_win']]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss_per_losing_trade = self.trades[~self.trades['is_win']]['pnl'].mean() if losing_trades > 0 else 0
        
        # Win/lose streaks
        win_lose_series = self.trades['is_win'].astype(int).diff().fillna(1)
        streak_changes = win_lose_series[win_lose_series != 0].index
        
        streak_types = []
        streak_lengths = []
        
        for i in range(len(streak_changes)):
            if i == 0:
                streak_start = 0
            else:
                streak_start = self.trades.index.get_loc(streak_changes[i])
            
            if i == len(streak_changes) - 1:
                streak_end = len(self.trades)
            else:
                streak_end = self.trades.index.get_loc(streak_changes[i+1])
            
            streak_type = self.trades.iloc[streak_start]['is_win']
            streak_length = streak_end - streak_start
            
            streak_types.append(streak_type)
            streak_lengths.append(streak_length)
        
        longest_win_streak = max([length for i, length in enumerate(streak_lengths) if streak_types[i]], default=0)
        longest_lose_streak = max([length for i, length in enumerate(streak_lengths) if not streak_types[i]], default=0)
        
        # Compile statistics
        self.stats = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_trade_duration': avg_trade_duration,
            'avg_profit_per_trade': avg_profit_per_trade,
            'avg_profit_per_winning_trade': avg_profit_per_winning_trade,
            'avg_loss_per_losing_trade': avg_loss_per_losing_trade,
            'longest_win_streak': longest_win_streak,
            'longest_lose_streak': longest_lose_streak
        }
        
        return self.stats
    
    def plot_results(self, figsize=(14, 10)):
        """
        Plot backtest results
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if self.equity_curve is None:
            print("No backtest results to plot. Run backtest first.")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        axes[0].plot(self.equity_curve.index, self.equity_curve.values)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity')
        axes[0].grid(True)
        
        # Plot drawdowns
        if len(self.equity_curve) > 0:
            peak = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve / peak - 1) * 100
            axes[1].fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
            axes[1].set_title('Drawdown')
            axes[1].set_ylabel('Drawdown %')
            axes[1].grid(True)
        
        # Plot trade results
        if len(self.trades) > 0:
            trade_results = pd.Series(index=self.trades['exit_time'], data=self.trades['pnl'].values)
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_results]
            axes[2].bar(trade_results.index, trade_results.values, color=colors, alpha=0.6)
            axes[2].set_title('Trade Results')
            axes[2].set_ylabel('Profit/Loss')
            axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    
    def monte_carlo_analysis(self, num_simulations=1000, confidence_level=0.95):
        """
        Perform Monte Carlo analysis by shuffling the order of trades
        
        Args:
            num_simulations (int): Number of Monte Carlo simulations
            confidence_level (float): Confidence level for the statistics
            
        Returns:
            dict: Monte Carlo analysis results
        """
        if len(self.trades) == 0:
            print("No trades to analyze. Run backtest first.")
            return None
        
        # Get PnL from trades
        pnl_series = self.trades['pnl']
        
        # Run simulations
        equity_curves = []
        for _ in range(num_simulations):
            # Shuffle the order of trades
            shuffled_pnl = pnl_series.sample(frac=1).reset_index(drop=True)
            
            # Create equity curve
            equity_curve = self.initial_capital + shuffled_pnl.cumsum()
            equity_curves.append(equity_curve)
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curves).T
        
        # Calculate final equity for each simulation
        final_equity = equity_df.iloc[-1]
        
        # Calculate drawdowns for each simulation
        max_drawdowns = []
        for col in equity_df.columns:
            peak = equity_df[col].expanding().max()
            drawdown = (equity_df[col] / peak - 1) * 100
            max_drawdowns.append(abs(drawdown.min()))
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        final_equity_lower = np.percentile(final_equity, lower_percentile)
        final_equity_upper = np.percentile(final_equity, upper_percentile)
        final_equity_mean = final_equity.mean()
        
        max_drawdown_lower = np.percentile(max_drawdowns, lower_percentile)
        max_drawdown_upper = np.percentile(max_drawdowns, upper_percentile)
        max_drawdown_mean = np.mean(max_drawdowns)
        
        # Calculate success probability (final equity > initial capital)
        success_prob = (final_equity > self.initial_capital).mean() * 100
        
        # Calculate expected return and risk
        expected_return = (final_equity_mean / self.initial_capital - 1) * 100
        risk = max_drawdown_mean
        
        # Return results
        return {
            'simulations': num_simulations,
            'confidence_level': confidence_level,
            'equity_curves': equity_df,
            'final_equity': {
                'mean': final_equity_mean,
                'lower': final_equity_lower,
                'upper': final_equity_upper
            },
            'max_drawdown': {
                'mean': max_drawdown_mean,
                'lower': max_drawdown_lower,
                'upper': max_drawdown_upper
            },
            'success_probability': success_prob,
            'expected_return': expected_return,
            'risk': risk
        }
    
    def plot_monte_carlo(self, mc_results, num_curves=100, figsize=(12, 10)):
        """
        Plot Monte Carlo simulation results
        
        Args:
            mc_results (dict): Monte Carlo analysis results
            num_curves (int): Number of equity curves to plot
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if mc_results is None:
            print("No Monte Carlo results to plot. Run monte_carlo_analysis first.")
            return None
        
        equity_df = mc_results['equity_curves']
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot a subset of equity curves
        for col in equity_df.sample(columns=num_curves, axis=1).columns:
            axes[0].plot(equity_df[col], color='blue', alpha=0.1)
        
        # Plot mean and confidence intervals
        mean_curve = equity_df.mean(axis=1)
        lower_curve = equity_df.quantile(q=(1 - mc_results['confidence_level']) / 2, axis=1)
        upper_curve = equity_df.quantile(q=(1 + mc_results['confidence_level']) / 2, axis=1)
        
        axes[0].plot(mean_curve, color='black', linewidth=2, label='Mean')
        axes[0].plot(lower_curve, color='red', linewidth=2, label=f"{(1 - mc_results['confidence_level']) / 2 * 100:.1f}%")
        axes[0].plot(upper_curve, color='green', linewidth=2, label=f"{(1 + mc_results['confidence_level']) / 2 * 100:.1f}%")
        
        axes[0].axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title(f'Monte Carlo Simulation: {mc_results["simulations"]} Runs')
        axes[0].set_ylabel('Equity')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot histogram of final equity
        final_equity = equity_df.iloc[-1]
        axes[1].hist(final_equity, bins=50, alpha=0.7)
        axes[1].axvline(x=self.initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[1].axvline(x=mc_results['final_equity']['lower'], color='red', linewidth=2, label=f"{(1 - mc_results['confidence_level']) / 2 * 100:.1f}%")
        axes[1].axvline(x=mc_results['final_equity']['mean'], color='black', linewidth=2, label='Mean')
        axes[1].axvline(x=mc_results['final_equity']['upper'], color='green', linewidth=2, label=f"{(1 + mc_results['confidence_level']) / 2 * 100:.1f}%")
        
        axes[1].set_title('Distribution of Final Equity')
        axes[1].set_xlabel('Equity')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
