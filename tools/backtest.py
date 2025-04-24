"""
Backtest script for PowerX strategy
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from powerx.backtest.engine import BacktestEngine
from powerx.backtest.data_loader import DataLoader
from powerx.mt5_handler import MT5Handler


def load_config(config_file):
    """
    Load configuration from YAML file
    
    Args:
        config_file (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_backtest(config):
    """
    Run backtest with configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Backtest results
    """
    # Setup data loader
    data_loader = DataLoader()
    
    # Load data based on config
    data_source = config.get('data_source', 'synthetic')
    
    if data_source == 'mt5':
        # Use MT5 connection for data
        mt5_config = config.get('mt5', {})
        mt5_handler = MT5Handler(
            mt5_config.get('login'),
            mt5_config.get('password'),
            mt5_config.get('server')
        )
        
        # Get data
        start_date = datetime.now() - timedelta(days=config.get('days_to_load', 365))
        data, higher_tf_data = data_loader.load_mt5_data(
            config['strategy']['symbol'],
            config['strategy']['timeframe'],
            start_date,
            higher_timeframe=config['strategy']['higher_timeframe']
        )
    
    elif data_source == 'csv':
        # Use CSV files for data
        data, higher_tf_data = data_loader.load_csv_data(
            config['data']['main_file'],
            config['data'].get('higher_tf_file')
        )
    
    else:  # Synthetic data
        # Generate synthetic data
        data, higher_tf_data = data_loader.generate_synthetic_data(
            num_bars=config.get('num_bars', 1000),
            trend_strength=config.get('trend_strength', 0.6),
            volatility=config.get('volatility', 1.0),
            timeframe=config['strategy']['timeframe'],
            higher_timeframe=config['strategy']['higher_timeframe']
        )
    
    # Setup backtest engine
    engine = BacktestEngine(
        data=data,
        higher_tf_data=higher_tf_data,
        initial_capital=config.get('initial_capital', 10000.0),
        commission=config.get('commission', 0.0),
        slippage=config.get('slippage', 0.0),
        rsi_period=config['strategy']['rsi_period'],
        stoch_k_period=config['strategy']['stoch_k_period'],
        stoch_smooth_period=config['strategy']['stoch_smooth_period'],
        macd_fast_period=config['strategy']['macd_fast_period'],
        macd_slow_period=config['strategy']['macd_slow_period'],
        macd_signal_period=config['strategy']['macd_signal_period'],
        atr_period=config['strategy']['atr_period'],
        supertrend_multiplier=config['strategy']['supertrend_multiplier'],
        supertrend_period=config['strategy']['supertrend_period'],
        sl_multiplier=config['strategy']['sl_multiplier'],
        tp_multiplier=config['strategy']['tp_multiplier'],
        allow_longs=config['strategy']['allow_longs'],
        allow_shorts=config['strategy']['allow_shorts']
    )
    
    # Run backtest
    backtest_results = engine.run_backtest()
    
    # Run Monte Carlo simulation if requested
    if config.get('monte_carlo', {}).get('enabled', False):
        mc_config = config.get('monte_carlo', {})
        mc_results = engine.monte_carlo_analysis(
            num_simulations=mc_config.get('num_simulations', 1000),
            confidence_level=mc_config.get('confidence_level', 0.95)
        )
        
        backtest_results['monte_carlo'] = mc_results
    
    # Plot results if requested
    if config.get('plot', {}).get('enabled', True):
        fig = engine.plot_results(figsize=(10, 8))
        
        # Save plot if requested
        if config.get('plot', {}).get('save', False):
            plot_file = config.get('plot', {}).get('file', f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            fig.savefig(plot_file)
            print(f"Plot saved to {plot_file}")
        
        # Show plot if requested
        if config.get('plot', {}).get('show', True):
            plt.show()
    
    # Save results if requested
    if config.get('save_results', False):
        results_file = config.get('results_file', f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        trades_file = config.get('trades_file', f'backtest_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        backtest_results['equity_curve'].to_csv(results_file)
        backtest_results['trades'].to_csv(trades_file)
        
        print(f"Results saved to {results_file}")
        print(f"Trades saved to {trades_file}")
    
    return backtest_results


def print_stats(stats):
    """
    Print backtest statistics
    
    Args:
        stats (dict): Backtest statistics
    """
    print("\n--- BACKTEST RESULTS ---")
    print(f"Total trades: {stats['total_trades']}")
    print(f"Win rate: {stats['win_rate']:.2%}")
    print(f"Profit factor: {stats['profit_factor']:.2f}")
    print(f"Total return: {stats['total_return']:.2f}%")
    print(f"Annualized return: {stats['annualized_return']:.2f}%")
    print(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {stats['max_drawdown']:.2f}%")
    print(f"Max drawdown duration: {stats['max_drawdown_duration']} days")
    print(f"Average trade duration: {stats['avg_trade_duration']}")
    print(f"Average profit per trade: {stats['avg_profit_per_trade']:.2f}")
    print(f"Average profit per winning trade: {stats['avg_profit_per_winning_trade']:.2f}")
    print(f"Average loss per losing trade: {stats['avg_loss_per_losing_trade']:.2f}")
    print(f"Longest win streak: {stats['longest_win_streak']}")
    print(f"Longest lose streak: {stats['longest_lose_streak']}")
    print("------------------------\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PowerX Strategy Backtester')
    parser.add_argument('-c', '--config', type=str, default='backtest_config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Create default config file if it doesn't exist
    if not os.path.exists(args.config):
        default_config = {
            'data_source': 'synthetic',  # 'synthetic', 'mt5', 'csv'
            'num_bars': 1000,
            'trend_strength': 0.6,
            'volatility': 1.0,
            'initial_capital': 10000.0,
            'commission': 0.0,
            'slippage': 0.0,
            'mt5': {
                'login': None,
                'password': None,
                'server': None
            },
            'data': {
                'main_file': 'data.csv',
                'higher_tf_file': None
            },
            'strategy': {
                'symbol': 'EURUSD',
                'timeframe': 'H1',
                'higher_timeframe': 'D1',
                'rsi_period': 7,
                'stoch_k_period': 14,
                'stoch_smooth_period': 3,
                'macd_fast_period': 12,
                'macd_slow_period': 26,
                'macd_signal_period': 9,
                'atr_period': 14,
                'supertrend_multiplier': 4.0,
                'supertrend_period': 10,
                'sl_multiplier': 1.5,
                'tp_multiplier': 3.0,
                'allow_longs': True,
                'allow_shorts': True
            },
            'monte_carlo': {
                'enabled': True,
                'num_simulations': 1000,
                'confidence_level': 0.95
            },
            'plot': {
                'enabled': True,
                'show': True,
                'save': False,
                'file': 'backtest_results.png'
            },
            'save_results': False,
            'results_file': 'backtest_results.csv',
            'trades_file': 'backtest_trades.csv'
        }
        
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default config file: {args.config}")
    
    # Load config
    config = load_config(args.config)
    
    # Run backtest
    results = run_backtest(config)
    
    # Print statistics
    print_stats(results['stats'])
    
    # Print Monte Carlo results if available
    if 'monte_carlo' in results:
        mc = results['monte_carlo']
        print("\n--- MONTE CARLO ANALYSIS ---")
        print(f"Number of simulations: {mc['simulations']}")
        print(f"Confidence level: {mc['confidence_level']:.0%}")
        print(f"Final equity (mean): {mc['final_equity']['mean']:.2f}")
        print(f"Final equity ({(1-mc['confidence_level'])/2*100:.1f}%): {mc['final_equity']['lower']:.2f}")
        print(f"Final equity ({(1+mc['confidence_level'])/2*100:.1f}%): {mc['final_equity']['upper']:.2f}")
        print(f"Max drawdown (mean): {mc['max_drawdown']['mean']:.2f}%")
        print(f"Max drawdown ({(1-mc['confidence_level'])/2*100:.1f}%): {mc['max_drawdown']['lower']:.2f}%")
        print(f"Max drawdown ({(1+mc['confidence_level'])/2*100:.1f}%): {mc['max_drawdown']['upper']:.2f}%")
        print(f"Success probability: {mc['success_probability']:.2f}%")
        print(f"Expected return: {mc['expected_return']:.2f}%")
        print(f"Risk (mean max drawdown): {mc['risk']:.2f}%")
        print("----------------------------\n")
    
    return results


if __name__ == '__main__':
    main()
