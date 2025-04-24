"""
PowerX Strategy with Basic Monte Carlo
Main executable file
"""

import argparse
import os
from datetime import datetime

from config.config_manager import ConfigManager
from brokers.mt5_broker import MT5Broker
from core.position_sizer import MonteCarloPositionSizer
from strategies.powerx_strategy import PowerXStrategy
from utils.logging_utils import setup_logging


def main():
    """Main function"""
    # Setup argument parser
    parser = argparse.ArgumentParser(description='PowerX Strategy with Basic Monte Carlo')
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    parser.add_argument('-s', '--symbol', type=str, help='Trading symbol')
    parser.add_argument('-t', '--timeframe', type=str, help='Trading timeframe (e.g. M15)')
    parser.add_argument('-ht', '--higher-timeframe', type=str, help='Higher timeframe (e.g. H1)')
    parser.add_argument('--login', type=int, help='MT5 account login')
    parser.add_argument('--password', type=str, help='MT5 account password')
    parser.add_argument('--server', type=str, help='MT5 server name')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting PowerX Strategy with Basic Monte Carlo")
    
    # Load configuration
    if args.config:
        config_file = args.config
    else:
        config_file = 'config.yaml'
        if not os.path.exists(config_file):
            # Create default config if it doesn't exist
            default_config = {
                'mt5': {
                    'login': None,
                    'password': None,
                    'server': None,
                },
                'strategy': {
                    'symbol': 'EURUSD',
                    'timeframe': 'M15',
                    'higher_timeframe': 'H1',
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
                    'allow_shorts': True,
                },
                'execution': {
                    'check_interval': 5,  # seconds
                }
            }
            ConfigManager.create_default_config(config_file, default_config)
            logger.info(f"Created default config file: {config_file}")
    
    # Load config
    config = ConfigManager.load_config(config_file)
    logger.info(f"Loaded configuration from {config_file}")
    
    # Override config with command line arguments
    strategy_config = config.get('strategy', {})
    if args.symbol:
        strategy_config['symbol'] = args.symbol
    if args.timeframe:
        strategy_config['timeframe'] = args.timeframe
    if args.higher_timeframe:
        strategy_config['higher_timeframe'] = args.higher_timeframe
    
    mt5_config = config.get('mt5', {})
    if args.login:
        mt5_config['login'] = args.login
    if args.password:
        mt5_config['password'] = args.password
    if args.server:
        mt5_config['server'] = args.server
    
    # Get strategy parameters
    strategy_params = ConfigManager.get_strategy_params(config)
    
    # Create broker instance
    broker = MT5Broker(
        login=mt5_config.get('login'),
        password=mt5_config.get('password'),
        server=mt5_config.get('server')
    )
    
    # Create position sizer
    position_sizer = MonteCarloPositionSizer()
    
    # Create strategy
    strategy = PowerXStrategy(
        broker=broker,
        position_sizer=position_sizer,
        symbol=strategy_params['symbol'],
        timeframe=strategy_params['timeframe'],
        higher_timeframe=strategy_params['higher_timeframe'],
        params=strategy_params
    )
    
    # Run strategy
    try:
        execution_params = ConfigManager.get_execution_params(config)
        strategy.run(interval_seconds=execution_params['check_interval'])
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        strategy.stop()
        logger.info("Strategy stopped")


if __name__ == '__main__':
    main()
