"""
PowerX Strategy with Basic Monte Carlo
Main executable file
"""

import argparse
import logging
import yaml
import os
from datetime import datetime

from powerx.strategy import PowerXStrategy


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup logging
    log_filename = f"logs/powerx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('powerx')


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


def create_default_config():
    """
    Create default configuration file if it doesn't exist
    
    Returns:
        str: Path to config file
    """
    config_file = 'config.yaml'
    
    if not os.path.exists(config_file):
        config = {
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
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    return config_file


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
        config_file = create_default_config()
        logger.info(f"Created default config file: {config_file}")
    
    config = load_config(config_file)
    logger.info(f"Loaded configuration from {config_file}")
    
    # Override config with command line arguments
    if args.symbol:
        config['strategy']['symbol'] = args.symbol
    if args.timeframe:
        config['strategy']['timeframe'] = args.timeframe
    if args.higher_timeframe:
        config['strategy']['higher_timeframe'] = args.higher_timeframe
    if args.login:
        config['mt5']['login'] = args.login
    if args.password:
        config['mt5']['password'] = args.password
    if args.server:
        config['mt5']['server'] = args.server
    
    # Create strategy
    strategy = PowerXStrategy(
        symbol=config['strategy']['symbol'],
        timeframe=config['strategy']['timeframe'],
        higher_timeframe=config['strategy']['higher_timeframe'],
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
        allow_shorts=config['strategy']['allow_shorts'],
        mt5_login=config['mt5']['login'],
        mt5_password=config['mt5']['password'],
        mt5_server=config['mt5']['server']
    )
    
    # Run strategy
    try:
        strategy.run(interval_seconds=config['execution']['check_interval'])
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        strategy.stop()
        logger.info("Strategy stopped")


if __name__ == '__main__':
    main()
