"""
Configuration management for PowerX strategy
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager class"""
    
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_file (str): Path to config file
            
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def create_default_config(config_file: str, default_config: Dict[str, Any]) -> None:
        """
        Create default configuration file if it doesn't exist
        
        Args:
            config_file (str): Path to config file
            default_config (dict): Default configuration dictionary
        """
        if not os.path.exists(config_file):
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            print(f"Created default config file: {config_file}")
    
    @staticmethod
    def get_strategy_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get strategy parameters from configuration
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Strategy parameters dictionary
        """
        strategy_config = config.get('strategy', {})
        return {
            'symbol': strategy_config.get('symbol', 'EURUSD'),
            'timeframe': strategy_config.get('timeframe', 'M15'),
            'higher_timeframe': strategy_config.get('higher_timeframe', 'H1'),
            'rsi_period': strategy_config.get('rsi_period', 7),
            'stoch_k_period': strategy_config.get('stoch_k_period', 14),
            'stoch_smooth_period': strategy_config.get('stoch_smooth_period', 3),
            'macd_fast_period': strategy_config.get('macd_fast_period', 12),
            'macd_slow_period': strategy_config.get('macd_slow_period', 26),
            'macd_signal_period': strategy_config.get('macd_signal_period', 9),
            'atr_period': strategy_config.get('atr_period', 14),
            'supertrend_multiplier': strategy_config.get('supertrend_multiplier', 4.0),
            'supertrend_period': strategy_config.get('supertrend_period', 10),
            'sl_multiplier': strategy_config.get('sl_multiplier', 1.5),
            'tp_multiplier': strategy_config.get('tp_multiplier', 3.0),
            'allow_longs': strategy_config.get('allow_longs', True),
            'allow_shorts': strategy_config.get('allow_shorts', True),
        }
    
    @staticmethod
    def get_broker_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get broker parameters from configuration
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Broker parameters dictionary
        """
        mt5_config = config.get('mt5', {})
        return {
            'login': mt5_config.get('login'),
            'password': mt5_config.get('password'),
            'server': mt5_config.get('server'),
        }
    
    @staticmethod
    def get_execution_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get execution parameters from configuration
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Execution parameters dictionary
        """
        execution_config = config.get('execution', {})
        return {
            'check_interval': execution_config.get('check_interval', 5),
        }
