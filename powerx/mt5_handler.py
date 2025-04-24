"""
MetaTrader 5 connection and order execution module
This module handles the connection to MT5 terminal and executes trading orders.
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5


class MT5Handler:
    """
    Handle MetaTrader 5 connection and trading operations
    """
    
    def __init__(self, login=None, password=None, server=None):
        """
        Initialize connection to MetaTrader 5 terminal
        
        Args:
            login (int): MetaTrader account login (optional)
            password (str): MetaTrader account password (optional)
            server (str): MetaTrader server name (optional)
        """
        self.connected = False
        self.login = login
        self.password = password
        self.server = server
        
        # Connect to MT5 terminal
        self.connect()
    
    def connect(self):
        """
        Connect to MetaTrader 5 terminal
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
            return False
        
        # If login credentials are provided, try to login
        if self.login and self.password:
            authorized = mt5.login(self.login, self.password, self.server)
            if not authorized:
                print(f"login failed, error code = {mt5.last_error()}")
                mt5.shutdown()
                return False
        
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from MetaTrader 5 terminal"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
    
    def get_account_info(self):
        """
        Get account information
        
        Returns:
            dict: Account info dictionary
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            print(f"Failed to get account info, error code = {mt5.last_error()}")
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage,
            'currency': account_info.currency
        }
    
    def get_symbol_info(self, symbol):
        """
        Get symbol information
        
        Args:
            symbol (str): Symbol name
            
        Returns:
            dict: Symbol info dictionary
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}, error code = {mt5.last_error()}")
            return None
        
        return {
            'tick_size': symbol_info.trade_tick_size,
            'contract_size': symbol_info.trade_contract_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
            'digits': symbol_info.digits,
            'point': symbol_info.point
        }
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date=None, include_partial=True):
        """
        Get historical price data from MT5
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe as string ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1')
            start_date (datetime): Start date
            end_date (datetime): End date (optional, defaults to now)
            include_partial (bool): Whether to include partial (current) candle
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        # Map timeframe string to MT5 timeframe enum
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        if timeframe not in timeframe_map:
            print(f"Invalid timeframe: {timeframe}")
            return None
        
        mt5_timeframe = timeframe_map[timeframe]
        
        # Set end_date to now if not provided
        if end_date is None:
            end_date = datetime.now()
        
        # Convert datetime to MT5 format
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Get rates
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to get rates for {symbol}, error code = {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Set time as index
        df = df.set_index('time')
        
        # Rename columns to match our convention
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        })
        
        # Drop incomplete candle if requested
        if not include_partial and df.index[-1].minute == datetime.now().minute:
            df = df[:-1]
        
        # Add symbol and tick_size
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            df['symbol'] = symbol
            df['tick_size'] = symbol_info['tick_size']
        
        return df
    
    def open_position(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=None):
        """
        Open a new position
        
        Args:
            symbol (str): Symbol name
            order_type (str): Order type ('BUY' or 'SELL')
            volume (float): Trade volume in lots
            price (float): Price for pending orders (optional)
            sl (float): Stop loss price (optional)
            tp (float): Take profit price (optional)
            comment (str): Order comment (optional)
            
        Returns:
            dict: Order result dictionary
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        # Map order type string to MT5 order type enum
        order_type_map = {
            'BUY': mt5.ORDER_TYPE_BUY,
            'SELL': mt5.ORDER_TYPE_SELL,
            'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
            'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
            'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
            'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
        }
        
        if order_type not in order_type_map:
            print(f"Invalid order type: {order_type}")
            return None
        
        mt5_order_type = order_type_map[order_type]
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found")
            return None
        
        # If the symbol is not available in MarketWatch, add it
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"Symbol {symbol} not found in Market Watch")
                return None
        
        # Get current price if not provided for market orders
        if price is None and mt5_order_type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            price = mt5.symbol_info_tick(symbol).ask if mt5_order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "deviation": 20,  # Maximum price deviation in points
            "magic": 12345,   # Magic number (ID)
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_IOC,  # Fill or kill
        }
        
        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Add comment if provided
        if comment:
            request["comment"] = comment
        
        # Send the request
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, error code = {result.retcode}")
            return None
        
        return {
            'ticket': result.order,
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'price': price,
            'sl': sl,
            'tp': tp,
            'comment': comment
        }
    
    def close_position(self, ticket):
        """
        Close an open position by ticket
        
        Args:
            ticket (int): Position ticket
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MT5")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"Position with ticket {ticket} not found")
            return False
        
        position = position[0]
        
        # Determine order type for closing
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "price": mt5.symbol_info_tick(position.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 12345,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send the request
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, error code = {result.retcode}")
            return False
        
        return True
    
    def modify_position(self, ticket, sl=None, tp=None):
        """
        Modify an open position (SL/TP)
        
        Args:
            ticket (int): Position ticket
            sl (float): New stop loss price (optional)
            tp (float): New take profit price (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MT5")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"Position with ticket {ticket} not found")
            return False
        
        position = position[0]
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }
        
        # Send the request
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, error code = {result.retcode}")
            return False
        
        return True
    
    def get_positions(self, symbol=None):
        """
        Get open positions
        
        Args:
            symbol (str): Symbol to filter positions (optional)
            
        Returns:
            list: List of open positions
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        # Get positions
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            print(f"Failed to get positions, error code = {mt5.last_error()}")
            return None
        
        # Convert to list of dictionaries
        positions_list = []
        for position in positions:
            positions_list.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': position.volume,
                'open_price': position.price_open,
                'current_price': position.price_current,
                'sl': position.sl,
                'tp': position.tp,
                'profit': position.profit,
                'comment': position.comment,
                'time': datetime.fromtimestamp(position.time)
            })
        
        return positions_list
