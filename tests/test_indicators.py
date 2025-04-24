"""
Unit tests for technical indicators
"""

import unittest
import pandas as pd
import numpy as np
from powerx.indicators import (
    calculate_rsi, calculate_stochastic, calculate_macd,
    calculate_atr, calculate_supertrend
)


class TestIndicators(unittest.TestCase):
    
    def setUp(self):
        """Setup test data"""
        # Create test data with OHLC prices
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with a trend and some oscillations
        close_prices = np.linspace(100, 200, 100) + np.sin(np.linspace(0, 10, 100)) * 10
        high_prices = close_prices + np.random.rand(100) * 5
        low_prices = close_prices - np.random.rand(100) * 5
        open_prices = close_prices - np.random.rand(100) * 10 + 5
        
        self.data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }, index=dates)
    
    def test_rsi(self):
        """Test RSI calculation"""
        rsi = calculate_rsi(self.data, period=14)
        
        # Check result is not None
        self.assertIsNotNone(rsi)
        
        # Check RSI is between 0 and 100
        self.assertTrue(all(0 <= r <= 100 for r in rsi[14:]))
        
        # Check NaN values at the beginning (due to the calculation)
        self.assertTrue(all(np.isnan(r) for r in rsi[:14]))
        
        # Check length
        self.assertEqual(len(rsi), len(self.data))
    
    def test_stochastic(self):
        """Test Stochastic calculation"""
        stoch = calculate_stochastic(self.data, k_period=14, smooth_period=3)
        
        # Check result is not None
        self.assertIsNotNone(stoch)
        
        # Check Stochastic is between 0 and 100
        self.assertTrue(all(0 <= s <= 100 for s in stoch[14:] if not np.isnan(s)))
        
        # Check NaN values at the beginning (due to the calculation)
        self.assertTrue(all(np.isnan(s) for s in stoch[:14]))
        
        # Check length
        self.assertEqual(len(stoch), len(self.data))
    
    def test_macd(self):
        """Test MACD calculation"""
        macd, signal, hist = calculate_macd(self.data, fast_period=12, slow_period=26, signal_period=9)
        
        # Check results are not None
        self.assertIsNotNone(macd)
        self.assertIsNotNone(signal)
        self.assertIsNotNone(hist)
        
        # Check lengths
        self.assertEqual(len(macd), len(self.data))
        self.assertEqual(len(signal), len(self.data))
        self.assertEqual(len(hist), len(self.data))
        
        # Check NaN values at the beginning (due to the calculation)
        self.assertTrue(all(np.isnan(m) for m in macd[:26]))
        self.assertTrue(all(np.isnan(s) for s in signal[:34]))  # 26 + 9 - 1
        self.assertTrue(all(np.isnan(h) for h in hist[:34]))
    
    def test_atr(self):
        """Test ATR calculation"""
        atr = calculate_atr(self.data, period=14)
        
        # Check result is not None
        self.assertIsNotNone(atr)
        
        # Check ATR values are positive
        self.assertTrue(all(a > 0 for a in atr[14:] if not np.isnan(a)))
        
        # Check NaN values at the beginning (due to the calculation)
        self.assertTrue(all(np.isnan(a) for a in atr[:14]))
        
        # Check length
        self.assertEqual(len(atr), len(self.data))
    
    def test_supertrend(self):
        """Test SuperTrend calculation"""
        supertrend, direction = calculate_supertrend(self.data, multiplier=3, period=10)
        
        # Check results are not None
        self.assertIsNotNone(supertrend)
        self.assertIsNotNone(direction)
        
        # Check lengths
        self.assertEqual(len(supertrend), len(self.data))
        self.assertEqual(len(direction), len(self.data))
        
        # Check direction is either 1 (downtrend) or -1 (uptrend)
        self.assertTrue(all(d in [1, -1, 0] for d in direction))
        
        # First value should be in downtrend as per implementation
        self.assertEqual(direction[0], 1)


if __name__ == '__main__':
    unittest.main()
