"""
Unit tests for Monte Carlo position sizing
"""

import unittest
from powerx.monte_carlo import MonteCarloPositionSizer


class TestMonteCarloPositionSizer(unittest.TestCase):
    
    def setUp(self):
        """Setup test case"""
        self.sizer = MonteCarloPositionSizer()
    
    def test_initial_sequence(self):
        """Test initial sequence is [0, 1]"""
        self.assertEqual(self.sizer.sequence, [0, 1])
    
    def test_initial_bet_amount(self):
        """Test initial bet amount is 1.0"""
        bet_amount = self.sizer.calculate_bet_amount()
        self.assertEqual(bet_amount, 1.0)
    
    def test_win_sequence_update(self):
        """Test sequence update after a win"""
        # Initial sequence [0, 1]
        self.sizer.update_sequence(True)  # Win
        # After winning with sequence [0, 1], sequence should be empty
        # then reset to [0, 1]
        self.assertEqual(self.sizer.sequence, [0, 1])
    
    def test_loss_sequence_update(self):
        """Test sequence update after a loss"""
        # Initial sequence [0, 1]
        # Initial bet amount is 1.0
        self.sizer.calculate_bet_amount()  # Calculate to set last_bet_amount
        self.sizer.update_sequence(False)  # Loss
        # After losing, bet amount (1.0) should be added to sequence
        self.assertEqual(self.sizer.sequence, [0, 1, 1])
    
    def test_bet_amount_calculation(self):
        """Test bet amount calculation"""
        # Initial sequence [0, 1]
        bet_amount = self.sizer.calculate_bet_amount()
        self.assertEqual(bet_amount, 1.0)
        
        # Simulate a loss, sequence becomes [0, 1, 1]
        self.sizer.update_sequence(False)
        bet_amount = self.sizer.calculate_bet_amount()
        # Bet amount should be first + last element: 0 + 1 = 1
        self.assertEqual(bet_amount, 1.0)
        
        # Simulate another loss, sequence becomes [0, 1, 1, 1]
        self.sizer.update_sequence(False)
        bet_amount = self.sizer.calculate_bet_amount()
        # Bet amount should be first + last element: 0 + 1 = 1
        self.assertEqual(bet_amount, 1.0)
        
        # Simulate another loss, sequence becomes [0, 1, 1, 1, 1]
        self.sizer.update_sequence(False)
        bet_amount = self.sizer.calculate_bet_amount()
        # Bet amount should be first + last element: 0 + 1 = 1
        self.assertEqual(bet_amount, 1.0)
        
        # Simulate a win, sequence becomes [1, 1, 1]
        self.sizer.update_sequence(True)
        bet_amount = self.sizer.calculate_bet_amount()
        # Bet amount should be first + last element: 1 + 1 = 2
        self.assertEqual(bet_amount, 2.0)
    
    def test_complex_sequence(self):
        """Test a more complex sequence of wins and losses"""
        # Starting with [0, 1]
        
        # Lose (bet 1)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(False)
        # Sequence should be [0, 1, 1]
        self.assertEqual(self.sizer.sequence, [0, 1, 1])
        
        # Lose (bet 1)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(False)
        # Sequence should be [0, 1, 1, 1]
        self.assertEqual(self.sizer.sequence, [0, 1, 1, 1])
        
        # Win (bet 1)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(True)
        # Sequence should be [1, 1]
        self.assertEqual(self.sizer.sequence, [1, 1])
        
        # Win (bet 2)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(True)
        # Sequence should be empty, then reset to [0, 1]
        self.assertEqual(self.sizer.sequence, [0, 1])
        
        # Lose (bet 1)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(False)
        # Sequence should be [0, 1, 1]
        self.assertEqual(self.sizer.sequence, [0, 1, 1])
        
        # Win (bet 1)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(True)
        # Sequence should be [1]
        self.assertEqual(self.sizer.sequence, [1])
        
        # Lose (bet 1)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(False)
        # Sequence should be [1, 1]
        self.assertEqual(self.sizer.sequence, [1, 1])
        
        # Win (bet 2)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(True)
        # Sequence should be empty, then reset to [0, 1]
        self.assertEqual(self.sizer.sequence, [0, 1])
    
    def test_decomposition_rule(self):
        """Test the special decomposition rule"""
        # Create custom sequence with one element > 1
        self.sizer.sequence = [4]
        
        # Win (sequence should decompose)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(True)
        # 4 should decompose to [2, 2]
        self.assertEqual(self.sizer.sequence, [2, 2])
        
        # Reset
        self.sizer.sequence = [5]
        
        # Win (sequence should decompose)
        self.sizer.calculate_bet_amount()
        self.sizer.update_sequence(True)
        # 5 should decompose to [2, 3]
        self.assertEqual(self.sizer.sequence, [2, 3])
    
    def test_minimum_bet_amount(self):
        """Test the minimum bet amount is 1.0"""
        # Create a sequence that would result in a bet of 0
        self.sizer.sequence = [0, 0]
        bet_amount = self.sizer.calculate_bet_amount()
        # Minimum bet should be 1.0
        self.assertEqual(bet_amount, 1.0)


if __name__ == '__main__':
    unittest.main()
