"""
Basic Decomposition Monte Carlo position sizing implementation
This module implements the position sizing algorithm based on the
Basic Decomposition Monte Carlo method.
"""


class MonteCarloPositionSizer:
    """
    A position sizer that implements the Basic Decomposition Monte Carlo method.
    """
    
    def __init__(self):
        """Initialize with default sequence [0, 1]"""
        self.sequence = [0, 1]
        self.last_bet_amount = 1.0
    
    def calculate_bet_amount(self):
        """
        Calculate the next bet amount based on the current sequence.
        
        Returns:
            float: The next bet amount (number of units)
        """
        seq_size = len(self.sequence)
        
        if seq_size >= 2:
            # Sum the first and last numbers in the sequence
            bet_amount = float(self.sequence[0] + self.sequence[-1])
        elif seq_size == 1:
            # If only one number left, use it
            bet_amount = float(self.sequence[0])
        else:
            # If sequence is empty, use default starting bet
            bet_amount = 1.0
        
        # Store this bet amount for reference
        self.last_bet_amount = max(1.0, bet_amount)
        return self.last_bet_amount
    
    def update_sequence(self, is_win):
        """
        Update the sequence based on the trade result.
        
        Args:
            is_win (bool): True if the trade was profitable, False otherwise
        """
        if is_win:
            # If win: remove first and last numbers from sequence
            if len(self.sequence) >= 2:
                self.sequence.pop(0)  # Remove first
                self.sequence.pop(-1)  # Remove last
            elif len(self.sequence) == 1:
                self.sequence.clear()
            
            # Special rule: If one number remains and it's > 1, decompose it
            if len(self.sequence) == 1 and self.sequence[0] > 1:
                left_num = self.sequence[0] // 2
                right_num = self.sequence[0] - left_num
                self.sequence.clear()
                self.sequence.append(left_num)
                self.sequence.append(right_num)
            elif len(self.sequence) == 1 and self.sequence[0] <= 1:
                self.sequence.clear()
        else:
            # If loss: add the bet amount to the end of the sequence
            self.sequence.append(int(self.last_bet_amount))
        
        # If sequence is empty, restart cycle with [0, 1]
        if len(self.sequence) == 0:
            self.sequence = [0, 1]
    
    def get_current_sequence(self):
        """
        Get the current sequence.
        
        Returns:
            list: The current sequence
        """
        return self.sequence.copy()
