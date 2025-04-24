"""
Logging utilities for PowerX strategy
"""

import os
import logging
from datetime import datetime


def setup_logging(log_dir: str = 'logs', log_level=logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir (str): Directory to store log files
        log_level: Logging level
        
    Returns:
        logging.Logger: Logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Setup logging
    log_filename = f"{log_dir}/powerx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('powerx')
