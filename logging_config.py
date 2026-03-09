"""
Logging configuration for the RAG-Based LLM Chatbot application.
"""

import logging
import os
from datetime import datetime
import sys

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"chatbot_{timestamp}.log")

# Ensure log file can be written
try:
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Log file initialized at {datetime.now()}\n")
except Exception as e:
    print(f"Error creating log file: {e}")

# Configure logging - set up root logger (only once)
def configure_logging():
    """Configure logging - called only once."""
    
    root_logger = logging.getLogger()
    
    # Skip if already configured
    if root_logger.handlers:
        return
    
    try:
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.stream.reconfigure(encoding='utf-8')  # Force UTF-8 for console
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
    except Exception as e:
        print(f"Error configuring logging: {e}")

# Initialize logging on import
configure_logging()

def get_logger(name):
    """Get a logger instance with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True  # Ensure logs propagate to root logger
    return logger
