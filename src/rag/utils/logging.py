"""
Simple logging utilities for RAG components.

Provides basic logging setup for consistent formatting across modules.
"""

import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with standardized formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
