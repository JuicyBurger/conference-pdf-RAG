"""
Standardized logging utilities for RAG components.

This module provides consistent logging setup and utilities for all RAG components.
"""

import logging
import functools
import time
import traceback
from typing import Any, Callable, TypeVar, cast

# Type variable for generic function
F = TypeVar('F', bound=Callable[..., Any])

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

def log_execution(logger: logging.Logger) -> Callable[[F], F]:
    """Decorator to log function execution with timing.
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            func_name = func.__qualname__
            logger.debug(f"Starting {func_name}")
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"Completed {func_name} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Error in {func_name} after {execution_time:.2f}s: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise
        return cast(F, wrapper)
    return decorator
