"""
Error handling utilities for RAG components.

This module provides utilities for consistent error handling across the RAG system.
"""

import logging
import functools
from typing import Any, Callable, Optional, TypeVar, cast, Union, Type

from .errors import RAGError

# Type variable for generic function
F = TypeVar('F', bound=Callable[..., Any])

def handle_errors(
    error_class: Type[Exception] = RAGError,
    fallback_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False
) -> Callable[[F], F]:
    """
    Decorator to standardize error handling.
    
    Args:
        error_class: Exception class to wrap errors in
        fallback_return: Value to return on error if not reraising
        log_level: Logging level for errors
        reraise: Whether to reraise the wrapped exception
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Error in {func.__qualname__}: {str(e)}",
                    exc_info=True
                )
                
                if reraise:
                    if isinstance(e, error_class):
                        raise
                    raise error_class(str(e)) from e
                return fallback_return
        return cast(F, wrapper)
    return decorator
