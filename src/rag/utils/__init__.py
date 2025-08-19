"""
Utility functions for RAG components.

This module provides shared utilities for logging, error handling, and other
common functionality across the RAG system.
"""

from .logging_utils import setup_logger, log_execution
from .error_utils import handle_errors
from ..errors import (
    RAGError, 
    EmbeddingError, 
    RetrievalError, 
    GenerationError, 
    DatabaseError, 
    ConfigurationError, 
    GraphError
)

__all__ = [
    # Logging utilities
    "setup_logger",
    "log_execution",
    
    # Error handling utilities
    "handle_errors",
    
    # Error types
    "RAGError",
    "EmbeddingError", 
    "RetrievalError",
    "GenerationError",
    "DatabaseError",
    "ConfigurationError",
    "GraphError"
]
