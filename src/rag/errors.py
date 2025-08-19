"""
Standardized error types for RAG components.

This module defines custom exception types for different error scenarios
in the RAG system, making it easier to handle and report errors consistently.
"""

class RAGError(Exception):
    """Base exception for all RAG-related errors."""
    pass

class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass

class RetrievalError(RAGError):
    """Raised when retrieval operations fail."""
    pass

class GenerationError(RAGError):
    """Raised when text generation fails."""
    pass

class DatabaseError(RAGError):
    """Raised when database operations fail."""
    pass

class ConfigurationError(RAGError):
    """Raised when configuration is invalid."""
    pass

class GraphError(RAGError):
    """Raised when graph operations fail."""
    pass
