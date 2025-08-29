"""
RAG engine implementations for different retrieval strategies.

This module provides:
- VectorRAGEngine: Dense vector similarity search using Qdrant
- GraphRAGEngine: Knowledge graph traversal using Neo4j  
- HybridRAGEngine: Combined vector + graph retrieval
- BaseRAGEngine: Abstract base class for all engines
- Standardized request/response models
"""

from .base import BaseRAGEngine
from .vector import VectorRAGEngine
from .graph import GraphRAGEngine  
from .hybrid import HybridRAGEngine
from .models import (
    EngineType,
    RetrievalRequest,
    RetrievalResponse,
    AnswerRequest,
    AnswerResponse,
    Evidence
)

__all__ = [
    # Engine implementations
    'BaseRAGEngine',
    'VectorRAGEngine',
    'GraphRAGEngine',
    'HybridRAGEngine',
    
    # Request/Response models
    'EngineType',
    'RetrievalRequest',
    'RetrievalResponse', 
    'AnswerRequest',
    'AnswerResponse',
    'Evidence'
]
