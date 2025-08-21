"""
Standardized models for RAG engine requests and responses.

This module defines the data structures used for communication between
RAG engines and their clients.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class EngineType(Enum):
    """Available RAG engine types."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"

@dataclass
class RetrievalRequest:
    """Standardized request for RAG engine retrieval."""
    query: str
    room_id: Optional[str] = None
    top_k: int = 10
    score_threshold: float = 0.3
    max_hops: Optional[int] = None
    expansion_top_k: Optional[int] = None
    prefer_chat_scope: bool = True
    scope_filter: Optional[str] = None
    extra_filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize request parameters."""
        if not self.query or not isinstance(self.query, str):
            raise ValueError("Query must be a non-empty string")
        
        self.top_k = max(1, min(50, self.top_k))
        self.score_threshold = max(0.0, min(1.0, self.score_threshold))

@dataclass
class RetrievalResponse:
    """Standardized response from RAG engine retrieval."""
    evidence: List[Evidence]
    engine_type: EngineType
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Number of evidence items returned."""
        return len(self.evidence)
    
    @property
    def has_results(self) -> bool:
        """Whether any evidence was found."""
        return len(self.evidence) > 0

@dataclass
class AnswerRequest:
    """Standardized request for RAG engine answer generation."""
    query: str
    room_id: Optional[str] = None
    pdf_summary: Optional[str] = None
    retrieval_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.query or not isinstance(self.query, str):
            raise ValueError("Query must be a non-empty string")

@dataclass
class AnswerResponse:
    """Standardized response from RAG engine answer generation."""
    answer: str
    evidence: List[Evidence]
    engine_type: EngineType
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class Evidence:
    """Standardized evidence container for RAG engines."""
    
    def __init__(self, raw_hit: Any, engine_type: EngineType):
        """Initialize evidence from a raw hit object.
        
        Args:
            raw_hit: Raw hit object from the underlying storage system
            engine_type: Type of engine that produced this evidence
        """
        self.raw = raw_hit
        self.engine_type = engine_type
        
        # Extract common fields for convenience
        self.id = getattr(raw_hit, 'id', None)
        self.score = getattr(raw_hit, 'score', 0.0)
        self.payload = getattr(raw_hit, 'payload', {})
        
    @property
    def doc_id(self) -> Optional[str]:
        """Get document ID from payload."""
        return self.payload.get('doc_id')
    
    @property
    def page(self) -> Optional[str]:
        """Get page from payload."""
        return self.payload.get('page')
    
    @property
    def text(self) -> str:
        """Get text content from payload."""
        return self.payload.get('text', '') or self.payload.get('content', '')
    
    @property
    def content(self) -> str:
        """Get content (alias for text)."""
        return self.text
    
    @property
    def type(self) -> str:
        """Get content type from payload."""
        return self.payload.get('type', 'unknown')
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata from payload."""
        return self.payload
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to dictionary for serialization."""
        return {
            'id': self.id,
            'score': self.score,
            'doc_id': self.doc_id,
            'page': self.page,
            'text': self.text,
            'type': self.type,
            'engine_type': self.engine_type.value
        }
