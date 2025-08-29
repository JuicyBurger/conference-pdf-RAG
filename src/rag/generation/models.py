"""
Shared data models for unified generation system.

This module defines common data structures used across QA generation,
suggestions generation, and other generation tasks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class Question:
    """Base question model."""
    text: str
    doc_id: str
    page: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QAPair:
    """Question-Answer pair for training data."""
    question: str
    answer: str
    doc_id: str
    sources: List[Dict[str, Any]] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "doc_id": self.doc_id,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class QuestionSuggestion:
    """Question suggestion for frontend display."""
    question: str
    doc_id: str
    popularity_score: float = 0.0
    created_at: Optional[datetime] = None
    page: Optional[int] = None
    room_id: Optional[str] = None
    suggestion_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.suggestion_id is None:
            self.suggestion_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "question": self.question,
            "doc_id": self.doc_id,
            "popularity_score": self.popularity_score,
            "created_at": self.created_at.isoformat(),
            "page": self.page,
            "room_id": self.room_id,
            "suggestion_id": self.suggestion_id,
            "metadata": self.metadata
        }


@dataclass
class GenerationRequest:
    """Request for question/QA generation."""
    doc_id: str
    num_items: int = 5
    timeout: float = 120.0
    context_top_k: int = 20
    chat_context: Optional[str] = None
    room_id: Optional[str] = None
    use_lightweight: bool = True
    custom_prompts: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.custom_prompts is None:
            self.custom_prompts = {}


@dataclass
class GenerationResult:
    """Result from generation operations."""
    success: bool
    items: List[Any]  # Questions, QAPairs, or QuestionSuggestions
    doc_id: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def count(self) -> int:
        """Number of generated items."""
        return len(self.items) if self.items else 0
