"""
Vector-based RAG engine implementation.

This module implements a RAG engine that uses vector search for retrieval.
It preserves the current behavior of using global Qdrant collection retrieval
and the existing generator with optional reranking.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRAGEngine
from .models import (
    RetrievalRequest, 
    RetrievalResponse, 
    Evidence, 
    EngineType
)
from ..retrieval.retrieval_service import retrieval_service
from ...models.reranker import rerank

# Configure logging
logger = logging.getLogger(__name__)

class VectorRAGEngine(BaseRAGEngine):
    """Vector-based RAG engine using dense retrieval.
    
    This engine uses vector search for retrieval and preserves the current behavior
    of using global Qdrant collection retrieval with optional reranking.
    """
    
    def __init__(self):
        """Initialize the vector RAG engine."""
        super().__init__(EngineType.VECTOR)
    
    def _initialize_engine(self):
        """Initialize vector engine-specific components."""
        # No special initialization needed for vector engine
        pass
    
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve context using vector search.
        
        Args:
            request: Standardized retrieval request
            
        Returns:
            Standardized retrieval response
        """
        logger.info(f"Vector retrieval: query='{request.query[:50]}...', top_k={request.top_k}")
        
        # Perform vector retrieval
        vector_hits = retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            room_id=request.room_id,
            prefer_chat_scope=request.prefer_chat_scope,
            scope_filter=request.scope_filter,
            extra_filters=request.extra_filters
        )
        
        # Convert hits to Evidence objects
        evidence = [Evidence(hit, self.engine_type) for hit in vector_hits]
        logger.info(f"Vector search returned {len(evidence)} results")
        
        # Apply reranking if we have multiple results
        if len(evidence) > 1:
            logger.info("Applying reranking to vector results")
            reranked_hits = rerank(request.query, [ev.raw for ev in evidence])
            evidence = [Evidence(hit, self.engine_type) for hit in reranked_hits]
            logger.info(f"After reranking: {len(evidence)} results")
        
        return RetrievalResponse(
            evidence=evidence,
            engine_type=self.engine_type,
            query=request.query,
            metadata={
                'reranked': len(evidence) > 1,
                'original_count': len(vector_hits)
            }
        )
    
    # Backward compatibility
    def retrieve_legacy(self, query: str, room_id: Optional[str] = None, **kwargs) -> List[Evidence]:
        """Legacy retrieval method for backward compatibility."""
        return super().retrieve_legacy(query, room_id, **kwargs)