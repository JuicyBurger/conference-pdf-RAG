"""
Base RAG engine class with common functionality.

This module defines the core RAG engine interface and provides a base implementation
with shared functionality across different engine types.
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict
import logging
from abc import ABC, abstractmethod

from .models import (
    RetrievalRequest, 
    RetrievalResponse, 
    AnswerRequest, 
    AnswerResponse, 
    Evidence, 
    EngineType
)

# Configure logging
logger = logging.getLogger(__name__)

class BaseRAGEngine(ABC):
    """Base class for RAG engines with common functionality.
    
    This abstract base class defines the interface for all RAG engines
    and provides common functionality that can be shared across different
    engine implementations.
    """
    
    def __init__(self, engine_type: EngineType):
        """Initialize the RAG engine.
        
        Args:
            engine_type: Type of this engine
        """
        self.engine_type = engine_type
        logger.info(f"Initializing {self.engine_type.value} engine")
        self._initialize()
        
    def _initialize(self):
        """Initialize engine-specific components."""
        try:
            # Call engine-specific initialization
            self._initialize_engine()
            logger.info(f"{self.engine_type.value} engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.engine_type.value} engine: {e}")
            raise
    
    @abstractmethod
    def _initialize_engine(self):
        """Initialize engine-specific components. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve relevant context for a query.
        
        Args:
            request: Standardized retrieval request
            
        Returns:
            Standardized retrieval response
        """
        pass
    
    def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Generate an answer for a query using retrieved context.
        
        Args:
            request: Standardized answer request
            
        Returns:
            Standardized answer response
        """
        logger.info(f"{self.engine_type.value} engine answering query: {request.query[:50]}...")
        
        # Create retrieval request from answer request
        retrieval_request = RetrievalRequest(
            query=request.query,
            room_id=request.room_id,
            **(request.retrieval_params or {})
        )
        
        # Retrieve context
        retrieval_response = self.retrieve(retrieval_request)
        
        # Handle empty results
        if not retrieval_response.has_results:
            answer = "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"
            return AnswerResponse(
                answer=answer,
                evidence=[],
                engine_type=self.engine_type,
                query=request.query
            )
        
        # Generate answer from retrieved evidence
        answer = self._generate_answer(request, retrieval_response.evidence)
        
        return AnswerResponse(
            answer=answer,
            evidence=retrieval_response.evidence,
            engine_type=self.engine_type,
            query=request.query
        )
    
    def _generate_answer(self, request: AnswerRequest, evidence: List[Evidence]) -> str:
        """Generate answer from evidence. To be implemented by subclasses.
        
        Args:
            request: Answer request
            evidence: Retrieved evidence
            
        Returns:
            Generated answer
        """
        # Import here to avoid circular dependencies
        from ..qa_generation import generate_answer
        
        # Convert evidence to raw hits for backward compatibility
        raw_hits = [ev.raw for ev in evidence]
        
        # Log the evidence that will be used for prompt building
        logger.info(f"🔍 FINAL EVIDENCE FOR PROMPT BUILDING:")
        logger.info(f"   - Engine type: {self.engine_type.value}")
        logger.info(f"   - Total evidence items: {len(evidence)}")
        logger.info(f"   - PDF summary provided: {bool(request.pdf_summary)}")
        
        # Log evidence order and types
        logger.info("🔍 EVIDENCE ORDER ANALYSIS:")
        vector_count = 0
        graph_count = 0
        for i, ev in enumerate(evidence):
            source_type = "vector"
            if hasattr(ev.raw, 'payload') and ev.raw.payload:
                source_type = ev.raw.payload.get('source', 'vector')
            
            doc_id = "unknown"
            page = "unknown"
            if hasattr(ev.raw, 'payload') and ev.raw.payload:
                doc_id = ev.raw.payload.get('doc_id', 'unknown')
                page = ev.raw.payload.get('page', 'unknown')
            
            score = getattr(ev.raw, 'score', 0.0)
            logger.info(f"   {i+1}. [{doc_id} p{page}] {source_type} (score: {score:.3f})")
            
            # Count evidence types
            if source_type == 'graph_knowledge':
                graph_count += 1
            else:
                vector_count += 1
        
        logger.info(f"📊 EVIDENCE TYPE SUMMARY: {vector_count} vector + {graph_count} graph = {len(evidence)} total")
        
        # Add PDF summary if provided
        if request.pdf_summary:
            logger.info("📄 ADDING PDF SUMMARY TO EVIDENCE")
            # Create a synthetic hit for the PDF summary
            pdf_hit = type('Hit', (), {
                'payload': {'content': request.pdf_summary[:1000], 'page': 'uploaded_pdf'}
            })()
            # Combine PDF summary with retrieved evidence
            combined = [pdf_hit] + raw_hits[:5]  # Limit to top 5 evidence items
            logger.info(f"📄 FINAL COMBINED EVIDENCE: 1 PDF + {len(raw_hits[:5])} retrieved = {len(combined)} total")
            return generate_answer(request.query, combined)
        
        # Generate answer from retrieved evidence
        logger.info(f"📄 FINAL EVIDENCE: {len(raw_hits)} retrieved items")
        return generate_answer(request.query, raw_hits)
    
    # Backward compatibility methods
    def retrieve_legacy(self, query: str, room_id: Optional[str] = None, **kwargs) -> List[Evidence]:
        """Legacy retrieval method for backward compatibility."""
        request = RetrievalRequest(
            query=query,
            room_id=room_id,
            **kwargs
        )
        response = self.retrieve(request)
        return response.evidence
    
    def answer_legacy(self, query: str, room_id: Optional[str] = None, pdf_summary: Optional[str] = None, **kwargs) -> str:
        """Legacy answer method for backward compatibility."""
        request = AnswerRequest(
            query=query,
            room_id=room_id,
            pdf_summary=pdf_summary,
            retrieval_params=kwargs
        )
        response = self.answer(request)
        return response.answer


# For backward compatibility
RAGEngine = BaseRAGEngine