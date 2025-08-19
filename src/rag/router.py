"""
RAG Router: delegates query answering to the appropriate engine based on room metadata.

This module provides a router that selects the appropriate RAG engine based on
room configuration or explicit mode selection.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging

from src.API.services.chat_service import chat_service
from src.config import RAG_DEFAULT_MODE
from .engine.base import BaseRAGEngine
from .engine.models import EngineType, RetrievalRequest, RetrievalResponse, AnswerRequest, AnswerResponse
from .utils import handle_errors, ConfigurationError, setup_logger

# Configure logging
logger = setup_logger(__name__)

class RAGRouter:
    """Router for selecting the appropriate RAG engine based on context.
    
    This router delegates query answering to the appropriate engine based on
    room metadata or explicit mode selection.
    """
    
    def __init__(self):
        """Initialize the RAG router with all available engines."""
        logger.info("Initializing RAG router")
        self._engines = {}
        self._initialize_engines()
    
    @handle_errors(error_class=ConfigurationError, reraise=True)
    def _initialize_engines(self):
        """Initialize all available RAG engines."""
        # Import engines lazily to avoid heavy deps at import time
        from .engine.vector import VectorRAGEngine
        from .engine.graph import GraphRAGEngine
        from .engine.hybrid import HybridRAGEngine
        
        # Initialize engines
        self._engines = {
            'vector': VectorRAGEngine(),
            'graph': GraphRAGEngine(),
            'hybrid': HybridRAGEngine()
        }
        
        # Set default mode
        self.default_mode = RAG_DEFAULT_MODE.lower() if hasattr(RAG_DEFAULT_MODE, 'lower') else 'vector'
        
        logger.info(f"RAG router initialized with default mode: {self.default_mode}")
        logger.info(f"Available engines: {list(self._engines.keys())}")
    
    def _get_room_mode(self, room_id: Optional[str]) -> str:
        """Get the RAG mode for a specific room.
        
        Args:
            room_id: Room ID to get mode for
            
        Returns:
            RAG mode (vector, graph, or hybrid)
        """
        if not room_id:
            return self.default_mode
            
        try:
            # Try to get room from cache first
            room = chat_service.rooms.get(room_id)
            
            if not room:
                # Attempt to fetch if not cached
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        data = loop.run_until_complete(chat_service.get_room(room_id))
                        room = data or {}
                    finally:
                        loop.close()
                except Exception as e:
                    logger.warning(f"Failed to fetch room {room_id}: {e}")
                    room = {}
                    
            # Get mode from room data
            mode = (room or {}).get("rag_mode")
            if mode and hasattr(mode, 'lower'):
                mode = mode.lower()
                if mode in self._engines:
                    logger.info(f"Room {room_id} configured for {mode} engine")
                    return mode
                else:
                    logger.warning(f"Invalid mode '{mode}' for room {room_id}, using default")
            
            logger.info(f"Room {room_id} using default mode: {self.default_mode}")
            return self.default_mode
            
        except Exception as e:
            logger.warning(f"Failed to get room mode for {room_id}: {e}")
            return self.default_mode
    
    def get_engine(self, room_id: Optional[str] = None, mode: Optional[str] = None) -> BaseRAGEngine:
        """Get the appropriate RAG engine based on context.
        
        Args:
            room_id: Optional room ID for context
            mode: Optional explicit mode override
            
        Returns:
            Selected RAG engine
        """
        # Use explicit mode if provided, otherwise get from room
        if mode and hasattr(mode, 'lower'):
            effective_mode = mode.lower()
        else:
            effective_mode = self._get_room_mode(room_id)
        
        # Validate mode
        if effective_mode not in self._engines:
            logger.warning(f"Invalid mode '{effective_mode}', falling back to default")
            effective_mode = self.default_mode
        
        engine = self._engines[effective_mode]
        logger.info(f"Selected {effective_mode} engine for room_id={room_id}")
        return engine
    
    def retrieve(self, request: RetrievalRequest, room_id: Optional[str] = None, mode: Optional[str] = None) -> RetrievalResponse:
        """Retrieve context using the appropriate engine.
        
        Args:
            request: Standardized retrieval request
            room_id: Optional room ID for context
            mode: Optional explicit mode override
            
        Returns:
            Standardized retrieval response
        """
        try:
            engine = self.get_engine(room_id, mode)
            return engine.retrieve(request)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Return empty response on error
            return RetrievalResponse(
                evidence=[],
                engine_type=EngineType.VECTOR,  # Default fallback
                query=request.query
            )
    
    def answer(self, request: AnswerRequest, room_id: Optional[str] = None, mode: Optional[str] = None) -> AnswerResponse:
        """Generate an answer using the appropriate engine.
        
        Args:
            request: Standardized answer request
            room_id: Optional room ID for context
            mode: Optional explicit mode override
            
        Returns:
            Standardized answer response
        """
        try:
            engine = self.get_engine(room_id, mode)
            return engine.answer(request)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Return error response
            return AnswerResponse(
                answer="抱歉，我在處理您的請求時遇到了問題。請再試一次或換一種問法。",
                evidence=[],
                engine_type=EngineType.VECTOR,  # Default fallback
                query=request.query
            )
    
    # Backward compatibility methods
    def retrieve_legacy(self, query: str, room_id: Optional[str] = None, mode: Optional[str] = None, **kwargs) -> List:
        """Legacy retrieval method for backward compatibility."""
        try:
            from .engine.models import RetrievalRequest
            request = RetrievalRequest(query=query, room_id=room_id, **kwargs)
            response = self.retrieve(request, room_id, mode)
            return response.evidence
        except Exception as e:
            logger.error(f"Legacy retrieval failed: {e}")
            return []
    
    def answer_legacy(self, query: str, room_id: Optional[str] = None, pdf_summary: Optional[str] = None, mode: Optional[str] = None, **kwargs) -> str:
        """Legacy answer method for backward compatibility."""
        try:
            from .engine.models import AnswerRequest
            request = AnswerRequest(
                query=query,
                room_id=room_id,
                pdf_summary=pdf_summary,
                retrieval_params=kwargs
            )
            response = self.answer(request, room_id, mode)
            return response.answer
        except Exception as e:
            logger.error(f"Legacy answer generation failed: {e}")
            return "抱歉，我在處理您的請求時遇到了問題。請再試一次或換一種問法。"
    
    # Convenience properties for direct engine access (for advanced use cases)
    @property
    def vector_engine(self) -> BaseRAGEngine:
        """Get the vector engine directly."""
        return self._engines.get('vector')
    
    @property
    def graph_engine(self) -> BaseRAGEngine:
        """Get the graph engine directly."""
        return self._engines.get('graph')
    
    @property
    def hybrid_engine(self) -> BaseRAGEngine:
        """Get the hybrid engine directly."""
        return self._engines.get('hybrid')

# Global router instance
router = RAGRouter()