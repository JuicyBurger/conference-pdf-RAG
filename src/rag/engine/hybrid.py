"""
Hybrid RAG engine implementation.

This module implements a RAG engine that combines graph and vector approaches.
It composes GraphRAG and VectorRAG retrievals at query time and synthesizes a
single answer.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRAGEngine
from .models import RetrievalRequest, RetrievalResponse, AnswerRequest, AnswerResponse, Evidence
from .vector import VectorRAGEngine
from .graph import GraphRAGEngine
from ..qa_generation import generate_answer

# Configure logging
logger = logging.getLogger(__name__)

class HybridRAGEngine(BaseRAGEngine):
    """Hybrid RAG engine combining vector and graph approaches.
    
    This engine combines the results from both vector and graph engines
    to provide the best possible answer.
    """
    
    def __init__(self):
        """Initialize the hybrid RAG engine."""
        # Initialize engines first so they're available in _initialize_engine
        self.vector_engine = VectorRAGEngine()
        self.graph_engine = GraphRAGEngine()
        
        # Then call parent constructor
        from .models import EngineType
        super().__init__(EngineType.HYBRID)
    
    def _initialize_engine(self):
        """Initialize hybrid engine-specific components."""
        # Engines are already initialized in __init__
        pass
    
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Enhanced hybrid retrieval with proper context integration.
        
        Implements the three-step hybrid RAG process:
        1. Vector similarity search over document chunks (Qdrant)
        2. Graph-based search over knowledge graph (Neo4j)
        3. Context integration: merge both sources into unified context
        
        Args:
            request: Standardized retrieval request
            
        Returns:
            Standardized retrieval response with integrated context
        """
        logger.info(f"Enhanced hybrid retrieval: query='{request.query[:50]}...'")
        
        # Step 1: Vector similarity search over document chunks (Qdrant)
        vector_response = self._vector_search(request)
        logger.info(f"Vector search: {len(vector_response.evidence)} results")
        
        # Step 2: Graph-based search over knowledge graph (Neo4j)
        graph_response = self._graph_search(request)
        logger.info(f"Graph search: {len(graph_response.evidence)} results")
        
        # Step 3: Context integration - merge both sources
        integrated_evidence = self._integrate_contexts(vector_response.evidence, graph_response.evidence, request)
        logger.info(f"Context integration: {len(integrated_evidence)} integrated results")
        
        return RetrievalResponse(
            evidence=integrated_evidence,
            engine_type=self.engine_type,
            query=request.query,
            metadata={
                'vector_count': len(vector_response.evidence),
                'graph_count': len(graph_response.evidence),
                'integrated_count': len(integrated_evidence),
                'integration_method': 'unified_context'
            }
        )
    
    def _vector_search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Step 1: Vector similarity search over document chunks using Qdrant."""
        # Use vector engine for document chunk similarity
        # Don't scope by room for vector search to ensure we get results
        vector_request = RetrievalRequest(
            query=request.query,
            room_id=None,  # Don't scope by room for vector retrieval
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            prefer_chat_scope=False,  # Don't prefer chat scope
            scope_filter=None  # No scope filter
        )
        
        return self.vector_engine.retrieve(vector_request)
    
    def _graph_search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Step 2: Graph-based search over knowledge graph using Neo4j."""
        # Use graph engine for knowledge graph traversal
        graph_request = RetrievalRequest(
            query=request.query,
            room_id=None,  # Don't scope graph by room
            top_k=request.top_k // 2,  # Allocate some results to graph
            score_threshold=0.25,  # Lower threshold for graph
            max_hops=request.max_hops or 2
        )
        
        return self.graph_engine.retrieve(graph_request)
    
    def _integrate_contexts(self, vector_evidence: List[Evidence], graph_evidence: List[Evidence], 
                          request: RetrievalRequest) -> List[Evidence]:
        """Step 3: Context integration - merge vector and graph results into unified context.
        
        Following the paper's recommendation: Qdrant results FIRST, then Neo4j results.
        This order is important for optimal RAG performance.
        """
        
        integrated_evidence = []
        seen_ids = set()
        
        # Step 1: Process vector evidence (document chunks) FIRST - as recommended by the paper
        logger.info(f"🔍 Adding {len(vector_evidence)} vector (Qdrant) results FIRST")
        for evidence in vector_evidence:
            if evidence.id and evidence.id not in seen_ids:
                seen_ids.add(evidence.id)
                # Keep vector evidence as-is (document chunks)
                integrated_evidence.append(evidence)
        
        # Step 2: Process graph evidence (knowledge graph triples) SECOND - as recommended by the paper
        logger.info(f"🔍 Adding {len(graph_evidence)} graph (Neo4j) results SECOND")
        for evidence in graph_evidence:
            if evidence.id and evidence.id not in seen_ids:
                seen_ids.add(evidence.id)
                # Convert graph triples to readable context
                enhanced_evidence = self._enhance_graph_evidence(evidence)
                integrated_evidence.append(enhanced_evidence)
        
        # Limit to requested top_k while preserving the order (Qdrant first, then Neo4j)
        # DO NOT sort by score as this would destroy the intended order
        integrated_evidence = integrated_evidence[:request.top_k]
        
        logger.info(f"🔍 Final integrated context: {len(integrated_evidence)} items (Qdrant first, then Neo4j)")
        
        return integrated_evidence
    
    def _enhance_graph_evidence(self, evidence: Evidence) -> Evidence:
        """Convert graph triples into readable sentences for context integration."""
        
        # Extract graph information from evidence
        graph_info = evidence.raw
        
        # If this is graph evidence, try to extract entities and relationships
        if hasattr(graph_info, 'payload'):
            payload = graph_info.payload
            
            # Look for entity/relationship information
            if 'entities' in payload or 'relationships' in payload:
                # Convert to readable format
                readable_context = self._convert_graph_to_readable(payload)
                
                # Create enhanced evidence with readable context
                enhanced_raw = type('EnhancedHit', (), {
                    'payload': {
                        'text': readable_context,
                        'source': 'graph_knowledge',
                        'original_evidence': payload
                    },
                    'score': evidence.score
                })()
                
                return Evidence(enhanced_raw, self.engine_type)
        
        # If no graph-specific enhancement possible, return as-is
        return evidence
    
    def _convert_graph_to_readable(self, payload: dict) -> str:
        """Convert graph payload into readable sentences."""
        
        readable_parts = []
        
        # Extract entities
        entities = payload.get('entities', [])
        if entities:
            entity_text = f"Entities mentioned: {', '.join(entities[:5])}"  # Limit to 5 entities
            readable_parts.append(entity_text)
        
        # Extract relationships
        relationships = payload.get('relationships', [])
        if relationships:
            rel_text = f"Key relationships: {', '.join(relationships[:3])}"  # Limit to 3 relationships
            readable_parts.append(rel_text)
        
        # Extract any additional context
        if 'context' in payload:
            readable_parts.append(payload['context'])
        
        # Combine into readable text
        if readable_parts:
            return ". ".join(readable_parts) + "."
        else:
            return "Graph-based knowledge context available."
    
    def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Generate an answer for a query using combined retrieval results.
        
        Args:
            request: Standardized answer request
            
        Returns:
            Standardized answer response
        """
        logger.info(f"Hybrid engine answering query: {request.query[:50]}...")
        
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
        """Generate answer from evidence.
        
        Args:
            request: Answer request
            evidence: Retrieved evidence
            
        Returns:
            Generated answer
        """
        # Convert to raw hits for generator
        combined_hits = [ev.raw for ev in evidence]
        
        # Add PDF summary if provided
        if request.pdf_summary:
            pdf_hit = type('Hit', (), {'payload': {'content': request.pdf_summary[:1000], 'page': 'uploaded_pdf'}})()
            combined_hits = [pdf_hit] + combined_hits[:11]  # Limit to 11 + PDF summary = 12 total
        
        # Handle empty results
        if not combined_hits:
            return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"

        # Generate answer
        return generate_answer(request.query, combined_hits)