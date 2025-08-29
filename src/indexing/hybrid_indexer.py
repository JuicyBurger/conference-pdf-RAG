"""
Hybrid indexer orchestrator.

This module coordinates both vector and graph indexing for hybrid RAG systems.
"""

from typing import List, Dict, Any, Optional, Union
from .base import BaseIndexer, IndexingResult
from .vector_indexer import VectorIndexer
from .graph_indexer import GraphIndexer


class HybridIndexingResult:
    """Result of hybrid indexing operation combining vector and graph results."""
    
    def __init__(self, 
                 vector_result: IndexingResult,
                 graph_result: IndexingResult,
                 doc_id: str):
        self.vector_result = vector_result
        self.graph_result = graph_result
        self.doc_id = doc_id
        
        # Compute overall success
        self.success = vector_result.success and graph_result.success
        self.indexed_count = vector_result.indexed_count + graph_result.indexed_count
        
        # Combine errors if any
        errors = []
        if not vector_result.success and vector_result.error:
            errors.append(f"Vector: {vector_result.error}")
        if not graph_result.success and graph_result.error:
            errors.append(f"Graph: {graph_result.error}")
        self.error = "; ".join(errors) if errors else None
        
        # Combine metadata
        self.metadata = {
            "vector": vector_result.metadata,
            "graph": graph_result.metadata
        }
    
    def __repr__(self):
        if self.success:
            return f"HybridIndexingResult(success=True, vector={self.vector_result.indexed_count}, graph={self.graph_result.indexed_count}, doc_id='{self.doc_id}')"
        else:
            return f"HybridIndexingResult(success=False, error='{self.error}', doc_id='{self.doc_id}')"


class HybridIndexer(BaseIndexer):
    """
    Orchestrator for hybrid indexing combining vector and graph approaches.
    
    This indexer coordinates both VectorIndexer and GraphIndexer to provide
    comprehensive indexing for hybrid RAG systems.
    """
    
    def __init__(self, 
                 collection_name: Optional[str] = None,
                 corpus_id: Optional[str] = None):
        super().__init__("HybridIndexer")
        
        # Initialize component indexers
        self.vector_indexer = VectorIndexer(collection_name) if collection_name else VectorIndexer()
        self.graph_indexer = GraphIndexer(corpus_id)
    
    def initialize(self) -> bool:
        """Initialize both vector and graph indexers."""
        vector_init = self.vector_indexer.initialize()
        graph_init = self.graph_indexer.initialize()
        
        success = vector_init and graph_init
        
        if success:
            self.logger.info("Hybrid indexer initialized successfully")
        else:
            self.logger.error(f"Hybrid indexer initialization failed: vector={vector_init}, graph={graph_init}")
        
        return success
    
    def index_nodes(self, 
                   nodes: List[Dict[str, Any]], 
                   doc_id: str,
                   extra_payload: Optional[Dict[str, Any]] = None) -> HybridIndexingResult:
        """
        Index nodes using both vector and graph indexers.
        
        Args:
            nodes: List of document nodes to index
            doc_id: Document identifier
            extra_payload: Additional metadata to include
            
        Returns:
            HybridIndexingResult with results from both indexers
        """
        self.logger.info(f"Starting hybrid indexing for doc_id='{doc_id}' with {len(nodes)} nodes")
        
        # Index with vector indexer
        self.logger.info("Phase 1: Vector indexing")
        vector_result = self.vector_indexer.index_nodes(nodes, doc_id, extra_payload)
        
        # Index with graph indexer  
        self.logger.info("Phase 2: Graph indexing")
        graph_result = self.graph_indexer.index_nodes(nodes, doc_id, extra_payload)
        
        # Create combined result
        result = HybridIndexingResult(vector_result, graph_result, doc_id)
        
        self.logger.info(f"Hybrid indexing completed: {result}")
        return result
    
    def index_nodes_vector_only(self, 
                               nodes: List[Dict[str, Any]], 
                               doc_id: str,
                               extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
        """Index nodes using only vector indexer."""
        self.logger.info(f"Vector-only indexing for doc_id='{doc_id}'")
        return self.vector_indexer.index_nodes(nodes, doc_id, extra_payload)
    
    def index_nodes_graph_only(self, 
                              nodes: List[Dict[str, Any]], 
                              doc_id: str,
                              extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
        """Index nodes using only graph indexer."""
        self.logger.info(f"Graph-only indexing for doc_id='{doc_id}'")
        return self.graph_indexer.index_nodes(nodes, doc_id, extra_payload)
    
    def get_vector_indexer(self) -> VectorIndexer:
        """Get the vector indexer component."""
        return self.vector_indexer
    
    def get_graph_indexer(self) -> GraphIndexer:
        """Get the graph indexer component."""
        return self.graph_indexer


# Convenience functions for backward compatibility
def index_nodes_hybrid(nodes: List[Dict[str, Any]], 
                      doc_id: str,
                      collection_name: Optional[str] = None,
                      corpus_id: Optional[str] = None,
                      extra_payload: Optional[Dict[str, Any]] = None) -> HybridIndexingResult:
    """
    Convenience function for hybrid indexing.
    
    Args:
        nodes: List of document nodes to index
        doc_id: Document identifier
        collection_name: Qdrant collection name (optional)
        corpus_id: Corpus identifier for graph indexing (optional)
        extra_payload: Additional metadata to include
        
    Returns:
        HybridIndexingResult with results from both indexers
    """
    indexer = HybridIndexer(collection_name, corpus_id)
    if not indexer.initialize():
        raise RuntimeError("Failed to initialize hybrid indexer")
    
    return indexer.index_nodes(nodes, doc_id, extra_payload)


def index_nodes_vector(nodes: List[Dict[str, Any]], 
                      doc_id: str,
                      collection_name: Optional[str] = None,
                      extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
    """
    Convenience function for vector-only indexing.
    
    Args:
        nodes: List of document nodes to index
        doc_id: Document identifier
        collection_name: Qdrant collection name (optional)
        extra_payload: Additional metadata to include
        
    Returns:
        IndexingResult from vector indexer
    """
    indexer = VectorIndexer(collection_name) if collection_name else VectorIndexer()
    if not indexer.initialize():
        raise RuntimeError("Failed to initialize vector indexer")
    
    return indexer.index_nodes(nodes, doc_id, extra_payload)


def index_nodes_graph(nodes: List[Dict[str, Any]], 
                     doc_id: str,
                     corpus_id: Optional[str] = None,
                     extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
    """
    Convenience function for graph-only indexing.
    
    Args:
        nodes: List of document nodes to index
        doc_id: Document identifier
        corpus_id: Corpus identifier for graph indexing (optional)
        extra_payload: Additional metadata to include
        
    Returns:
        IndexingResult from graph indexer
    """
    indexer = GraphIndexer(corpus_id)
    if not indexer.initialize():
        raise RuntimeError("Failed to initialize graph indexer")
    
    return indexer.index_nodes(nodes, doc_id, extra_payload)
