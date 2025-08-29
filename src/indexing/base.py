"""
Base indexer interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IndexingResult:
    """Result of an indexing operation."""
    
    def __init__(self, 
                 success: bool,
                 indexed_count: int = 0,
                 doc_id: str = None,
                 metadata: Dict[str, Any] = None,
                 error: str = None):
        self.success = success
        self.indexed_count = indexed_count
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.error = error
    
    def __repr__(self):
        if self.success:
            return f"IndexingResult(success=True, indexed={self.indexed_count}, doc_id='{self.doc_id}')"
        else:
            return f"IndexingResult(success=False, error='{self.error}')"


class BaseIndexer(ABC):
    """Abstract base class for all indexers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def index_nodes(self, 
                   nodes: List[Dict[str, Any]], 
                   doc_id: str,
                   extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
        """
        Index a list of document nodes.
        
        Args:
            nodes: List of document nodes to index
            doc_id: Document identifier
            extra_payload: Additional metadata to include
            
        Returns:
            IndexingResult with operation details
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the indexer (create collections, ensure indexes, etc.).
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    def validate_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter nodes before indexing.
        
        Args:
            nodes: Raw nodes to validate
            
        Returns:
            List of valid nodes
        """
        valid_nodes = []
        for node in nodes:
            if self._is_valid_node(node):
                valid_nodes.append(node)
            else:
                self.logger.warning(f"Skipping invalid node: {node.get('id', 'unknown')}")
        
        return valid_nodes
    
    def _is_valid_node(self, node: Dict[str, Any]) -> bool:
        """Check if a node is valid for indexing."""
        # Basic validation - must have text and page
        if not node.get('text') or not str(node.get('text')).strip():
            return False
        if node.get('page') is None:
            return False
        return True
