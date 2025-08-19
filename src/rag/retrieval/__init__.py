"""
Retrieval module for RAG components.

This module provides standardized retrieval functionality for RAG engines.
"""

from .retrieval_service import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalService,
    retrieval_service
)

# Re-export functions for convenience
retrieve = retrieval_service.retrieve
retrieve_with_request = retrieval_service.retrieve_with_request
parse_constraints_for_text = retrieval_service.parse_constraints_for_text
add_new_doc_id_to_parser = retrieval_service.add_new_doc_id_to_parser
get_all_doc_ids = retrieval_service.get_all_doc_ids
bootstrap_jieba_from_qdrant = retrieval_service.bootstrap_jieba_from_qdrant

__all__ = [
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalService",
    "retrieval_service",
    "retrieve",
    "retrieve_with_request",
    "parse_constraints_for_text",
    "add_new_doc_id_to_parser",
    "get_all_doc_ids",
    "bootstrap_jieba_from_qdrant"
]