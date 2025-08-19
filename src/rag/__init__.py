"""
RAG pipeline components

This package now exposes a routing layer that can select between different
RAG engines (vector / graph / hybrid) without changing API routes.
Existing vector components remain available for backward compatibility.
"""

from .retrieval.retrieval_service import retrieval_service  # unified retrieval service
from .qa_generation import generate_answer, generate_qa_pairs_for_doc  # refactored generator
from .indexing.indexer import index_pdf, index_pdfs, init_collection  # legacy vector indexing

__all__ = [
    'retrieval_service',
    'generate_answer',
    'generate_qa_pairs_for_doc', 
    'index_pdf',
    'index_pdfs',
    'init_collection'
]