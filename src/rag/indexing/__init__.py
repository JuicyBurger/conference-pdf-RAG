"""
Indexing module for RAG pipeline.

This module contains:
- indexer.py: Main indexing functionality for PDFs
- add_indexes.py: Utility to add missing indexes to existing collections
"""

from .indexer import index_pdf, index_pdfs, init_collection
from .add_indexes import add_missing_indexes

__all__ = [
    'index_pdf',
    'index_pdfs', 
    'init_collection',
    'add_missing_indexes'
] 