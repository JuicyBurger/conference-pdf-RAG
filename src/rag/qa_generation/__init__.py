"""
Generation module for RAG components.

This module provides functionality for generating answers and QA pairs.
"""

from .answer_generator import generate_answer
from .qa_generator import generate_qa_pairs_for_doc, fetch_doc_chunks

__all__ = [
    "generate_answer",
    "generate_qa_pairs_for_doc",
    "fetch_doc_chunks"
]
