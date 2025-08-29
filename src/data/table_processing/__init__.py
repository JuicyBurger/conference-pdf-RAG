"""
Table processing module for hybrid RAG system.

This module provides comprehensive table processing capabilities:
- Table detection and extraction from PDFs
- Image-based table processing using LLMs 
- Structured data conversion and validation
- Knowledge graph extraction from tables
- Table summarization and indexing

Workflow:
PDF → Table Detection → Image Cropping → LLM Extraction → JSON Structure → KG Extraction → Vector Indexing
"""

from .pipeline import TableProcessingPipeline
from .detector import TableDetector
from .extractor import TableExtractor
from .converter import TableConverter
from .kg_extractor import TableKGExtractor
from .summarizer import TableSummarizer

__all__ = [
    'TableProcessingPipeline',
    'TableDetector',
    'TableExtractor', 
    'TableConverter',
    'TableKGExtractor',
    'TableSummarizer'
]
