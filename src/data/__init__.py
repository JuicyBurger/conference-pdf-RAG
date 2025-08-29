"""
Data extraction and processing modules.

This module provides core data processing functionality for the document analyzer:
- PDF text and image extraction
- Table detection and processing 
- Text chunking and preprocessing
- Document summarization
- Node building for indexing

Note: For high-level table processing workflows, use src.table_processing instead.
"""

# Core PDF processing
from .pdf_ingestor import build_page_nodes, extract_text_pages
from .pdf_summarizer import summarize_pdf_content, extract_pdf_text_for_chat

# Text processing utilities  
from .chunker import chunk_text
from .node_builder import clean_text_for_comparison, process_table_to_json

# Table processing components (now from consolidated module)
from .table_processing.extractor import extract_tables_from_image_transformers
from .table_processing.kg_extractor import extract_table_kg
from .table_processing.summarizer import summarize_table
from .table_processing.detector import extract_tables_to_images

__all__ = [
    # Core PDF processing
    'build_page_nodes', 
    'extract_text_pages',
    'summarize_pdf_content',
    'extract_pdf_text_for_chat',
    
    # Text processing utilities
    'chunk_text',
    'clean_text_for_comparison',
    'process_table_to_json',
    
    # Table processing components
    'extract_tables_from_image_transformers',
    'extract_table_kg', 
    'summarize_table',
    'extract_tables_to_images'
]