#!/usr/bin/env python3
"""
Simple test script for GraphRAG ingestion pipeline.
Uses an existing PDF file to test the ingestion process.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import TRAINING_CORPUS_ID
from src.rag.graph.training_ingestion import ingest_pdfs_to_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_pdf_files():
    """Find available PDF files in the current directory."""
    # Look for PDF files in current directory and data subdirectories
    pdf_files = []
    
    # Check current directory
    current_pdfs = list(Path(".").glob("*.pdf"))
    pdf_files.extend(current_pdfs)
    
    # Check data directory if it exists
    data_dir = Path("data/training")
    if data_dir.exists():
        data_pdfs = list(data_dir.rglob("*.pdf"))
        pdf_files.extend(data_pdfs)
    
    if not pdf_files:
        logger.error("❌ No PDF files found in current directory or data subdirectories")
        return []
    
    logger.info(f"📄 Found {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files):
        logger.info(f"  {i+1}. {pdf.name}")
    
    return pdf_files


def main():
    """Main test function."""
    logger.info("🧪 Starting simple GraphRAG ingestion test...")
    
    # Find PDF files
    pdf_files = find_pdf_files()
    if not pdf_files:
        return False
    
    # Use the first PDF file
    pdf_path = str(pdf_files[0])
    logger.info(f"🚀 Testing with: {pdf_path}")
    
    try:
        # Run the ingestion
        result = ingest_pdfs_to_graph(
            pdf_paths=[pdf_path],
            training_corpus_id=TRAINING_CORPUS_ID
        )
        
        logger.info(f"✅ Ingestion completed successfully!")
        logger.info(f"📊 Results: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        sys.exit(1)
