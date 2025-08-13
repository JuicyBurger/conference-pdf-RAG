#!/usr/bin/env python3
"""
Launch script for Sinon-RAG API

Simple script to start the API server with proper configuration.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('api.log', mode='a')
        ]
    )

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        'data',
        'data/uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

def main():
    """Main entry point"""
    print("🚀 Starting Sinon-RAG API Server...")
    
    # Set CUDA memory allocation configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✅ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # Bootstrap jieba with document IDs from Qdrant
    try:
        from src.rag.retriever import bootstrap_jieba_from_qdrant
        bootstrap_success = bootstrap_jieba_from_qdrant()
        if bootstrap_success:
            print("✅ Successfully bootstrapped jieba with document IDs")
        else:
            print("⚠️ Jieba bootstrap failed, will use fallback segmentation")
    except Exception as e:
        print(f"⚠️ Failed to bootstrap jieba: {e}")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Ensure directories
    ensure_directories()
    
    # Configuration
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Debug: {debug}")
    
    # Start server
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 