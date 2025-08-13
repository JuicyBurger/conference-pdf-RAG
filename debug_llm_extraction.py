#!/usr/bin/env python3
"""
Debug script to test LLM extraction step by step.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag.embeddings.jina_adapter import configure_llamaindex_for_local_models
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm_configuration():
    """Test if LLM configuration is working."""
    logger.info("🔧 Testing LLM configuration...")
    
    try:
        configure_llamaindex_for_local_models()
        logger.info(f"✅ LLM configured: {type(Settings.llm).__name__}")
        logger.info(f"✅ Embedding model: {type(Settings.embed_model).__name__}")
        return True
    except Exception as e:
        logger.error(f"❌ LLM configuration failed: {e}")
        return False


def test_extractor_creation():
    """Test if DynamicLLMPathExtractor can be created."""
    logger.info("🔧 Testing extractor creation...")
    
    try:
        extractor = DynamicLLMPathExtractor(
            llm=Settings.llm,
            max_triplets_per_chunk=5,
            num_workers=1,
        )
        logger.info(f"✅ Extractor created: {type(extractor).__name__}")
        return extractor
    except Exception as e:
        logger.error(f"❌ Extractor creation failed: {e}")
        return None


def test_extraction():
    """Test the actual extraction process."""
    logger.info("🔧 Testing extraction process...")
    
    # Create a test text node
    test_text = """
    Company: TechCorp Inc.
    CEO: John Smith
    Founded: 2020
    Industry: Technology
    Revenue: $10M
    Employees: 100
    Location: San Francisco, CA
    Products: AI Platform, Cloud Services
    Partners: Microsoft, Google
    """
    
    text_node = TextNode(text=test_text, metadata={"doc_id": "test"})
    logger.info(f"✅ Created text node with {len(test_text)} characters")
    
    # Create extractor
    extractor = test_extractor_creation()
    if not extractor:
        return False
    
    # Test extraction
    try:
        logger.info("🔧 Running extraction...")
        result = asyncio.run(extractor._aextract(text_node))
        logger.info(f"✅ Extraction completed: {type(result)}")
        
        # Check what the result contains
        logger.info(f"Result attributes: {dir(result)}")
        
        # Try to access entities and relations
        if hasattr(result, 'entities'):
            entities = result.entities or []
            logger.info(f"🏷️ Found {len(entities)} entities")
            for i, entity in enumerate(entities[:3]):  # Show first 3
                logger.info(f"  Entity {i+1}: {entity}")
        
        if hasattr(result, 'relations'):
            relations = result.relations or []
            logger.info(f"🔗 Found {len(relations)} relations")
            for i, relation in enumerate(relations[:3]):  # Show first 3
                logger.info(f"  Relation {i+1}: {relation}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}", exc_info=True)
        return False


def main():
    """Main debug function."""
    logger.info("🧪 Starting LLM extraction debug...")
    
    # Test 1: LLM Configuration
    if not test_llm_configuration():
        return False
    
    # Test 2: Extractor Creation
    if not test_extractor_creation():
        return False
    
    # Test 3: Actual Extraction
    if not test_extraction():
        return False
    
    logger.info("🎉 All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        sys.exit(1)
