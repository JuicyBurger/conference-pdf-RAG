#!/usr/bin/env python3
"""
Test script for GraphRAG ingestion pipeline.
Tests the entire end-to-end process from PDF upload to Neo4j graph and Qdrant vectors.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    TRAINING_CORPUS_ID
)
from src.rag.graph.training_ingestion import ingest_pdfs_to_graph
from src.rag.graph.graph_store import get_driver, ensure_graph_indexes
from src.rag.indexing.indexer import ensure_indexes_exist
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_connections():
    """Test connections to Qdrant and Neo4j."""
    logger.info("🔌 Testing connections...")
    
    # Test Qdrant
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        logger.info(f"✅ Qdrant connected. Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        return False
    
    # Test Neo4j
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 as test")
            result.single()
        logger.info(f"✅ Neo4j connected to database: {NEO4J_DATABASE}")
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        return False
    
    return True


def test_indexes():
    """Test and ensure indexes exist."""
    logger.info("📊 Testing indexes...")
    
    try:
        # Ensure Qdrant indexes
        ensure_indexes_exist()
        logger.info("✅ Qdrant indexes ensured")
        
        # Ensure Neo4j indexes
        ensure_graph_indexes()
        logger.info("✅ Neo4j indexes ensured")
        
        return True
    except Exception as e:
        logger.error(f"❌ Index setup failed: {e}")
        return False


def create_test_pdf():
    """Create a simple test PDF for ingestion."""
    logger.info("📄 Creating test PDF...")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        test_pdf_path = "test_document.pdf"
        
        # Create a simple PDF with some test content
        c = canvas.Canvas(test_pdf_path, pagesize=letter)
        c.drawString(100, 750, "Test Document for GraphRAG")
        c.drawString(100, 720, "This is a test document to verify GraphRAG ingestion.")
        c.drawString(100, 690, "It contains various entities and relationships.")
        c.drawString(100, 660, "Company: TechCorp Inc.")
        c.drawString(100, 630, "CEO: John Smith")
        c.drawString(100, 600, "Founded: 2020")
        c.drawString(100, 570, "Industry: Technology")
        c.drawString(100, 540, "Revenue: $10M")
        c.drawString(100, 510, "Employees: 100")
        c.drawString(100, 480, "Location: San Francisco, CA")
        c.drawString(100, 450, "Products: AI Platform, Cloud Services")
        c.drawString(100, 420, "Partners: Microsoft, Google")
        c.drawString(100, 390, "Funding: Series A - $5M")
        c.drawString(100, 360, "Investors: Sequoia Capital, Andreessen Horowitz")
        c.drawString(100, 330, "Competitors: OpenAI, Anthropic")
        c.drawString(100, 300, "Technology: Machine Learning, Natural Language Processing")
        c.drawString(100, 270, "Use Cases: Customer Service, Data Analysis")
        c.drawString(100, 240, "Market: B2B SaaS")
        c.drawString(100, 210, "Growth: 200% YoY")
        c.drawString(100, 180, "Challenges: Talent Acquisition, Market Competition")
        c.drawString(100, 150, "Future Plans: International Expansion, Product Development")
        c.save()
        
        logger.info(f"✅ Test PDF created: {test_pdf_path}")
        return test_pdf_path
        
    except ImportError:
        logger.warning("⚠️ reportlab not available, using existing PDF if available")
        # Check if there's an existing test PDF
        existing_pdfs = list(Path(".").glob("*.pdf"))
        if existing_pdfs:
            logger.info(f"✅ Using existing PDF: {existing_pdfs[0]}")
            return str(existing_pdfs[0])
        else:
            logger.error("❌ No PDF available for testing")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to create test PDF: {e}")
        return None


def test_ingestion(pdf_path: str):
    """Test the complete GraphRAG ingestion pipeline."""
    logger.info(f"🚀 Testing GraphRAG ingestion with: {pdf_path}")
    
    try:
        # Run the ingestion
        result = ingest_pdfs_to_graph(
            file_paths=[pdf_path],
            corpus_id=TRAINING_CORPUS_ID
        )
        
        logger.info(f"✅ Ingestion completed successfully!")
        logger.info(f"📊 Results: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        return None


def verify_neo4j_data():
    """Verify that data was properly ingested into Neo4j."""
    logger.info("🔍 Verifying Neo4j data...")
    
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            
            # Check Corpus
            result = session.run("MATCH (c:Corpus) RETURN count(c) as count")
            corpus_count = result.single()["count"]
            logger.info(f"📁 Corpus nodes: {corpus_count}")
            
            # Check Documents
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            doc_count = result.single()["count"]
            logger.info(f"📄 Document nodes: {doc_count}")
            
            # Check Chunks
            result = session.run("MATCH (ch:Chunk) RETURN count(ch) as count")
            chunk_count = result.single()["count"]
            logger.info(f"📝 Chunk nodes: {chunk_count}")
            
            # Check Entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()["count"]
            logger.info(f"🏷️ Entity nodes: {entity_count}")
            
            # Check Relationships
            result = session.run("MATCH ()-[r]->() WHERE NOT type(r) IN ['HAS_DOCUMENT', 'HAS_CHUNK'] RETURN count(r) as count")
            relation_count = result.single()["count"]
            logger.info(f"🔗 LLM-generated relationships: {relation_count}")
            
            # Show some sample entities
            if entity_count > 0:
                result = session.run("MATCH (e:Entity) RETURN e.name, e.description LIMIT 5")
                entities = list(result)
                logger.info("🏷️ Sample entities:")
                for entity in entities:
                    logger.info(f"  - {entity['e.name']}: {entity['e.description']}")
            
            # Show some sample relationships
            if relation_count > 0:
                result = session.run("""
                    MATCH (s:Entity)-[r]->(t:Entity) 
                    WHERE NOT type(r) IN ['HAS_DOCUMENT', 'HAS_CHUNK']
                    RETURN s.name, type(r), t.name LIMIT 5
                """)
                relations = list(result)
                logger.info("🔗 Sample relationships:")
                for rel in relations:
                    logger.info(f"  - {rel['s.name']} --[{rel['type(r)']}]--> {rel['t.name']}")
            
            return {
                "corpus": corpus_count,
                "documents": doc_count,
                "chunks": chunk_count,
                "entities": entity_count,
                "relationships": relation_count
            }
            
    except Exception as e:
        logger.error(f"❌ Neo4j verification failed: {e}")
        return None


def verify_qdrant_data():
    """Verify that data was properly indexed in Qdrant."""
    logger.info("🔍 Verifying Qdrant data...")
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Get collection info
        collection_info = client.get_collection(QDRANT_COLLECTION)
        vector_count = collection_info.vectors_count
        logger.info(f"📊 Total vectors in Qdrant: {vector_count}")
        
        # Count vectors by scope
        result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter={"must": [{"key": "scope", "match": {"value": "graph"}}]},
            limit=1000
        )
        graph_vectors = len(result[0])
        logger.info(f"🕸️ Graph-scoped vectors: {graph_vectors}")
        
        # Show some sample payloads
        if graph_vectors > 0:
            result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter={"must": [{"key": "scope", "match": {"value": "graph"}}]},
                limit=3
            )
            logger.info("📝 Sample graph vectors:")
            for point in result[0]:
                payload = point.payload
                logger.info(f"  - Kind: {payload.get('kind', 'unknown')}")
                logger.info(f"    Text: {payload.get('text', '')[:100]}...")
                logger.info(f"    Corpus: {payload.get('corpus_id', 'unknown')}")
        
        return {
            "total_vectors": vector_count,
            "graph_vectors": graph_vectors
        }
        
    except Exception as e:
        logger.error(f"❌ Qdrant verification failed: {e}")
        return None


def cleanup_test_data():
    """Clean up test data from databases."""
    logger.info("🧹 Cleaning up test data...")
    
    try:
        # Clean up Neo4j
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            # Delete test corpus and all related data
            result = session.run("""
                MATCH (c:Corpus {id: $corpus_id})
                OPTIONAL MATCH (c)-[:HAS_DOCUMENT]->(d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(ch:Chunk)
                OPTIONAL MATCH (e:Entity {corpus_id: $corpus_id})
                OPTIONAL MATCH ()-[r]-() WHERE r.corpus_id = $corpus_id
                DELETE ch, d, c, e, r
                RETURN count(ch) + count(d) + count(c) + count(e) as deleted_nodes
            """, {"corpus_id": TRAINING_CORPUS_ID})
            deleted_nodes = result.single()["deleted_nodes"]
            logger.info(f"🗑️ Deleted {deleted_nodes} nodes from Neo4j")
        
        # Clean up Qdrant
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        result = client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector={"filter": {"must": [{"key": "corpus_id", "match": {"value": TRAINING_CORPUS_ID}}]}}
        )
        logger.info(f"🗑️ Deleted test vectors from Qdrant")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🧪 Starting GraphRAG ingestion test...")
    
    # Test connections
    if not test_connections():
        logger.error("❌ Connection tests failed. Exiting.")
        return False
    
    # Test indexes
    if not test_indexes():
        logger.error("❌ Index tests failed. Exiting.")
        return False
    
    # Create test PDF
    pdf_path = create_test_pdf()
    if not pdf_path:
        logger.error("❌ No test PDF available. Exiting.")
        return False
    
    # Test ingestion
    ingestion_result = test_ingestion(pdf_path)
    if not ingestion_result:
        logger.error("❌ Ingestion test failed. Exiting.")
        return False
    
    # Verify Neo4j data
    neo4j_data = verify_neo4j_data()
    if not neo4j_data:
        logger.error("❌ Neo4j verification failed.")
        return False
    
    # Verify Qdrant data
    qdrant_data = verify_qdrant_data()
    if not qdrant_data:
        logger.error("❌ Qdrant verification failed.")
        return False
    
    # Summary
    logger.info("🎉 Test completed successfully!")
    logger.info("📊 Summary:")
    logger.info(f"  - Neo4j: {neo4j_data['entities']} entities, {neo4j_data['relationships']} relationships")
    logger.info(f"  - Qdrant: {qdrant_data['graph_vectors']} graph vectors")
    
    # Ask user if they want to clean up
    try:
        cleanup = input("\n🧹 Do you want to clean up test data? (y/N): ").strip().lower()
        if cleanup in ['y', 'yes']:
            cleanup_test_data()
            logger.info("✅ Test data cleaned up")
    except KeyboardInterrupt:
        logger.info("⏹️ Skipping cleanup")
    
    return True


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
