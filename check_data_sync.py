#!/usr/bin/env python3
"""
Script to check data synchronization between Qdrant and Neo4j after PDF ingestion.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from src.rag.graph.graph_store import get_driver

def check_data_sync():
    """Check data synchronization between Qdrant and Neo4j."""
    print("🔍 Checking data synchronization between Qdrant and Neo4j...")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"✅ Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False
    
    # Initialize Neo4j driver
    try:
        driver = get_driver()
        print(f"✅ Connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return False
    
    # Check Qdrant data
    print(f"\n📊 Checking Qdrant collection: {QDRANT_COLLECTION}")
    try:
        collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
        qdrant_points = collection_info.points_count
        qdrant_vector_size = collection_info.config.params.vectors.size
        print(f"   Points: {qdrant_points}")
        print(f"   Vector size: {qdrant_vector_size}")
        
        # Get unique doc_ids from Qdrant
        qdrant_doc_ids = set()
        offset = None
        limit = 100
        
        while True:
            response = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=limit,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False
            )
            
            points, offset = response
            if not points:
                break
                
            for point in points:
                doc_id = point.payload.get("doc_id", "")
                if doc_id:
                    qdrant_doc_ids.add(doc_id)
            
            if not offset:
                break
        
        print(f"   Unique doc_ids: {len(qdrant_doc_ids)}")
        if qdrant_doc_ids:
            print(f"   Doc IDs: {list(qdrant_doc_ids)}")
            
    except Exception as e:
        print(f"   ❌ Error checking Qdrant: {e}")
        return False
    
    # Check Neo4j data
    print(f"\n📊 Checking Neo4j database: {NEO4J_DATABASE}")
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Count Document nodes
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            doc_count = result.single()["count"]
            print(f"   Document nodes: {doc_count}")
            
            # Count Chunk nodes
            result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            chunk_count = result.single()["count"]
            print(f"   Chunk nodes: {chunk_count}")
            
            # Get unique doc_ids from Neo4j
            result = session.run("MATCH (d:Document) RETURN DISTINCT d.doc_id as doc_id")
            neo4j_doc_ids = {record["doc_id"] for record in result if record["doc_id"]}
            print(f"   Unique doc_ids: {len(neo4j_doc_ids)}")
            if neo4j_doc_ids:
                print(f"   Doc IDs: {list(neo4j_doc_ids)}")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            relationship_count = result.single()["count"]
            print(f"   Relationships: {relationship_count}")
            
            # Check for specific relationship types
            result = session.run("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) as count")
            has_chunk_count = result.single()["count"]
            print(f"   HAS_CHUNK relationships: {has_chunk_count}")
            
    except Exception as e:
        print(f"   ❌ Error checking Neo4j: {e}")
        return False
    
    # Compare synchronization
    print(f"\n🔍 Synchronization Analysis:")
    
    # Check doc_id consistency
    doc_id_match = qdrant_doc_ids == neo4j_doc_ids
    print(f"   Doc ID consistency: {'✅' if doc_id_match else '❌'}")
    
    if not doc_id_match:
        print(f"   Qdrant doc_ids: {qdrant_doc_ids}")
        print(f"   Neo4j doc_ids: {neo4j_doc_ids}")
        print(f"   Missing in Neo4j: {qdrant_doc_ids - neo4j_doc_ids}")
        print(f"   Missing in Qdrant: {neo4j_doc_ids - qdrant_doc_ids}")
    
    # Check chunk count consistency
    chunk_match = qdrant_points == chunk_count
    print(f"   Chunk count consistency: {'✅' if chunk_match else '❌'}")
    print(f"   Qdrant chunks: {qdrant_points}")
    print(f"   Neo4j chunks: {chunk_count}")
    
    # Check if chunks have proper relationships
    if chunk_count > 0 and has_chunk_count > 0:
        relationship_ratio = has_chunk_count / chunk_count
        print(f"   Chunk relationship ratio: {relationship_ratio:.2f}")
        if relationship_ratio >= 0.8:  # Allow some flexibility
            print(f"   ✅ Good chunk relationship coverage")
        else:
            print(f"   ⚠️ Low chunk relationship coverage")
    
    # Overall sync status
    sync_status = doc_id_match and chunk_match
    print(f"\n🎯 Overall Sync Status: {'✅ SYNCED' if sync_status else '❌ NOT SYNCED'}")
    
    if sync_status:
        print(f"✅ Data is properly synchronized between Qdrant and Neo4j!")
    else:
        print(f"❌ Data synchronization issues detected!")
        print(f"💡 Consider re-running the ingestion process.")
    
    return sync_status

if __name__ == "__main__":
    print("🔍 Data Synchronization Check")
    print("=" * 50)
    
    success = check_data_sync()
    
    print("=" * 50)
    if success:
        print("🎉 Data synchronization check completed successfully!")
    else:
        print("❌ Data synchronization issues found!")
