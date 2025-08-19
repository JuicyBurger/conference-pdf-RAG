#!/usr/bin/env python3
"""
Script to clean up orphaned chunks from previous ingestions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import NEO4J_DATABASE
from src.rag.graph.graph_store import get_driver

def cleanup_orphaned_chunks():
    """Clean up orphaned chunks that don't have Document relationships."""
    print("🧹 Cleaning up orphaned chunks...")
    
    try:
        driver = get_driver()
        print(f"✅ Connected to Neo4j at {NEO4J_DATABASE}")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return False
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Count orphaned chunks first
            result = session.run("""
                MATCH (c:Chunk)
                WHERE NOT (c)<-[:HAS_CHUNK]-()
                RETURN count(c) as orphaned_count
            """)
            
            orphaned_count = result.single()["orphaned_count"]
            print(f"📊 Found {orphaned_count} orphaned chunks")
            
            if orphaned_count == 0:
                print("✅ No orphaned chunks to clean up!")
                return True
            
            # Show what we're about to delete
            print(f"\n🗑️ Orphaned chunk details:")
            result = session.run("""
                MATCH (c:Chunk)
                WHERE NOT (c)<-[:HAS_CHUNK]-()
                RETURN c.doc_id as doc_id, c.page as page, c.type as type
                LIMIT 10
            """)
            
            for record in result:
                doc_id = record["doc_id"]
                page = record["page"]
                chunk_type = record["type"]
                print(f"   Doc: {doc_id}, Page: {page}, Type: {chunk_type}")
            
            if orphaned_count > 10:
                print(f"   ... and {orphaned_count - 10} more")
            
            # Confirm deletion
            print(f"\n⚠️ This will delete {orphaned_count} orphaned chunks.")
            print("💡 These are chunks from previous ingestions that are no longer linked to documents.")
            
            # Delete orphaned chunks
            result = session.run("""
                MATCH (c:Chunk)
                WHERE NOT (c)<-[:HAS_CHUNK]-()
                DETACH DELETE c
                RETURN count(c) as deleted_count
            """)
            
            deleted_count = result.single()["deleted_count"]
            print(f"✅ Deleted {deleted_count} orphaned chunks")
            
            # Verify cleanup
            result = session.run("MATCH (c:Chunk) RETURN count(c) as remaining_chunks")
            remaining_chunks = result.single()["remaining_chunks"]
            print(f"📊 Remaining chunks: {remaining_chunks}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧹 Cleanup Orphaned Chunks")
    print("=" * 40)
    
    success = cleanup_orphaned_chunks()
    
    print("=" * 40)
    if success:
        print("🎉 Cleanup completed successfully!")
    else:
        print("❌ Cleanup failed!")

