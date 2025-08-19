#!/usr/bin/env python3
"""
Detailed analysis of Neo4j nodes to understand the synchronization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import NEO4J_DATABASE
from src.rag.graph.graph_store import get_driver

def detailed_sync_analysis():
    """Detailed analysis of Neo4j nodes."""
    print("🔍 Detailed Neo4j Node Analysis...")
    
    try:
        driver = get_driver()
        print(f"✅ Connected to Neo4j at {NEO4J_DATABASE}")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return False
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Get all node types and their counts
            print(f"\n📊 Node Type Analysis:")
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
            """)
            
            for record in result:
                labels = record["labels"]
                count = record["count"]
                print(f"   {labels}: {count}")
            
            # Get Document nodes with their properties
            print(f"\n📄 Document Node Details:")
            result = session.run("MATCH (d:Document) RETURN d.doc_id as doc_id, d.filename as filename")
            for record in result:
                print(f"   Doc ID: {record['doc_id']}")
                print(f"   Filename: {record['filename']}")
            
            # Get Chunk nodes with their properties
            print(f"\n📝 Chunk Node Details:")
            result = session.run("""
                MATCH (c:Chunk)
                RETURN c.doc_id as doc_id, c.page as page, c.type as type, count(c) as count
                ORDER BY c.doc_id, c.page
            """)
            
            chunk_summary = {}
            for record in result:
                doc_id = record['doc_id']
                page = record['page']
                chunk_type = record['type']
                count = record['count']
                
                if doc_id not in chunk_summary:
                    chunk_summary[doc_id] = {}
                if page not in chunk_summary[doc_id]:
                    chunk_summary[doc_id][page] = {}
                
                chunk_summary[doc_id][page][chunk_type] = count
            
            for doc_id, pages in chunk_summary.items():
                print(f"   Document: {doc_id}")
                for page, types in pages.items():
                    print(f"     Page {page}: {types}")
            
            # Get relationship types and counts
            print(f"\n🔗 Relationship Analysis:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            
            for record in result:
                rel_type = record["type"]
                count = record["count"]
                print(f"   {rel_type}: {count}")
            
            # Check Document -> Chunk relationships specifically
            print(f"\n📋 Document-Chunk Relationship Details:")
            result = session.run("""
                MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk)
                RETURN d.doc_id as doc_id, count(r) as chunk_count
            """)
            
            for record in result:
                doc_id = record["doc_id"]
                chunk_count = record["chunk_count"]
                print(f"   Document {doc_id} has {chunk_count} chunks")
            
            # Check for orphaned chunks (chunks without Document relationships)
            print(f"\n🔍 Orphaned Chunk Analysis:")
            result = session.run("""
                MATCH (c:Chunk)
                WHERE NOT (c)<-[:HAS_CHUNK]-()
                RETURN count(c) as orphaned_count
            """)
            
            orphaned_count = result.single()["orphaned_count"]
            print(f"   Orphaned chunks (no Document relationship): {orphaned_count}")
            
            # Check for knowledge graph entities
            print(f"\n🧠 Knowledge Graph Entity Analysis:")
            result = session.run("""
                MATCH (n)
                WHERE NOT n:Document AND NOT n:Chunk
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
            """)
            
            kg_entities = 0
            for record in result:
                labels = record["labels"]
                count = record["count"]
                kg_entities += count
                print(f"   {labels}: {count}")
            
            print(f"   Total KG entities: {kg_entities}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Detailed Neo4j Node Analysis")
    print("=" * 50)
    
    success = detailed_sync_analysis()
    
    print("=" * 50)
    if success:
        print("✅ Analysis completed!")
    else:
        print("❌ Analysis failed!")
