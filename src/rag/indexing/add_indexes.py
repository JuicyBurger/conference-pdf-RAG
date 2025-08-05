#!/usr/bin/env python3
"""
Utility script to add missing indexes to existing Qdrant collections.
This is useful when you have existing data but need to add filtering capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

def add_missing_indexes(collection_name: str = QDRANT_COLLECTION):
    """Add missing indexes to an existing collection."""
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0,
    )
    
    print(f"🔍 Checking collection '{collection_name}'...")
    
    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection exists with {collection_info.points_count} points")
    except Exception as e:
        print(f"❌ Collection '{collection_name}' not found: {e}")
        return
    
    # List of indexes to create
    indexes_to_create = [
        ("doc_id", "keyword", "Document ID filtering"),
        ("page", "integer", "Page number filtering"), 
        ("type", "keyword", "Content type filtering"),
        ("text", "text", "Full-text search"),
    ]
    
    print("\n🔧 Adding missing indexes...")
    
    for field_name, field_schema, description in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ Created {field_schema} index on '{field_name}' - {description}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ Index on '{field_name}' already exists")
            else:
                print(f"⚠️ Failed to create index on '{field_name}': {e}")
    
    print("\n🎉 Index creation completed!")
    print("\n📋 Available filtering capabilities:")
    print("  • doc_id: Filter by specific documents")
    print("  • page: Filter by page numbers (e.g., page 43)")
    print("  • type: Filter by content type (paragraph, table_record)")
    print("  • text: Full-text search within content")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add missing indexes to Qdrant collection")
    parser.add_argument("--collection", default=QDRANT_COLLECTION, 
                       help="Collection name (default: from config)")
    
    args = parser.parse_args()
    add_missing_indexes(args.collection) 