#!/usr/bin/env python3
"""
Script to check what indexes exist in a Qdrant collection.
This helps diagnose filtering issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

def check_collection_indexes(collection_name: str = QDRANT_COLLECTION):
    """Check what indexes exist in a collection."""
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0,
    )
    
    print(f"🔍 Checking indexes for collection '{collection_name}'...")
    
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection exists with {collection_info.points_count} points")
        
        # Check for indexes by trying to get collection info
        print("\n📋 Checking for indexes...")
        
        # Try to get collection info which includes index information
        collection_info = client.get_collection(collection_name)
        
        # Check what filtering capabilities we have
        print("\n🎯 Filtering capabilities:")
        
        expected_indexes = {
            "doc_id": "Document filtering",
            "page": "Page number filtering", 
            "type": "Content type filtering",
            "text": "Full-text search"
        }
        
        # We'll test indexes by trying to use them
        existing_fields = set()
        
        # Test each expected index by trying to use it
        for field, description in expected_indexes.items():
            try:
                # Try to create a simple filter to test if the index exists
                if field == "text":
                    # For text field, try a MatchText query
                    from qdrant_client.http.models import FieldCondition, MatchText, Filter
                    test_filter = Filter(must=[FieldCondition(key=field, match=MatchText(text="test"))])
                else:
                    # For other fields, try a simple value match
                    from qdrant_client.http.models import FieldCondition, MatchValue, Filter
                    # Use appropriate test values based on field type
                    if field == "page":
                        test_value = 1  # Integer for page numbers
                    else:
                        test_value = "test"  # String for other fields
                    test_filter = Filter(must=[FieldCondition(key=field, match=MatchValue(value=test_value))])
                
                # Try to search with this filter (this will fail if index doesn't exist)
                # Use query_points with the correct API - query instead of query_vector
                client.query_points(
                    collection_name=collection_name,
                    query=[0.0] * 768,  # Dummy vector as query
                    query_filter=test_filter,
                    limit=1
                )
                print(f"  ✅ {field}: {description}")
                existing_fields.add(field)
            except Exception as e:
                if "Index required" in str(e) or "not found" in str(e):
                    print(f"  ❌ {field}: {description} (MISSING)")
                else:
                    print(f"  ⚠️ {field}: {description} (UNKNOWN - {str(e)}...)")
                
        # Check if we have any data to test with
        if collection_info.points_count > 0:
            print(f"\n🧪 Testing with sample data...")
            try:
                # Try to get a sample point to see payload structure
                sample_points = client.scroll(
                    collection_name=collection_name,
                    limit=1
                )[0]
                
                if sample_points:
                    sample_payload = sample_points[0].payload
                    print(f"📄 Sample payload fields: {list(sample_payload.keys())}")
                    
                    # Check if our expected fields exist in payload
                    for field in expected_indexes.keys():
                        if field in sample_payload:
                            print(f"  ✅ {field} exists in payload")
                        else:
                            print(f"  ❌ {field} missing from payload")
                    
                    # Show what fields are actually available
                    print(f"\n📄 Available payload fields: {list(sample_payload.keys())}")
                else:
                    print("⚠️ No sample data found")
                    
            except Exception as e:
                print(f"⚠️ Could not get sample data: {e}")
        
    except Exception as e:
        print(f"❌ Error checking collection: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check indexes in Qdrant collection")
    parser.add_argument("--collection", default=QDRANT_COLLECTION, 
                       help="Collection name (default: from config)")
    
    args = parser.parse_args()
    check_collection_indexes(args.collection) 