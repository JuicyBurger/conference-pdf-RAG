#!/usr/bin/env python3
"""
Test script to verify table ingestion fix
"""

import json
import os

def test_table_ingestion_fix():
    """Test that table ingestion works with the fixed structured_data"""
    
    try:
        print("🧪 Testing Table Ingestion Fix")
        print("=" * 50)
        
        # Load test data
        test_data_path = "test_table_extractor_results.json"
        if not os.path.exists(test_data_path):
            print(f"❌ Test data not found: {test_data_path}")
            return False
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Test the KG extraction directly
        from src.data.table_kg_extractor import extract_table_kg
        
        table_data = test_data[0]  # Use first table
        print(f"📊 Testing KG extraction for table with {len(table_data.get('records', []))} records")
        
        # Extract KG
        kg_result = extract_table_kg(table_data, "test_doc", "table_1")
        
        print(f"✅ KG Extraction Results:")
        print(f"   Table Type: {kg_result['table_type']}")
        print(f"   Confidence: {kg_result['confidence']}")
        print(f"   Observations: {kg_result['observation_count']}")
        print(f"   Panels: {kg_result['panel_count']}")
        print(f"   Metrics: {kg_result['metric_count']}")
        print(f"   Dimensions: {kg_result['dimension_count']}")
        
        if kg_result['observation_count'] > 0:
            print(f"   ✅ SUCCESS: Found {kg_result['observation_count']} observations")
            
            # Show first observation
            if kg_result['observations']:
                first_obs = kg_result['observations'][0]
                print(f"\n📋 Sample Observation:")
                print(f"   Metric: {first_obs.get('metric_leaf')}")
                print(f"   Value: {first_obs.get('value_raw')}")
                print(f"   Dimensions: {first_obs.get('dimensions')}")
                print(f"   Panel: {first_obs.get('panel')}")
        else:
            print(f"   ❌ FAILED: No observations extracted")
            return False
        
        # Test the embedding creation
        from src.rag.graph.table_ingestion import create_row_level_embedding, create_table_level_embedding
        
        print(f"\n🔍 Testing Embedding Creation:")
        
        # Create row-level embeddings
        row_embeddings = create_row_level_embedding(
            kg_result['observations'], 
            "test_doc", 
            "table_1", 
            page=1, 
            structured_data=table_data
        )
        
        print(f"   Row Embeddings: {len(row_embeddings)}")
        if row_embeddings:
            print(f"   Sample Row Text: {row_embeddings[0]['text'][:100]}...")
        
        # Create table-level embedding
        table_embedding = create_table_level_embedding(
            kg_result['observations'],
            table_data,
            "test_doc",
            "table_1",
            page=1,
            analysis={'table_type': kg_result['table_type']}
        )
        
        print(f"   Table Embedding: {table_embedding['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_ingestion_fix()
    if success:
        print("\n✅ Table ingestion fix test PASSED")
    else:
        print("\n❌ Table ingestion fix test FAILED")
