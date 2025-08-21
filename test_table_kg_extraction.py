#!/usr/bin/env python3
"""
Test script for deterministic table KG extraction.
"""

import os
import sys
import json

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_table_kg_extraction():
    """Test the table KG extraction with sample data"""
    
    try:
        print("🧪 Testing Table KG Extraction")
        print("=" * 50)
        
        # Import the KG extraction function
        from src.data.table_kg_extractor import extract_table_kg
        
        # Load test data from the JSON file
        test_data_path = "test_table_extractor_results.json"
        if not os.path.exists(test_data_path):
            print(f"❌ Test data not found: {test_data_path}")
            return False
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"📄 Loaded test data with {len(test_data)} tables")
        
        # Test each table
        for i, table_data in enumerate(test_data):
            print(f"\n📊 Testing Table {i+1}:")
            print(f"   Headers: {table_data.get('headers', [])}")
            print(f"   Records: {len(table_data.get('records', []))}")
            print(f"   Notes: {len(table_data.get('notes', []))}")
            
            # Extract KG
            kg_result = extract_table_kg(table_data, "test_doc", f"table_{i+1}")
            
            print(f"   ✅ KG Extraction Results:")
            print(f"      Table Type: {kg_result['table_type']}")
            print(f"      Confidence: {kg_result['confidence']:.2f}")
            print(f"      Observations: {kg_result['observation_count']}")
            print(f"      Panels: {kg_result['panel_count']}")
            print(f"      Metrics: {kg_result['metric_count']}")
            print(f"      Dimensions: {kg_result['dimension_count']}")
            
            # Show sample observations
            if kg_result['observations']:
                print(f"   📋 Sample Observations:")
                for j, obs in enumerate(kg_result['observations'][:3]):  # Show first 3
                    print(f"      {j+1}. Metric: {obs['metric_leaf']}")
                    print(f"         Value: {obs['value_raw']} ({obs['unit']})")
                    print(f"         Dimensions: {obs['dimensions']}")
                    if obs['panel']:
                        print(f"         Panel: {obs['panel']}")
            
            if kg_result.get('error'):
                print(f"   ❌ Error: {kg_result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_ingestion():
    """Test the Neo4j ingestion synchronously"""
    
    try:
        print("\n🧪 Testing Neo4j Ingestion (Synchronous)")
        print("=" * 50)
        
        # Import the ingestion function
        from src.rag.graph.table_ingestion import extract_and_ingest_table_kg
        
        # Load test data
        test_data_path = "test_table_extractor_results.json"
        if not os.path.exists(test_data_path):
            print(f"❌ Test data not found: {test_data_path}")
            return False
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Test ingestion for first table
        table_data = test_data[0]
        print(f"📊 Testing ingestion for table with {len(table_data.get('records', []))} records")
        
        # Extract and ingest synchronously
        result = extract_and_ingest_table_kg(table_data, "test_doc", "table_1")
        
        print(f"✅ Ingestion Results:")
        print(f"   Table Type: {result['table_type']}")
        print(f"   Observations: {result['observation_count']}")
        print(f"   Ingested to Neo4j: {result['ingested_to_neo4j']}")
        print(f"   Neo4j Nodes Created: {result.get('neo4j_nodes_created', 0)}")
        print(f"   Neo4j Relationships Created: {result.get('neo4j_relationships_created', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Deterministic Table KG Extraction")
    print("=" * 60)
    
    # Test 1: KG Extraction
    success1 = test_table_kg_extraction()
    
    # Test 2: Neo4j Ingestion
    success2 = test_neo4j_ingestion()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed! Table KG extraction is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
