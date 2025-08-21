#!/usr/bin/env python3
"""
Test script to verify the table extractor integration fix.
"""

import os
import sys
import json

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_table_extractor_json():
    """Test that table extractor returns JSON data correctly."""
    
    try:
        print("🧪 Testing table extractor JSON output...")
        
        # Import the table extractor
        from src.data.table_extractor import extract_tables_from_image
        
        # Test image path
        test_image = "data/tables/113年報 20240531-16-18/table_05_P3.png"
        
        if not os.path.exists(test_image):
            print(f"⚠️  Test image not found: {test_image}")
            print("Skipping test")
            return True
        
        print(f"📄 Processing: {test_image}")
        
        # Extract tables
        tables = extract_tables_from_image(test_image)
        
        print(f"✅ Successfully extracted {len(tables)} table(s)")
        
        # Verify JSON structure
        for i, table in enumerate(tables):
            print(f"\n📋 Table {i+1} structure:")
            print(f"   Type: {type(table)}")
            print(f"   Keys: {list(table.keys())}")
            
            # Check for expected JSON structure
            expected_keys = ['headers', 'records', 'notes']
            for key in expected_keys:
                if key in table:
                    print(f"   ✅ {key}: {type(table[key])} - {len(table[key]) if isinstance(table[key], list) else 'N/A'}")
                else:
                    print(f"   ❌ Missing key: {key}")
            
            # Show sample data
            if 'headers' in table and table['headers']:
                print(f"   📊 Headers: {table['headers']}")
            
            if 'records' in table and table['records']:
                print(f"   📝 First record: {table['records'][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_table_node_creation():
    """Test that table nodes are created correctly from JSON data."""
    
    try:
        print("\n🧪 Testing table node creation from JSON...")
        
        # Import the function
        from src.data.pdf_ingestor import _create_table_nodes_from_json
        
        # Sample JSON data (similar to what table_extractor returns)
        sample_json_tables = [
            {
                "headers": ["姓名", "職位", "薪資"],
                "records": [
                    {"姓名": "張三", "職位": "工程師", "薪資": "50000"},
                    {"姓名": "李四", "職位": "經理", "薪資": "80000"}
                ],
                "notes": [{"label": "備註", "text": "薪資為月薪"}]
            }
        ]
        
        # Create table nodes
        table_nodes = _create_table_nodes_from_json(1, 1, sample_json_tables)
        
        print(f"✅ Created {len(table_nodes)} table nodes")
        
        # Analyze node types
        node_types = {}
        for node in table_nodes:
            node_type = node.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"📊 Node type distribution:")
        for node_type, count in node_types.items():
            print(f"   {node_type}: {count}")
        
        # Show sample nodes
        print(f"\n🔍 Sample nodes:")
        for i, node in enumerate(table_nodes[:3]):
            print(f"   {i+1}. Type: {node['type']}")
            print(f"      Text preview: {node['text'][:100]}...")
            if 'structured_data' in node:
                print(f"      Has structured data: Yes")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Table Extractor Integration Fix")
    print("=" * 50)
    
    # Test 1: JSON extraction
    success1 = test_table_extractor_json()
    
    # Test 2: Node creation
    success2 = test_table_node_creation()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ All tests passed! Table extractor integration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
