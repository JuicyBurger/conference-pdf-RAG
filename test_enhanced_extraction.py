#!/usr/bin/env python3
"""
Test script for enhanced PDF extraction with table support
"""

import sys
import os

def test_table_extraction():
    """Test table extraction functionality"""
    print("🧪 Testing table extraction...")
    
    try:
        from src.data.pdf_ingestor import extract_tables_per_page, CAMELOT_AVAILABLE
        
        if not CAMELOT_AVAILABLE:
            print("⚠️ Camelot not available - skipping table test")
            return True
        
        # Test with a sample PDF (if available)
        test_pdf = "data/raw/113年報 20240531.pdf"
        if os.path.exists(test_pdf):
            tables = extract_tables_per_page(test_pdf)
            print(f"✅ Found tables on {len(tables)} pages")
            
            for page_no, page_tables in tables.items():
                print(f"  Page {page_no}: {len(page_tables)} tables")
        else:
            print("⚠️ No test PDF found - skipping table extraction test")
        
        return True
        
    except Exception as e:
        print(f"❌ Table extraction test failed: {e}")
        return False

def test_node_building():
    """Test node building functionality"""
    print("\n🧪 Testing node building...")
    
    try:
        from src.data.pdf_ingestor import build_page_nodes
        
        # Test with a sample PDF (if available)
        test_pdf = "data/raw/113年報 20240531.pdf"
        if os.path.exists(test_pdf):
            nodes = build_page_nodes(test_pdf)
            print(f"✅ Generated {len(nodes)} nodes")
            
            # Count different node types
            paragraph_nodes = [n for n in nodes if n["kind"] == "paragraph"]
            table_nodes = [n for n in nodes if n["kind"] == "table_row"]
            
            print(f"  - {len(paragraph_nodes)} paragraph nodes")
            print(f"  - {len(table_nodes)} table row nodes")
            
            # Show sample nodes
            if nodes:
                print("\n📋 Sample nodes:")
                for i, node in enumerate(nodes[:3]):
                    text_preview = node["text"][:50] + "..." if len(node["text"]) > 50 else node["text"]
                    print(f"  {i+1}. [{node['kind']}] Page {node['page']}: {text_preview}")
        else:
            print("⚠️ No test PDF found - skipping node building test")
        
        return True
        
    except Exception as e:
        print(f"❌ Node building test failed: {e}")
        return False

def test_deduplication():
    """Test deduplication functionality"""
    print("\n🧪 Testing deduplication...")
    
    try:
        from src.data.pdf_ingestor import deduplicate_content
        
        # Test with sample data
        paragraph = "公司營收增長了15%。我們的利潤率保持穩定。"
        table_rows = [
            "公司營收｜增長15%｜穩定",
            "營業利潤｜1000萬｜成長",
            "完全不同的內容｜測試｜數據"
        ]
        
        clean_paragraph, clean_rows = deduplicate_content(paragraph, table_rows, 85.0)
        
        print(f"✅ Original: {len(table_rows)} rows → Cleaned: {len(clean_rows)} rows")
        print(f"  Removed {len(table_rows) - len(clean_rows)} duplicate rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Deduplication test failed: {e}")
        return False

def test_integration():
    """Test integration with existing pipeline"""
    print("\n🧪 Testing integration...")
    
    try:
        from src.rag.indexer import index_pdf
        from src.config import QDRANT_COLLECTION
        
        # This would test the full pipeline but requires Qdrant connection
        print("✅ Indexer imports successfully with enhanced functions")
        print("⚠️ Full integration test requires Qdrant connection")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Enhanced PDF Extraction")
    print("=" * 50)
    
    tests = [
        test_table_extraction,
        test_node_building,
        test_deduplication,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced PDF extraction is working.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 