#!/usr/bin/env python3
"""
Simple test script for the table extractor
"""

import json
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.table_extractor import TableExtractor

def test_table_extractor():
    """Test the table extractor with a specific image file."""
    
    # Test image path
    test_image = "data/tables/113年報 20240531-16-18/table_05_P3.png"
    
    print(f"🔍 Testing table extractor with image: {test_image}")
    
    # Check if the image file exists
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please make sure the image file exists in the specified path.")
        return False
    
    try:
        # Initialize the table extractor
        print("🔧 Initializing TableExtractor...")
        extractor = TableExtractor()
        
        # Extract tables from the image
        print("📊 Extracting tables from image...")
        tables = extractor.extract_tables_from_image(test_image)
        
        # Display results
        print(f"✅ Successfully extracted {len(tables)} table(s)")
        
        for i, table in enumerate(tables):
            print(f"\n📋 Table {i+1}:")
            print(f"   Headers: {table.get('headers', [])}")
            print(f"   Records: {len(table.get('records', []))}")
            print(f"   Notes: {len(table.get('notes', []))}")
            
            # Show first few records
            records = table.get('records', [])
            if records:
                print(f"   First record: {records[0]}")
                if len(records) > 1:
                    print(f"   Second record: {records[1]}")
        
        # Save results to JSON file
        output_file = "test_table_extractor_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Also get the raw HTML content and save to text file
        print("📄 Extracting raw HTML content...")
        try:
            # Get the raw HTML from the OCR step
            html_content = extractor._ocr_image_to_html(test_image)
            
            # Save HTML to text file
            html_output_file = "test_table_extractor_html.txt"
            with open(html_output_file, 'w', encoding='utf-8') as f:
                f.write("=== RAW HTML CONTENT FROM OCR ===\n\n")
                f.write(html_content)
                f.write("\n\n=== END OF HTML CONTENT ===\n")
            
            print(f"💾 Raw HTML content saved to: {html_output_file}")
            
        except Exception as e:
            print(f"⚠️  Could not extract raw HTML content: {e}")
        return True
        
    except Exception as e:
        print(f"❌ Error during table extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_extractor()
    sys.exit(0 if success else 1)
