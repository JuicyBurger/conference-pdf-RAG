#!/usr/bin/env python3
"""
Simple test script for the table summarizer
"""

import json
import sys
import os
from pathlib import Path
from bs4 import BeautifulSoup

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.table_summarizer import summarize_table

def parse_html_to_table_data(html_file_path: str):
    """Parse HTML file to table data structure."""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    if not tables:
        print("❌ No tables found in HTML file")
        return []
    
    table_data_list = []
    
    for table in tables:
        # Extract headers
        headers = []
        header_rows = table.find_all("tr")[:3]  # Look at first 3 rows for headers
        
        for row in header_rows:
            cells = row.find_all(["th", "td"])
            row_headers = []
            for cell in cells:
                text = cell.get_text(strip=True)
                if text:
                    row_headers.append(text)
            if row_headers:
                headers.extend(row_headers)
        
        # Extract records
        records = []
        data_rows = table.find_all("tr")[3:]  # Skip first 3 rows (headers)
        
        for row in data_rows:
            cells = row.find_all(["td", "th"])
            record = {}
            for i, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                if text:
                    # Use header if available, otherwise use column index
                    header = headers[i] if i < len(headers) else f"Column_{i+1}"
                    record[header] = text
            if record:
                records.append(record)
        
        # Extract notes (long text cells)
        notes = []
        for row in data_rows:
            cells = row.find_all(["td", "th"])
            for cell in cells:
                text = cell.get_text(strip=True)
                if len(text) > 50:  # Long text might be notes
                    notes.append(text)
        
        table_data = {
            "columns": headers,  # Changed from "headers" to "columns" to match new summarizer
            "records": records,
            "notes": notes
        }
        
        table_data_list.append(table_data)
    
    return table_data_list

def test_table_summarizer():
    """Test the table summarizer with HTML file."""
    
    # HTML file path
    html_file = "test_table_extractor_html.html"
    
    print(f"🔍 Testing table summarizer with HTML file: {html_file}")
    
    # Check if the HTML file exists
    if not Path(html_file).exists():
        print(f"❌ HTML file not found: {html_file}")
        return False
    
    try:
        # Parse HTML to table data
        print("📊 Parsing HTML to table data...")
        table_data_list = parse_html_to_table_data(html_file)
        
        if not table_data_list:
            print("❌ No table data extracted from HTML")
            return False
        
        print(f"✅ Successfully parsed {len(table_data_list)} table(s)")
        
        # Test summarization for each table
        for i, table_data in enumerate(table_data_list):
            print(f"\n📋 Testing Table {i+1}:")
            print(f"   Columns: {table_data.get('columns', [])}")
            print(f"   Records: {len(table_data.get('records', []))}")
            print(f"   Notes: {len(table_data.get('notes', []))}")
            
            # Generate summary
            print("📝 Generating summary...")
            summary = summarize_table(table_data)
            
            print(f"\n📄 Summary for Table {i+1}:")
            print("=" * 50)
            print(summary)
            print("=" * 50)
            
            # Save summary to file
            summary_file = f"table_summary_{i+1}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Table {i+1} Summary:\n")
                f.write("=" * 50 + "\n")
                f.write(summary)
                f.write("\n" + "=" * 50 + "\n")
            
            print(f"💾 Summary saved to: {summary_file}")
        
        # Save all table data to JSON for inspection
        json_file = "test_table_summarizer_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(table_data_list, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Table data saved to: {json_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error during table summarization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_summarizer()
    sys.exit(0 if success else 1)
