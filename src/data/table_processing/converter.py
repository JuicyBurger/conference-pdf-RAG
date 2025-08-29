"""
Table data conversion and structuring.

This module handles the conversion of raw extracted table data into
structured formats suitable for knowledge graph extraction and indexing.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

# Import existing conversion functionality
from ..node_builder import process_table_to_json

logger = logging.getLogger(__name__)


class TableConverter:
    """
    Handles conversion of raw table data to structured formats.
    
    This class provides functionality to convert raw extracted table data
    into structured JSON format suitable for further processing.
    """
    
    def __init__(self):
        """Initialize the table converter."""
        logger.info("Initialized table converter")
    
    def convert_to_structured_format(self, 
                                   raw_table_data: List[Dict[str, Any]], 
                                   table_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert raw extracted table data to structured format.
        
        Args:
            raw_table_data: Raw table data from extractor
            table_info: Additional table metadata
            
        Returns:
            Structured table data dictionary or None if conversion failed
        """
        try:
            if not raw_table_data:
                logger.warning("No table data to convert")
                return None
            
            # Take the best table (highest confidence)
            best_table = max(raw_table_data, key=lambda x: x.get('confidence', 0.0))
            
            logger.debug(f"Converting table with confidence {best_table.get('confidence', 0.0):.3f}")
            
            # Convert to DataFrame for processing
            df = self._extract_dataframe_from_table(best_table)
            if df is None or df.empty:
                logger.warning("Failed to extract DataFrame from table data")
                return None
            
            # Use existing conversion functionality
            structured_data = process_table_to_json(df)
            
            # Add metadata from table_info
            structured_data.update({
                "table_id": table_info.get("table_id", "unknown"),
                "page": table_info.get("page", "unknown"),
                "bbox": table_info.get("bbox", []),
                "confidence": best_table.get("confidence", 0.0),
                "extraction_metadata": {
                    "image_path": table_info.get("image_path"),
                    "image_width": table_info.get("image_width"),
                    "image_height": table_info.get("image_height"),
                    "raw_cells": len(best_table.get("cells", [])),
                    "raw_rows": len(best_table.get("rows", [])),
                    "raw_columns": len(best_table.get("columns", []))
                }
            })
            
            logger.debug(f"Converted table to structured format: {len(structured_data.get('records', []))} records, "
                        f"{len(structured_data.get('columns', []))} columns")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Failed to convert table data to structured format: {e}")
            return None
    
    def _extract_dataframe_from_table(self, table_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Extract pandas DataFrame from raw table data.
        
        Args:
            table_data: Raw table data dictionary
            
        Returns:
            DataFrame or None if extraction failed
        """
        try:
            # Try different extraction methods based on available data
            
            # Method 1: Use cells data if available
            cells = table_data.get("cells", [])
            if cells:
                return self._dataframe_from_cells(cells)
            
            # Method 2: Use rows data if available
            rows = table_data.get("rows", [])
            if rows:
                return self._dataframe_from_rows(rows)
            
            # Method 3: Parse text content if available
            text = table_data.get("text", "")
            if text:
                return self._dataframe_from_text(text)
            
            logger.warning("No usable data found in table")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract DataFrame from table data: {e}")
            return None
    
    def _dataframe_from_cells(self, cells: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Extract DataFrame from cell data."""
        try:
            # Build a grid from cell positions
            max_row = max(cell.get("row", 0) for cell in cells) + 1
            max_col = max(cell.get("col", 0) for cell in cells) + 1
            
            # Initialize grid
            grid = [["" for _ in range(max_col)] for _ in range(max_row)]
            
            # Fill grid with cell values
            for cell in cells:
                row = cell.get("row", 0)
                col = cell.get("col", 0)
                text = cell.get("text", "").strip()
                if 0 <= row < max_row and 0 <= col < max_col:
                    grid[row][col] = text
            
            # Convert to DataFrame
            df = pd.DataFrame(grid)
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Failed to create DataFrame from cells: {e}")
            return None
    
    def _dataframe_from_rows(self, rows: List[List[str]]) -> Optional[pd.DataFrame]:
        """Extract DataFrame from row data."""
        try:
            if not rows:
                return None
            
            # Ensure all rows have the same length
            max_cols = max(len(row) for row in rows) if rows else 0
            normalized_rows = []
            
            for row in rows:
                normalized_row = row + [""] * (max_cols - len(row))
                normalized_rows.append(normalized_row)
            
            df = pd.DataFrame(normalized_rows)
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Failed to create DataFrame from rows: {e}")
            return None
    
    def _dataframe_from_text(self, text: str) -> Optional[pd.DataFrame]:
        """Extract DataFrame from text content."""
        try:
            # Simple text parsing - split by lines and then by common delimiters
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines:
                return None
            
            # Try common delimiters
            delimiters = ['\t', '|', ',', ';']
            
            for delimiter in delimiters:
                if delimiter in lines[0]:
                    rows = []
                    for line in lines:
                        row = [cell.strip() for cell in line.split(delimiter)]
                        rows.append(row)
                    
                    if rows:
                        return self._dataframe_from_rows(rows)
            
            # If no delimiter works, treat each line as a single column
            df = pd.DataFrame({'content': lines})
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Failed to create DataFrame from text: {e}")
            return None
