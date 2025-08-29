"""
Knowledge graph extraction from structured table data.

This module handles the deterministic extraction of knowledge graphs
from structured table data without using LLMs using rule-based analysis.
"""

import re
import hashlib
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Aggregate row detection
AGG_RE = re.compile(r'(總計|合計|平均|小計|Total|Average)', re.I)


@dataclass
class UnitInfo:
    """Unit extraction result with precedence tracking"""
    value_num: Optional[float]
    unit: str
    unit_source: str  # cell|leaf|ancestor|caption|inferred
    unit_confidence: float


@dataclass
class PanelInfo:
    """Panel detection result"""
    panel: Optional[str]
    panel_confidence: float


class TableKGExtractor:
    """
    Handles knowledge graph extraction from structured table data.
    
    Uses deterministic rule-based analysis to extract observations, entities,
    and relationships from structured table data.
    """
    
    def __init__(self, include_aggregates: bool = True):
        """
        Initialize the table KG extractor.
        
        Args:
            include_aggregates: Whether to include aggregate observations
        """
        self.include_aggregates = include_aggregates
        logger.info(f"Initialized table KG extractor (include_aggregates={include_aggregates})")
    
    def extract_table_kg(self, 
                        structured_data: Dict[str, Any], 
                        doc_id: str, 
                        table_id: str) -> Dict[str, Any]:
        """
        Extract knowledge graph from structured table data.
        
        Args:
            structured_data: Structured table data dictionary
            doc_id: Document identifier
            table_id: Table identifier
            
        Returns:
            Dictionary containing KG extraction results
        """
        try:
            logger.debug(f"Extracting KG from table {table_id} in document {doc_id}")
            
            # Use core extraction logic
            kg_result = self._extract_table_kg_impl(
                structured_data=structured_data,
                doc_id=doc_id,
                table_id=table_id,
                include_aggregates=self.include_aggregates
            )
            
            if kg_result:
                observations_count = kg_result.get('observation_count', 0)
                logger.debug(f"Extracted {observations_count} observations from table {table_id}")
            else:
                logger.warning(f"No KG data extracted from table {table_id}")
                kg_result = {
                    'table_type': 'unknown',
                    'confidence': 0.0,
                    'observation_count': 0,
                    'observations': []
                }
            
            return kg_result
            
        except Exception as e:
            logger.error(f"Failed to extract KG from table {table_id}: {e}")
            return {
                'table_type': 'error',
                'confidence': 0.0,
                'observation_count': 0,
                'observations': [],
                'error': str(e)
            }
    
    def _extract_table_kg_impl(self, structured_data: dict, doc_id: str, table_id: str, include_aggregates: bool = True) -> dict:
        """Main function to extract KG from table structured_data"""
        
        try:
            # 1. Analyze table structure
            analysis = self._analyze_table_structure(structured_data)
            
            # 2. Generate observations
            observations = self._generate_observations(structured_data, analysis, doc_id, table_id, include_aggregates)
            
            # 3. Return summary for immediate use
            return {
                'table_type': analysis['table_type'],
                'confidence': analysis['confidence'],
                'observation_count': len(observations),
                'panel_count': len(set(o['panel'] for o in observations if o['panel'])),
                'metric_count': len(set(o['metric_leaf'] for o in observations)),
                'dimension_count': len(set(dim for o in observations for dim in o['dimensions'].keys())),
                'observations': observations  # Include for async processing
            }
            
        except Exception as e:
            logger.error(f"Error extracting table KG: {e}")
            return {
                'table_type': 'error',
                'confidence': 0.0,
                'observation_count': 0,
                'panel_count': 0,
                'metric_count': 0,
                'dimension_count': 0,
                'observations': [],
                'error': str(e)
            }
    
    def _analyze_table_structure(self, structured_data: dict) -> dict:
        """Simplified table structure analysis"""
        # Accept both 'headers' (legacy) and 'columns' (current converter output)
        headers = structured_data.get('headers') or structured_data.get('columns') or []
        records = structured_data.get('records', [])
        
        # Simple classification: if has numeric data, it's fact_quant, otherwise generic
        numeric_count = 0
        total_cells = 0
        
        for record in records:
            for header in headers:
                if header in record:
                    total_cells += 1
                    value = record[header]
                    if self._is_numeric(value):
                        numeric_count += 1
        
        numeric_density = numeric_count / total_cells if total_cells > 0 else 0.0
        
        if numeric_density > 0.3:
            table_type = "fact_quant"
            confidence = 0.8
        else:
            table_type = "generic"
            confidence = 0.6
        
        return {
            'table_type': table_type,
            'confidence': confidence,
            'numeric_density': numeric_density,
            'patterns': {}
        }
    
    def _generate_observations(self, structured_data: dict, analysis: dict, doc_id: str, table_id: str, include_aggregates: bool) -> List[dict]:
        """Simplified observation generation"""
        # Accept both 'headers' (legacy) and 'columns' (current converter output)
        headers = structured_data.get('headers') or structured_data.get('columns') or []
        records = structured_data.get('records', []) or []
        
        observations = []
        
        for row_idx, record in enumerate(records):
            for col_idx, header in enumerate(headers):
                if header not in record:
                    continue
                
                value = record.get(header, "")
                
                # Try to convert string values to numbers; keep non-numeric as categorical
                numeric_value = None
                if isinstance(value, (int, float)):
                    numeric_value = float(value)
                elif isinstance(value, str):
                    # Clean the string and try to convert to number
                    cleaned_value = value.strip()
                    if cleaned_value:
                        cleaned_numeric = cleaned_value.replace(',', '').replace(' ', '').rstrip('%元萬億千')
                        try:
                            numeric_value = float(cleaned_numeric)
                        except (ValueError, TypeError):
                            # Keep as categorical/textual observation (numeric_value remains None)
                            pass
                    else:
                        # Empty string cell, skip
                        continue
                else:
                    # Convert other types to string if meaningful; skip if falsy
                    if value is None:
                        continue
                    value = str(value)
                
                # Create a simple observation
                observation = {
                    'doc_id': doc_id,
                    'table_id': table_id,
                    'page': structured_data.get('page'),
                    'anchor': structured_data.get('anchor'),
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'metric_path': [header],
                    'metric_path_str': header,
                    'metric_leaf': header,
                    'dimensions': {k: v for k, v in record.items() if k != header and not self._is_numeric(v)},
                    'value_raw': value,
                    'value_num': numeric_value,
                    'unit': None,
                    'unit_source': 'inferred',
                    'unit_confidence': 0.0,
                    'panel': None,
                    'panel_confidence': 0.0,
                    'aggregate': False,
                    'source_hash': hashlib.md5(f"{doc_id}|{table_id}|{row_idx}|{col_idx}|{header}|{value}".encode("utf-8")).hexdigest()
                }
                
                observations.append(observation)
        
        return observations
    
    def _is_numeric(self, value) -> bool:
        """Check if a value is numeric (including string representations)"""
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, str):
            cleaned_value = value.strip().replace(',', '').replace(' ', '')
            # Remove common suffixes like %, 元, 萬, etc.
            cleaned_value = cleaned_value.rstrip('%元萬億千')
            try:
                float(cleaned_value)
                return True
            except (ValueError, TypeError):
                return False
        return False
    
    def extract_multiple_tables_kg(self, 
                                  tables_data: List[Dict[str, Any]], 
                                  doc_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract knowledge graphs from multiple tables.
        
        Args:
            tables_data: List of structured table data dictionaries
            doc_id: Document identifier
            
        Returns:
            Dictionary mapping table IDs to KG extraction results
        """
        results = {}
        
        for table_data in tables_data:
            table_id = table_data.get('table_id', 'unknown')
            try:
                kg_result = self.extract_table_kg(table_data, doc_id, table_id)
                results[table_id] = kg_result
            except Exception as e:
                logger.error(f"Failed to process table {table_id}: {e}")
                results[table_id] = {
                    'table_type': 'error',
                    'confidence': 0.0,
                    'observation_count': 0,
                    'observations': [],
                    'error': str(e)
                }
        
        return results


# Convenience function for backward compatibility
def extract_table_kg(structured_data: dict, doc_id: str, table_id: str, include_aggregates: bool = True) -> dict:
    """
    Convenience function to extract KG from table structured_data.
    
    Args:
        structured_data: Structured table data dictionary
        doc_id: Document identifier
        table_id: Table identifier
        include_aggregates: Whether to include aggregate observations
        
    Returns:
        Dictionary containing KG extraction results
    """
    extractor = TableKGExtractor(include_aggregates=include_aggregates)
    return extractor.extract_table_kg(structured_data, doc_id, table_id)
