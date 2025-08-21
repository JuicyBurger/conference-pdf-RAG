"""
Deterministic Table KG Extraction

Rule-based table structure analysis and schema-free observation generation
for knowledge graph extraction from table structured_data.
"""

import re
import hashlib
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Aggregate row detection
AGG_RE = re.compile(r'(總計|合計|平均|小計|Total|Average)', re.I)

def is_aggregate_row(rec: dict, headers: List[str], numeric_cols: set) -> bool:
    """Detect if row is an aggregate/total row"""
    # Only look at non-numeric fields (likely row headers)
    txt = ' '.join(str(rec.get(h,'')).strip() for h in headers if h not in numeric_cols)
    return bool(AGG_RE.search(txt))


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


def compute_numeric_density(headers: List[str], records: List[dict]) -> float:
    """Compute numeric density across all cells, handling ragged rows"""
    # Count actual present cells and numeric cells
    present = sum(1 for r in records for h in headers if h in r)
    numeric = sum(1 for r in records for h in headers if h in r and is_numeric_value(r[h]))
    
    return numeric / present if present else 0.0


def is_numeric_value(v):
    if v is None: return False
    if isinstance(v, (int, float)): return True
    s = str(v).strip()
    if not s: return False
    
    # Remove common non-numeric characters but preserve decimal points and signs
    s2 = re.sub(r'[,\s%¥$€£]', '', s)
    
    # Check for numeric patterns including scientific notation
    numeric_pattern = r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$'
    return bool(re.match(numeric_pattern, s2))


def detect_thead_pattern(headers: List[str]) -> bool:
    """Detect if table has a thead pattern"""
    # Check for header-like patterns (short, descriptive, no numbers)
    header_like_count = 0
    for header in headers:
        if header and len(header) < 50 and not re.search(r'\d+', header):
            header_like_count += 1
    
    return header_like_count / len(headers) > 0.7 if headers else False


def detect_rowspan_pattern(records: List[dict], headers: List[str]) -> bool:
    """Detect rowspan patterns in records"""
    if not records or not headers:
        return False
    
    # Check for repeated values in first column (common rowspan pattern)
    first_col_values = [str(r.get(headers[0], "")).strip() for r in records]
    
    # Count consecutive repeated values
    consecutive_repeats = sum(1 for i in range(1, len(first_col_values))
                              if first_col_values[i] and first_col_values[i] == first_col_values[i-1])
    
    return consecutive_repeats > len(records) * 0.3


def detect_colspan_pattern(headers: List[str]) -> bool:
    """Detect colspan patterns in headers"""
    # Check for empty or merged headers
    empty_headers = sum(1 for h in headers if not h or h.strip() == "")
    return empty_headers > len(headers) * 0.2


def detect_header_repetition(headers: List[str]) -> bool:
    """Detect repeated header patterns"""
    if len(headers) < 2:
        return False
    
    # Check for repeated header patterns
    for i in range(len(headers) - 1):
        if headers[i] and headers[i] == headers[i + 1]:
            return True
    
    return False


def detect_matrix_pattern(headers: List[str], records: List[dict]) -> bool:
    """Detect matrix/crosstab patterns"""
    if not headers or not records:
        return False
    
    # Check for matrix-like structure (similar number of columns and rows)
    col_count = len(headers)
    row_count = len(records)
    
    # Matrix typically has similar dimensions
    ratio = min(col_count, row_count) / max(col_count, row_count)
    return ratio > 0.3


def classify_table_rules(numeric_density: float, patterns: dict) -> Tuple[str, float]:
    """Rule-based table classification"""
    
    # Fact/quant table: high numeric density, clear headers
    if numeric_density > 0.6 and patterns['has_thead']:
        return "fact_quant", 0.9
    
    # Matrix/crosstab: similar dimensions, some numeric content
    if patterns['matrix_pattern'] and numeric_density > 0.3:
        return "matrix_crosstab", 0.8
    
    # Form/directory: low numeric density, structured layout
    if numeric_density < 0.3 and (patterns['has_rowspan'] or patterns['has_colspan']):
        return "form_directory", 0.7
    
    # Generic table: fallback
    return "generic", 0.5


def create_stats_sketch(headers: List[str], records: List[dict], patterns: dict) -> dict:
    """Create compact stats sketch for LLM classification"""
    return {
        "header_count": len(headers),
        "record_count": len(records),
        "numeric_density": compute_numeric_density(headers, records),
        "patterns": patterns,
        "sample_headers": headers[:5] if headers else [],
        "sample_records": records[:3] if records else []
    }


def classify_with_llm(stats_sketch: dict) -> Tuple[str, float]:
    """LLM fallback for table classification (placeholder)"""
    # TODO: Implement LLM classification
    # For now, return generic with low confidence
    return "generic", 0.4


def build_metric_path(headers: List[str], col_idx: int, structured_data: dict = None) -> List[str]:
    """Build metric path from column hierarchy"""
    if col_idx >= len(headers):
        return []
    
    header = headers[col_idx]
    if not header:
        return []
    
    # Check for hierarchical headers if available
    header_paths = structured_data.get("header_paths") if structured_data else None
    if header_paths and col_idx < len(header_paths):
        return header_paths[col_idx]
    
    # For now, treat single header as path
    return [header]


def get_metric_path_str(metric_path: List[str], header: str) -> str:
    """Get joined metric path string for convenience"""
    return " > ".join(metric_path) if metric_path else header


def build_dimensions(headers, records, row_idx, col_idx, numeric_cols):
    dims = {}
    record = records[row_idx]
    for i, h in enumerate(headers):
        if i == col_idx: 
            continue  # skip metric column
        if h in numeric_cols: 
            continue  # numeric cols are metrics, not dims
        val = record.get(h)
        if val is None: 
            continue
        s = str(val).strip()
        if not s: 
            continue
        if not is_numeric_value(val):  # keep categorical only
            dims[h] = s
    return dims


def is_data_cell(value: Any) -> bool:
    """Determine if a cell contains numeric data (not header/empty/narrative)"""
    # Only treat numeric-like cells as facts; dates/currency already pass is_numeric_value
    return is_numeric_value(value)


def extract_unit_from_cell(value: str) -> Optional[Dict[str, Any]]:
    """Extract unit from cell text (highest precedence)"""
    if not value or not isinstance(value, str):
        return None
    
    # Common unit patterns
    unit_patterns = [
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*%', '%', 0.9),
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*千元', '千元', 0.9),
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*仟元', '仟元', 0.9),
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*萬元', '萬元', 0.9),
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*億元', '億元', 0.9),
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*元', '元', 0.8),
    ]
    
    for pattern, unit, confidence in unit_patterns:
        match = re.search(pattern, value)
        if match:
            return {
                'unit': unit,
                'confidence': confidence,
                'value': match.group(1)
            }
    
    return None


def extract_unit_from_header(header: str) -> Optional[Dict[str, Any]]:
    """Extract unit from column header"""
    if not header or not isinstance(header, str):
        return None
    
    # First check for parentheses units (e.g., 平均分數（分）)
    paren_match = re.search(r'[\(（]\s*([^()（）]+?)\s*[\)）]', header)
    if paren_match:
        return {
            'unit': paren_match.group(1).strip(),
            'confidence': 0.8
        }
    
    # Check for unit indicators in header
    unit_indicators = {
        '百分比': '%',
        '比率': '%',
        '金額': '元',
        '營收': '元',
        '成本': '元',
        '費用': '元',
        '利潤': '元',
        '資產': '元',
        '負債': '元',
    }
    
    for indicator, unit in unit_indicators.items():
        if indicator in header:
            return {
                'unit': unit,
                'confidence': 0.7
            }
    
    return None


def extract_unit_from_ancestors(headers: List[str], col_idx: int) -> Optional[Dict[str, Any]]:
    """Extract unit from ancestor headers (placeholder)"""
    # TODO: Implement ancestor header unit extraction
    return None


def extract_unit_from_caption(notes: List[dict]) -> Optional[Dict[str, Any]]:
    """Extract unit from caption/notes"""
    if not notes:
        return None
    
    # Check notes for unit information
    for note in notes:
        note_text = note.get('text', '')
        if '單位' in note_text or 'unit' in note_text.lower():
            # Extract unit from note
            unit_match = re.search(r'單位[：:]\s*([^\s,，。]+)', note_text)
            if unit_match:
                return {
                    'unit': unit_match.group(1),
                    'confidence': 0.6
                }
    
    return None


def infer_unit_from_context(value: str, header: str, headers: List[str]) -> Dict[str, Any]:
    """Infer unit from context (lowest precedence)"""
    # Return None unit with low confidence when no reliable unit is found
    return {'unit': None, 'confidence': 0.0}


def parse_numeric_value(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    if not s: return None
    
    # Remove common non-numeric characters but preserve decimal points and signs
    s2 = re.sub(r'[,\s%¥$€£]', '', s)
    
    # Extract numeric value with scientific notation support
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s2)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    
    return None


def has_large_rowspan(records: List[dict], row_idx: int, headers: List[str]) -> bool:
    """Check if row has large rowspan pattern"""
    if row_idx >= len(records) or not headers:
        return False
    
    record = records[row_idx]
    if not record:
        return False
    
    # Use headers[0] to get first column value
    first_value = record.get(headers[0], "")
    return len(str(first_value)) > 20  # Long text suggests rowspan


def extract_panel_from_rowspan(records: List[dict], row_idx: int, headers: List[str]) -> Optional[str]:
    """Extract panel name from rowspan"""
    if row_idx >= len(records) or not headers:
        return None
    
    record = records[row_idx]
    if not record:
        return None
    
    # Use headers[0] to get first column value
    v = record.get(headers[0], "")
    v = str(v).strip()
    return v or None


def is_header_repeat(headers: List[str], row_idx: int) -> bool:
    """Check if row contains header repetition"""
    # TODO: Implement header repetition detection
    return False


def extract_panel_from_header_repeat(headers: List[str], row_idx: int) -> Optional[str]:
    """Extract panel from header repetition"""
    # TODO: Implement panel extraction from header repetition
    return None


def has_horizontal_rule(records: List[dict], row_idx: int) -> bool:
    """Check for horizontal rules (placeholder)"""
    # TODO: Implement horizontal rule detection
    return False


def extract_panel_from_rules(records: List[dict], row_idx: int) -> Optional[str]:
    """Extract panel from rules (placeholder)"""
    # TODO: Implement panel extraction from rules
    return None


def has_run_length_segment(records: List[dict], row_idx: int) -> bool:
    """Check for run-length segments (placeholder)"""
    # TODO: Implement run-length segment detection
    return False


def extract_panel_from_run_length(records: List[dict], row_idx: int) -> Optional[str]:
    """Extract panel from run-length segments (placeholder)"""
    # TODO: Implement panel extraction from run-length
    return None


def generate_observation_hash(rec, header, value, *, doc_id="", table_id="", row_idx=-1, col_idx=-1):
    """Generate deterministic hash for observation with provenance"""
    s = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    key = f"{doc_id}|{table_id}|{row_idx}|{col_idx}|{header}|{value}|{s}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def analyze_table_structure(structured_data: dict) -> dict:
    """Analyze table structure to determine type and extraction strategy"""
    headers = structured_data.get('headers', [])
    records = structured_data.get('records', [])
    
    # Compute numeric density
    numeric_density = compute_numeric_density(headers, records)
    
    # Check for patterns
    patterns = {
        'has_thead': detect_thead_pattern(headers),
        'has_rowspan': detect_rowspan_pattern(records, headers),
        'has_colspan': detect_colspan_pattern(headers),
        'repeated_headers': detect_header_repetition(headers),
        'matrix_pattern': detect_matrix_pattern(headers, records)
    }
    
    # Rule-based classification
    table_type, confidence = classify_table_rules(numeric_density, patterns)
    
    # LLM fallback if confidence < 0.6
    if confidence < 0.6:
        stats_sketch = create_stats_sketch(headers, records, patterns)
        table_type, confidence = classify_with_llm(stats_sketch)
    
    return {
        'table_type': table_type,
        'confidence': confidence,
        'numeric_density': numeric_density,
        'patterns': patterns
    }


def generate_observations(structured_data: dict, analysis: dict, doc_id: str, table_id: str, include_aggregates: bool = True) -> List[dict]:
    """Generate schema-free observation JSON for each data cell"""
    headers = structured_data.get('headers', []) or []
    records = structured_data.get('records', []) or []
    
    # Precompute numeric-cols once (fast + correct)
    numeric_cols = {h for h in headers if any(is_numeric_value(r.get(h)) for r in records)}
    
    # Compute row panels once
    row_panels = compute_row_panels(headers, records)
    
    observations = []
    
    for row_idx, rec in enumerate(records):
        # (optional) skip/mark aggregates
        is_agg = is_aggregate_row(rec, headers, numeric_cols)
        
        # Skip aggregate rows if not included
        if is_agg and not include_aggregates:
            continue
        
        for col_idx, header in enumerate(headers):
            if header not in rec: 
                continue
            
            value = rec.get(header, "")
            
            # Data cell gating: numeric only
            if not is_data_cell(value):
                continue
            
            # Skip if this column is not numeric → not a metric cell
            if header not in numeric_cols:
                continue
            
            # Build metric path (column hierarchy)
            metric_path = build_metric_path(headers, col_idx, structured_data)
            metric_leaf = metric_path[-1] if metric_path else header
            metric_path_str = get_metric_path_str(metric_path, header)
            
            # Build dimensions (row hierarchy + categorical columns)
            dimensions = build_dimensions(headers, records, row_idx, col_idx, numeric_cols)
            
            # Extract unit information
            unit_info = extract_unit_info(value, header, headers, col_idx, structured_data)
            
            # Use row-level panel assignment
            panel = row_panels[row_idx] if row_idx < len(row_panels) else None
            panel_confidence = 0.9 if panel else 0.0
            
            observation = {
                'doc_id': doc_id, 
                'table_id': table_id,
                'page': structured_data.get('page'), 
                'anchor': structured_data.get('anchor'),
                'row_idx': row_idx, 
                'col_idx': col_idx,
                'metric_path': metric_path,
                'metric_path_str': metric_path_str,
                'metric_leaf': metric_leaf,
                'dimensions': dimensions,
                'value_raw': value,
                'value_num': unit_info.value_num,
                'unit': unit_info.unit,
                'unit_source': unit_info.unit_source,
                'unit_confidence': unit_info.unit_confidence,
                'panel': panel,
                'panel_confidence': panel_confidence,
                'aggregate': is_agg,
                'source_hash': generate_observation_hash(rec, header, value, doc_id=doc_id, table_id=table_id, row_idx=row_idx, col_idx=col_idx)
            }
            
            observations.append(observation)
    
    return observations


def extract_unit_info(value: str, header: str, headers: List[str], col_idx: int, structured_data: dict) -> UnitInfo:
    """Extract unit information with strict precedence"""
    
    # 1. Cell text first (e.g., "10.0%", "1,838 仟元")
    cell_unit = extract_unit_from_cell(value)
    if cell_unit:
        return UnitInfo(
            value_num=parse_numeric_value(value),
            unit=cell_unit['unit'],
            unit_source='cell',
            unit_confidence=cell_unit['confidence']
        )
    
    # 2. Leaf column header
    leaf_unit = extract_unit_from_header(header)
    if leaf_unit:
        return UnitInfo(
            value_num=parse_numeric_value(value),
            unit=leaf_unit['unit'],
            unit_source='leaf',
            unit_confidence=leaf_unit['confidence']
        )
    
    # 3. Ancestor headers
    ancestor_unit = extract_unit_from_ancestors(headers, col_idx)
    if ancestor_unit:
        return UnitInfo(
            value_num=parse_numeric_value(value),
            unit=ancestor_unit['unit'],
            unit_source='ancestor',
            unit_confidence=ancestor_unit['confidence']
        )
    
    # 4. Caption/notes (if available)
    caption_unit = extract_unit_from_caption(structured_data.get('notes', []))
    if caption_unit:
        return UnitInfo(
            value_num=parse_numeric_value(value),
            unit=caption_unit['unit'],
            unit_source='caption',
            unit_confidence=caption_unit['confidence']
        )
    
    # 5. Inference (last resort)
    inferred_unit = infer_unit_from_context(value, header, headers)
    return UnitInfo(
        value_num=parse_numeric_value(value),
        unit=inferred_unit['unit'],
        unit_source='inferred',
        unit_confidence=inferred_unit['confidence']
    )


def compute_row_panels(headers: List[str], records: List[dict]) -> List[Optional[str]]:
    """Compute panel assignment for each row using run-length and left-band heuristics"""
    row_panels = []
    
    for row_idx in range(len(records)):
        panel = None
        
        # Check for large rowspan labels in left band (column 0)
        if has_large_rowspan(records, row_idx, headers):
            panel = extract_panel_from_rowspan(records, row_idx, headers)
        
        # Check for mid-table header repeats
        if not panel and is_header_repeat(headers, row_idx):
            panel = extract_panel_from_header_repeat(headers, row_idx)
        
        # Check for thick horizontal rules (if available from PDF)
        if not panel and has_horizontal_rule(records, row_idx):
            panel = extract_panel_from_rules(records, row_idx)
        
        # Check for long run-length segments in first column
        if not panel and has_run_length_segment(records, row_idx):
            panel = extract_panel_from_run_length(records, row_idx)
        
        row_panels.append(panel)
    
    return row_panels


def detect_panel(headers: List[str], records: List[dict], row_idx: int, col_idx: int) -> PanelInfo:
    """Detect panel using layout/repetition heuristics"""
    
    # Check for large rowspan labels in left band
    if col_idx == 0 and has_large_rowspan(records, row_idx, headers):
        panel = extract_panel_from_rowspan(records, row_idx, headers)
        return PanelInfo(panel=panel, panel_confidence=0.9)
    
    # Check for mid-table header repeats
    if is_header_repeat(headers, row_idx):
        panel = extract_panel_from_header_repeat(headers, row_idx)
        return PanelInfo(panel=panel, panel_confidence=0.8)
    
    # Check for thick horizontal rules (if available from PDF)
    if has_horizontal_rule(records, row_idx):
        panel = extract_panel_from_rules(records, row_idx)
        return PanelInfo(panel=panel, panel_confidence=0.7)
    
    # Check for long run-length segments in first column
    if col_idx == 0 and has_run_length_segment(records, row_idx):
        panel = extract_panel_from_run_length(records, row_idx)
        return PanelInfo(panel=panel, panel_confidence=0.6)
    
    return PanelInfo(panel=None, panel_confidence=0.0)


def extract_table_kg(structured_data: dict, doc_id: str, table_id: str, include_aggregates: bool = True) -> dict:
    """Main function to extract KG from table structured_data"""
    
    try:
        # 1. Analyze table structure
        analysis = analyze_table_structure(structured_data)
        
        # 2. Generate observations
        observations = generate_observations(structured_data, analysis, doc_id, table_id, include_aggregates)
        
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
