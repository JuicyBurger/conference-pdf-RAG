"""
Node builder helpers for converting paragraphs and tables to structured nodes
"""

import pandas as pd
import json
import re
from typing import List, Dict, Any


def paragraphs_to_nodes(page_no: int, paragraph_text: str) -> List[Dict[str, Any]]:
    """
    Convert paragraph text to nodes.
    For now, we create one node per page's paragraph content.
    
    Args:
        page_no: Page number (1-based)
        paragraph_text: Clean paragraph text for the page
        
    Returns:
        List of paragraph nodes
    """
    if not paragraph_text or not paragraph_text.strip():
        return []
    
    return [{
        "page": page_no,
        "type": "paragraph", 
        "text": paragraph_text.strip()
    }]


def clean_cell(cell: Any) -> str:
    """Clean and normalize cell content."""
    s = str(cell)
    return re.sub(r'\s+', ' ', s.replace('\n', ' ')).strip()


def is_number(s: str) -> bool:
    """Return True if s is purely a number/comma/dot string."""
    return bool(re.fullmatch(r'[\d,\.%]+', s))


def detect_header_rows(df: pd.DataFrame) -> List[int]:
    """
    Treat every row up until the first one containing ANY numeric cell
    as a header row.
    """
    header_idxs = []
    for idx, row in df.iterrows():
        # clean each cell and test
        cells = [clean_cell(c) for c in row]
        if any(is_number(c) for c in cells if c):
            break
        header_idxs.append(idx)
    return header_idxs


def collapse_headers(df: pd.DataFrame, header_idxs: List[int]) -> List[str]:
    """
    Concatenate the text of every header row into a single column name for each column index.
    """
    # Extract the header rows as clean text
    rows = df.loc[header_idxs].map(clean_cell).values.tolist()
    if not rows:
        return []
    ncol = len(rows[0])
    cols = [''] * ncol
    for row in rows:
        for j, cell in enumerate(row):
            if cell:
                cols[j] = (cols[j] + ' ' + cell).strip()
    return cols


def process_table_to_json(table_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a DataFrame and convert it to structured JSON format.
    
    Args:
        table_df: DataFrame containing table data
        
    Returns:
        Dictionary with structured table data
    """
    try:
        # 1) Clean all cells
        df = table_df.map(clean_cell)
        
        # 2) Auto-detect header rows by numeric detection
        header_idxs = detect_header_rows(df)
        if not header_idxs:
            header_idxs = [0]  # fallback
        
        # 3) Collapse header lines into a single list of column names
        columns = collapse_headers(df, header_idxs)
        
        # Fill-forward blank headers
        for j in range(1, len(columns)):
            if columns[j] == "":
                columns[j] = columns[j-1]
        
        # Drop any truly empty columns
        keep = [i for i, c in enumerate(columns) if c.strip()]
        columns = [columns[i] for i in keep]
        data_df = df.drop(index=header_idxs).reset_index(drop=True)
        data_df = data_df.iloc[:, keep]
        data_df.columns = columns
        
        # 4) Split narrative rows into notes vs. real records
        records = []
        notes = []
        first_col = columns[0] if columns else ""
        
        for rec in data_df.to_dict(orient='records'):
            val0 = rec.get(first_col, "")
            other_vals = [v for k, v in rec.items() if k != first_col]
            
            # If first column is blank or starts with summary keyword,
            # AND all other cells are blank, treat as note
            if (not val0 or val0.startswith("二年度")) and all(not v for v in other_vals):
                notes.append(val0 or "".join(other_vals))
            else:
                # Only keep rows with at least one non-blank value
                if any(str(v).strip() for v in rec.values()):
                    records.append(rec)
        
        # 5) Create output structure
        table_data = {
            "columns": columns,
            "records": records,
            "notes": notes
        }
        
        return table_data
        
    except Exception as e:
        print(f"⚠️ Error processing table: {e}")
        return {"columns": [], "records": [], "notes": []}


def table_to_nodes(page_no: int, table_df: pd.DataFrame, table_id: int) -> List[Dict[str, Any]]:
    """
    Convert a table DataFrame to structured nodes with JSON format.
    
    Args:
        page_no: Page number (1-based)
        table_df: DataFrame containing table data
        table_id: Table ID within the page (0-based)
        
    Returns:
        List of structured table nodes
    """
    nodes = []
    
    if table_df.empty:
        return nodes
    
    # Process table to structured JSON
    table_json = process_table_to_json(table_df)
    
    # Create different types of nodes for better searchability
    
    # 1. Table summary node (overview of the entire table)
    if table_json["records"]:
        summary_text = f"表格包含 {len(table_json['columns'])} 個欄位: {', '.join(table_json['columns'])}。共有 {len(table_json['records'])} 筆記錄。"
        if table_json["notes"]:
            summary_text += f" 備註: {' '.join(table_json['notes'])}"
        
        nodes.append({
            "page": page_no,
            "type": "table_summary",
            "table_id": table_id,
            "text": summary_text,
            "structured_data": table_json
        })
    
    # 2. Individual record nodes (each row as a meaningful sentence)
    for row_idx, record in enumerate(table_json["records"]):
        # Convert record to natural language
        record_parts = []
        for col, value in record.items():
            if value and str(value).strip():
                record_parts.append(f"{col}: {value}")
        
        if record_parts:
            record_text = "。".join(record_parts) + "。"
            
            nodes.append({
                "page": page_no,
                "type": "table_record",
                "table_id": table_id,
                "row_idx": row_idx,
                "text": record_text,
                "structured_data": record
            })
    
    # 3. Column-specific nodes (for column-based queries)
    for col_idx, column in enumerate(table_json["columns"]):
        col_values = [str(record.get(column, "")).strip() for record in table_json["records"]]
        col_values = [v for v in col_values if v]
        
        if col_values:
            col_text = f"在{column}欄位中，包含以下數據: {', '.join(col_values[:10])}" # Limit to first 10 values
            if len(col_values) > 10:
                col_text += f"等，共{len(col_values)}筆資料"
            col_text += "。"
            
            nodes.append({
                "page": page_no,
                "type": "table_column",
                "table_id": table_id,
                "column_idx": col_idx,
                "column_name": column,
                "text": col_text,
                "structured_data": {"column": column, "values": col_values}
            })
    
    return nodes


def clean_text_for_comparison(text: str) -> str:
    """
    Clean text for deduplication comparison.
    Remove extra whitespace and normalize punctuation.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text for comparison
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize common punctuation
    text = text.replace('|', '｜')  # Normalize pipe characters
    
    return text.strip().lower() 