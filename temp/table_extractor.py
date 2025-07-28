import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import camelot
import pandas as pd

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extracts tables from PDF pages using Camelot with intelligent header detection
    and LLM-friendly JSON output formatting.
    Based on the experiment in camelot_table_detection_test.py
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the table extractor.
        
        Args:
            output_dir: Directory to save extracted table JSON files. If None, uses default.
        """
        self.output_dir = output_dir or Path("data/processed/tables")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_cell(self, cell: Any) -> str:
        """Clean and normalize cell content."""
        s = str(cell)
        return re.sub(r'\s+', ' ', s.replace('\n', ' ')).strip()
    
    def is_number(self, s: str) -> bool:
        """Return True if s is purely a number/comma/dot string."""
        return bool(re.fullmatch(r'[\d,\.]+', s))
    
    def detect_header_rows(self, df: pd.DataFrame) -> List[int]:
        """
        Treat every row up until the first one containing ANY numeric cell
        as a header row.
        """
        header_idxs = []
        for idx, row in df.iterrows():
            # clean each cell and test
            cells = [self.clean_cell(c) for c in row]
            if any(self.is_number(c) for c in cells if c):
                break
            header_idxs.append(idx)
        return header_idxs
    
    def collapse_headers(self, df, header_idxs: list) -> list:
        """
        Concatenate the text of every header row into a single column name for each column index.
        """
        # Extract the header rows as clean text
        rows = df.loc[header_idxs].map(self.clean_cell).values.tolist()
        if not rows:
            return []
        ncol = len(rows[0])
        cols = [''] * ncol
        for row in rows:
            for j, cell in enumerate(row):
                if cell:
                    cols[j] = (cols[j] + ' ' + cell).strip()
        return cols
    
    def get_table_text(self, table) -> str:
        """
        Get the text representation of a table (for removal from paragraph).
        """
        # Join all cells row-wise
        return '\n'.join(['\t'.join(map(str, row)) for row in table.df.values.tolist()])

    def detect_tables_on_page(self, pdf_path: str, page_number: int) -> list:
        """
        Detect tables on a specific page using Camelot (lattice flavor only).
        """
        tables = []
        try:
            kwargs = {
                'pages': str(page_number),
                'strip_text': '\n',
                'split_text': True,
                'flavor': 'lattice'
            }
            camelot_tables = camelot.read_pdf(str(pdf_path), **kwargs)
            if camelot_tables.n > 0:
                tables.extend(camelot_tables)
                logger.debug(f"Found {camelot_tables.n} tables on page {page_number} with flavor lattice")
        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_number} with flavor lattice: {e}")
        return tables
    
    def process_table_to_json(self, table) -> str:
        """
        Process a single table and convert it to LLM-friendly JSON string.
        Based on the experiment logic from camelot_table_detection_test.py
        
        Args:
            table: Camelot table object
            
        Returns:
            JSON string representation of the table
        """
        try:
            # 1) Get raw DataFrame & clean all cells
            df_raw = table.df
            df = df_raw.map(self.clean_cell)
            
            # 2) Auto-detect header rows by numeric detection
            header_idxs = self.detect_header_rows(df)
            if not header_idxs:
                header_idxs = [0]  # fallback
            
            # 3) Collapse header lines into a single list of column names
            columns = self.collapse_headers(df, header_idxs)
            
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
            
            # 5) Create output structure and convert to JSON string
            table_data = {
                "columns": columns,
                "records": records,
                "notes": notes
            }
            
            return json.dumps(table_data, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return "{}"
    
    def process_page_with_tables(self, pdf_path: str, page_number: int, page_text: str) -> str:
        """
        Process a single page by combining text and tables into a single string.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to process
            page_text: Text extracted from the page using PyMuPDF
            
        Returns:
            Combined string with text and inline JSON tables
        """
        # Detect tables on this page
        tables = self.detect_tables_on_page(pdf_path, page_number)
        
        if not tables:
            # No tables found, return just the text
            return page_text
        
        # Process tables and combine with text
        result = page_text
        
        for table in tables:
            # Convert table to JSON string
            table_json = self.process_table_to_json(table)
            
            # Add table delimiter and JSON
            result += f"\n[TABLE]\n{table_json}\n[/TABLE]\n"
        
        return result
    
    @staticmethod
    def is_table_page(text: str) -> bool:
        """
        Heuristic to detect if a page contains a table based on column consistency or common table header keywords.
        Returns True if the text looks like a table, else False.
        Note: PyMuPDF text extraction is limited; this may miss some tables.
        """
        import re
        from collections import Counter
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        # Try to detect table-like structure by column count
        col_counts = [len([col for col in re.split(r'\s{2,}|\t|\|', line) if col.strip()]) for line in lines]
        col_counter = Counter(col_counts)
        if col_counter:
            most_common_cols, count = col_counter.most_common(1)[0]
            if most_common_cols >= 4 and count >= 3:
                return True
        # Additional heuristic: look for common table header keywords (customize as needed)
        table_keywords = ['職稱', '姓名', '國籍', '年齡', '持股', '比率', '日期', '公司', '股份']
        header_matches = sum(any(kw in line for kw in table_keywords) for line in lines[:5])
        if header_matches >= 2:
            return True
        return False
    
    def remove_table_text_from_paragraph(self, paragraph: str, tables: list, threshold: int = 85) -> str:
        """
        Remove table-like text from the paragraph using rapidfuzz fuzzy matching.
        Args:
            paragraph: The full page text.
            tables: List of Camelot table objects.
            threshold: Similarity threshold for removal (0-100).
        Returns:
            Cleaned paragraph string.
        """
        from rapidfuzz import fuzz
        para_lines = paragraph.splitlines()
        table_lines = []
        for table in tables:
            for row in table.df.values.tolist():
                row_text = ' '.join(map(str, row)).strip()
                if row_text:
                    table_lines.append(row_text)
        # Remove lines from para_lines that are similar to any table line
        cleaned_lines = []
        for pline in para_lines:
            if not pline.strip():
                cleaned_lines.append(pline)
                continue
            # If any table line is similar, skip this line
            if any(fuzz.partial_ratio(pline, tline) >= threshold for tline in table_lines):
                continue
            cleaned_lines.append(pline)
        return '\n'.join(cleaned_lines).strip()
    
 