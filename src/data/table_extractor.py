import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import defaultdict
from bs4 import BeautifulSoup, Tag
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import platform

# Import our Ollama client and config
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.client_factory import get_llm_client
from src.config import OCR_MODEL

# Transformers model constants
TRANSFORMERS_MODEL_REPO = "nanonets/Nanonets-OCR-s"
TRANSFORMERS_PROMPT = """
Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.
"""

# Global variables for Transformers model
_transformers_model = None
_transformers_tokenizer = None
_transformers_processor = None

def _load_transformers_model():
    """Load the Transformers model for OCR."""
    global _transformers_model, _transformers_tokenizer, _transformers_processor
    
    if _transformers_model is not None:
        return _transformers_model, _transformers_tokenizer, _transformers_processor

    # Windows: skip flash_attn, use built-in SDPA attention
    if platform.system().lower().startswith("win"):
        print("Windows 環境：略過 flash_attn，使用內建 SDPA 注意力。")

    kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }

    _transformers_model = AutoModelForImageTextToText.from_pretrained(TRANSFORMERS_MODEL_REPO, **kwargs)
    _transformers_model.eval()
    _transformers_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_REPO)
    _transformers_processor = AutoProcessor.from_pretrained(TRANSFORMERS_MODEL_REPO)
    
    return _transformers_model, _transformers_tokenizer, _transformers_processor

class TableExtractor:
    """
    Extract table data from images using Ollama vision model and convert to JSON.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize the table extractor.
        
        Args:
            model: Ollama model name to use for vision tasks (defaults to OCR_MODEL from config)
        """
        # Get the client using our standard factory
        self.client = get_llm_client()
        
        # Use OCR_MODEL from config
        self.model = model or OCR_MODEL
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            # Use our client to test connection
            response = self.client.list()
            print(f"✅ Connected to Ollama server")
            print(f"🔧 Using OCR model: {self.model}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama server: {e}")
    

    def extract_tables_from_image(self, image_path: str) -> List[Dict]:
        """
        Extract table data from an image and return as JSON.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing table data
        """
        # Step 1: Extract HTML from image using Ollama
        html_content = self._ocr_image_to_html(image_path)
        
        # Step 2: Parse HTML tables to JSON
        return self._parse_html_tables(html_content)
    
    def _ocr_image_to_html(self, image_path: str) -> str:
        """Use Ollama to extract table HTML from image."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Prepare prompt for table extraction
        system_prompt = """只要判斷表格內容"""
        user_prompt = """
        Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.
        """ 
        
        try:
            # Use our standard LLM function for consistency
            from src.models.LLM import LLM as llm_function
            
            response = llm_function(
                client=self.client,
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                img=image_path,
                options={"temperature": 0.1},  # Low temperature for consistent output
                raw=True  # Get raw text response
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from image: {e}")
    
    def _ocr_image_to_html_transformers(self, image_path: str) -> str:
        """Use Transformers model to extract table HTML from image."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load Transformers model
            model, _, processor = _load_transformers_model()
            img = Image.open(image_path)

            messages = [
                {"role": "system", "content": "只要判斷表格內容"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                        {"type": "text", "text": TRANSFORMERS_PROMPT},
                    ],
                },
            ]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[img], padding=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=4000, do_sample=False)

            gen_only = output_ids[0][len(inputs.input_ids[0]):]
            text = processor.batch_decode([gen_only], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            # Debug: Print the raw response for troubleshooting
            print(f"🔍 Transformers OCR Response Length: {len(text)} characters")
            print(f"🔍 Transformers Response Preview: {text[:200]}...")
            
            # Validate that we got HTML content
            if not text or len(text.strip()) < 10:
                raise RuntimeError("Transformers OCR returned empty or very short response")
            
            if "<table" not in text.lower():
                print("⚠️  Warning: Transformers response doesn't contain <table> tags")
                print(f"⚠️  Transformers response content: {text}")
            
            return text
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from image using Transformers: {e}")
    
    def extract_tables_from_image_transformers(self, image_path: str) -> List[Dict]:
        """
        Extract table data from an image using Transformers model and return as JSON.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing table data
        """
        # Step 1: Extract HTML from image using Transformers
        html_content = self._ocr_image_to_html_transformers(image_path)
        
        # Step 2: Parse HTML tables to JSON
        return self._parse_html_tables(html_content)

    def _parse_html_tables(self, html: str) -> List[Dict]:
        """
        Parse HTML tables into structured JSON data.
        
        Args:
            html: HTML string containing table elements
            
        Returns:
            List of table data dictionaries
        """
        soup = BeautifulSoup(html, "html.parser")
        tables = []
        
        for table in soup.find_all("table"):
            table_data = self._parse_single_table(table)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _parse_single_table(self, table: Tag) -> Optional[Dict]:
        """Parse a single table element into structured data."""
        trs = table.find_all("tr")
        if not trs:
            return None
        
        # Detect header rows
        header_rows = self._infer_header_rows(trs)
        header_trs = trs[:header_rows]
        data_trs = trs[header_rows:]
        
        # Build header structure
        header_grid = self._build_header_grid(header_trs)
        if not header_grid:
            return None
        
        header_paths = self._flatten_header_paths(header_grid)
        header_paths = self._dedup_headers(header_paths)
        
        # Detect and separate notes/footnotes
        note_indices, notes = self._detect_notes(data_trs, len(header_paths))
        
        # Filter out note rows from data
        data_trs = [tr for i, tr in enumerate(data_trs) if i not in note_indices]
        
        # Expand data rows
        data_matrix = self._expand_rows(data_trs, len(header_paths))
        
        # Convert to JSON records
        records = []
        for row in data_matrix:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            
            record = {}
            for header, value in zip(header_paths, row):
                if header is None:
                    continue
                cleaned_value = self._clean_cell(value)
                record[header] = self._maybe_parse_number(cleaned_value)
            records.append(record)
        
        # Add notes as special records if any
        for note in notes:
            note_record = {h: "" for h in header_paths if h}
            if header_paths:
                note_record[header_paths[0]] = note.get("label", "")
            note_record["__note__"] = True
            note_record["note_text"] = note.get("text", "")
            records.append(note_record)
        
        return {
            "headers": [h for h in header_paths if h],
            "records": records,
            "notes": notes
        }
    
    def _infer_header_rows(self, tr_tags: List[Tag]) -> int:
        """Infer how many rows are headers based on content patterns."""
        def is_numeric_like(text: str) -> bool:
            text = text.strip()
            if not text:
                return False
            # Check for numeric patterns
            numeric_pattern = re.compile(r"[0-9０-９]")
            if numeric_pattern.search(text):
                core = text.replace(" ", "")
                if re.fullmatch(r"[()％%0-9０-９.\-+,/]+", core):
                    return True
                if '%' in text or '％' in text:
                    return True
            return False
        
        count = 0
        first_row_numeric = 0
        
        for tr in tr_tags:
            cells = tr.find_all(["td", "th"])
            if not cells:
                break
            
            texts = [cell.get_text(strip=True) for cell in cells]
            numeric_count = sum(1 for text in texts if is_numeric_like(text))
            total = len(cells)
            numeric_ratio = numeric_count / total if total else 0.0
            
            if count == 0:
                first_row_numeric = numeric_count
                count += 1
                continue
            
            # Rule 1: If first row has no numbers, stop when numbers appear
            if first_row_numeric == 0 and numeric_count > 0:
                break
            
            # Rule 2: Stop if numeric ratio is high
            if numeric_ratio >= 0.5:
                break
            
            count += 1
        
        return max(1, min(count, len(tr_tags) - 1))
    
    def _build_header_grid(self, tr_tags: List[Tag]) -> List[List[Optional[str]]]:
        """Build header grid with rowspan/colspan expansion."""
        if not tr_tags:
            return []
        
        # Calculate total columns including active rowspans
        total_cols = 0
        active_spans = []  # (remaining_rows, colspan)
        
        for tr in tr_tags:
            inherited_width = sum(cs for rem, cs in active_spans if rem > 0)
            row_width = sum(int(cell.get("colspan", 1)) for cell in tr.find_all(["td", "th"]))
            leaf_width = inherited_width + row_width
            total_cols = max(total_cols, leaf_width)
            
            # Update active spans
            active_spans = [(rem - 1, cs) for rem, cs in active_spans if rem - 1 > 0]
            
            # Add new rowspans
            for cell in tr.find_all(["td", "th"]):
                rs = int(cell.get("rowspan", 1))
                cs = int(cell.get("colspan", 1))
                if rs > 1:
                    active_spans.append((rs - 1, cs))
        
        # Build the grid
        rows_n = len(tr_tags)
        grid = [[None] * total_cols for _ in range(rows_n)]
        pending = defaultdict(list)
        
        for r, tr in enumerate(tr_tags):
            # Apply pending rowspans
            if r in pending:
                for col_idx, text in pending[r]:
                    if 0 <= col_idx < total_cols:
                        grid[r][col_idx] = text
            
            cells = tr.find_all(["td", "th"])
            c = 0
            
            for cell in cells:
                # Find next empty position
                while c < total_cols and grid[r][c] is not None:
                    c += 1
                if c >= total_cols:
                    break
                
                rs = int(cell.get("rowspan", 1))
                cs = int(cell.get("colspan", 1))
                text = self._cell_text(cell)
                
                # Fill current row
                for i in range(cs):
                    if c + i < total_cols:
                        grid[r][c + i] = text
                
                # Schedule rowspans for future rows
                if rs > 1:
                    for rr in range(r + 1, r + rs):
                        for i in range(cs):
                            if c + i < total_cols:
                                pending[rr].append((c + i, text))
                c += cs
        
        return grid
    
    def _flatten_header_paths(self, grid: List[List[Optional[str]]]) -> List[Optional[str]]:
        """Flatten header grid into column paths."""
        if not grid:
            return []
        
        cols_n = len(grid[0])
        headers = []
        
        for col in range(cols_n):
            parts = []
            last = None
            
            for row in range(len(grid)):
                val = grid[row][col]
                if not val:
                    continue
                v = val.strip()
                if v == "":
                    continue
                if v != last:
                    parts.append(v)
                    last = v
            
            headers.append(" / ".join(parts) if parts else None)
        
        return headers
    
    def _detect_notes(self, data_trs: List[Tag], col_count: int) -> tuple[set, List[Dict]]:
        """Detect footnote/explanation rows."""
        note_indices = set()
        notes = []
        
        for idx, tr in enumerate(data_trs):
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue
            
            cell_meta = []
            for cell in tds:
                cs = int(cell.get("colspan", 1))
                txt = self._cell_text(cell)
                cell_meta.append((txt, cs))
            
            # Check for long-span cells (potential notes)
            long_span_found = False
            for i, (txt, cs) in enumerate(cell_meta):
                ratio = cs / max(1, col_count)
                if (i == 0 and len(cell_meta) == 1 and cs >= col_count and len(txt) >= 20):
                    long_span_found = True
                elif i > 0 and ratio >= 0.55 and len(txt) >= 20:
                    long_span_found = True
            
            if long_span_found:
                note_indices.add(idx)
                if len(cell_meta) == 1:
                    notes.append({"label": "", "text": cell_meta[0][0]})
                else:
                    first = cell_meta[0][0].strip()
                    rest_text = " ".join(m[0] for m in cell_meta[1:]).strip()
                    if len(first) <= 12 and rest_text:
                        notes.append({"label": first, "text": rest_text})
                    else:
                        notes.append({"label": "", "text": " ".join(m[0] for m in cell_meta).strip()})
        
        return note_indices, notes
    
    def _expand_rows(self, tr_tags: List[Tag], total_cols: int) -> List[List[Optional[str]]]:
        """Expand data rows handling rowspan/colspan."""
        if not tr_tags:
            return []
        
        pending = defaultdict(list)
        matrix = []
        
        for r, tr in enumerate(tr_tags):
            row = [None] * total_cols
            
            # Apply pending rowspans
            if r in pending:
                for col_idx, text in pending[r]:
                    if 0 <= col_idx < total_cols:
                        row[col_idx] = text
            
            cells = tr.find_all(["td", "th"])
            c = 0
            
            for cell in cells:
                while c < total_cols and row[c] is not None:
                    c += 1
                if c >= total_cols:
                    break
                
                rs = int(cell.get("rowspan", 1))
                cs = int(cell.get("colspan", 1))
                text = self._cell_text(cell)
                
                for i in range(cs):
                    if c + i < total_cols:
                        row[c + i] = text
                
                if rs > 1:
                    for rr in range(r + 1, r + rs):
                        for i in range(cs):
                            if c + i < total_cols:
                                pending[rr].append((c + i, text))
                c += cs
            
            matrix.append(row)
        
        return matrix
    
    def _cell_text(self, cell: Tag) -> str:
        """Extract clean text from cell."""
        for br in cell.find_all("br"):
            br.replace_with("\n")
        txt = cell.get_text(separator=" ").strip()
        return re.sub(r"\s+", " ", txt)
    
    def _dedup_headers(self, headers: List[Optional[str]]) -> List[Optional[str]]:
        """Remove duplicate headers by adding suffixes."""
        seen = {}
        out = []
        
        for h in headers:
            if h is None:
                out.append(None)
                continue
            if h not in seen:
                seen[h] = 1
                out.append(h)
            else:
                seen[h] += 1
                out.append(f"{h}_{seen[h]}")
        
        return out
    
    def _clean_cell(self, val: Optional[str]) -> str:
        """Clean cell value."""
        return "" if val is None else val.strip()
    
    def _maybe_parse_number(self, s: str):
        """Parse string to number if possible."""
        if not s:
            return s
        
        # Remove commas and try to parse
        s_clean = s.replace(",", "")
        try:
            if "." in s_clean:
                return float(s_clean)
            else:
                return int(s_clean)
        except ValueError:
            return s


def extract_tables_from_image(image_path: str, model: str = None) -> List[Dict]:
    """
    Convenience function to extract tables from an image using Ollama.
    
    Args:
        image_path: Path to the image file
        model: Ollama model name (defaults to OCR_MODEL from config)
        
    Returns:
        List of table data dictionaries
    """
    extractor = TableExtractor(model)
    return extractor.extract_tables_from_image(image_path)

def extract_tables_from_image_transformers(image_path: str) -> List[Dict]:
    """
    Convenience function to extract tables from an image using Transformers.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of table data dictionaries
    """
    extractor = TableExtractor()
    return extractor.extract_tables_from_image_transformers(image_path)


if __name__ == "__main__":
    # Example usage
    test_image = "data/tables/113年報 20240531-16-18/table_05_P3.png"
    
    if Path(test_image).exists():
        try:
            tables = extract_tables_from_image(test_image)
            print(json.dumps(tables, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run table_to_image_camelot.py first to generate test images")