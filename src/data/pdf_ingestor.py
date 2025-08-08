# src/ingestion/pdf_ingestor.py

import os
import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process
from .node_builder import paragraphs_to_nodes, table_to_nodes, clean_text_for_comparison

# Import camelot with error handling
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️ Camelot not available. Table extraction will be skipped.")

def _detect_common_headers_footers(doc: fitz.Document, top_margin: float = 60.0, bottom_margin: float = 60.0) -> Tuple[set, set]:
    """Detect repeated header/footer lines that appear on many pages and should be stripped."""
    from collections import Counter
    head_cnt: Counter = Counter()
    foot_cnt: Counter = Counter()
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text_dict = page.get_text("dict")
        height = page.rect.height
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                span_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not span_text.strip():
                    continue
                # Compute line y0 using span bbox values
                y_candidates = []
                for span in line.get("spans", []):
                    bbox = span.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                        y_candidates.append(bbox[1])
                y0 = min(y_candidates) if y_candidates else 0
                # Rough top/bottom detection using line bbox approximations
                if y0 <= top_margin:
                    head_cnt[span_text.strip()] += 1
                if y0 >= (height - bottom_margin):
                    foot_cnt[span_text.strip()] += 1
    # Keep strings that appear on > 70% of pages
    threshold = max(2, int(len(doc) * 0.7))
    headers = {s for s, c in head_cnt.items() if c >= threshold}
    footers = {s for s, c in foot_cnt.items() if c >= threshold}
    return headers, footers


def _reconstruct_paragraph_from_dict(page: fitz.Page, headers: set, footers: set) -> str:
    """Reconstruct a cleaner paragraph string from PyMuPDF dict blocks, skipping common headers/footers."""
    blocks = page.get_text("dict").get("blocks", [])
    parts: List[str] = []
    last_y = None
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = text.strip()
            if not text:
                continue
            # Skip repetitive headers/footers
            if text in headers or text in footers:
                continue
            # Simple join heuristic: if vertical gap small, add space; otherwise newline (later collapsed)
            y_candidates = []
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                    y_candidates.append(bbox[1])
            y = min(y_candidates) if y_candidates else None
            sep = " "
            if last_y is not None and y is not None and abs(y - last_y) > 18:  # new paragraph-ish
                sep = "\n"
            # Fix hyphenated breaks (mainly Western text); for CJK we simply concatenate
            if parts:
                parts.append(sep)
            parts.append(text)
            last_y = y
    # Collapse whitespace but keep sentence boundaries somewhat
    merged = " ".join(s for s in " ".join(parts).split())
    return merged


def extract_text_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of {"page": int, "text": str} for each page using block-aware reconstruction,
    skipping repetitive headers/footers and reducing PyMuPDF line noise.
    """
    pages: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        print(f"📄 Extracting text from {total_pages} pages...")
        headers, footers = _detect_common_headers_footers(doc)
        if headers:
            print(f"🔖 Detected headers to strip: {list(headers)[:3]}{'...' if len(headers)>3 else ''}")
        if footers:
            print(f"🔖 Detected footers to strip: {list(footers)[:3]}{'...' if len(footers)>3 else ''}")
        for pno in range(total_pages):
            page = doc.load_page(pno)
            text = _reconstruct_paragraph_from_dict(page, headers, footers)
            pages.append({"page": pno + 1, "text": text})
    return pages

def extract_image_blocks(pdf_path: str, output_dir: str = "extracted_images") -> List[Dict[str, Any]]:
    """
    Returns a list of image‐blocks with:
      - page: int (1-based)
      - bbox: (x0,y0,x1,y1) if available, else None
      - ext: "png"
      - img_bytes: PNG bytes
      - file_path: saved image file path (if output_dir provided)
    Falls back to page.get_images() if no dict‐blocks found.
    """
    img_blocks: List[Dict[str,Any]] = []
    doc = fitz.open(pdf_path)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    img_counter = 0
    total_pages = len(doc)
    print(f"🖼️  Extracting images from {total_pages} pages...")
    
    for pno in range(total_pages):
        page = doc[pno]
        found_on_page = False

        # 1) Try the text-dict blocks for images (type==1)
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 1:
                continue
            img_info = block.get("image")
            if not img_info or not isinstance(img_info, dict) or "xref" not in img_info:
                continue

            x0, y0, x1, y1 = block.get("bbox", (None,)*4)
            xref = img_info["xref"]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # CMYK or alpha
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            
            # Save image to file if output_dir is specified
            file_path = None
            if output_dir:
                img_counter += 1
                filename = f"page_{pno + 1}_img_{img_counter}.png"
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(img_bytes)

            img_blocks.append({
                "page": pno + 1,
                "bbox": (x0, y0, x1, y1),
                "ext": "png",
                "img_bytes": img_bytes,
                "file_path": file_path
            })
            found_on_page = True

        # 2) Fallback: use page.get_images() if no images found via dict
        if not found_on_page:
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                
                # Save image to file if output_dir is specified
                file_path = None
                if output_dir:
                    img_counter += 1
                    filename = f"page_{pno + 1}_img_{img_counter}.png"
                    file_path = os.path.join(output_dir, filename)
                    with open(file_path, "wb") as f:
                        f.write(img_bytes)

                # No reliable bbox from get_images(), so set None
                img_blocks.append({
                    "page": pno + 1,
                    "bbox": None,
                    "ext": "png",
                    "img_bytes": img_bytes,
                    "file_path": file_path
                })

    doc.close()
    return img_blocks

def extract_tables_per_page(pdf_path: str) -> Dict[int, List[pd.DataFrame]]:
    """
    Extract tables from PDF using Camelot (lattice method).
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dict mapping page number (1-based) to list of DataFrames
    """
    if not CAMELOT_AVAILABLE:
        return {}
    
    tables_by_page: Dict[int, List[pd.DataFrame]] = {}
    
    try:
        print("📊 Extracting tables using Camelot (lattice)...")
        # Detect text-based pages only to avoid image-based warnings and wasted work
        text_pages: List[int] = []
        with fitz.open(pdf_path) as doc:
            for pno in range(len(doc)):
                raw = doc.load_page(pno).get_text("text").strip()
                if raw:
                    text_pages.append(pno + 1)  # 1-based
        page_spec = 'all' if not text_pages else ",".join(str(i) for i in text_pages)
        tables = camelot.read_pdf(pdf_path, flavor='lattice', pages=page_spec)
        
        print(f"📊 Processing {len(tables)} tables...")
        for table in tables:
            page_no = table.page
            if page_no not in tables_by_page:
                tables_by_page[page_no] = []
            tables_by_page[page_no].append(table.df)
            
    except Exception as e:
        print(f"⚠️ Table extraction failed for {pdf_path}: {e}")
        return {}
    
    return tables_by_page


def deduplicate_content(paragraph_text: str, table_rows: List[str], threshold: float = 85.0) -> tuple[str, List[str]]:
    """
    Remove duplicate content between paragraphs and table rows.
    Uses RapidFuzz to find similar content and removes duplicates.
    
    Args:
        paragraph_text: Full paragraph text
        table_rows: List of table row texts
        threshold: Similarity threshold (default 85%)
        
    Returns:
        Tuple of (clean_paragraph_text, clean_table_rows)
    """
    if not table_rows or not paragraph_text:
        return paragraph_text, table_rows
    
    # Split paragraphs into sentences for comparison
    import re
    sentences = re.split(r'[。！？；]', paragraph_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Clean texts for comparison
    clean_sentences = [clean_text_for_comparison(s) for s in sentences]
    clean_rows = [clean_text_for_comparison(row) for row in table_rows]
    
    # Find duplicates
    unique_rows = []
    for i, row in enumerate(table_rows):
        clean_row = clean_rows[i]
        
        # Check if this row is similar to any sentence
        matches = process.extract(clean_row, clean_sentences, scorer=fuzz.token_set_ratio, limit=1)
        
        if not matches or matches[0][1] < threshold:
            unique_rows.append(row)
        else:
            print(f"🔄 Removed duplicate table row...")
    
    return paragraph_text, unique_rows


def build_page_nodes(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Build enhanced nodes from PDF pages including both paragraphs and tables.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of nodes ready for indexing
    """
    # Extract existing data
    pages = extract_text_pages(pdf_path)
    tables_map = extract_tables_per_page(pdf_path)
    
    all_nodes = []
    total_pages = len(pages)
    print(f"🏗️  Building nodes from {total_pages} pages...")
    
    for page_data in pages:
        page_no = page_data["page"]
        paragraph_text = page_data["text"]
        
        # Get tables for this page
        page_tables = tables_map.get(page_no, [])
        
        # Convert tables to row texts
        all_table_rows = []
        table_nodes = []
        
        for table_id, table_df in enumerate(page_tables):
            # Convert table to nodes
            table_row_nodes = table_to_nodes(page_no, table_df, table_id)
            table_nodes.extend(table_row_nodes)
            
            # Collect row texts for deduplication
            row_texts = [node["text"] for node in table_row_nodes]
            all_table_rows.extend(row_texts)
        
        # Deduplicate content
        clean_paragraph, clean_table_rows = deduplicate_content(paragraph_text, all_table_rows)
        
        # Create paragraph nodes
        paragraph_nodes = paragraphs_to_nodes(page_no, clean_paragraph)
        all_nodes.extend(paragraph_nodes)
        
        # Filter table nodes to keep only non-duplicated ones
        clean_table_rows_set = set(clean_table_rows)
        filtered_table_nodes = [
            node for node in table_nodes 
            if node["text"] in clean_table_rows_set
        ]
        all_nodes.extend(filtered_table_nodes)
        
        # Show page info
        print(f"📄 Page {page_no}: {len(paragraph_nodes)} paragraphs, {len(filtered_table_nodes)} table rows")
    
    print(f"✅ Built {len(all_nodes)} total nodes")
    return all_nodes






if __name__ == "__main__":
    data = ingest_pdf("data/raw/2024_03_14法人說明會簡報.pdf")
    print(f"Extracted {len(data['pages'])} pages and {len(data['images'])} images")
    # Display image information grouped by page
    print("\n=== Image Distribution by Page ===")
    
    # Create page to images mapping
    page_images = {}
    for img in data['images']:
        page_num = img['page']
        if page_num not in page_images:
            page_images[page_num] = []
        page_images[page_num].append(img)
    
    # Display by page order
    for page_num in sorted(page_images.keys()):
        images = page_images[page_num]
        print(f"\nPage {page_num} - {len(images)} image(s):")
        
        for i, img in enumerate(images, 1):
            print(f"  Image {i}:")
            print(f"    BBox: {img['bbox']}")
            print(f"    Format: {img['ext']}")
            print(f"    Size: {len(img['img_bytes'])} bytes")
            if img['file_path']:
                print(f"    Saved to: {img['file_path']}")
            else:
                print(f"    Saved to: Not saved")
    
    # Show pages without images
    all_pages = set(range(1, len(data['pages']) + 1))
    pages_with_images = set(page_images.keys())
    pages_without_images = all_pages - pages_with_images
    
    if pages_without_images:
        print(f"\nPages without images: {sorted(pages_without_images)}")
    
    # Statistics summary
    print(f"\n=== Summary Statistics ===")
    print(f"Total pages: {len(data['pages'])}")
    print(f"Pages with images: {len(pages_with_images)}")
    print(f"Pages without images: {len(pages_without_images)}")
    print(f"Total images: {len(data['images'])}")
    
    # Image distribution per page
    if page_images:
        print(f"Average images per page: {len(data['images']) / len(pages_with_images):.1f}")
        max_images_page = max(page_images.keys(), key=lambda x: len(page_images[x]))
        print(f"Page with most images: Page {max_images_page} ({len(page_images[max_images_page])} images)")