# src/ingestion/pdf_ingestor.py

import os
import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from rapidfuzz import fuzz, process
from .node_builder import paragraphs_to_nodes, table_to_nodes, clean_text_for_comparison

# Import table processing components
from .table_processing.detector import extract_tables_to_images
from .table_processing.extractor import extract_tables_from_image
from .table_processing.summarizer import summarize_table
from .table_processing.change_html_to_json import parse_html_tables
from .table_processing.pipeline import TableProcessingPipeline
from .table_chunker import extract_table_chunks
from ..config import USE_TABLE_HTML_CHUNKS, RETAIN_TABLE_HTML_FILES

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


def _span_intersects_any(span_bbox: Tuple[float, float, float, float], rects: List[Tuple[float, float, float, float]]) -> bool:
    """Return True if span_bbox intersects any rect in rects."""
    if not rects:
        return False
    x0, y0, x1, y1 = span_bbox
    for rx0, ry0, rx1, ry1 in rects:
        if not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1):
            return True
    return False


def _reconstruct_paragraph_from_dict(
    page: fitz.Page,
    headers: set,
    footers: set,
    skip_rects_mupdf: Optional[List[Tuple[float, float, float, float]]] = None,
) -> str:
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
            # Skip if any span falls inside a table region
            if skip_rects_mupdf:
                spans = line.get("spans", [])
                intersects = False
                for sp in spans:
                    bbox = sp.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        if _span_intersects_any(tuple(bbox), skip_rects_mupdf):
                            intersects = True
                            break
                if intersects:
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


def extract_text_pages(
    pdf_path: str,
    skip_regions_pdf_by_page: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
) -> List[Dict[str, Any]]:
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
            # Convert PDF-space skip rects (origin bottom-left) to MuPDF space (origin top-left)
            skip_rects_mupdf: List[Tuple[float, float, float, float]] = []
            if skip_regions_pdf_by_page and (pno + 1) in skip_regions_pdf_by_page:
                height = page.rect.height
                for (x0, y0, x1, y1) in skip_regions_pdf_by_page[pno + 1]:
                    my0 = height - y1
                    my1 = height - y0
                    skip_rects_mupdf.append((x0, my0, x1, my1))
            text = _reconstruct_paragraph_from_dict(page, headers, footers, skip_rects_mupdf)
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
            page_no = int(table.page)
            if page_no not in tables_by_page:
                tables_by_page[page_no] = []
            tables_by_page[page_no].append(table.df)
        
        # Fallback to stream if lattice found nothing
        if not any(tables_by_page.values()):
            print("📊 Lattice found no tables, falling back to Camelot (stream)...")
            tables_stream = camelot.read_pdf(pdf_path, flavor='stream', pages=page_spec)
            for table in tables_stream:
                page_no = int(table.page)
                if page_no not in tables_by_page:
                    tables_by_page[page_no] = []
                tables_by_page[page_no].append(table.df)
            
    except Exception as e:
        print(f"⚠️ Table extraction failed for {pdf_path}: {e}")
        return {}
    
    return tables_by_page


def extract_tables_with_meta(pdf_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """Extract tables along with flavor, bbox (PDF coords), and Camelot accuracy.

    Returns: { page_no: [ { 'df': DataFrame, 'flavor': str, 'accuracy': float|None, 'bbox_pdf': (x0,y0,x1,y1)|None } ] }
    """
    if not CAMELOT_AVAILABLE:
        return {}

    meta_by_page: Dict[int, List[Dict[str, Any]]] = {}
    try:
        # Select candidate pages
        with fitz.open(pdf_path) as doc:
            text_pages = [pno + 1 for pno in range(len(doc)) if doc.load_page(pno).get_text("text").strip()]
        page_spec = 'all' if not text_pages else ",".join(str(i) for i in text_pages)

        # Try lattice first
        tables_lattice = camelot.read_pdf(pdf_path, flavor='lattice', pages=page_spec)
        lattice_pages: Dict[int, List[Any]] = {}
        for t in tables_lattice:
            page_no = int(t.page)
            lattice_pages.setdefault(page_no, []).append(t)

        # Optionally also get stream if lattice is empty
        stream_pages: Dict[int, List[Any]] = {}
        if not lattice_pages:
            print("📊 Using Camelot (stream) due to no lattice tables...")
            tables_stream = camelot.read_pdf(pdf_path, flavor='stream', pages=page_spec)
            for t in tables_stream:
                page_no = int(t.page)
                stream_pages.setdefault(page_no, []).append(t)

        pages_set = set(list(lattice_pages.keys()) + list(stream_pages.keys()))
        for page_no in sorted(pages_set):
            entries: List[Dict[str, Any]] = []
            for t in lattice_pages.get(page_no, []):
                bbox = getattr(t, 'bbox', getattr(t, '_bbox', None))
                acc = None
                try:
                    acc = (t.parsing_report or {}).get('accuracy')
                except Exception:
                    pass
                entries.append({
                    'df': t.df,
                    'flavor': 'lattice',
                    'accuracy': float(acc) if acc is not None else None,
                    'bbox_pdf': tuple(bbox) if bbox else None,
                })
            for t in stream_pages.get(page_no, []):
                bbox = getattr(t, 'bbox', getattr(t, '_bbox', None))
                acc = None
                try:
                    acc = (t.parsing_report or {}).get('accuracy')
                except Exception:
                    pass
                entries.append({
                    'df': t.df,
                    'flavor': 'stream',
                    'accuracy': float(acc) if acc is not None else None,
                    'bbox_pdf': tuple(bbox) if bbox else None,
                })
            if entries:
                meta_by_page[page_no] = entries
    except Exception as e:
        print(f"⚠️ Table meta extraction failed: {e}")
        return {}

    return meta_by_page


def deduplicate_content(paragraph_text: str, table_rows: List[str], threshold: float = 85.0) -> tuple[str, List[str]]:
    """
    Remove duplicate content by:
      - Stripping sentences from paragraph_text that are highly similar to any table row text
      - Keeping all table_rows (we prefer to retain structured table data)

    Returns: (clean_paragraph_text, table_rows)
    """
    if not paragraph_text:
        return paragraph_text, table_rows

    import re
    # Split paragraphs into sentences and normalize
    sentences = re.split(r'[。！？；]', paragraph_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return paragraph_text, table_rows

    clean_sentences = [clean_text_for_comparison(s) for s in sentences]
    clean_rows = [clean_text_for_comparison(r) for r in (table_rows or [])]

    keep_mask: List[bool] = [True] * len(sentences)
    if clean_rows:
        for i, cs in enumerate(clean_sentences):
            # Compare this sentence to all row texts
            match = process.extract(cs, clean_rows, scorer=fuzz.token_set_ratio, limit=1)
            if match and match[0][1] >= threshold:
                keep_mask[i] = False

    kept_sentences = [s for s, k in zip(sentences, keep_mask) if k]
    clean_paragraph_text = "。".join(kept_sentences)
    if paragraph_text.endswith("。") and clean_paragraph_text and not clean_paragraph_text.endswith("。"):
        clean_paragraph_text += "。"

    return clean_paragraph_text, (table_rows or [])


def build_page_nodes(pdf_path: str, save_prepared: bool = True, doc_id_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build enhanced nodes from PDF pages including both paragraphs and tables.
    Now includes advanced table processing: table-to-image conversion and JSON extraction.
    Optionally dumps prepared artifacts to PREPARED_DIR/{doc_id} for later batched processing.
    
    Args:
        pdf_path: Path to PDF file
        save_prepared: If True, write prepared artifacts to disk for batching
        
    Returns:
        List of nodes ready for indexing
    """
    all_nodes = []
    total_pages = 0

    # Cache check: if prepared nodes already exist for this document, load and return
    try:
        from ..config import PREPARED_DIR
        preferred_doc_id = doc_id_override or os.path.splitext(os.path.basename(pdf_path))[0]
        base_dir = os.path.join(PREPARED_DIR, preferred_doc_id)
        nodes_path = os.path.join(base_dir, "nodes_all.json")
        if os.path.exists(nodes_path):
            import json
            print(f"🔍 CACHE HIT: Found existing prepared data for '{preferred_doc_id}'")
            print(f"   📁 Cache location: {nodes_path}")
            with open(nodes_path, "r", encoding="utf-8") as f:
                cached_nodes = json.load(f)
            print(f"   📊 Loaded {len(cached_nodes)} nodes from cache")
            print(f"   ⚡ Skipping PDF extraction and table processing")
            return cached_nodes
        else:
            print(f"🔍 CACHE MISS: No prepared data found for '{preferred_doc_id}'")
            print(f"   📁 Expected location: {nodes_path}")
            print(f"   🚀 Proceeding with full PDF extraction and processing")
    except Exception:
        print(f"⚠️  Cache check failed, proceeding with extraction")
    
    # Step 1: Extract tables to images and get metadata
    print("🔄 Step 1: Extracting tables to images...")
    preferred_doc_id = doc_id_override or os.path.splitext(os.path.basename(pdf_path))[0]
    table_images = extract_tables_to_images(pdf_path, output_dir="data/temp_tables", base_name=preferred_doc_id)
    
    if table_images:
        print(f"📊 Found {len(table_images)} tables to process")
        
        # Group tables by page for processing
        tables_by_page = {}
        for table_info in table_images:
            page_num = table_info['page']
            if page_num not in tables_by_page:
                tables_by_page[page_num] = []
            tables_by_page[page_num].append(table_info)
        
        total_pages = max(tables_by_page.keys()) if tables_by_page else 0
    else:
        print("📊 No tables found, proceeding with text-only extraction")
        # Fallback to original method if no tables found
        return _build_page_nodes_fallback(pdf_path)
    
    # Step 2: Extract text pages (excluding table regions)
    print("🔄 Step 2: Extracting text content...")
    # Create skip regions from table bounding boxes
    skip_regions_pdf_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for table_info in table_images:
        page_num = table_info['page']
        # The coordinates from table_to_image_camelot are already in PyMuPDF space
        # We need to convert them to PDF space for extract_text_pages
        x0, y0, x1, y1 = table_info['x0'], table_info['y0'], table_info['x1'], table_info['y1']
        
        # Convert PyMuPDF coordinates (top-left origin) to PDF coordinates (bottom-left origin)
        # We need to get the page height to do this conversion
        try:
            with fitz.open(pdf_path) as doc:
                if page_num <= len(doc):
                    page = doc[page_num - 1]  # 0-based indexing
                    page_height = page.rect.height
                    # Convert y coordinates: PyMuPDF y0 (top) -> PDF y1 (top)
                    # PyMuPDF y1 (bottom) -> PDF y0 (bottom)
                    pdf_y0 = page_height - y1  # Convert bottom to bottom
                    pdf_y1 = page_height - y0  # Convert top to top
                    skip_regions_pdf_by_page.setdefault(page_num, []).append((x0, pdf_y0, x1, pdf_y1))
        except Exception as e:
            print(f"⚠️  Could not convert coordinates for page {page_num}: {e}")
            # Fallback: use coordinates as-is
            skip_regions_pdf_by_page.setdefault(page_num, []).append((x0, y0, x1, y1))
    
    pages = extract_text_pages(pdf_path, skip_regions_pdf_by_page=skip_regions_pdf_by_page)
    total_pages = max(total_pages, len(pages))

    # Run new two-phase table pipeline once for the whole PDF
    try:
        pipeline = TableProcessingPipeline(output_dir="data/temp_tables")
        doc_id = preferred_doc_id
        # Use pre-extracted table images to avoid duplicate extraction
        tp_result = pipeline.process_pdf_with_tables(pdf_path, table_images, doc_id=doc_id)
        table_nodes_new = tp_result.nodes_generated or []
        all_nodes.extend(table_nodes_new)
        print(f"   📊 Added {len(table_nodes_new)} table nodes via pipeline")

        # Optional: generate table HTML chunk nodes using pipeline raw output
        if USE_TABLE_HTML_CHUNKS:
            try:
                raw = (tp_result.metadata or {}).get("raw_extracted", {}) or {}
                timgs = (tp_result.metadata or {}).get("table_images", []) or []
                id_to_page = {str(info.get("table_id")): info.get("page") for info in timgs if isinstance(info, dict)}
                for tid, payload in raw.items():
                    html_text = ""
                    if isinstance(payload, dict):
                        html_text = payload.get("text") or payload.get("html") or ""
                    else:
                        html_text = str(payload)
                    if not html_text or not str(html_text).strip():
                        continue
                    page_num_for_tid = id_to_page.get(str(tid), 'unknown')
                    source_id = f"table_{tid}_P{page_num_for_tid}.html"
                    chunks = extract_table_chunks(
                        html_text,
                        source_id=source_id,
                        max_rows_per_chunk=15,
                        max_chars=1500,
                        row_level=False,
                        include_notes=False,
                        char_chunk=True,
                        max_chars_per_chunk=1200,
                        overlap_chars=150,
                        char_only_if_long=True,
                    )
                    for c in chunks:
                        all_nodes.append({
                            "type": "table_chunk",
                            "page": page_num_for_tid,
                            "text": c.text,
                            "metadata": {
                                **c.meta,
                                "table_id": tid,
                                "source_id": c.source_id,
                                "scope": "table",
                            },
                        })
                print(f"   🧩 Added table_chunk nodes via pipeline")
            except Exception as e:
                print(f"   ⚠️  Table HTML chunking (pipeline) failed: {e}")
    except Exception as e:
        print(f"   ⚠️  Pipeline table processing failed: {e}")

    print(f"🏗️  Building nodes from {total_pages} pages...")
    
    # Step 3: Process each page
    for page_num in range(1, total_pages + 1):
        print(f"📄 Processing page {page_num}...")
        
        # Get text content for this page
        page_text = ""
        for page_data in pages:
            if page_data["page"] == page_num:
                page_text = page_data["text"]
                break
        
        # Get tables for this page
        page_tables = tables_by_page.get(page_num, [])
        
        # Process paragraph content
        if page_text.strip():
            paragraph_nodes = paragraphs_to_nodes(page_num, page_text)
            all_nodes.extend(paragraph_nodes)
            print(f"   📝 Added {len(paragraph_nodes)} paragraph nodes")
        
        # Table nodes are now added via TableProcessingPipeline; skip per-table processing here
        
        # Clean up temporary table images
        for table_info in page_tables:
            try:
                if os.path.exists(table_info['image_path']):
                    os.remove(table_info['image_path'])
            except Exception as e:
                print(f"   ⚠️  Could not clean up {table_info['image_path']}: {e}")
    
    # Optionally clean up temporary directory (retain by default)
    if not RETAIN_TABLE_HTML_FILES:
        try:
            temp_dir = "data/temp_tables"
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️  Could not clean up temp directory: {e}")
    
    # Save prepared artifacts
    try:
        if save_prepared:
            from ..config import PREPARED_DIR
            import json
            doc_id = preferred_doc_id
            base_dir = os.path.join(PREPARED_DIR, doc_id)
            os.makedirs(base_dir, exist_ok=True)
            # Save pages
            with open(os.path.join(base_dir, "pages.json"), "w", encoding="utf-8") as f:
                json.dump(pages, f, ensure_ascii=False)
            # Save raw extracted tables (if available from pipeline)
            try:
                raw = (tp_result.metadata or {}).get("raw_extracted", {}) if 'tp_result' in locals() else {}
                with open(os.path.join(base_dir, "tables_raw.json"), "w", encoding="utf-8") as f:
                    json.dump(raw, f, ensure_ascii=False)
            except Exception:
                pass
            # Save all nodes
            with open(os.path.join(base_dir, "nodes_all.json"), "w", encoding="utf-8") as f:
                json.dump(all_nodes, f, ensure_ascii=False)
            print(f"💾 Prepared artifacts saved to {base_dir}")
    except Exception as e:
        print(f"⚠️  Failed to save prepared artifacts: {e}")
    
    print(f"✅ Built {len(all_nodes)} total nodes")
    return all_nodes


def _build_page_nodes_fallback(pdf_path: str) -> List[Dict[str, Any]]:
    """Fallback to original table extraction method when no tables are found."""
    print("🔄 Using fallback table extraction method...")
    
    # Extract tables with metadata first and mask their regions from paragraph text
    tables_meta = extract_tables_with_meta(pdf_path)
    skip_regions_pdf_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for pno, entries in (tables_meta or {}).items():
        for ent in entries:
            bbox = ent.get('bbox_pdf')
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                skip_regions_pdf_by_page.setdefault(pno, []).append(tuple(bbox))

    pages = extract_text_pages(pdf_path, skip_regions_pdf_by_page=skip_regions_pdf_by_page)
    # Build simple tables map for node conversion
    tables_map: Dict[int, List[pd.DataFrame]] = {}
    for pno, entries in (tables_meta or {}).items():
        tables_map[pno] = [ent['df'] for ent in entries]
    
    all_nodes = []
    total_pages = len(pages)
    print(f"🏗️  Building nodes from {total_pages} pages (fallback method)...")
    
    for page_data in pages:
        page_no = page_data["page"]
        paragraph_text = page_data["text"]
        
        # Get tables for this page
        page_tables = tables_map.get(page_no, [])
        
        # Convert tables to row texts
        all_table_rows = []
        all_table_cells = []
        table_nodes = []
        
        for table_id, table_df in enumerate(page_tables):
            # Convert table to nodes
            table_row_nodes = table_to_nodes(page_no, table_df, table_id)
            table_nodes.extend(table_row_nodes)
            
            # Collect row texts for deduplication
            row_texts = [node["text"] for node in table_row_nodes]
            all_table_rows.extend(row_texts)
            # Collect raw cell texts for stricter dedup
            try:
                for cell in table_df.values.flatten():
                    s = str(cell).strip()
                    if s:
                        all_table_cells.append(s)
            except Exception:
                pass
        
        # Deduplicate content using both generated row texts and raw cell texts
        dedup_basis = list(dict.fromkeys(all_table_rows + all_table_cells))
        clean_paragraph, clean_table_rows = deduplicate_content(paragraph_text, dedup_basis)
        
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
    
    print(f"✅ Built {len(all_nodes)} total nodes (fallback)")
    return all_nodes


def _create_table_nodes_from_json(page_num: int, table_id: int, json_tables: List[Dict]) -> List[Dict[str, Any]]:
    """
    Create table nodes from extracted JSON content.
    
    Args:
        page_num: Page number
        table_id: Table identifier
        json_tables: List of table data from JSON extraction
        
    Returns:
        List of table nodes ready for indexing
    """
    table_nodes = []
    
    for table_idx, table_data in enumerate(json_tables):
        headers = table_data.get('headers', [])
        records = table_data.get('records', [])
        notes = table_data.get('notes', [])
        
        # Create complete structured_data for KG extraction
        complete_structured_data = {
            "headers": headers,
            "records": records,
            "notes": notes,
            "page": page_num,
            "anchor": f"table_{table_id}_{table_idx}"
        }
        
        # Create table summary node with intelligent summarization
        if headers and records:
            # Prepare table data for summarization
            table_data_for_summary = {
                "columns": headers,
                "records": records,
                "notes": notes
            }
            
            try:
                # Generate intelligent summary using LLM
                print(f"   🤖 Generating intelligent summary for table {table_id}_{table_idx}...")
                summary_text = summarize_table(table_data_for_summary)
                print(f"   ✅ Summary generated: {len(summary_text)} characters")
            except Exception as e:
                print(f"   ⚠️  Summary generation failed, using fallback: {e}")
                # Fallback to basic summary
                summary_text = f"Table with {len(headers)} columns and {len(records)} rows. Headers: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}"
            
            table_nodes.append({
                "type": "table_summary",
                "page": page_num,
                "table_id": f"{table_id}_{table_idx}",
                "text": summary_text,
                "structured_data": complete_structured_data  # Use complete data for KG extraction
            })
        
        # Create table record nodes
        for record_idx, record in enumerate(records):
            # Handle both dict and list records
            if isinstance(record, dict):
                if record.get('__note__'):
                    # Skip note records for now (they're handled separately)
                    continue
                    
                # Create a structured text representation
                record_parts = []
                for header, value in record.items():
                    if header != '__note__':
                        record_parts.append(f"{header}: {value}")
                
                record_text = " | ".join(record_parts)
            elif isinstance(record, list):
                # Convert list to structured text
                record_parts = []
                for j, value in enumerate(record):
                    if j < len(headers):
                        header = headers[j]
                        record_parts.append(f"{header}: {value}")
                    else:
                        record_parts.append(f"欄位{j+1}: {value}")
                
                record_text = " | ".join(record_parts)
            else:
                # Skip unknown record types
                continue
            
            table_nodes.append({
                "type": "table_record",
                "page": page_num,
                "table_id": f"{table_id}_{table_idx}",
                "row_idx": record_idx,
                "text": record_text,
                "structured_data": complete_structured_data  # Use complete data for KG extraction
            })
        
        # Create table column nodes
        for col_idx, header in enumerate(headers):
            if not header:
                continue
                
            # Collect all values for this column
            column_values = []
            for record in records:
                # Handle both dict and list records
                if isinstance(record, dict):
                    if not record.get('__note__') and header in record:
                        value = record[header]
                        if value and str(value).strip():
                            column_values.append(str(value))
                elif isinstance(record, list):
                    # Find the column index for this header
                    try:
                        col_idx = headers.index(header)
                        if col_idx < len(record):
                            value = record[col_idx]
                            if value and str(value).strip():
                                column_values.append(str(value))
                    except ValueError:
                        # Header not found in headers list
                        continue
            
            if column_values:
                column_text = f"Column '{header}': {', '.join(column_values[:10])}{'...' if len(column_values) > 10 else ''}"
                
                table_nodes.append({
                    "type": "table_column",
                    "page": page_num,
                    "table_id": f"{table_id}_{table_idx}",
                    "column_idx": col_idx,
                    "column_name": header,
                    "text": column_text,
                    "structured_data": complete_structured_data  # Use complete data for KG extraction
                })
        
        # Create note nodes if any
        for note_idx, note in enumerate(notes):
            note_text = note.get('text', '')
            if note_text:
                table_nodes.append({
                    "type": "table_note",
                    "page": page_num,
                    "table_id": f"{table_id}_{table_idx}",
                    "note_idx": note_idx,
                    "text": note_text,
                    "structured_data": complete_structured_data  # Use complete data for KG extraction
                })
    
    return table_nodes
