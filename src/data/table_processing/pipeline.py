"""
Complete table processing pipeline.

This module orchestrates the entire table processing workflow from PDF input
to structured knowledge extraction and indexing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .detector import TableDetector
from .extractor import TableExtractor  
from .converter import TableConverter
from .kg_extractor import TableKGExtractor
from .summarizer import TableSummarizer
from .change_html_to_json import parse_html_tables
from urllib.parse import urlparse, urlunparse
from ...config import OCR_API_URL, OCR_CLEANUP_URL

logger = logging.getLogger(__name__)


class TableProcessingResult:
    """Result of table processing pipeline."""
    
    def __init__(self):
        self.success: bool = False
        self.tables_found: int = 0
        self.tables_processed: int = 0
        self.tables_indexed: int = 0
        self.nodes_generated: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        logger.error(f"Table processing error: {error}")
    
    def __repr__(self):
        return (f"TableProcessingResult(success={self.success}, "
                f"found={self.tables_found}, processed={self.tables_processed}, "
                f"indexed={self.tables_indexed}, errors={len(self.errors)})")


class TableProcessingPipeline:
    """
    Complete table processing pipeline for PDF documents.
    
    Handles the entire workflow:
    1. Table detection and cropping
    2. Image-to-JSON extraction using LLMs
    3. Structured data conversion
    4. Knowledge graph extraction
    5. Table summarization
    6. Node generation for indexing
    """
    
    def __init__(self, 
                 output_dir: str = "data/temp_tables",
                 min_accuracy: float = 50.0,
                 dpi: int = 300):
        """
        Initialize the table processing pipeline.
        
        Args:
            output_dir: Directory for temporary table images
            min_accuracy: Minimum accuracy threshold for table detection
            dpi: DPI for image extraction
        """
        self.output_dir = Path(output_dir)
        self.min_accuracy = min_accuracy
        self.dpi = dpi
        
        # Initialize components
        self.detector = TableDetector(output_dir=str(self.output_dir), 
                                    min_accuracy=min_accuracy, 
                                    dpi=dpi)
        self.extractor = TableExtractor()
        self.converter = TableConverter()
        self.kg_extractor = TableKGExtractor()
        self.summarizer = TableSummarizer()
        
        logger.info(f"Initialized table processing pipeline (output_dir={output_dir}, "
                   f"min_accuracy={min_accuracy}, dpi={dpi})")
    
    def process_pdf(self, 
                   pdf_path: str,
                   doc_id: Optional[str] = None) -> TableProcessingResult:
        """
        Process all tables in a PDF document with a two-phase flow to reduce GPU OOM:
        1) Detect tables and extract images
        2) Phase A: Extract raw table HTML/JSON for all images (OCR model loads once)
        3) Cleanup OCR memory
        4) Phase B: Convert all extracted tables, summarize, extract KG, and build nodes (LLM loads once)
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier (defaults to filename)
            
        Returns:
            TableProcessingResult with processing details
        """
        return self._process_pdf_impl(pdf_path, doc_id, pre_extracted_tables=None)

    def process_pdf_with_tables(self, 
                               pdf_path: str,
                               table_images: List[Dict[str, Any]],
                               doc_id: Optional[str] = None) -> TableProcessingResult:
        """
        Process tables using pre-extracted table images to avoid duplicate extraction.
        
        Args:
            pdf_path: Path to PDF file
            table_images: Pre-extracted table image metadata
            doc_id: Document identifier (defaults to filename)
            
        Returns:
            TableProcessingResult with processing details
        """
        return self._process_pdf_impl(pdf_path, doc_id, pre_extracted_tables=table_images)

    def _process_pdf_impl(self, 
                         pdf_path: str,
                         doc_id: Optional[str] = None,
                         pre_extracted_tables: Optional[List[Dict[str, Any]]] = None) -> TableProcessingResult:
        """
        Internal implementation for PDF processing with optional pre-extracted tables.
        """
        result = TableProcessingResult()
        
        try:
            # Set doc_id if not provided
            if doc_id is None:
                doc_id = Path(pdf_path).stem
            
            logger.info(f"Starting table processing for PDF: {pdf_path}")
            
            # Step 1: Get table images (either pre-extracted or detect now)
            if pre_extracted_tables:
                logger.info("Step 1: Using pre-extracted table images...")
                table_images = pre_extracted_tables
            else:
                logger.info("Step 1: Detecting and extracting table images...")
                # Use doc_id as the base folder name for temp images to keep naming consistent
                table_images = self.detector.extract_tables_to_images(pdf_path, base_name=doc_id)
            
            result.tables_found = len(table_images)
            if not table_images:
                logger.info("No tables found in PDF")
                result.success = True
                return result
            
            logger.info(f"Found {len(table_images)} tables to process")

            # Phase A: Extract raw table data for all images first
            logger.info("Phase A: Extracting raw table data for all images ...")
            extracted_raw_by_id: Dict[str, Any] = {}
            for table_info in table_images:
                image_path = table_info.get('image_path')
                table_id = str(table_info.get('table_id', 'unknown'))
                if not image_path or not Path(image_path).exists():
                    result.add_error(f"Image not found for table {table_id}: {image_path}")
                    continue
                try:
                    raw_data = self.extractor.extract_tables_via_api(image_path) or {
                        "text": self.extractor.extract_tables_from_image(image_path) or ""
                    }
                    extracted_raw_by_id[table_id] = raw_data
                except Exception as e:
                    result.add_error(f"Raw extraction failed for table {table_id}: {e}")
                    continue

            # Call memory cleanup after ALL tables are extracted
            try:
                # Resolve cleanup endpoint: explicit env or derive from OCR_API_URL
                cleanup_url = OCR_CLEANUP_URL
                if not cleanup_url:
                    try:
                        parsed = urlparse(OCR_API_URL)
                        cleanup_url = urlunparse((parsed.scheme, parsed.netloc, "/memory/clear", "", "", ""))
                    except Exception:
                        cleanup_url = None
                if cleanup_url:
                    logger.info("🧹 Calling memory cleanup after all table extractions...")
                    import httpx
                    with httpx.Client(timeout=10.0) as client:
                        resp = client.post(cleanup_url)
                        if resp.status_code == 200:
                            logger.info("✅ Memory cleanup successful")
                        else:
                            logger.warning(f"⚠️ Memory cleanup returned {resp.status_code}: {resp.text}")
                else:
                    logger.info("🧹 Skipping OCR memory cleanup (no endpoint configured)")
            except Exception as e:
                logger.warning(f"⚠️ Memory cleanup failed: {e}")

            # Phase B: Convert all, summarize in batch, then extract KG and create nodes
            logger.info("Phase B: Converting, summarizing, and indexing tables ...")
            structured_by_id: Dict[str, Dict[str, Any]] = {}

            def _as_rows_candidate_from_html(html_text: str) -> Dict[str, Any]:
                try:
                    tables = parse_html_tables(html_text or "", separate_notes=True)
                    # choose the largest rows table as primary
                    best = None
                    best_len = -1
                    for t in tables:
                        if isinstance(t, dict) and isinstance(t.get("rows"), list):
                            ln = len(t.get("rows") or [])
                            if ln > best_len:
                                best = t
                                best_len = ln
                    if best:
                        rows = best.get("rows") or []
                        notes = best.get("notes") or []
                        return {"rows": rows, "notes": notes}
                except Exception:
                    pass
                return {"text": html_text or ""}

            # Convert raw payloads for all tables first
            for table_info in table_images:
                try:
                    table_id = str(table_info.get('table_id', 'unknown'))
                    raw_payload = extracted_raw_by_id.get(table_id)
                    if not raw_payload:
                        result.add_error(f"No raw data for table {table_id}")
                        continue
                    candidates: List[Dict[str, Any]] = []
                    if isinstance(raw_payload, dict):
                        if 'tables' in raw_payload and isinstance(raw_payload['tables'], list):
                            for t in raw_payload['tables']:
                                if isinstance(t, dict) and ('rows' in t or 'cells' in t):
                                    candidates.append(t)
                                elif isinstance(t, dict) and (isinstance(t.get('html'), str) or isinstance(t.get('text'), str)):
                                    candidates.append(_as_rows_candidate_from_html(t.get('html') or t.get('text') or ""))
                        elif 'rows' in raw_payload or 'cells' in raw_payload:
                            candidates = [raw_payload]
                        elif isinstance(raw_payload.get('text'), str) or isinstance(raw_payload.get('html'), str):
                            text_val = raw_payload.get('text') or raw_payload.get('html') or ""
                            candidates = [_as_rows_candidate_from_html(text_val)]
                        else:
                            candidates = [raw_payload]
                    else:
                        candidates = [_as_rows_candidate_from_html(str(raw_payload))]

                    structured_data = self.converter.convert_to_structured_format(candidates, table_info)
                    if not structured_data:
                        result.add_error(f"Conversion failed for table {table_id}")
                        continue
                    structured_by_id[table_id] = structured_data
                except Exception as e:
                    result.add_error(f"Conversion error for table {table_info.get('table_id', 'unknown')}: {e}")
                    continue

            # Summarize in batch (single summarizer instance)
            summaries_by_id: Dict[str, str] = {}
            if structured_by_id:
                try:
                    summaries_by_id = self.summarizer.summarize_multiple_tables(list(structured_by_id.values()))
                except Exception as e:
                    result.add_error(f"Batch summarization failed: {e}")
                    # Fallback: summarize individually
                    for tid, sdata in structured_by_id.items():
                        try:
                            summaries_by_id[tid] = self.summarizer.summarize_table(sdata)
                        except Exception as e2:
                            result.add_error(f"Summarization failed for table {tid}: {e2}")

            # Create nodes and extract KG per table
            for table_info in table_images:
                try:
                    table_id = str(table_info.get('table_id', 'unknown'))
                    structured_data = structured_by_id.get(table_id)
                    if not structured_data:
                        continue
                    summary_text = summaries_by_id.get(table_id, "")
                    kg_result = self.kg_extractor.extract_table_kg(structured_data, doc_id, table_id)
                    nodes = self._create_table_nodes(
                        structured_data=structured_data,
                        summary_text=summary_text,
                        kg_result=kg_result,
                        doc_id=doc_id,
                        table_id=table_id,
                        page_num=table_info.get('page', 'unknown')
                    )
                    result.nodes_generated.extend(nodes)
                    result.tables_processed += 1
                    result.tables_indexed += 1
                except Exception as e:
                    result.add_error(f"Node creation failed for table {table_info.get('table_id', 'unknown')}: {e}")
                    continue

            # Attempt to cleanup LLM memory on server if supported
            try:
                self.summarizer.cleanup_llm()
            except Exception:
                pass
            
            result.success = result.tables_processed > 0 or result.tables_found == 0
            result.metadata = {
                "doc_id": doc_id,
                "pdf_path": pdf_path,
                "output_dir": str(self.output_dir),
                "table_images": table_images,
                # Expose raw extracted payloads for downstream consumers (e.g., HTML chunking)
                "raw_extracted": extracted_raw_by_id
            }
            
            logger.info(f"Table processing completed: {result}")
            return result
            
        except Exception as e:
            result.add_error(f"Pipeline failed: {e}")
            return result
    
    def _process_single_table(self, 
                             table_info: Dict[str, Any], 
                             doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single table through the complete pipeline.
        
        Args:
            table_info: Table information from detector
            doc_id: Document identifier
            
        Returns:
            List of nodes generated for this table, or None if processing failed
        """
        table_id = table_info.get('table_id', 'unknown')
        page_num = table_info.get('page', 'unknown')
        image_path = table_info.get('image_path')
        
        logger.debug(f"Processing table {table_id} on page {page_num}")
        
        if not image_path or not Path(image_path).exists():
            logger.warning(f"Table image not found: {image_path}")
            return None
        
        # Step 2a: Extract table structure from image using LLM
        logger.debug(f"Extracting table structure from image: {image_path}")
        raw_table_data = self.extractor.extract_tables_from_image(image_path)
        
        if not raw_table_data:
            logger.warning(f"No table data extracted from image: {image_path}")
            return None
        
        # Step 2b: Convert to structured format
        logger.debug(f"Converting table data to structured format")
        structured_data = self.converter.convert_to_structured_format(raw_table_data, table_info)
        
        if not structured_data:
            logger.warning(f"Failed to convert table data to structured format")
            return None
        
        # Step 2c: Generate table summary
        logger.debug(f"Generating table summary")
        summary_text = self.summarizer.summarize_table(structured_data)
        
        # Step 2d: Extract knowledge graph
        logger.debug(f"Extracting knowledge graph from table")
        kg_result = self.kg_extractor.extract_table_kg(structured_data, doc_id, table_id)
        
        # Step 2e: Create nodes for indexing
        nodes = self._create_table_nodes(
            structured_data=structured_data,
            summary_text=summary_text,
            kg_result=kg_result,
            doc_id=doc_id,
            table_id=table_id,
            page_num=page_num
        )
        
        logger.debug(f"Generated {len(nodes)} nodes for table {table_id}")
        return nodes
    
    def _create_table_nodes(self,
                           structured_data: Dict[str, Any],
                           summary_text: str,
                           kg_result: Dict[str, Any],
                           doc_id: str,
                           table_id: str,
                           page_num: int) -> List[Dict[str, Any]]:
        """
        Create indexable nodes from processed table data.
        
        Args:
            structured_data: Structured table data
            summary_text: Generated summary
            kg_result: Knowledge graph extraction result
            doc_id: Document identifier
            table_id: Table identifier
            page_num: Page number
            
        Returns:
            List of nodes ready for indexing
        """
        nodes = []
        
        # Ensure summary text is non-empty; fallback to basic summary if needed
        if not (summary_text or "").strip():
            try:
                headers = structured_data.get('headers') or structured_data.get('columns') or []
                records = structured_data.get('records') or []
                notes = structured_data.get('notes') or []
                basic = f"表格包含 {len(headers)} 個欄位: {', '.join(headers[:6])}{'...' if len(headers) > 6 else ''}。共有 {len(records)} 筆記錄。"
                if notes:
                    basic += f" 備註: {' '.join([str(n) for n in notes if str(n).strip()][:2])}{'...' if len(notes) > 2 else ''}"
                summary_text = basic
            except Exception:
                summary_text = "表格摘要：自動生成失敗，使用備用摘要。"

        # Create table summary node
        summary_node = {
            "page": page_num,
            "type": "table_summary",
            "text": summary_text,
            "table_id": table_id,
            "structured_data": structured_data,
            "kg_observations": kg_result.get('observations', [])
        }
        nodes.append(summary_node)
        
        # Create nodes for table records (if detailed indexing is needed)
        records = structured_data.get('records', [])
        for row_idx, record in enumerate(records[:10]):  # Limit to first 10 records
            record_text = " | ".join([f"{k}: {v}" for k, v in record.items() if v])
            if record_text.strip():
                record_node = {
                    "page": page_num,
                    "type": "table_record", 
                    "text": record_text,
                    "table_id": table_id,
                    "row_idx": row_idx,
                    "record_data": record
                }
                nodes.append(record_node)
        
        # Create nodes for table notes if present
        notes = structured_data.get('notes', [])
        for note_idx, note in enumerate(notes):
            if note and note.strip():
                note_node = {
                    "page": page_num,
                    "type": "table_note",
                    "text": note,
                    "table_id": table_id,
                    "note_idx": note_idx
                }
                nodes.append(note_node)
        
        return nodes
