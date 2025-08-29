"""
Table detection and extraction from PDF documents.

This module handles the detection of tables in PDFs and their conversion to images
for further processing using Camelot and PyMuPDF.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Import required libraries with error handling
try:
    import camelot
    import fitz  # PyMuPDF for image rendering
    CAMELOT_AVAILABLE = True
except ImportError as e:
    CAMELOT_AVAILABLE = False
    camelot = None
    fitz = None

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _convert_bbox_to_fitz_rect(bbox: Tuple[float, float, float, float], page_height: float) -> "fitz.Rect":
    """Convert Camelot bbox (x1, y1, x2, y2) to PyMuPDF rect (x0, y0, x1, y1).
    
    Camelot uses (x1, y1, x2, y2) where y1 is top, y2 is bottom.
    PyMuPDF uses (x0, y0, x1, y1) where y0 is top, y1 is bottom.
    """
    x1, y1, x2, y2 = bbox
    # Convert y coordinates: Camelot y1 (top) -> PyMuPDF y0 (top)
    # Camelot y2 (bottom) -> PyMuPDF y1 (bottom)
    y0 = page_height - y2  # Convert bottom to top
    y1_fitz = page_height - y1  # Convert top to bottom
    return fitz.Rect(x1, y0, x2, y1_fitz)


def _detect_tables_with_camelot(pdf_path: str, pages: str = "all") -> List[Dict[str, Any]]:
    """
    Detect tables using Camelot with lattice flavor.
    
    Args:
        pdf_path: Path to PDF file
        pages: Pages to process (default: "all")
        
    Returns:
        List of table dictionaries with metadata
    """
    if not CAMELOT_AVAILABLE:
        raise ImportError("Camelot is not available. Install with: pip install camelot-py[pdf]")
    
    logger.info(f"Detecting tables with Camelot")
    
    try:
        # Read tables with Camelot using lattice flavor
        tables = camelot.read_pdf(
            pdf_path,
            pages=pages,
            flavor="lattice",
            suppress_stdout=True,
            copy_text=['v']
        )
        
        logger.info(f"Camelot found {len(tables)} tables")
        
        # Convert to our format
        table_data = []
        for i, table in enumerate(tables):
            try:
                # Get bbox - try multiple approaches
                bbox = None
                
                # Method 1: Direct bbox attribute
                if hasattr(table, 'bbox'):
                    bbox = table.bbox
                # Method 2: Private bbox attribute
                elif hasattr(table, '_bbox'):
                    bbox = table._bbox
                # Method 3: From parsing report
                elif hasattr(table, 'parsing_report'):
                    report = table.parsing_report
                    if 'bbox' in report:
                        bbox = report['bbox']
                
                if bbox is None:
                    logger.warning(f"Could not get bbox for table {i}, skipping")
                    continue
                
                # Get other attributes with safe defaults
                accuracy = getattr(table, 'accuracy', 100.0)
                whitespace = getattr(table, 'whitespace', 0.0)
                page = getattr(table, 'page', 1)
                order = getattr(table, 'order', i)
                
                table_info = {
                    'index': i,
                    'table': table,
                    'bbox': bbox,
                    'accuracy': accuracy,
                    'whitespace': whitespace,
                    'page': page,
                    'order': order
                }
                table_data.append(table_info)
                
            except Exception as e:
                logger.error(f"Error processing table {i}: {e}")
                continue
            
        return table_data
        
    except Exception as e:
        logger.error(f"Error detecting tables with Camelot: {e}")
        return []


class TableDetector:
    """
    Handles table detection and image extraction from PDF documents.
    
    This class provides table detection using Camelot and extracts table regions
    as images for further processing.
    """
    
    def __init__(self, 
                 output_dir: str = "data/temp_tables",
                 dpi: int = 300,
                 min_accuracy: float = 50.0):
        """
        Initialize the table detector.
        
        Args:
            output_dir: Directory for temporary table images
            dpi: DPI for image extraction  
            min_accuracy: Minimum accuracy threshold for table detection
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.min_accuracy = min_accuracy
        
        if not CAMELOT_AVAILABLE:
            raise ImportError("This module requires camelot-py and PyMuPDF. Install with: pip install camelot-py[pdf] pymupdf")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized table detector (output_dir={output_dir}, "
                   f"dpi={dpi}, min_accuracy={min_accuracy})")
    
    def extract_tables_to_images(self, pdf_path: str, *, base_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF and convert to images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of table information dictionaries with image paths
        """
        try:
            logger.info(f"Extracting tables from PDF: {pdf_path}")
            
            # Extract tables to images
            table_images = self._extract_tables_to_images_impl(
                pdf_path=pdf_path,
                output_dir=str(self.output_dir),
                dpi=self.dpi,
                min_accuracy=self.min_accuracy,
                base_name=base_name
            )
            
            logger.info(f"Found {len(table_images)} tables in PDF")
            
            # Add any additional metadata processing here if needed
            for table_info in table_images:
                self._enhance_table_info(table_info)
            
            return table_images
            
        except Exception as e:
            logger.error(f"Failed to extract tables from PDF {pdf_path}: {e}")
            return []
    
    def _extract_tables_to_images_impl(
        self,
        pdf_path: str,
        output_dir: str = "data/tables",
        dpi: int = 300,
        min_accuracy: float = 50.0,
        base_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Core implementation for extracting tables from PDF and saving as images.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory (default: data/tables)
            dpi: DPI for rendered images (default: 300)
            min_accuracy: Minimum accuracy threshold (default: 50.0)
            
        Returns:
            List of table metadata dictionaries
        """
        base = base_name or os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Open PDF with PyMuPDF for rendering
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Cannot open PDF: {pdf_path} ({e})")
            return []
        
        # Detect tables
        tables = _detect_tables_with_camelot(pdf_path)
        
        # Filter tables by quality
        filtered_tables = []
        for table_info in tables:
            if table_info['accuracy'] >= min_accuracy:
                filtered_tables.append(table_info)
            else:
                logger.info(f"Skipping table {table_info['index']} on page {table_info['page']}: "
                           f"accuracy={table_info['accuracy']:.1f}%")
        
        if not filtered_tables:
            logger.warning("No tables found")
            doc.close()
            return []
        
        # Sort tables by page and order
        filtered_tables.sort(key=lambda x: (x['page'], x['order']))
        
        # Create output directory
        pdf_dir = os.path.join(output_dir, base)
        _ensure_dir(pdf_dir)
        
        # Process each table
        results = []
        table_counter = 1
        
        for table_info in filtered_tables:
            page_num = table_info['page']
            bbox = table_info['bbox']
            
            try:
                # Get the page for rendering
                page = doc[page_num - 1]  # Camelot uses 1-based page numbers
                
                # Convert Camelot bbox to PyMuPDF rect
                page_height = page.rect.height
                rect = _convert_bbox_to_fitz_rect(bbox, page_height)
                
                # Round the rect for cleaner output
                rounded_rect = fitz.Rect(
                    round(rect.x0, 2),
                    round(rect.y0, 2),
                    round(rect.x1, 2),
                    round(rect.y1, 2)
                )
                
                # Create pixmap with clip
                pix = page.get_pixmap(clip=rounded_rect, dpi=dpi, alpha=False)
                
                # Save image with the required naming format
                img_path = os.path.join(pdf_dir, f"table_{table_counter:02d}_P{page_num}.png")
                pix.save(img_path)
                
                logger.info(f"Exported table {table_counter} from page {page_num}: {img_path}")
                
                # Create metadata
                result = {
                    "file_name": os.path.basename(pdf_path),
                    "page": page_num,
                    "table_id": table_counter,
                    "accuracy": round(table_info['accuracy'], 2),
                    "whitespace": round(table_info['whitespace'], 2),
                    "x0": rounded_rect.x0,
                    "y0": rounded_rect.y0,
                    "x1": rounded_rect.x1,
                    "y1": rounded_rect.y1,
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "dpi": dpi,
                    "image_path": img_path,
                    "camelot_index": table_info['index']
                }
                results.append(result)
                table_counter += 1
                
            except Exception as e:
                logger.error(f"Failed to export table {table_counter} from page {page_num}: {e}")
                continue
        
        doc.close()
        logger.info(f"Successfully exported {len(results)} tables")
        return results
    
    def _enhance_table_info(self, table_info: Dict[str, Any]) -> None:
        """
        Enhance table information with additional metadata.
        
        Args:
            table_info: Table information dictionary to enhance
        """
        # Add image size information if image exists
        image_path = table_info.get('image_path')
        if image_path and Path(image_path).exists():
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    table_info['image_width'] = img.width
                    table_info['image_height'] = img.height
                    table_info['image_mode'] = img.mode
            except Exception as e:
                logger.debug(f"Could not read image metadata for {image_path}: {e}")
        
        # Ensure table_id is present
        if 'table_id' not in table_info:
            page = table_info.get('page', 'unknown')
            bbox = table_info.get('bbox', 'unknown')
            table_info['table_id'] = f"table_p{page}_{bbox}"


# Convenience function for backward compatibility
def extract_tables_to_images(
    pdf_path: str,
    output_dir: str = "data/tables",
    dpi: int = 300,
    min_accuracy: float = 50.0,
    base_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract tables from PDF and save as images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory (default: data/tables)
        dpi: DPI for rendered images (default: 300)
        min_accuracy: Minimum accuracy threshold (default: 50.0)
        
    Returns:
        List of table metadata dictionaries
    """
    detector = TableDetector(output_dir=output_dir, dpi=dpi, min_accuracy=min_accuracy)
    return detector.extract_tables_to_images(pdf_path, base_name=base_name)