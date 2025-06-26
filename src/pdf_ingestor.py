# src/ingestion/pdf_ingestor.py

import os
import fitz  # PyMuPDF
from typing import List, Dict, Any

def extract_text_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of {"page": int, "text": str} for each page,
    collapsing intra-page linebreaks into spaces.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            # get all text in reading order
            raw = page.get_text("text")
            # collapse lines
            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            merged = " ".join(lines)
            pages.append({"page": pno + 1, "text": merged})
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
    
    for pno in range(len(doc)):
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

def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Full ingestion: returns dict with
      - pages: List[{"page","text"}]
      - images: List[{"page","bbox","ext","img_bytes"}]
    """
    return {
        "pages": extract_text_pages(pdf_path),
        "images": extract_image_blocks(pdf_path),
    }



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