# src/ingestion/pdf_ingestor.py

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

def extract_image_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of image‐blocks with:
      - page: int (1-based)
      - bbox: (x0,y0,x1,y1) if available, else None
      - ext: "png"
      - img_bytes: PNG bytes
    Falls back to page.get_images() if no dict‐blocks found.
    """
    img_blocks: List[Dict[str,Any]] = []
    doc = fitz.open(pdf_path)

    for pno in range(len(doc)):
        page = doc[pno]
        found_on_page = False

        # 1) Try the text-dict blocks for images (type==1)
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 1:
                continue
            img_info = block.get("image")
            if not img_info or "xref" not in img_info:
                continue

            x0, y0, x1, y1 = block.get("bbox", (None,)*4)
            xref = img_info["xref"]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # CMYK or alpha
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")

            img_blocks.append({
                "page": pno + 1,
                "bbox": (x0, y0, x1, y1),
                "ext": "png",
                "img_bytes": img_bytes
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

                # No reliable bbox from get_images(), so set None
                img_blocks.append({
                    "page": pno + 1,
                    "bbox": None,
                    "ext": "png",
                    "img_bytes": img_bytes
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