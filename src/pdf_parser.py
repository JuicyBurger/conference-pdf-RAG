# src/pdf_parser.py

import logging, os
import pdfplumber

# silence pdfminer warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def parse_pdf(path: str) -> str:
    """
    Extracts text from each page, merges intra-page lines into
    a single paragraph, and then joins pages with double-newlines.
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            # collapse single newlines into spaces, drop empty lines
            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            merged = " ".join(lines)
            pages.append(merged)
    # join pages with blank line to signal page break
    return "\n\n".join(pages)

if __name__ == "__main__":
    # quick sanity check
    text = parse_pdf("data/raw/2024_03_14法人說明會簡報.pdf")
    print(text[:500], "…")
    print("\n---PAGE BREAK---\n".join(f"[{i}]" for i in range(len(text.split('\n\n')))))
