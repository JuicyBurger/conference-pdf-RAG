# src/indexer.py

# stdlib
import os, sys, glob, uuid
from typing import List

# third-party
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# local
from .config        import QDRANT_URL, QDRANT_API_KEY
from .embedder      import model, embed
from .pdf_ingestor  import extract_text_pages
from .chunker       import chunk_text

# ─── Setup Qdrant client ────────────────────────────────────────────────────────
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def init_collection(collection_name: str = "docs") -> None:
    """
    (Re)create the Qdrant collection with the proper vector size & distance metric.
    """
    vector_size = model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        ),
    )
    print(f"✅ Collection '{collection_name}' initialized: vectors={vector_size}, distance=Cosine")
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        # keyword is the right type for exact matches on strings
        field_schema="keyword",
    )
    print(f"🔑 Payload index created on field 'doc_id'")


def index_pdf(pdf_path: str, collection_name: str = "docs") -> int:
    """
    1) Extract pages (with page numbers)
    2) Chunk each page
    3) Embed & upsert, including 'page' in payload
    """
    pages = extract_text_pages(pdf_path)  # [{"page":1,"text":...}, ...]
    points = []
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    for page in pages:
        page_no = page["page"]
        page_text = page["text"]
        
        chunks = chunk_text(page_text)
        
        for chunk_idx, (lang, text) in enumerate(chunks):
            vec = embed(text)
            payload = {
                "doc_id": doc_id,
                "page":   page_no,
                "lang":   lang,
                "text":   text,
            }
            points.append({
                "id":      str(uuid.uuid4()),
                "vector":  vec,
                "payload": payload,
            })

    BATCH_SIZE = 500
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"🔹 Indexed {len(points)} chunks from '{pdf_path}'")
    
    return len(points)


def index_pdfs(
    pdf_paths: List[str],
    collection_name: str = "docs",
    recreate: bool = True
) -> None:
    """
    Optionally recreate the collection, then index multiple PDFs.
    """
    if recreate:
        init_collection(collection_name)

    total = 0
    for path in pdf_paths:
        try:
            total += index_pdf(path, collection_name=collection_name)
        except Exception as e:
            print(f"❌ Failed to index {path}: {e}")
        
    print(f"🎉 Done! Total chunks indexed: {total}")


if __name__ == "__main__":
    args = sys.argv[1:]
    recreate = "--no-recreate" not in args
    # Remove the flag from the list
    args = [a for a in args if a != "--no-recreate"]

    # Expand any globs into real file paths
    pdfs = []
    for a in args:
        if any(ch in a for ch in "*?[]"):
            pdfs.extend(glob.glob(a))
        else:
            pdfs.append(a)

    if not pdfs:
        print("Usage: python indexer.py [--no-recreate] <pdf1> [pdf2 ...]")
        sys.exit(1)

    index_pdfs(pdfs, recreate=recreate)