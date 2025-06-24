# src/indexer.py

import os
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from config import QDRANT_URL, QDRANT_API_KEY
from embedder import model, embed
from pdf_parser import parse_pdf
from chunker import chunk_text

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


def index_pdf(pdf_path: str, collection_name: str = "docs") -> int:
    """
    Parse → chunk → embed → upsert a single PDF.
    Returns the number of chunks indexed.
    """
    # 1. Parse PDF to raw text
    full_text = parse_pdf(pdf_path)

    # 2. Chunk text into (lang, chunk_str) tuples
    chunks = chunk_text(full_text)
    print(f"🔍 Parsed into {len(chunks)} chunks")
    # 3. Build upsert payload
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    points = []
    for idx, (lang, chunk_str) in enumerate(chunks):
        vec = embed(chunk_str)
        payload = {
            "doc_id": doc_id,
            "lang": lang,
            "text": chunk_str,
        }
        # valid UUID for Qdrant
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": vec,
            "payload": payload,
        })

    # 4. Upsert into Qdrant
    client.upsert(collection_name=collection_name, points=points)
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
        total += index_pdf(path, collection_name=collection_name)

    print(f"🎉 Done! Total chunks indexed: {total}")


# ─── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python indexer.py <pdf1> [pdf2 ...]")
        sys.exit(1)

    pdf_files = sys.argv[1:]
    index_pdfs(pdf_files, recreate=True)