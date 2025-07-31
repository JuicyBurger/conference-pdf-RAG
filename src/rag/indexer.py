# src/indexer.py

# stdlib
import os, sys, glob, uuid
from typing import List

# third-party
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# local
from ..config        import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from ..models.embedder      import model, embed
from ..data.pdf_ingestor  import build_page_nodes
from ..data.chunker       import chunk_text

# ─── Setup Qdrant client ────────────────────────────────────────────────────────
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,  # 60 second timeout for operations
)


def init_collection(collection_name: str = QDRANT_COLLECTION) -> None:
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


def index_pdf(pdf_path: str, collection_name: str = QDRANT_COLLECTION) -> int:
    """
    1) Extract and build enhanced nodes (paragraphs + tables)
    2) Chunk paragraphs if needed (tables are already row-based)
    3) Embed & upsert with enhanced metadata
    """
    nodes = build_page_nodes(pdf_path)  # Enhanced nodes with tables
    points = []
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Collect all texts for batch embedding
    all_texts = []
    text_metadata = []  # Store metadata for each text
    
    print("🔄 Preparing texts for batch embedding...")
    
    for node in nodes:
        node_text = node["text"]
        
        # For paragraphs, apply chunking if text is too long
        if node["type"] == "paragraph":
            chunks = chunk_text(node_text)
            for chunk_idx, chunk_content in enumerate(chunks):
                all_texts.append(chunk_content)
                text_metadata.append({
                    "type": "paragraph",
                    "page": node["page"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_content,
                })
        
        # For table nodes, use as-is (no chunking needed)
        elif node["type"] in ["table_summary", "table_record", "table_column"]:
            all_texts.append(node_text)
            metadata = {
                "type": node["type"],
                "page": node["page"],
                "table_id": node.get("table_id"),
                "text": node_text,
            }
            
            # Add type-specific metadata
            if node["type"] == "table_record":
                metadata["row_idx"] = node.get("row_idx")
            elif node["type"] == "table_column":
                metadata["column_idx"] = node.get("column_idx")
                metadata["column_name"] = node.get("column_name")
            
            # Include structured data if available
            if "structured_data" in node:
                metadata["structured_data"] = node["structured_data"]
            
            text_metadata.append(metadata)
    
    # Batch embed all texts at once (MUCH FASTER!)
    print(f"🧠 Generating embeddings for {len(all_texts)} texts...")
    
    # Process in smaller batches for better memory management
    batch_size = 100  # Process 100 texts at a time
    vectors = []
    
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_vectors = embed(batch_texts)
        vectors.extend(batch_vectors)
        print(f"   Processed {min(i + batch_size, len(all_texts))}/{len(all_texts)} texts...")
    
    # Create points with batch-generated vectors
    for i, (text, metadata) in enumerate(zip(all_texts, text_metadata)):
        payload = {
            "doc_id": doc_id,
            "page": metadata["page"],
            "type": metadata["type"],
            "text": metadata["text"],
        }
        
        # Add type-specific fields
        if metadata["type"] == "paragraph":
            payload["chunk_idx"] = metadata["chunk_idx"]
        elif metadata["type"] in ["table_summary", "table_record", "table_column"]:
            payload["table_id"] = metadata.get("table_id")
            if metadata["type"] == "table_record":
                payload["row_idx"] = metadata.get("row_idx")
            elif metadata["type"] == "table_column":
                payload["column_idx"] = metadata.get("column_idx")
                payload["column_name"] = metadata.get("column_name")
            # Store structured data if available
            if "structured_data" in metadata:
                payload["structured_data"] = metadata["structured_data"]
        
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vectors[i],
            "payload": payload,
        })

    # Optimize batch size for better performance
    BATCH_SIZE = 200  # Optimal balance between speed and memory usage
    total_indexed = 0
    total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"📤 Uploading {len(points)} chunks in {total_batches} batches...")
    
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        # Retry logic for failed uploads
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure vectors are lists of floats
                batch = [
                    {
                        'id': point['id'],
                        'vector': [float(v) for v in point['vector']] if isinstance(point['vector'], (list, tuple)) else point['vector'].tolist(),
                        'payload': point['payload']
                    }
                    for point in batch
                ]

                client.upsert(collection_name=collection_name, points=batch)
                total_indexed += len(batch)
                print(f"✅ Batch {batch_num} uploaded successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    # If it's a VectorStruct error and this is a large batch, try smaller chunks
                    if "VectorStruct" in str(e) and len(batch) > 10:
                        print(f"🔄 VectorStruct error detected. Splitting batch into smaller chunks...")
                        # Split the batch in half and retry each part
                        mid = len(batch) // 2
                        batch1 = batch[:mid]
                        batch2 = batch[mid:]
                        
                        try:
                            client.upsert(collection_name=collection_name, points=batch1)
                            total_indexed += len(batch1)
                            print(f"✅ Batch {batch_num}a uploaded successfully ({len(batch1)} chunks)")
                            
                            client.upsert(collection_name=collection_name, points=batch2)
                            total_indexed += len(batch2)
                            print(f"✅ Batch {batch_num}b uploaded successfully ({len(batch2)} chunks)")
                            break
                        except Exception as split_error:
                            print(f"⚠️ Split batch also failed: {split_error}")
                            
                    print("🔄 Retrying in 2 seconds...")  # Reduced from 5s to 2s
                    import time
                    time.sleep(2)
                else:
                    print(f"❌ Batch {batch_num} failed after {max_retries} attempts: {e}")
                    raise
    
    print(f"✅ Successfully indexed {total_indexed}/{len(points)} chunks from '{pdf_path}'")
    return total_indexed


def index_pdfs(
    pdf_paths: List[str],
    collection_name: str = QDRANT_COLLECTION,
    recreate: bool = True
) -> None:
    """
    Optionally recreate the collection, then index multiple PDFs.
    """
    if recreate:
        init_collection(collection_name)

    total = 0
    print(f"📚 Indexing {len(pdf_paths)} PDF files...")
    
    # Process PDFs sequentially (parallel processing can cause memory issues)
    for i, path in enumerate(pdf_paths, 1):
        print(f"📚 Processing PDF {i}/{len(pdf_paths)}: {os.path.basename(path)}")
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