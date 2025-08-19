# src/rag/indexing/indexer.py

# stdlib
import os, sys, glob, uuid, hashlib
from typing import List

def md5_to_uuid(md5_hash: str) -> str:
    """Convert an MD5 hash to UUID format by inserting hyphens."""
    if not md5_hash or len(md5_hash) != 32:
        raise ValueError(f"Invalid MD5 hash: {md5_hash}")
    
    # Insert hyphens at positions 8, 12, 16, 20
    uuid_parts = [
        md5_hash[0:8],
        md5_hash[8:12], 
        md5_hash[12:16],
        md5_hash[16:20],
        md5_hash[20:32]
    ]
    
    return "-".join(uuid_parts)

# third-party
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# local
from ...config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from ...models.embedder      import model, embed
from ...data.pdf_ingestor  import build_page_nodes
from ...data.chunker       import chunk_text

# ─── Setup Qdrant client ────────────────────────────────────────────────────────
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,  # 60 second timeout for operations
)


def ensure_indexes_exist(collection_name: str = QDRANT_COLLECTION) -> None:
    """
    Ensure all necessary indexes exist for the collection.
    This is safe to call multiple times - existing indexes are ignored.
    """
    print(f"🔍 Ensuring indexes exist for collection '{collection_name}'...")
    
    # List of required indexes
    required_indexes = [
        ("doc_id", "keyword", "Document ID filtering"),
        ("page", "integer", "Page number filtering"),
        ("type", "keyword", "Content type filtering"),
        ("text", "text", "Full-text search"),
        ("room_id", "keyword", "Room scoping for chat uploads"),
        ("scope", "keyword", "Data scope flag: chat/global/graph"),
        ("corpus_id", "keyword", "Corpus scoping for training graph"),
    ]
    
    for field_name, field_schema, description in required_indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ Created {field_schema} index on '{field_name}' - {description}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ Index on '{field_name}' already exists")
            else:
                print(f"⚠️ Could not create index on '{field_name}': {e}")
    
    print("🎉 All indexes ensured!")


def add_missing_indexes(collection_name: str = QDRANT_COLLECTION):
    """Add missing indexes to an existing collection.
    
    This is a utility function for adding indexes to existing collections
    that may have been created without proper indexing.
    """
    print(f"🔍 Checking collection '{collection_name}'...")
    
    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection exists with {collection_info.points_count} points")
    except Exception as e:
        print(f"❌ Collection '{collection_name}' not found: {e}")
        return
    
    # List of indexes to create
    indexes_to_create = [
        ("doc_id", "keyword", "Document ID filtering"),
        ("page", "integer", "Page number filtering"), 
        ("type", "keyword", "Content type filtering"),
        ("text", "text", "Full-text search"),
    ]
    
    print("\n🔧 Adding missing indexes...")
    
    for field_name, field_schema, description in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ Created {field_schema} index on '{field_name}' - {description}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ Index on '{field_name}' already exists")
            else:
                print(f"⚠️ Failed to create index on '{field_name}': {e}")
    
    print("\n🎉 Index creation completed!")
    print("\n📋 Available filtering capabilities:")
    print("  • doc_id: Filter by specific documents")
    print("  • page: Filter by page numbers (e.g., page 43)")
    print("  • type: Filter by content type (paragraph, table_record)")
    print("  • text: Full-text search within content")


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
    
    # Ensure all indexes exist after creating the collection
    ensure_indexes_exist(collection_name)


def index_nodes(nodes: List[dict], collection_name: str = QDRANT_COLLECTION, doc_id: str | None = None, extra_payload: dict | None = None) -> int:
    """Index pre-extracted nodes (avoid re-reading the PDF)."""
    # Ensure indexes exist before indexing
    ensure_indexes_exist(collection_name)

    points = []
    if not doc_id:
        raise ValueError("doc_id is required when indexing pre-extracted nodes")
    
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
    
    # Batch embed all texts at once (balanced for speed/memory)
    print(f"🧠 Generating embeddings for {len(all_texts)} texts...")
    vectors = []
    batch_size = 100
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
        if extra_payload:
            try:
                # Merge but do not overwrite core fields unintentionally
                for k, v in extra_payload.items():
                    if k not in payload:
                        payload[k] = v
            except Exception:
                pass
        
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
        
        # Generate consistent ID using same method as Neo4j
        seed = f"{doc_id}|{metadata['page']}|{metadata['type']}|{metadata.get('chunk_idx', 0)}|{metadata['text'][:64]}".encode("utf-8", errors="ignore")
        md5_hash = hashlib.md5(seed).hexdigest()
        chunk_id = md5_to_uuid(md5_hash)  # Convert to UUID format
        
        points.append({
            "id": chunk_id,
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
    
    print(f"✅ Successfully indexed {total_indexed}/{len(points)} chunks for doc_id '{doc_id}'")
    
    # Add the document ID to the query parser for future queries
    try:
        from ..retrieval.retrieval_service import retrieval_service
        retrieval_service.add_new_doc_id_to_parser(doc_id)
        print(f"✅ Added '{doc_id}' to query parser vocabulary")
    except Exception as e:
        print(f"⚠️ Failed to add '{doc_id}' to query parser: {e}")
    
    # Generate suggestions for this document (async/background process)
    try:
        from ..suggestions import generate_suggestions_for_doc
        print(f"🤖 Generating question suggestions for '{doc_id}'...")
        
        # Generate suggestions in background (don't block indexing if it fails)
        suggestion_success = generate_suggestions_for_doc(
            doc_id=doc_id,
            num_questions=8,  # Generate 8 questions per document
            auto_init_collection=True,
            use_lightweight=True  # Use lightweight generation for speed
        )
        
        if suggestion_success:
            print(f"✅ Question suggestions generated for '{doc_id}'")
        else:
            print(f"⚠️ Failed to generate suggestions for '{doc_id}' (indexing still successful)")
            
    except Exception as e:
        print(f"⚠️ Suggestion generation failed for '{doc_id}': {e} (indexing still successful)")
    
    return total_indexed


def index_pdf(pdf_path: str, collection_name: str = QDRANT_COLLECTION, doc_id: str | None = None, extra_payload: dict | None = None) -> int:
    """
    Backwards-compatible helper that extracts nodes, then delegates to index_nodes.
    """
    # Ensure indexes exist before indexing
    ensure_indexes_exist(collection_name)

    nodes = build_page_nodes(pdf_path)
    # Prefer provided doc_id (e.g., original filename without extension)
    if not doc_id:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    return index_nodes(nodes, collection_name=collection_name, doc_id=doc_id, extra_payload=extra_payload)


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
    else:
        # Ensure indexes exist even when not recreating
        ensure_indexes_exist(collection_name)

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


def index_text_payloads(
    payloads: list[dict],
    collection_name: str = QDRANT_COLLECTION,
    batch_size: int = 200
) -> int:
    """Index arbitrary text payloads into Qdrant.

    Each payload must contain at least: { 'text': str }
    Any additional keys are stored in the payload.
    Returns number of points indexed.
    """
    texts = []
    metas = []
    for p in payloads:
        t = (p or {}).get("text", "")
        if not t:
            continue
        texts.append(t)
        metas.append({k: v for k, v in p.items() if k != "text"})

    if not texts:
        return 0

    vectors = []
    for i in range(0, len(texts), 100):
        vectors.extend(embed(texts[i:i+100]))

    points = []
    for i, (text, meta) in enumerate(zip(texts, metas)):
        payload = {**meta, "text": text}
        points.append({
            "id": str(uuid.uuid4()),
            "vector": [float(v) for v in vectors[i]],
            "payload": payload,
        })

    total_indexed = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total_indexed += len(batch)
    return total_indexed

if __name__ == "__main__":
    args = sys.argv[1:]
    
    # Check if user wants to just ensure indexes
    if "--ensure-indexes" in args:
        args.remove("--ensure-indexes")
        print("🔧 Ensuring indexes exist...")
        ensure_indexes_exist()
        print("✅ Indexes ensured!")
        sys.exit(0)
    
    # Check if user wants to add missing indexes
    if "--add-missing-indexes" in args:
        args.remove("--add-missing-indexes")
        print("🔧 Adding missing indexes...")
        add_missing_indexes()
        print("✅ Missing indexes added!")
        sys.exit(0)
    
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
        print("Usage: python -m src.rag.indexing.indexer [--ensure-indexes] [--add-missing-indexes] [--no-recreate] <pdf1> [pdf2 ...]")
        print("  --ensure-indexes: Just ensure indexes exist without indexing files")
        print("  --add-missing-indexes: Add missing indexes to existing collection")
        sys.exit(1)

    index_pdfs(pdfs, recreate=recreate)