"""
Vector indexer for Qdrant vector search.

This module handles indexing documents into Qdrant for vector similarity search.
"""

import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from .base import BaseIndexer, IndexingResult
from ..config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from ..models.embedder import model, embed
from ..data.chunker import chunk_text


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


class VectorIndexer(BaseIndexer):
    """Handles vector indexing to Qdrant."""
    
    def __init__(self, collection_name: str = QDRANT_COLLECTION):
        super().__init__("VectorIndexer")
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=120.0
        )
    
    def initialize(self) -> bool:
        """Initialize Qdrant collection and indexes."""
        try:
            # Create/recreate collection if needed
            vector_size = model.get_sentence_embedding_dimension()
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
            )
            self.logger.info(f"Collection '{self.collection_name}' initialized: vectors={vector_size}, distance=Cosine")
            
            # Ensure indexes exist
            self._ensure_indexes_exist()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector indexer: {e}")
            return False
    
    def _ensure_indexes_exist(self) -> None:
        """Ensure all necessary payload indexes exist for the collection (idempotent).

        This is called before every upsert to guarantee TEXT index on 'text' is available
        for BM25/MatchText queries. Includes simple retries for flaky networks.
        """
        self.logger.info(f"Ensuring payload indexes for collection '{self.collection_name}' (including TEXT on 'text')...")

        required_indexes = [
            ("doc_id", "keyword", "Document ID filtering"),
            ("page", "integer", "Page number filtering"),
            ("type", "keyword", "Content type filtering"),
            ("text", "text", "Full-text search (BM25/MatchText)"),
            ("room_id", "keyword", "Room scoping for chat uploads"),
            ("scope", "keyword", "Data scope flag: chat/global/graph"),
            ("corpus_id", "keyword", "Corpus scoping for training graph"),
        ]

        # Lazy import at top-level for clarity
        from qdrant_client.http.models import PayloadSchemaType

        for field_name, field_schema, description in required_indexes:
            schema = getattr(PayloadSchemaType, field_schema.upper(), PayloadSchemaType.KEYWORD)
            last_err = None
            for attempt in range(1, 4):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema,
                    )
                    self.logger.debug(f"Created {field_schema} index on '{field_name}' - {description}")
                    last_err = None
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "already exists" in msg or "exists" in msg:
                        self.logger.debug(f"Index on '{field_name}' already exists")
                        last_err = None
                        break
                    last_err = e
                    if attempt < 3:
                        self.logger.warning(f"Retry {attempt}/3 creating index '{field_name}' ({field_schema}): {e}")
                        try:
                            import time as _time
                            _time.sleep(1.0 * attempt)
                        except Exception:
                            pass
                    else:
                        self.logger.warning(f"Could not create index on '{field_name}': {e}")
            if last_err:
                # Continue attempting other indexes even if one failed
                continue
    
    def index_nodes(self, 
                   nodes: List[Dict[str, Any]], 
                   doc_id: str,
                   extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
        """
        Index nodes to Qdrant vector store.
        
        Args:
            nodes: List of document nodes to index
            doc_id: Document identifier  
            extra_payload: Additional metadata to include
            
        Returns:
            IndexingResult with operation details
        """
        try:
            # Ensure needed payload indexes (including full-text on 'text') exist
            try:
                self._ensure_indexes_exist()
            except Exception as e:
                self.logger.warning(f"Could not ensure payload indexes before indexing: {e}")
            # Visibility: target collection and endpoint
            self.logger.info(f"Qdrant target: url={QDRANT_URL}, collection={self.collection_name}")
            # Validate nodes
            valid_nodes = self.validate_nodes(nodes)
            if not valid_nodes:
                return IndexingResult(
                    success=False,
                    error="No valid nodes to index",
                    doc_id=doc_id
                )
            
            # Prepare texts and metadata for batch embedding
            all_texts = []
            text_metadata = []
            
            # Summarize types
            try:
                from collections import Counter
                type_counts = Counter([n.get("type", "unknown") for n in valid_nodes])
                self.logger.info(f"Preparing {len(valid_nodes)} nodes (types={dict(type_counts)})")
            except Exception:
                self.logger.info(f"Preparing {len(valid_nodes)} nodes for batch embedding...")
            
            for node in valid_nodes:
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
                
                # Include table summaries/notes as retrievable text; skip raw table records/columns
                elif node["type"] in ["table_summary", "table_note"]:
                    chunks = chunk_text(node_text, content_type="table")
                    for chunk_idx, chunk_content in enumerate(chunks):
                        all_texts.append(chunk_content)
                        text_metadata.append({
                            "type": node["type"],
                            "page": node["page"],
                            "chunk_idx": chunk_idx,
                            "text": chunk_content,
                            # carry table_id when present for traceability
                            **({"table_id": node.get("table_id")} if node.get("table_id") else {})
                        })
                # Table HTML/text chunks produced by pipeline → treat with table chunking as well
                elif node["type"] == "table_chunk":
                    chunks = chunk_text(node_text, content_type="table")
                    for chunk_idx, chunk_content in enumerate(chunks):
                        all_texts.append(chunk_content)
                        meta_extra = {}
                        # propagate useful metadata when present
                        try:
                            if isinstance(node.get("metadata"), dict):
                                for k in ("table_id", "source_id", "scope"):
                                    if k in node["metadata"]:
                                        meta_extra[k] = node["metadata"][k]
                        except Exception:
                            pass
                        text_metadata.append({
                            "type": node["type"],
                            "page": node["page"],
                            "chunk_idx": chunk_idx,
                            "text": chunk_content,
                            **meta_extra,
                        })
                elif node["type"] in ["table_record", "table_column"]:
                    self.logger.debug(f"Skipping {node['type']} node - handled by KG pipeline")
                    continue
                
                # Handle other node types as-is
                else:
                    all_texts.append(node_text)
                    base_meta = {
                        "type": node["type"],
                        "page": node["page"],
                        "chunk_idx": node.get("chunk_idx", 0),
                        "text": node_text,
                    }
                    # carry arbitrary metadata if provided
                    if isinstance(node.get("metadata"), dict):
                        for k, v in node["metadata"].items():
                            if k not in base_meta:
                                base_meta[k] = v
                    text_metadata.append(base_meta)
            
            if not all_texts:
                self.logger.warning("No texts to embed after filtering (only non-embeddable nodes?)")
                return IndexingResult(
                    success=True,
                    indexed_count=0,
                    doc_id=doc_id,
                    metadata={"message": "No texts to embed after filtering"}
                )
            
            # Generate embeddings in batches
            vectors = self._batch_embed_texts(all_texts)
            
            # Create points for Qdrant
            points = self._create_points(all_texts, text_metadata, vectors, doc_id, extra_payload)
            
            # Upload to Qdrant in batches
            indexed_count = self._upload_points(points)
            
            # Update query parser vocabulary
            self._update_query_parser(doc_id)
            
            return IndexingResult(
                success=True,
                indexed_count=indexed_count,
                doc_id=doc_id,
                metadata={
                    "total_texts": len(all_texts),
                    "total_points": len(points),
                    "collection": self.collection_name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Vector indexing failed for doc_id '{doc_id}': {e}")
            return IndexingResult(
                success=False,
                error=str(e),
                doc_id=doc_id
            )
    
    def _batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts in batches."""
        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Skipping empty text at index {i}")
        
        if not valid_texts:
            return []
        
        self.logger.info(f"Generating embeddings for {len(valid_texts)} valid texts...")
        
        import time
        start_time = time.time()
        vectors = []
        batch_size = 200  # Increased from 100 for better throughput
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(valid_texts), batch_size):
            batch_start = time.time()
            batch_texts = valid_texts[i:i + batch_size]
            batch_vectors = embed(batch_texts)
            vectors.extend(batch_vectors)
            batch_time = time.time() - batch_start
            batch_num = (i // batch_size) + 1
            
            self.logger.debug(f"Batch {batch_num}/{total_batches}: {len(batch_texts)} texts in {batch_time:.2f}s "
                            f"({len(batch_texts)/batch_time:.1f} texts/sec)")
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ Generated {len(vectors)} embeddings in {total_time:.2f}s "
                        f"({len(vectors)/total_time:.1f} embeddings/sec)")
        
        # Validate and fix vector dimensions
        expected_dimension = model.get_sentence_embedding_dimension()
        for i, vector in enumerate(vectors):
            if not vector or len(vector) == 0:
                self.logger.warning(f"Empty vector generated for text {i}, using zero vector")
                vectors[i] = [0.0] * expected_dimension
            elif len(vector) != expected_dimension:
                self.logger.warning(f"Wrong vector dimension {len(vector)} for text {i}, expected {expected_dimension}")
                if len(vector) < expected_dimension:
                    vectors[i] = vector + [0.0] * (expected_dimension - len(vector))
                else:
                    vectors[i] = vector[:expected_dimension]
        
        return vectors
    
    def _create_points(self, 
                      texts: List[str], 
                      metadata: List[Dict[str, Any]], 
                      vectors: List[List[float]], 
                      doc_id: str, 
                      extra_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create Qdrant points from texts, metadata, and vectors."""
        points = []
        
        for i, (text, meta, vector) in enumerate(zip(texts, metadata, vectors)):
            payload = {
                "doc_id": doc_id,
                "page": meta["page"],
                "type": meta["type"],
                "text": meta["text"],
            }
            
            # Add extra payload if provided
            if extra_payload:
                for k, v in extra_payload.items():
                    if k not in payload:
                        payload[k] = v
            
            # Add type-specific fields
            if meta["type"] == "paragraph":
                payload["chunk_idx"] = meta["chunk_idx"]
            else:
                # include chunk_idx if present for non-paragraph types too
                if "chunk_idx" in meta:
                    payload["chunk_idx"] = meta["chunk_idx"]

            # Merge arbitrary metadata captured earlier
            for k, v in meta.items():
                if k not in payload and k not in ("text",):
                    payload[k] = v
            
            # Generate consistent ID
            seed = f"{doc_id}|{meta.get('page')}|{meta.get('type')}|{meta.get('chunk_idx', 0)}|{meta.get('text','')[:64]}".encode("utf-8", errors="ignore")
            md5_hash = hashlib.md5(seed).hexdigest()
            chunk_id = md5_to_uuid(md5_hash)
            
            points.append({
                "id": chunk_id,
                "vector": [float(v) for v in vector],
                "payload": payload,
            })
        
        return points
    
    def _upload_points(self, points: List[Dict[str, Any]]) -> int:
        """Upload points to Qdrant in batches."""
        batch_size = 200
        total_indexed = 0
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        self.logger.info(f"Uploading {len(points)} chunks in {total_batches} batches...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            # Retry logic for failed uploads
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.client.upsert(collection_name=self.collection_name, points=batch)
                    total_indexed += len(batch)
                    self.logger.debug(f"Batch {batch_num}/{total_batches} uploaded successfully")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {e}")
                        # Try splitting batch if it's a vector error
                        if "VectorStruct" in str(e) and len(batch) > 10:
                            mid = len(batch) // 2
                            try:
                                self.client.upsert(collection_name=self.collection_name, points=batch[:mid])
                                self.client.upsert(collection_name=self.collection_name, points=batch[mid:])
                                total_indexed += len(batch)
                                self.logger.debug(f"Batch {batch_num} uploaded in two parts")
                                break
                            except Exception as split_error:
                                self.logger.warning(f"Split batch also failed: {split_error}")
                    else:
                        self.logger.error(f"Batch {batch_num} failed after {max_retries} attempts: {e}")
                        raise
        
        self.logger.info(f"Successfully indexed {total_indexed}/{len(points)} chunks")
        return total_indexed
    
    def _update_query_parser(self, doc_id: str) -> None:
        """Add document ID to query parser vocabulary."""
        try:
            from ..rag.retrieval.retrieval_service import retrieval_service
            retrieval_service.add_new_doc_id_to_parser(doc_id)
            self.logger.debug(f"Added '{doc_id}' to query parser vocabulary")
        except Exception as e:
            self.logger.warning(f"Failed to add '{doc_id}' to query parser: {e}")
