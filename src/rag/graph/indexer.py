from __future__ import annotations

from typing import List, Dict, Any
import hashlib
import logging
import os

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

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core import Settings
from src.rag.indexing.indexer import index_text_payloads, index_nodes
from src.models.embedder import configure_llamaindex_for_local_models
from src.data.pdf_ingestor import build_page_nodes
from .graph_store import get_driver
from src.config import (
    NEO4J_DATABASE,
    GRAPH_SPLIT_CHUNK_SIZE,
    GRAPH_SPLIT_OVERLAP,
    GRAPH_EXTRACT_MAX_TRIPLETS,
    GRAPH_EXTRACT_NUM_WORKERS,
    GRAPH_ALLOWED_ENTITY_TYPES,
    GRAPH_ALLOWED_RELATION_TYPES,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    TRAINING_CORPUS_ID,
    GRAPH_WRITE_CHUNKS_TO_NEO4J,
    QDRANT_COLLECTION,
)
from .sanitizer import SanitizeKGExtractor

logger = logging.getLogger(__name__)


def build_graph_from_nodes(nodes: List[Dict[str, Any]], doc_id: str, batch_size: int = 100, corpus_id: str | None = None) -> int:
    """Upsert Document and Chunk nodes to Neo4j so retrieval can traverse them.

    Returns number of chunks written.
    """
    if not nodes:
        return 0

    driver = get_driver()
    chunks: List[dict] = []
    
    # Import chunking function to ensure consistency with Qdrant
    from src.data.chunker import chunk_text
    
    for n in nodes:
        raw_text = (n.get("text") or "").strip()
        if not raw_text:
            continue
        page = n.get("page")
        ntype = n.get("type") or "paragraph"
        
        # Use same chunking strategy as Qdrant for consistency
        if ntype == "paragraph":
            # Apply chunking to paragraphs (same as Qdrant)
            text_chunks = chunk_text(raw_text)
            for chunk_idx, chunk_content in enumerate(text_chunks):
                seed = f"{doc_id}|{page}|{ntype}|{chunk_idx}|{chunk_content[:64]}".encode("utf-8", errors="ignore")
                md5_hash = hashlib.md5(seed).hexdigest()
                cid = md5_to_uuid(md5_hash)  # Convert to UUID format
                chunks.append({
                    "id": cid,
                    "doc_id": doc_id,
                    "page": int(page) if isinstance(page, int) or (isinstance(page, str) and page.isdigit()) else None,
                    "type": ntype,
                    "name": f"{doc_id} p{page}",
                    "text": chunk_content,
                    "corpus_id": corpus_id,
                    "chunk_idx": chunk_idx,
                })
        else:
            # For non-paragraph nodes, use as-is (same as Qdrant)
            chunk_idx = n.get("chunk_idx", 0)
            seed = f"{doc_id}|{page}|{ntype}|{chunk_idx}|{raw_text[:64]}".encode("utf-8", errors="ignore")
            md5_hash = hashlib.md5(seed).hexdigest()
            cid = md5_to_uuid(md5_hash)  # Convert to UUID format
            chunks.append({
                "id": cid,
                "doc_id": doc_id,
                "page": int(page) if isinstance(page, int) or (isinstance(page, str) and page.isdigit()) else None,
                "type": ntype,
                "name": f"{doc_id} p{page}",
                "text": raw_text,
                "corpus_id": corpus_id,
                "chunk_idx": chunk_idx,
            })

    if not chunks:
        return 0

    cypher = (
        "UNWIND $chunks AS ch "
        "MERGE (d:Document {doc_id: $doc_id}) "
        "ON CREATE SET d.name = $doc_id, d.corpus_id = $corpus_id "
        "MERGE (c:Chunk {id: ch.id}) "
        "SET c.doc_id = ch.doc_id, c.page = ch.page, c.type = ch.type, c.name = ch.name, c.text = ch.text, c.corpus_id = ch.corpus_id "
        "MERGE (d)-[:HAS_CHUNK]->(c)"
    )
    with driver.session(database=NEO4J_DATABASE) as session:
        for i in range(0, len(chunks), batch_size):
            session.run(cypher, {"chunks": chunks[i:i+batch_size], "doc_id": doc_id, "corpus_id": corpus_id})

    logger.info(f"[GraphIngest] Upserted {len(chunks)} chunks to Neo4j for doc_id={doc_id}")
    return len(chunks)


def extract_entities_relations_and_index(
    texts: List[str],
    doc_id: str,
    corpus_id: str,
    graph_store: Neo4jPropertyGraphStore,
) -> dict:
    """Use LlamaIndex DynamicLLMPathExtractor to build entities/relations and communities, and index summaries to Qdrant.

    Returns a summary dict.
    """
    if not texts:
        return {"entities": 0, "relations": 0, "communities": 0, "qdrant_vectors": 0}


    # Temporarily switch to extractor-specific LLM if available
    llm_backup = getattr(Settings, "llm", None)
    extract_llm = getattr(Settings, "_extract_llm", None)
    if extract_llm is not None:
        Settings.llm = extract_llm
    # Safe logging of model info
    try:
        meta = getattr(Settings.llm, "metadata", None)
        if meta and hasattr(meta, "model_name"):
            logger.info(f"[GraphExtractor] Using LLM for extraction: {meta.model_name}")
        else:
            model_attr = getattr(Settings.llm, "model", None)
            logger.info(f"[GraphExtractor] Using LLM for extraction: {model_attr or type(Settings.llm).__name__}")
    except Exception:
        logger.info(f"[GraphExtractor] Using LLM for extraction: {type(Settings.llm).__name__}")
    
    # Create documents from texts
    documents = [Document(text=t, metadata={"doc_id": doc_id}) for t in texts if t.strip()]
    if not documents:
        return {"entities": 0, "relations": 0, "communities": 0, "qdrant_vectors": 0}

    try:
        extractor_kwargs = {
            "llm": Settings.llm,
            "max_triplets_per_chunk": GRAPH_EXTRACT_MAX_TRIPLETS,
            "num_workers": GRAPH_EXTRACT_NUM_WORKERS,
        }
        if GRAPH_ALLOWED_ENTITY_TYPES:
            extractor_kwargs["allowed_entity_types"] = GRAPH_ALLOWED_ENTITY_TYPES
        if GRAPH_ALLOWED_RELATION_TYPES:
            extractor_kwargs["allowed_relation_types"] = GRAPH_ALLOWED_RELATION_TYPES

        dynamic_extractor = DynamicLLMPathExtractor(**extractor_kwargs)

        sanitizer = SanitizeKGExtractor()

        logger.info(
            f"[GraphExtractor] Building PropertyGraphIndex with DynamicLLMPathExtractor"
            f"(triplets={GRAPH_EXTRACT_MAX_TRIPLETS}, workers={GRAPH_EXTRACT_NUM_WORKERS})"
            f" and SanitizeKGExtractor for {len(documents)} documents"
        )
        try:

            splitter = SentenceSplitter(
                chunk_size=GRAPH_SPLIT_CHUNK_SIZE,
                chunk_overlap=GRAPH_SPLIT_OVERLAP,
            )
            split_nodes = IngestionPipeline(transformations=[splitter]).run(documents=documents)

            pgi = PropertyGraphIndex.from_documents(
                split_nodes,
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                embed_kg_nodes=True,  # Re-enable embedding for KG nodes now that the issue is fixed
                kg_extractors=[dynamic_extractor, sanitizer],
                property_graph_store=graph_store,
                show_progress=True,
            )
            logger.info("[GraphExtractor] PropertyGraphIndex created successfully")
        except Exception as bulk_err:
            logger.warning(f"[GraphExtractor] Bulk extraction failed, falling back to per-node: {bulk_err}")
            # Fallback: split and process nodes with dynamic extractor + sanitizer
            splitter = SentenceSplitter(
                chunk_size=GRAPH_SPLIT_CHUNK_SIZE,
                chunk_overlap=GRAPH_SPLIT_OVERLAP,
            )
            split_nodes = IngestionPipeline(transformations=[splitter]).run(documents=documents)
            processed_nodes = []
            for node in split_nodes:
                try:
                    # Use synchronous calls to avoid nested event loop issues
                    step1_nodes = dynamic_extractor([node])
                    step2_nodes = sanitizer(step1_nodes)
                    if step2_nodes:
                        processed_nodes.extend(step2_nodes)
                except Exception as node_err:
                    logger.debug(f"[GraphExtractor] Skipping node due to error: {node_err}")
            pgi = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                embed_kg_nodes=True,  # Re-enable embedding for KG nodes now that the issue is fixed
                kg_extractors=[dynamic_extractor, sanitizer],
            )
            pgi.insert_nodes(split_nodes)

        # After index is built, add corpus/doc metadata to entities and relations
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            # Add metadata to all entities and relations created by the extractors
            session.run(
                """
                MATCH (e:Entity) WHERE e.doc_id IS NULL
                SET e.doc_id = $doc_id, e.corpus_id = $corpus_id
                """,
                {"doc_id": doc_id, "corpus_id": corpus_id},
            )
            session.run(
                """
                MATCH ()-[r]->() WHERE r.doc_id IS NULL
                SET r.doc_id = $doc_id, r.corpus_id = $corpus_id
                """,
                {"doc_id": doc_id, "corpus_id": corpus_id},
            )

            # Count only semantic entities/relations (exclude structural ones)
            entity_count = session.run(
                "MATCH (e:Entity {doc_id: $doc_id}) RETURN count(e) AS c",
                {"doc_id": doc_id},
            ).single().get("c", 0)
            relation_count = session.run(
                "MATCH ()-[r]->() WHERE r.doc_id = $doc_id AND NOT type(r) IN ['HAS_CHUNK', 'HAS_DOCUMENT'] RETURN count(r) AS c",
                {"doc_id": doc_id},
            ).single().get("c", 0)
            logger.info(f"[GraphExtractor] Semantic extraction counts: entities={entity_count}, relations={relation_count}")

            # Debug: Check what types of semantic relationships were created
            if relation_count > 0:
                relation_types = session.run(
                    "MATCH ()-[r]->() WHERE r.doc_id = $doc_id AND NOT type(r) IN ['HAS_CHUNK', 'HAS_DOCUMENT'] RETURN DISTINCT type(r) AS rel_type, count(*) AS count ORDER BY count DESC",
                    {"doc_id": doc_id},
                ).data()
                logger.info(f"[GraphExtractor] Semantic relationship types: {relation_types}")
            else:
                logger.warning("[GraphExtractor] No semantic relationships found - LLM extraction may have failed")

        # Build communities if supported
        if hasattr(pgi.property_graph_store, "build_communities"):
            try:
                pgi.property_graph_store.build_communities()
                logger.info("[GraphExtractor] Built communities")
            except Exception as e:
                logger.warning(f"[GraphExtractor] Failed to build communities: {e}")

        logger.info("[GraphExtractor] PropertyGraphIndex build complete and metadata updated")
    except Exception as e:
        logger.error(f"[GraphExtractor] Entity/relation extraction failed: {e}", exc_info=True)
        # Continue gracefully
    finally:
        # Restore default LLM after extraction to ensure answering uses DEFAULT_MODEL
        if llm_backup is not None:
            Settings.llm = llm_backup

    # Prepare Qdrant indexing payloads for entity descriptions and community summaries
    payloads: List[Dict[str, Any]] = []

    qdrant_vectors = index_text_payloads(payloads) if payloads else 0
    logger.info(f"[GraphExtractor] Indexed {qdrant_vectors} entity/community vectors to Qdrant")

    # Return counts from Neo4j for visibility
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            e_cnt = session.run(
                "MATCH (e:Entity {doc_id: $doc_id}) RETURN count(e) AS c",
                {"doc_id": doc_id},
            ).single().get("c", 0)
            r_cnt = session.run(
                "MATCH ()-[r]->() WHERE r.doc_id = $doc_id RETURN count(r) AS c",
                {"doc_id": doc_id},
            ).single().get("c", 0)
    except Exception:
        e_cnt, r_cnt = None, None

    return {
        "entities": e_cnt,
        "relations": r_cnt,
        "communities": None,
        "qdrant_vectors": qdrant_vectors,
    }


def ingest_pdfs_to_graph(pdf_paths: List[str], training_room_id: str | None = None, training_corpus_id: str | None = None) -> dict:
    """Ingest PDFs into Neo4j property graph and Qdrant (scope="graph").

    - Writes chunks to Neo4j as (:Document)-[:HAS_CHUNK]->(:Chunk)
    - Indexes vectors to Qdrant with payload tags {room_id, scope:"graph"}

    Returns summary dict with counts per file.
    """
    results = {}
    for pdf_path in pdf_paths:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            logger.info(f"[TrainingIngest] Extracting nodes from {pdf_path}")
            nodes = build_page_nodes(pdf_path)
        except Exception as e:
            logger.error(f"[TrainingIngest] Node build failed for {pdf_path}: {e}")
            results[pdf_path] = {"error": f"node_build_failed: {e}"}
            continue

        corpus_id = training_corpus_id or TRAINING_CORPUS_ID

        # 1) Optionally write chunks to Neo4j (disabled by default to avoid super-node stars)
        chunk_count = 0
        if GRAPH_WRITE_CHUNKS_TO_NEO4J:
            try:
                logger.info(f"[TrainingIngest] Writing chunks to Neo4j (db={NEO4J_DATABASE}, corpus={corpus_id})")
                chunk_count = build_graph_from_nodes(nodes, doc_id, corpus_id=corpus_id)
            except Exception as e:
                logger.error(f"[TrainingIngest] Neo4j chunk ingest failed: {e}")
                # Don't fail the whole ingestion; proceed to KG extraction and Qdrant

        # 2) Extract entities/relations/communities and index summaries to Qdrant
        try:
            logger.info("[TrainingIngest] Extracting entities/relations and indexing summaries")
            graph_store = Neo4jPropertyGraphStore(
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                url=NEO4J_URI,
                database=NEO4J_DATABASE,
            )
            # Collect texts for extractor from nodes
            texts = [n.get("text", "") for n in nodes if n.get("text")]
            erc = extract_entities_relations_and_index(texts, doc_id, corpus_id, graph_store)
        except Exception as e:
            logger.error(f"[TrainingIngest] Graph extractor failed: {e}")
            results[pdf_path] = {"neo4j_chunks": chunk_count, "error": f"graph_extractor_failed: {e}"}
            continue

        # 3) Index raw vectors (chunks) to Qdrant under scope=graph for grounding (reuse extracted nodes)
        try:
            logger.info("[TrainingIngest] Indexing chunk vectors to Qdrant (scope=graph)")
            extra_payload = {"corpus_id": corpus_id, "scope": "graph"}
            vec_count = index_nodes(nodes, collection_name=QDRANT_COLLECTION, doc_id=doc_id, extra_payload=extra_payload)
        except Exception as e:
            logger.error(f"[TrainingIngest] Qdrant index failed: {e}")
            results[pdf_path] = {"neo4j_chunks": chunk_count, "erc": erc, "error": f"qdrant_index_failed: {e}"}
            continue

        results[pdf_path] = {"neo4j_chunks": chunk_count, "qdrant_chunks": vec_count, "erc": erc}
        logger.info(f"[TrainingIngest] Completed {pdf_path}: neo4j_chunks={chunk_count}, qdrant_chunks={vec_count}, erc={erc}")
    return results


