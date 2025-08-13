from __future__ import annotations

from typing import List, Dict, Any
import asyncio
import logging

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor, SimpleLLMPathExtractor
from llama_index.core import Settings
from src.rag.indexing.indexer import index_text_payloads
from src.rag.embeddings.jina_adapter import configure_llamaindex_for_local_models
from .graph_store import get_driver
from src.config import NEO4J_DATABASE
from src.rag.graph.sanitizer import SanitizeKGExtractor

logger = logging.getLogger(__name__)


def build_graph_from_nodes(nodes: List[Dict[str, Any]], doc_id: str, batch_size: int = 100, corpus_id: str | None = None) -> int:
    """Build property graph in Neo4j using LlamaIndex abstractions from existing nodes.

    Returns number of units processed.
    """
    if not nodes:
        return 0

    # Prepare documents by page/chunk
    texts: List[str] = []
    metadatas: List[dict] = []
    for n in nodes:
        text = n.get("text", "").strip()
        if not text:
            continue
        meta = {
            "doc_id": doc_id,
            "page": n.get("page"),
            "type": n.get("type"),
        }
        # carry optional keys
        for k in ("chunk_idx", "table_id", "row_idx", "column_idx", "column_name"):
            if k in n:
                meta[k] = n[k]
        texts.append(text)
        metadatas.append(meta)

    if not texts:
        return 0

    docs = [Document(text=t, metadata=m) for t, m in zip(texts, metadatas)]

    # Smaller chunks for faster LLM processing
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=20)
    pipeline = IngestionPipeline(transformations=[splitter])
    nodes = pipeline.run(documents=docs)

    # Skip structural relationships - focus only on semantic extraction
    logger.info(f"[GraphIngest] Prepared {len(nodes)} chunks for semantic extraction (doc_id={doc_id}, corpus_id={corpus_id})")
    return len(nodes)


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

    # Configure LlamaIndex to use our local models (Jina embeddings + local LLM)
    configure_llamaindex_for_local_models()
    logger.info(f"[GraphExtractor] Using LLM: {type(Settings.llm).__name__}")
    
    # Create documents from texts
    documents = [Document(text=t, metadata={"doc_id": doc_id}) for t in texts if t.strip()]
    if not documents:
        return {"entities": 0, "relations": 0, "communities": 0, "qdrant_vectors": 0}

    try:
        simple_extractor = SimpleLLMPathExtractor(
            llm=Settings.llm,
            max_paths_per_chunk=5,
            num_workers=1,
        )

        dynamic_extractor = DynamicLLMPathExtractor(
            llm=Settings.llm,
            max_triplets_per_chunk=8,
            num_workers=1,
        )

        sanitizer = SanitizeKGExtractor()

        logger.info(f"[GraphExtractor] Building PropertyGraphIndex with DynamicLLMPathExtractor, SimpleLLMPathExtractor, and SanitizeKGExtractor for {len(documents)} documents")
        try:
            pgi = PropertyGraphIndex.from_documents(
                documents,
                llm=Settings.llm,
                embed_kg_nodes=False,
                kg_extractors=[dynamic_extractor, simple_extractor, sanitizer],
                property_graph_store=graph_store,
                show_progress=True,
            )
            logger.info("[GraphExtractor] PropertyGraphIndex created successfully")
        except Exception as bulk_err:
            logger.warning(f"[GraphExtractor] Bulk extraction failed, falling back to per-node: {bulk_err}")
            # Fallback: per-node extractor with timeout
            splitter = SentenceSplitter(chunk_size=400, chunk_overlap=60)
            split_nodes = IngestionPipeline(transformations=[splitter]).run(documents=documents)
            processed_nodes = []
            for node in split_nodes:
                try:
                    # Use synchronous calls to avoid nested event loop issues
                    step1_nodes = simple_extractor([node])
                    step2_nodes = sanitizer(step1_nodes)
                    if step2_nodes:
                        processed_nodes.extend(step2_nodes)
                except Exception as node_err:
                    logger.debug(f"[GraphExtractor] Skipping node due to error: {node_err}")
            pgi = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                llm=Settings.llm,
                embed_kg_nodes=False,
            )
            if processed_nodes:
                pgi.insert_nodes(processed_nodes)


        
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


