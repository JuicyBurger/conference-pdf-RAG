from __future__ import annotations

import os
import logging
from typing import List

from src.data.pdf_ingestor import build_page_nodes
from src.rag.graph.indexer import build_graph_from_nodes, extract_entities_relations_and_index
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE, TRAINING_CORPUS_ID
from src.rag.indexing.indexer import index_pdf
from src.config import QDRANT_COLLECTION

logger = logging.getLogger(__name__)


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

        # 1) Write to Neo4j
        try:
            logger.info(f"[TrainingIngest] Writing to Neo4j (db={NEO4J_DATABASE}, corpus={corpus_id})")
            chunk_count = build_graph_from_nodes(nodes, doc_id, corpus_id=corpus_id)
        except Exception as e:
            logger.error(f"[TrainingIngest] Neo4j ingest failed: {e}")
            results[pdf_path] = {"error": f"neo4j_ingest_failed: {e}"}
            continue

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

        # 3) Index raw vectors (chunks) to Qdrant under scope=graph for grounding
        try:
            logger.info("[TrainingIngest] Indexing chunk vectors to Qdrant (scope=graph)")
            extra_payload = {"corpus_id": corpus_id, "scope": "graph"}
            vec_count = index_pdf(pdf_path, collection_name=QDRANT_COLLECTION, doc_id=doc_id, extra_payload=extra_payload)
        except Exception as e:
            logger.error(f"[TrainingIngest] Qdrant index failed: {e}")
            results[pdf_path] = {"neo4j_chunks": chunk_count, "erc": erc, "error": f"qdrant_index_failed: {e}"}
            continue

        results[pdf_path] = {"neo4j_chunks": chunk_count, "qdrant_chunks": vec_count, "erc": erc}
        logger.info(f"[TrainingIngest] Completed {pdf_path}: neo4j_chunks={chunk_count}, qdrant_chunks={vec_count}, erc={erc}")
    return results


