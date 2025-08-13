"""
GraphRAGEngine scaffolding.

Implementation will use LlamaIndex GraphRAG components (GraphRAGExtractor,
PropertyGraphIndex with Neo4jGraphStore) and QdrantVectorStore for grounding.
"""

from __future__ import annotations

from typing import List, Optional

from .base import RAGEngine, Evidence
from ..retriever import retrieve as vector_retrieve
from ..graph.graph_store import ensure_graph_indexes
from src.config import GRAPH_MAX_HOPS, GRAPH_EXPANSION_TOP_K, NEO4J_DATABASE, TRAINING_CORPUS_ID
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE


class GraphRAGEngine(RAGEngine):
    def __init__(self):
        # Defer heavy imports/setup until first use in future steps
        ensure_graph_indexes()

    def retrieve(self, room_id: str, query: str) -> List[Evidence]:
        # 1) Retrieve graph-scoped vectors (entity descriptions, community summaries, chunks)
        vector_hits = vector_retrieve(
            query=query,
            top_k=12,
            score_threshold=0.25,
            room_id=None,
            prefer_chat_scope=False,
            scope_filter="graph",
            extra_filters={"corpus_id": TRAINING_CORPUS_ID},
        )
        evidence = [Evidence(h) for h in vector_hits]

        # 2) Graph traversal: use Neo4j to expand from matched entities/communities
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            with driver.session(database=NEO4J_DATABASE) as session:
                # Simple expansion: search entity nodes by name similarity using fulltext (if exists), else by property
                # Then expand up to GRAPH_MAX_HOPS and collect related Chunk nodes for grounding
                # Note: assuming nodes labeled :Entity, :Community, :Chunk and relations (:MENTIONED_IN, :HAS_CHUNK)
                # This is a best-effort; schema may be tuned later
                cypher = f"""
                MATCH (c:Chunk)
                WHERE c.doc_id IS NOT NULL
                WITH collect(c) as chunks  // placeholder to avoid unused var in editor
                RETURN 1
                """  # placeholder to validate session
                session.run(cypher)

                # If we had entity names from vector payloads, we could seed expansion. For now, skip to return vectors
        except Exception:
            pass

        return evidence

    def answer(self, room_id: str, query: str, pdf_summary: Optional[str] = None) -> str:
        # Temporary: Use retrieved graph-scoped vectors to answer; next iteration will use graph traversal
        hits = self.retrieve(room_id, query)
        if not hits:
            return "目前沒有圖譜資料可用，請先在訓練空間進行知識圖譜建立。"
        from ..generator import generate_answer
        return generate_answer(query, [ev.raw for ev in hits])


