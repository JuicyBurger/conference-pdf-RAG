from __future__ import annotations

from typing import Optional
import logging

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core import Settings

from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from src.models.embedder import configure_llamaindex_for_local_models


logger = logging.getLogger(__name__)


class LlamaIndexGraphQuery:
    """Thin wrapper around LlamaIndex PropertyGraphIndex query engine.

    Builds from existing Neo4j graph and exposes a simple query() API that returns
    a LlamaIndex response object (with optional source_nodes).
    """

    def __init__(self) -> None:

        # Create graph store and PG index from existing Neo4j graph
        self._graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database=NEO4J_DATABASE,
        )
        # From existing graph; do not re-extract
        self._index = PropertyGraphIndex.from_existing(
            property_graph_store=self._graph_store,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            embed_kg_nodes=False,  # Don't re-embed existing nodes
        )
        # Build a default query engine; prefer to return sources if available
        self._qe = self._index.as_query_engine(llm=Settings.llm)

    def query(self, question: str):
        return self._qe.query(question)


def get_graph_query_engine() -> Optional[LlamaIndexGraphQuery]:
    try:
        return LlamaIndexGraphQuery()
    except Exception as e:
        logger.warning(f"LlamaIndex graph query engine unavailable: {e}")
        return None