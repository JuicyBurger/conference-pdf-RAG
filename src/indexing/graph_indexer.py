"""
Graph indexer for Neo4j knowledge graph.

This module handles indexing documents into Neo4j for graph-based search and knowledge extraction.
"""

import hashlib
from typing import List, Dict, Any, Optional

from .base import BaseIndexer, IndexingResult
from ..rag.graph.graph_store import get_driver, ensure_graph_indexes
from ..data.chunker import chunk_text
from ..config import (
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
    GRAPH_WRITE_CHUNKS_TO_NEO4J,
)


def md5_to_uuid(md5_hash: str) -> str:
    """Convert an MD5 hash to UUID format by inserting hyphens."""
    if not md5_hash or len(md5_hash) != 32:
        raise ValueError(f"Invalid MD5 hash: {md5_hash}")
    
    uuid_parts = [
        md5_hash[0:8],
        md5_hash[8:12], 
        md5_hash[12:16],
        md5_hash[16:20],
        md5_hash[20:32]
    ]
    
    return "-".join(uuid_parts)


class GraphIndexer(BaseIndexer):
    """Handles graph indexing to Neo4j with knowledge extraction."""
    
    def __init__(self, corpus_id: Optional[str] = None):
        super().__init__("GraphIndexer")
        self.corpus_id = corpus_id
        
    def initialize(self) -> bool:
        """Initialize Neo4j graph indexes and constraints."""
        try:
            ensure_graph_indexes()
            self.logger.info("Neo4j graph indexes and constraints ensured")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize graph indexer: {e}")
            return False
    
    def index_nodes(self, 
                   nodes: List[Dict[str, Any]], 
                   doc_id: str,
                   extra_payload: Optional[Dict[str, Any]] = None) -> IndexingResult:
        """
        Index nodes to Neo4j graph store with knowledge extraction.
        
        Args:
            nodes: List of document nodes to index
            doc_id: Document identifier
            extra_payload: Additional metadata (e.g., corpus_id)
            
        Returns:
            IndexingResult with operation details
        """
        try:
            # Validate nodes
            valid_nodes = self.validate_nodes(nodes)
            if not valid_nodes:
                return IndexingResult(
                    success=False,
                    error="No valid nodes to index",
                    doc_id=doc_id
                )
            
            corpus_id = (extra_payload or {}).get('corpus_id', self.corpus_id)
            
            total_indexed = 0
            
            # Step 1: Optionally write chunks to Neo4j (controlled by config)
            chunk_count = 0
            if GRAPH_WRITE_CHUNKS_TO_NEO4J:
                self.logger.info(f"Writing chunks to Neo4j (corpus={corpus_id})")
                chunk_count = self._build_graph_from_nodes(valid_nodes, doc_id, corpus_id)
                total_indexed += chunk_count
            
            # Step 2: Extract entities/relations using LlamaIndex
            self.logger.info("Extracting entities and relations using LlamaIndex")
            extraction_result = self._extract_entities_relations(valid_nodes, doc_id, corpus_id)
            total_indexed += extraction_result.get('entities', 0) + extraction_result.get('relations', 0)
            
            # Step 3: Handle table KG extraction if present
            table_kg_count = self._process_table_knowledge_graphs(valid_nodes, doc_id, corpus_id)
            total_indexed += table_kg_count
            
            return IndexingResult(
                success=True,
                indexed_count=total_indexed,
                doc_id=doc_id,
                metadata={
                    "chunk_count": chunk_count,
                    "extraction_result": extraction_result,
                    "table_kg_count": table_kg_count,
                    "corpus_id": corpus_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Graph indexing failed for doc_id '{doc_id}': {e}")
            return IndexingResult(
                success=False,
                error=str(e),
                doc_id=doc_id
            )
    
    def _build_graph_from_nodes(self, 
                               nodes: List[Dict[str, Any]], 
                               doc_id: str, 
                               corpus_id: Optional[str],
                               batch_size: int = 100) -> int:
        """Build Document and Chunk nodes in Neo4j."""
        driver = get_driver()
        chunks: List[dict] = []
        
        for n in nodes:
            raw_text = (n.get("text") or "").strip()
            if not raw_text:
                continue
                
            page = n.get("page")
            ntype = n.get("type") or "paragraph"
            
            # Use same chunking strategy as vector indexer for consistency
            if ntype == "paragraph":
                text_chunks = chunk_text(raw_text)
                for chunk_idx, chunk_content in enumerate(text_chunks):
                    seed = f"{doc_id}|{page}|{ntype}|{chunk_idx}|{chunk_content[:64]}".encode("utf-8", errors="ignore")
                    md5_hash = hashlib.md5(seed).hexdigest()
                    cid = md5_to_uuid(md5_hash)
                    chunks.append({
                        "id": cid,
                        "doc_id": doc_id,
                        "page": int(page) if isinstance(page, (int, str)) and str(page).isdigit() else None,
                        "type": ntype,
                        "name": f"{doc_id} p{page}",
                        "text": chunk_content,
                        "corpus_id": corpus_id,
                        "chunk_idx": chunk_idx,
                    })
            else:
                # For non-paragraph nodes, use as-is
                chunk_idx = n.get("chunk_idx", 0)
                seed = f"{doc_id}|{page}|{ntype}|{chunk_idx}|{raw_text[:64]}".encode("utf-8", errors="ignore")
                md5_hash = hashlib.md5(seed).hexdigest()
                cid = md5_to_uuid(md5_hash)
                chunks.append({
                    "id": cid,
                    "doc_id": doc_id,
                    "page": int(page) if isinstance(page, (int, str)) and str(page).isdigit() else None,
                    "type": ntype,
                    "name": f"{doc_id} p{page}",
                    "text": raw_text,
                    "corpus_id": corpus_id,
                    "chunk_idx": chunk_idx,
                })
        
        if not chunks:
            return 0
        
        # Upsert chunks to Neo4j
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
                session.run(cypher, {
                    "chunks": chunks[i:i+batch_size], 
                    "doc_id": doc_id, 
                    "corpus_id": corpus_id
                })
        
        self.logger.info(f"Upserted {len(chunks)} chunks to Neo4j for doc_id={doc_id}")
        return len(chunks)
    
    def _extract_entities_relations(self, 
                                   nodes: List[Dict[str, Any]], 
                                   doc_id: str, 
                                   corpus_id: Optional[str]) -> Dict[str, Any]:
        """Extract entities and relations using LlamaIndex."""
        try:
            # Import LlamaIndex components lazily to avoid heavy deps at module load
            from llama_index.core import Document, Settings
            from llama_index.core.ingestion import IngestionPipeline
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            from llama_index.core.indices.property_graph import PropertyGraphIndex, DynamicLLMPathExtractor
            from ..rag.graph.sanitizer import SanitizeKGExtractor
            from ..models.embedder import configure_llamaindex_for_local_models
            
            # Configure LlamaIndex for local models
            configure_llamaindex_for_local_models()
            
            # Collect texts from nodes
            texts = [n.get("text", "") for n in nodes if n.get("text")]
            if not texts:
                return {"entities": 0, "relations": 0, "communities": 0}
            
            # Create documents
            documents = [Document(text=t, metadata={"doc_id": doc_id}) for t in texts if t.strip()]
            if not documents:
                return {"entities": 0, "relations": 0, "communities": 0}
            
            # Set up extractors
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
            
            # Create graph store
            graph_store = Neo4jPropertyGraphStore(
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                url=NEO4J_URI,
                database=NEO4J_DATABASE,
            )
            
            # Process documents
            splitter = SentenceSplitter(
                chunk_size=GRAPH_SPLIT_CHUNK_SIZE,
                chunk_overlap=GRAPH_SPLIT_OVERLAP,
            )
            split_nodes = IngestionPipeline(transformations=[splitter]).run(documents=documents)
            
            # Build property graph index
            pgi = PropertyGraphIndex.from_documents(
                split_nodes,
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                embed_kg_nodes=True,
                kg_extractors=[dynamic_extractor, sanitizer],
                property_graph_store=graph_store,
                show_progress=True,
            )
            
            # Add corpus/doc metadata to entities and relations
            driver = get_driver()
            with driver.session(database=NEO4J_DATABASE) as session:
                session.run(
                    """
                    MATCH (e:Entity) WHERE e.doc_id IS NULL
                    SET e.doc_id = $doc_id, e.corpus_id = $corpus_id
                    """,
                    {"doc_id": doc_id, "corpus_id": corpus_id}
                )
                session.run(
                    """
                    MATCH ()-[r]->() WHERE r.doc_id IS NULL
                    SET r.doc_id = $doc_id, r.corpus_id = $corpus_id
                    """,
                    {"doc_id": doc_id, "corpus_id": corpus_id}
                )
                
                # Count results
                entity_count = session.run(
                    "MATCH (e:Entity {doc_id: $doc_id}) RETURN count(e) AS c",
                    {"doc_id": doc_id}
                ).single().get("c", 0)
                
                relation_count = session.run(
                    "MATCH ()-[r]->() WHERE r.doc_id = $doc_id AND NOT type(r) IN ['HAS_CHUNK', 'HAS_DOCUMENT'] RETURN count(r) AS c",
                    {"doc_id": doc_id}
                ).single().get("c", 0)
            
            self.logger.info(f"Extracted {entity_count} entities, {relation_count} relations for doc_id={doc_id}")
            
            return {
                "entities": entity_count,
                "relations": relation_count,
                "communities": 0  # Community building is optional
            }
            
        except Exception as e:
            self.logger.error(f"Entity/relation extraction failed: {e}")
            return {"entities": 0, "relations": 0, "communities": 0, "error": str(e)}
    
    def _process_table_knowledge_graphs(self, 
                                       nodes: List[Dict[str, Any]], 
                                       doc_id: str, 
                                       corpus_id: Optional[str]) -> int:
        """Process table knowledge graphs if table nodes are present."""
        try:
            from ..rag.graph.table_ingestion import extract_and_ingest_table_kg
            
            processed_tables = set()
            kg_count = 0
            
            for node in nodes:
                if node.get('type') in ['table_summary', 'table_record', 'table_column', 'table_note']:
                    table_id = node.get('table_id', 'unknown')
                    
                    # Only process each table once (use table_summary as the primary node)
                    if node.get('type') == 'table_summary' and table_id not in processed_tables:
                        structured_data = node.get('structured_data')
                        summary_text = (node.get('text') or "").strip()
                        
                        if structured_data:
                            kg_result = extract_and_ingest_table_kg(structured_data, doc_id, table_id, summary_text)
                            kg_count += kg_result.get('observations', 0)
                            self.logger.info(f"Ingested {kg_result.get('observations', 0)} table observations for doc_id={doc_id}, table_id={table_id}")
                            processed_tables.add(table_id)
            
            return kg_count
            
        except Exception as e:
            self.logger.error(f"Table KG processing failed: {e}")
            return 0
