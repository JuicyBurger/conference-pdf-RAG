"""
Graph-based RAG engine implementation.

This module implements a RAG engine that uses graph traversal for retrieval.
It combines vector search with Neo4j graph traversal to find relevant context.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRAGEngine
from .models import RetrievalRequest, RetrievalResponse, Evidence
from ..retrieval.retrieval_service import retrieval_service
from ..graph.graph_store import ensure_graph_indexes, get_driver
from src.config import (
    GRAPH_MAX_HOPS, 
    GRAPH_EXPANSION_TOP_K, 
    NEO4J_DATABASE,
    GRAPH_ALLOWED_RELATION_TYPES
)

# Configure logging
logger = logging.getLogger(__name__)

class GraphRAGEngine(BaseRAGEngine):
    """Graph-based RAG engine using Neo4j.
    
    This engine combines vector search with Neo4j graph traversal to find
    relevant context for a query.
    """
    
    def __init__(self):
        """Initialize the graph RAG engine."""
        from .models import EngineType
        super().__init__(EngineType.GRAPH)
    
    def _initialize_engine(self):
        """Initialize graph engine-specific components."""
        try:
            ensure_graph_indexes()
            logger.info("Neo4j graph indexes ensured")
        except Exception as e:
            logger.warning(f"Failed to ensure Neo4j indexes: {e}")
    
    def discover_relationship_types(self) -> List[str]:
        """Discover all relationship types present in the graph.
        
        Returns:
            List of relationship type names found in the graph
        """
        try:
            cypher = """
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) AS rel_type
            ORDER BY rel_type
            """
            with get_driver().session(database=NEO4J_DATABASE) as ses:
                result = ses.run(cypher).data()
                rel_types = [row['rel_type'] for row in result]
                logger.info(f"Discovered {len(rel_types)} relationship types in graph: {rel_types}")
                return rel_types
        except Exception as e:
            logger.warning(f"Failed to discover relationship types: {e}")
            return []
    
    def get_optimal_relationship_types(self, query: str = None) -> List[str]:
        """Get optimal relationship types for traversal based on query and graph content.
        
        Args:
            query: Optional query to help determine relevant relationship types
            
        Returns:
            List of relationship types to use for traversal
        """
        # Start with configured allowed types
        allowed_types = set(GRAPH_ALLOWED_RELATION_TYPES)
        
        # Discover actual types in the graph
        discovered_types = set(self.discover_relationship_types())
        
        # Filter to only use types that exist in the graph
        available_types = allowed_types.intersection(discovered_types)
        
        # If no configured types are available, use all discovered types
        if not available_types and discovered_types:
            available_types = discovered_types
            logger.info(f"No configured relationship types found in graph, using all discovered types: {list(available_types)}")
        
        # If still no types, fall back to a minimal set
        if not available_types:
            fallback_types = ["HAS_CHUNK", "HAS_DOCUMENT", "RELATED_TO"]
            available_types = set(fallback_types).intersection(discovered_types)
            if available_types:
                logger.warning(f"Using fallback relationship types: {list(available_types)}")
        
        # Convert to list and sort for consistency
        result = sorted(list(available_types))
        logger.info(f"Using {len(result)} relationship types for traversal: {result}")
        return result
    
    def expand_from_vector_seeds(self, seed_hits, top_k=8, hops=2, allowed_rels=None, query: str = None):
        """Expand from vector search results using Neo4j k-hop traversal.
        
        Args:
            seed_hits: Vector search results from Qdrant
            top_k: Maximum number of results to return
            hops: Number of hops for graph traversal
            allowed_rels: List of allowed relationship types (if None, will be discovered)
            query: Optional query to help determine relevant relationship types
            
        Returns:
            List of expanded chunks with bonus scores
        """
        seed_ids = [h.payload.get("id") or h.id for h in seed_hits if h]  # ensure you pass your Chunk.id
        if not seed_ids:
            logger.warning("No seed IDs found in vector hits")
            return []
        
        logger.info(f"🔍 Debug: Starting expansion with {len(seed_ids)} seed IDs: {seed_ids[:3]}...")
        
        # Get optimal relationship types
        if allowed_rels is None:
            allowed_rels = self.get_optimal_relationship_types(query)
        
        if not allowed_rels:
            logger.warning("No relationship types available for traversal")
            return []
        
        logger.info(f"🔍 Debug: Using {len(allowed_rels)} relationship types: {allowed_rels}")
        
        # First, let's check if the seed chunks exist in Neo4j
        try:
            with get_driver().session(database=NEO4J_DATABASE) as ses:
                # Check if seed chunks exist
                check_cypher = """
                UNWIND $seed_ids AS sid
                MATCH (c:Chunk {id: sid})
                RETURN c.id AS chunk_id, c.text AS text_preview
                LIMIT 5
                """
                existing_chunks = ses.run(check_cypher, {"seed_ids": seed_ids}).data()
                logger.info(f"🔍 Debug: Found {len(existing_chunks)} seed chunks in Neo4j: {existing_chunks}")
                
                if not existing_chunks:
                    logger.warning("🔍 Debug: No seed chunks found in Neo4j - this is the problem!")
                    
                    # Try alternative approach: look for chunks by doc_id and page instead of ID
                    logger.info("🔍 Debug: Trying alternative lookup by doc_id and page...")
                    alt_check_cypher = """
                    UNWIND $seed_hits AS hit
                    MATCH (c:Chunk {doc_id: hit.doc_id, page: hit.page})
                    WHERE c.text CONTAINS hit.text_preview
                    RETURN c.id AS chunk_id, c.text AS text_preview
                    LIMIT 5
                    """
                    
                    # Prepare seed hits with doc_id, page, and text preview
                    seed_hit_data = []
                    for hit in seed_hits[:3]:  # Try first 3 hits
                        payload = hit.payload if hasattr(hit, 'payload') else {}
                        seed_hit_data.append({
                            'doc_id': payload.get('doc_id', ''),
                            'page': payload.get('page', ''),
                            'text_preview': payload.get('text', '')[:100] if payload.get('text') else ''
                        })
                    
                    if seed_hit_data:
                        alt_existing_chunks = ses.run(alt_check_cypher, {"seed_hits": seed_hit_data}).data()
                        logger.info(f"🔍 Debug: Alternative lookup found {len(alt_existing_chunks)} chunks: {alt_existing_chunks}")
                        
                        if alt_existing_chunks:
                            # Use the found chunk IDs instead
                            seed_ids = [chunk['chunk_id'] for chunk in alt_existing_chunks]
                            logger.info(f"🔍 Debug: Using alternative chunk IDs: {seed_ids}")
                        else:
                            logger.warning("🔍 Debug: No chunks found even with alternative lookup")
                            
                            # Special handling for table embeddings
                            # Check if any of the seed hits are table embeddings
                            table_embeddings = []
                            for hit in seed_hits:
                                payload = hit.payload if hasattr(hit, 'payload') else {}
                                if payload.get('level') in ['row', 'table'] or 'table' in payload.get('type', ''):
                                    table_embeddings.append(hit)
                            
                            if table_embeddings:
                                logger.info(f"🔍 Debug: Found {len(table_embeddings)} table embeddings, creating temporary chunks")
                                # Create temporary chunk nodes for table embeddings
                                temp_chunk_ids = []
                                for hit in table_embeddings[:3]:  # Limit to 3
                                    payload = hit.payload if hasattr(hit, 'payload') else {}
                                    temp_chunk_id = f"temp_{payload.get('doc_id', '')}_{payload.get('table_id', '')}_{payload.get('level', '')}"
                                    
                                    # Create temporary chunk node
                                    temp_cypher = """
                                    MERGE (c:Chunk {id: $chunk_id})
                                    ON CREATE SET 
                                        c.text = $text,
                                        c.doc_id = $doc_id,
                                        c.page = $page,
                                        c.table_id = $table_id,
                                        c.type = $level,
                                        c.temp = true
                                    """
                                    ses.run(temp_cypher, {
                                        'chunk_id': temp_chunk_id,
                                        'text': payload.get('text', '')[:200],
                                        'doc_id': payload.get('doc_id', ''),
                                        'page': payload.get('page', ''),
                                        'table_id': payload.get('table_id', ''),
                                        'level': payload.get('level', 'table')
                                    })
                                    temp_chunk_ids.append(temp_chunk_id)
                                
                                if temp_chunk_ids:
                                    seed_ids = temp_chunk_ids
                                    logger.info(f"🔍 Debug: Created temporary chunk IDs: {temp_chunk_ids}")
                                else:
                                    return []
                            else:
                                return []
                
                # Now check what relationships exist from these chunks
                rel_check_cypher = """
                UNWIND $seed_ids AS sid
                MATCH (c:Chunk {id: sid})-[r]->(n)
                RETURN DISTINCT type(r) AS rel_type, count(*) AS count
                ORDER BY count DESC
                LIMIT 10
                """
                rel_counts = ses.run(rel_check_cypher, {"seed_ids": seed_ids}).data()
                logger.info(f"🔍 Debug: Relationship types from seed chunks: {rel_counts}")
                
                # Check if any of our allowed relationships exist from these chunks
                allowed_rel_check_cypher = """
                UNWIND $seed_ids AS sid
                MATCH (c:Chunk {id: sid})-[r]->(n)
                WHERE type(r) IN $allowedRels
                RETURN DISTINCT type(r) AS rel_type, count(*) AS count
                ORDER BY count DESC
                """
                allowed_rel_counts = ses.run(allowed_rel_check_cypher, {
                    "seed_ids": seed_ids,
                    "allowedRels": allowed_rels
                }).data()
                logger.info(f"🔍 Debug: Allowed relationship types from seed chunks: {allowed_rel_counts}")
                
                if not allowed_rel_counts:
                    logger.warning("🔍 Debug: No allowed relationships found from seed chunks!")
                    # Let's try with ALL relationships to see if there are any connections
                    all_rel_check_cypher = """
                    UNWIND $seed_ids AS sid
                    MATCH (c:Chunk {id: sid})-[r*1..2]-(n)
                    RETURN DISTINCT type(r[0]) AS rel_type, count(*) AS count
                    ORDER BY count DESC
                    LIMIT 10
                    """
                    all_rel_counts = ses.run(all_rel_check_cypher, {"seed_ids": seed_ids}).data()
                    logger.info(f"🔍 Debug: ALL relationship types from seed chunks (2 hops): {all_rel_counts}")
                    
                    # If we have relationships but none are in our allowed list, let's use ALL relationships
                    if all_rel_counts:
                        logger.info("🔍 Debug: Found relationships but none in allowed list. Using ALL relationships.")
                        # Extract all relationship types found
                        all_rel_types = [rel['rel_type'] for rel in all_rel_counts]
                        logger.info(f"🔍 Debug: Using ALL discovered relationship types: {all_rel_types}")
                        allowed_rels = all_rel_types
                    else:
                        logger.warning("🔍 Debug: No relationships found at all from seed chunks")
                        return []
                
        except Exception as e:
            logger.error(f"🔍 Debug: Error checking seed chunks: {e}")
            return []
        
        # Define relationship importance weights for scoring
        important_rels = {
            "ANSWERS": 0.3,
            "REPORTED": 0.3, 
            "FOR_PERIOD": 0.3,
            "SUBSIDIARY_OF": 0.2,
            "OPERATES_IN": 0.2,
            "IN_CATEGORY": 0.2,
            "BELONGS_TO": 0.2,
            "CONTAINS": 0.1,
            "HAS_CHUNK": 0.05,  # Structural relationships get lower weight
            "HAS_DOCUMENT": 0.05
        }
        
        # Build the Cypher query with literal hops value
        cypher = f"""
        UNWIND $seed_ids AS sid
        MATCH (c:Chunk {{id: sid}})
        WITH DISTINCT c

        MATCH p=(c)-[rels*1..{hops}]-(nbr)
        WHERE all(r IN rels WHERE type(r) IN $allowedRels)

        WITH c, nbr, rels
        // pick a chunk to return
        WITH CASE WHEN 'Chunk' IN labels(nbr) THEN nbr ELSE c END AS chunkNode,
             // Calculate path bonus based on relationship types
             reduce(score = 0.0, r IN rels | 
               score + CASE type(r)
                 WHEN 'ANSWERS' THEN 0.3
                 WHEN 'REPORTED' THEN 0.3
                 WHEN 'FOR_PERIOD' THEN 0.3
                 WHEN 'SUBSIDIARY_OF' THEN 0.2
                 WHEN 'OPERATES_IN' THEN 0.2
                 WHEN 'IN_CATEGORY' THEN 0.2
                 WHEN 'BELONGS_TO' THEN 0.2
                 WHEN 'CONTAINS' THEN 0.1
                 WHEN 'HAS_CHUNK' THEN 0.05
                 WHEN 'HAS_DOCUMENT' THEN 0.05
                 ELSE 0.1  // Default weight for other relationships
               END
             ) AS pathBonus

        RETURN chunkNode{{.*, id: chunkNode.id}} AS chunk,
               max(pathBonus) AS bonus
        ORDER BY bonus DESC
        LIMIT $top_k;
        """
        
        logger.info(f"🔍 Debug: Executing Cypher query with {len(seed_ids)} seed IDs and {len(allowed_rels)} relationship types")
        
        try:
            with get_driver().session(database=NEO4J_DATABASE) as ses:
                rows = ses.run(cypher, {
                    "seed_ids": seed_ids,
                    "allowedRels": allowed_rels, 
                    "top_k": top_k
                }).data()
            
            logger.info(f"🔍 Debug: Cypher query returned {len(rows)} results")
            if rows:
                logger.info(f"🔍 Debug: First result: {rows[0]}")
            
            return rows
        except Exception as e:
            logger.warning(f"Neo4j expansion failed: {e}")
            return []
    
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve context using simple hybrid: Vector seed (Qdrant) → Neo4j k-hop traverse.
        
        Args:
            request: Standardized retrieval request
            
        Returns:
            Standardized retrieval response
        """
        # Extract parameters with defaults
        top_k = request.top_k
        score_threshold = request.score_threshold
        max_hops = request.max_hops or GRAPH_MAX_HOPS
        expansion_top_k = request.expansion_top_k or GRAPH_EXPANSION_TOP_K
        
        logger.info(f"Graph retrieval: query='{request.query[:50]}...', max_hops={max_hops}, top_k={top_k}")
        
        # 1) Retrieve vector seeds from Qdrant
        vector_hits = retrieval_service.retrieve(
            query=request.query,
            top_k=top_k,
            score_threshold=score_threshold,
            room_id=None,  # Don't scope by room for graph retrieval
            prefer_chat_scope=False,
            scope_filter="graph",
        )
        
        # Debug: Check vector hits structure
        logger.info(f"🔍 Debug: Vector hits structure check:")
        for i, hit in enumerate(vector_hits[:3]):  # Check first 3 hits
            logger.info(f"🔍 Debug: Hit {i}: id={getattr(hit, 'id', 'NO_ID')}, "
                       f"payload_id={hit.payload.get('id', 'NO_PAYLOAD_ID') if hasattr(hit, 'payload') else 'NO_PAYLOAD'}, "
                       f"payload_keys={list(hit.payload.keys()) if hasattr(hit, 'payload') else 'NO_PAYLOAD'}")
        
        # Convert vector hits to Evidence objects
        evidence = [Evidence(h, self.engine_type) for h in vector_hits]
        logger.info(f"Vector search returned {len(evidence)} graph-scoped results")
        
        # 2) Expand from vector seeds using Neo4j k-hop traversal
        neo4j_count = 0
        try:
            logger.info("🕸️ Starting Neo4j k-hop expansion from vector seeds")
            
            # Expand from vector seeds with dynamic relationship discovery
            expanded_chunks = self.expand_from_vector_seeds(
                seed_hits=vector_hits,
                top_k=expansion_top_k,
                hops=max_hops,
                allowed_rels=None,  # Let the method discover optimal types
                query=request.query
            )
            
            # Convert expanded chunks to Evidence objects
            for chunk_data in expanded_chunks:
                chunk = chunk_data.get('chunk', {})
                bonus = chunk_data.get('bonus', 0.0)
                
                # Create a hit-like object for the expanded chunk
                hit = type("Hit", (), {
                    'id': chunk.get('id', ''),
                    'score': bonus,  # Use the path bonus as score
                        'payload': {
                        'doc_id': chunk.get('doc_id', ''),
                        'page': chunk.get('page', ''),
                        'text': chunk.get('text', ''),
                        'type': 'graph_chunk',
                        'source': 'neo4j_expansion',
                        'corpus_id': chunk.get('corpus_id', ''),
                        'name': chunk.get('name', '')
                        }
                    })()
                
                evidence.append(Evidence(hit, self.engine_type))
                neo4j_count += 1
            
            logger.info(f"🧩 Neo4j expansion added {neo4j_count} graph-derived hits")
            
        except Exception as e:
            # Best effort: ignore traversal errors and return vector-only evidence
            logger.warning(f"⚠️ Neo4j expansion failed: {e}")
        
        return RetrievalResponse(
            evidence=evidence,
            engine_type=self.engine_type,
            query=request.query,
            metadata={
                'neo4j_count': neo4j_count,
                'vector_count': len(vector_hits)
            }
        )