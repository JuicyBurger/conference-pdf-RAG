# src/RAG/retriever.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import MatchText, FieldCondition, Filter, MatchValue, MatchAny
from ..config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from ..models.embedder import embed
from .query_parser import QueryParser
import logging

logger = logging.getLogger(__name__)

# Set up logging for retriever
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Global query parser instance
_query_parser = None

def bootstrap_jieba_from_qdrant():
    """Bootstrap jieba dictionary with all document IDs from Qdrant."""
    try:
        doc_ids = get_all_doc_ids()
        if doc_ids:
            # Update the global parser with all doc IDs
            global _query_parser
            _query_parser = QueryParser(doc_ids)
            logger.info(f"✅ Bootstrapped jieba with {len(doc_ids)} document IDs from Qdrant")
            return True
        else:
            logger.warning("No document IDs found in Qdrant for bootstrapping")
            return False
    except Exception as e:
        logger.error(f"Failed to bootstrap jieba from Qdrant: {e}")
        return False

def _get_query_parser():
    """Get or create the global query parser instance with known doc IDs."""
    global _query_parser
    if _query_parser is None:
        # Get all unique document IDs from Qdrant for fuzzy matching
        known_doc_ids = get_all_doc_ids()
        _query_parser = QueryParser(known_doc_ids)
    return _query_parser

def add_new_doc_id_to_parser(doc_id: str):
    """Add a new document ID to the global query parser."""
    parser = _get_query_parser()
    return parser.add_new_doc_id(doc_id)

def get_all_doc_ids():
    """Retrieve all unique document IDs from the Qdrant collection."""
    try:
        # Use scroll to get all points and extract unique doc_ids
        doc_ids = set()
        limit = 1000
        offset = None
        
        while True:
            response = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=limit,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False
            )
            
            points, offset = response
            if not points:
                break
                
            for point in points:
                if point.payload and "doc_id" in point.payload:
                    doc_ids.add(point.payload["doc_id"])
            
            if offset is None:
                break
        
        logger.info(f"Found {len(doc_ids)} unique document IDs in Qdrant")
        return list(doc_ids)
        
    except Exception as e:
        logger.warning(f"Failed to retrieve doc IDs from Qdrant: {e}")
        return []



def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3):
    """
    Retrieve relevant documents with deduplication, score filtering, and constraint support.
    
    Args:
        query: Search query (natural language with potential constraints)
        top_k: Number of results to return
        score_threshold: Minimum similarity score threshold
        
    Returns:
        List of deduplicated, high-quality hits
    """
    # -------------------- Parse query for constraints --------------------
    parser = _get_query_parser()
    cleaned_query, constraints = parser.parse_query(query)
    
    logger.info(f"Original query: {query}")
    logger.info(f"Cleaned query: {cleaned_query}")
    logger.info(f"Extracted constraints: {constraints}")
    
    # Use cleaned query for embedding if we have semantic content, else use original
    search_query = cleaned_query if cleaned_query.strip() else query
    q_vec = embed(search_query)

    # -------------------- Build constraint filters --------------------
    must_filters = []
    
    # Document ID constraints (support multiple docs)
    doc_id_constraints_applied = False
    if doc_ids := constraints.get('doc_ids'):
        # Clean and validate doc_ids
        cleaned_doc_ids = []
        for doc_id in doc_ids:
            # Remove any trailing punctuation or extra text
            cleaned_id = doc_id.strip().rstrip(',.!?;:')
            if cleaned_id and len(cleaned_id) > 0:
                cleaned_doc_ids.append(cleaned_id)
        
        if cleaned_doc_ids:
            logger.info(f"Using document ID constraints: {cleaned_doc_ids}")
            if len(cleaned_doc_ids) == 1:
                must_filters.append(FieldCondition(key="doc_id", match=MatchValue(value=cleaned_doc_ids[0])))
            else:
                must_filters.append(FieldCondition(key="doc_id", match=MatchAny(any=cleaned_doc_ids)))
            doc_id_constraints_applied = True
        else:
            logger.warning("No valid document IDs found in constraints")
    
    # Page number constraints
    if pages := constraints.get('pages'):
        if len(pages) == 1:
            must_filters.append(FieldCondition(key="page", match=MatchValue(value=pages[0])))
        else:
            must_filters.append(FieldCondition(key="page", match=MatchAny(any=pages)))
    
    # Content type constraints
    if content_types := constraints.get('content_types'):
        if len(content_types) == 1:
            must_filters.append(FieldCondition(key="type", match=MatchValue(value=content_types[0])))
        else:
            must_filters.append(FieldCondition(key="type", match=MatchAny(any=content_types)))
    
    # Create filter object if we have constraints
    constraint_filter = Filter(must=must_filters) if must_filters else None

    # -------------------- Dense search --------------------
    initial_limit = min(top_k * 3, 50)  # Get 3x more results for filtering
    
    # Try with constraints first
    dense_hits = []
    try:
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_vec,
            query_filter=constraint_filter,
            limit=initial_limit,
            score_threshold=score_threshold
        )
        dense_hits = response.points if hasattr(response, 'points') else []
    except Exception as e:
        logger.warning(f"Score threshold not supported, using manual filtering: {e}")
        try:
            response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q_vec,
                query_filter=constraint_filter,
                limit=initial_limit,
            )
            dense_hits = response.points if hasattr(response, 'points') else []
            dense_hits = [h for h in dense_hits if h.score >= score_threshold]
        except Exception as e2:
            logger.warning(f"Constrained search failed: {e2}")
            # Graceful fallback: try without constraints
            if constraint_filter:
                logger.info("Retrying dense search without constraints...")
                try:
                    response = client.query_points(
                        collection_name=QDRANT_COLLECTION,
                        query=q_vec,
                        limit=initial_limit,
                        score_threshold=score_threshold
                    )
                    dense_hits = response.points if hasattr(response, 'points') else []
                except Exception as e3:
                    logger.warning(f"Score threshold not supported in fallback: {e3}")
                    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_vec,
                        limit=initial_limit,
                    )
                    dense_hits = response.points if hasattr(response, 'points') else []
                    dense_hits = [h for h in dense_hits if h.score >= score_threshold]
            else:
                dense_hits = []

    # -------------------- Graceful fallback for doc_id constraints --------------------
    # If we applied doc_id constraints but got no results, try without them
    if doc_id_constraints_applied and not dense_hits:
        logger.info("No results with doc_id constraints, trying without them...")
        try:
            # Remove doc_id constraints but keep other constraints
            fallback_filters = [f for f in must_filters if not isinstance(f, FieldCondition) or f.key != "doc_id"]
            fallback_filter = Filter(must=fallback_filters) if fallback_filters else None
            
            response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q_vec,
                query_filter=fallback_filter,
                limit=initial_limit,
                score_threshold=score_threshold
            )
            fallback_hits = response.points if hasattr(response, 'points') else []
            if fallback_hits:
                logger.info(f"Found {len(fallback_hits)} results without doc_id constraints")
                dense_hits = fallback_hits
        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")

    # -------------------- Keyword / full-text search --------------------
    keyword_hits = []
    try:
        # Combine text search with constraint filters
        keyword_filter_conditions = [FieldCondition(key="text", match=MatchText(text=search_query))]
        keyword_filter_conditions.extend(must_filters)
        
        # For keyword search, we need a dummy vector since query_points requires it
        dummy_vector = [0.0] * len(q_vec)  # Same dimension as query vector
        
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=dummy_vector,
            query_filter=Filter(must=keyword_filter_conditions),
            limit=initial_limit,
        )
        keyword_hits = response.points if hasattr(response, 'points') else []
    except Exception as e:
        logger.warning(f"Keyword search failed (text index missing?): {e}")
        # Graceful fallback: try keyword search without constraints
        if must_filters:
            logger.info("Retrying keyword search without constraints...")
            try:
                response = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=dummy_vector,
                    query_filter=Filter(must=[FieldCondition(key="text", match=MatchText(text=search_query))]),
                    limit=initial_limit,
                )
                keyword_hits = response.points if hasattr(response, 'points') else []
            except Exception as e2:
                logger.warning(f"Keyword search fallback also failed: {e2}")
                keyword_hits = []
        else:
            keyword_hits = []

    # -------------------- Merge & deduplicate --------------------
    all_hits = {}
    for h in dense_hits + keyword_hits:
        # Use best score if duplicate
        if (prev := all_hits.get(h.id)) is None or h.score > prev.score:
            all_hits[h.id] = h

    # Sort by score desc and keep top_k unique (dedup by doc_id+page)
    seen_sources = set()
    final = []
    for hit in sorted(all_hits.values(), key=lambda x: x.score, reverse=True):
        doc_id = hit.payload.get("doc_id", "unknown")
        page   = hit.payload.get("page", "unknown")
        key    = f"{doc_id}:{page}"
        if key in seen_sources:
            continue
        seen_sources.add(key)
        final.append(hit)
        if len(final) >= top_k:
            break

    constraint_info = f" [constraints: {constraints}]" if constraints else ""
    logger.info(f"🔍 dense={len(dense_hits)} keyword={len(keyword_hits)} → {len(final)} unique hits{constraint_info}")
    
    return final
