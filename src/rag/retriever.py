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


def _log_hits(prefix: str, hits: list, limit: int = 5):
    """Log a compact list of hits with score, doc_id, page, and id."""
    try:
        if not hits:
            logger.info(f"{prefix} hits: 0")
            return
        lines = []
        for idx, h in enumerate(hits[:limit]):
            try:
                pid = getattr(h, 'id', 'unknown')
                score = getattr(h, 'score', 0.0)
                payload = getattr(h, 'payload', {}) or {}
                doc_id = payload.get('doc_id', '')
                page = payload.get('page', '')
                ctype = payload.get('type', '')
                lines.append(f"{idx+1}. score={score:.3f} doc_id='{doc_id}' page={page} type='{ctype}' id={pid}")
            except Exception as e:
                lines.append(f"{idx+1}. <error formatting hit: {e}>")
        more = f" (+{len(hits)-limit} more)" if len(hits) > limit else ""
        logger.info(f"{prefix} hits: {len(hits)}\n  " + "\n  ".join(lines) + more)
    except Exception:
        # Best-effort logging only
        pass

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


def parse_constraints_for_text(text: str):
    """Parse a free-form query into (cleaned_query, constraints) using the global parser.

    Exposed for upstream callers (e.g., chat route) to decide gating based on constraints
    such as explicit doc_id or page references.
    """
    parser = _get_query_parser()
    return parser.parse_query(text)

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



def retrieve(
    query: str,
    top_k: int = 8,
    score_threshold: float = 0.3,
    constraints_override: dict | None = None,
    search_query_override: str | None = None,
):
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
    # Merge any caller-provided constraints (e.g., doc_ids parsed from original user input)
    if constraints_override:
        try:
            # Merge doc_ids
            if 'doc_ids' in constraints_override and constraints_override['doc_ids']:
                base = set(constraints.get('doc_ids', []) or [])
                base.update(constraints_override['doc_ids'])
                constraints['doc_ids'] = list(base)
            # Merge pages
            if 'pages' in constraints_override and constraints_override['pages']:
                base = set(constraints.get('pages', []) or [])
                base.update(constraints_override['pages'])
                constraints['pages'] = list(base)
            # Merge content_types
            if 'content_types' in constraints_override and constraints_override['content_types']:
                base = set(constraints.get('content_types', []) or [])
                base.update(constraints_override['content_types'])
                constraints['content_types'] = list(base)
        except Exception as e:
            logger.warning(f"Failed to merge constraints_override: {e}")
    
    logger.info(f"Original query: {query}")
    logger.info(f"Cleaned query: {cleaned_query}")
    logger.info(f"Extracted constraints: {constraints}")
    if constraints_override:
        logger.info(f"Constraints override provided: {constraints_override}")
    
    # Centroid-based low-signal gate, but DO NOT gate when explicit constraints exist
    search_basis = cleaned_query if cleaned_query.strip() else query
    has_explicit_constraints = bool(
        (constraints.get('doc_ids') and len(constraints['doc_ids']) > 0) or
        (constraints.get('pages') and len(constraints['pages']) > 0) or
        (constraints.get('content_types') and len(constraints['content_types']) > 0)
    )
    # Removed trivial guard: always attempt retrieval; rely on thresholds and overrides

    # Use cleaned query for embedding if we have semantic content, else use original
    search_query = search_query_override or search_basis
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
    
    # Content type constraints (treat as soft preference; do not hard-filter)
    prefer_content_types = constraints.get('content_types') or []
    
    # Create filter object if we have constraints
    constraint_filter = Filter(must=must_filters) if must_filters else None

    # -------------------- Dense search --------------------
    initial_limit = min(max(top_k * 3, 10), 50)  # Slightly higher floor for more chances
    
    # Try with constraints first
    dense_hits = []
    try:
        # If we have doc_id constraints, be more permissive with threshold
        effective_threshold = min(score_threshold, 0.25) if doc_id_constraints_applied else score_threshold
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_vec,
            query_filter=constraint_filter,
            limit=initial_limit,
            score_threshold=effective_threshold,
        )
        dense_hits = response.points if hasattr(response, 'points') else []
        _log_hits("Dense (constrained)", dense_hits)
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
            dense_hits = [h for h in dense_hits if h.score >= (min(score_threshold, 0.25) if doc_id_constraints_applied else score_threshold)]
            _log_hits("Dense (manual filter)", dense_hits)
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
                    _log_hits("Dense (no constraints)", dense_hits)
                except Exception as e3:
                    logger.warning(f"Score threshold not supported in fallback: {e3}")
                    response = client.query_points(
                        collection_name=QDRANT_COLLECTION,
                        query=q_vec,
                        limit=initial_limit,
                    )
                    dense_hits = response.points if hasattr(response, 'points') else []
                    dense_hits = [h for h in dense_hits if h.score >= score_threshold]
                    _log_hits("Dense (no constraints, manual filter)", dense_hits)
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
        # Build keyword filter (with doc_id if present, but without hard content_type filtering)
        keyword_must = [FieldCondition(key="text", match=MatchText(text=search_query))]
        # Only include doc_id/page filters
        if doc_id_constraints_applied:
            if len(cleaned_doc_ids) == 1:
                keyword_must.append(FieldCondition(key="doc_id", match=MatchValue(value=cleaned_doc_ids[0])))
            else:
                keyword_must.append(FieldCondition(key="doc_id", match=MatchAny(any=cleaned_doc_ids)))
        if pages := constraints.get('pages'):
            if len(pages) == 1:
                keyword_must.append(FieldCondition(key="page", match=MatchValue(value=pages[0])))
            else:
                keyword_must.append(FieldCondition(key="page", match=MatchAny(any=pages)))

        dummy_vector = [0.0] * len(q_vec)
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=dummy_vector,
            query_filter=Filter(must=keyword_must),
            limit=initial_limit,
        )
        keyword_hits = response.points if hasattr(response, 'points') else []
        _log_hits("Keyword", keyword_hits)
    except Exception as e:
        logger.warning(f"Keyword search failed (text index missing?): {e}")
        keyword_hits = []

    # If still nothing and doc_id constraints applied, scroll a few points from the doc to ensure some context
    scroll_hits = []
    if doc_id_constraints_applied and not (dense_hits or keyword_hits):
        try:
            scroll_filter = FieldCondition(key="doc_id", match=MatchAny(any=cleaned_doc_ids)) if len(cleaned_doc_ids) > 1 else FieldCondition(key="doc_id", match=MatchValue(value=cleaned_doc_ids[0]))
            points, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(must=[scroll_filter]),
                with_payload=True,
                with_vectors=False,
                limit=max(top_k, 3),
            )
            scroll_hits = points or []
            if scroll_hits:
                logger.info(f"Scroll fallback returned {len(scroll_hits)} points for doc_ids={cleaned_doc_ids}")
                _log_hits("Scroll", scroll_hits)
        except Exception as e:
            logger.warning(f"Scroll fallback failed: {e}")

    # -------------------- Merge & deduplicate --------------------
    all_hits = {}
    for h in dense_hits + keyword_hits + scroll_hits:
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

    # High-confidence gating: relax behavior — if nothing meets threshold, return best 1-2 hits instead of empty
    if not final:
        logger.info("No final results after deduplication")
        return []
    top_score = max(getattr(h, 'score', 0.0) for h in final)
    if top_score < score_threshold:
        logger.info(f"Top score {top_score:.3f} below threshold {score_threshold:.3f}; relaxing to return top-1")
        final = sorted(final, key=lambda x: x.score, reverse=True)[:1]

    # Soft prefer content types by promoting matching hits to the front
    if prefer_content_types:
        def type_rank(hit):
            t = hit.payload.get('type')
            return 0 if t in prefer_content_types else 1
        final = sorted(final, key=type_rank)

    constraint_info = f" [constraints: {constraints}]" if constraints else ""
    logger.info(f"🔍 dense={len(dense_hits)} keyword={len(keyword_hits)} → {len(final)} unique hits; top_score={top_score:.3f}{constraint_info}")
    _log_hits("Final", final)
    
    return final
