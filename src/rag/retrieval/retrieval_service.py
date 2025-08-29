"""
Unified retrieval service for RAG components.

This module provides a standardized retrieval interface that can be used
by different RAG engines to retrieve relevant context.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Set
import logging
from dataclasses import dataclass, field
import math

# New: BM25 + cross-encoder rerank imports
import jieba
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils import handle_errors, RetrievalError, DatabaseError, setup_logger

# Configure logging
logger = setup_logger(__name__)

@dataclass
class RetrievalRequest:
    """Standard retrieval request with unified parameters."""
    query: str
    top_k: int = 5
    score_threshold: float = 0.3
    filters: Optional[Dict[str, Any]] = None
    room_id: Optional[str] = None
    scope: Optional[str] = None
    constraints_override: Optional[Dict[str, Any]] = None
    search_query_override: Optional[str] = None
    prefer_chat_scope: bool = True
    scope_filter: Optional[str] = None
    extra_filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize request parameters."""
        if not self.query or not isinstance(self.query, str):
            raise ValueError("Query must be a non-empty string")
        
        self.top_k = max(1, min(50, self.top_k))  # Limit between 1-50
        self.score_threshold = max(0.0, min(1.0, self.score_threshold))  # Limit between 0-1
        
        # Initialize filters if None
        if self.filters is None:
            self.filters = {}

@dataclass
class RetrievalResult:
    """Standard retrieval result with unified structure."""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    raw: Any = None  # Original hit object for backward compatibility
    
    @classmethod
    def from_qdrant_hit(cls, hit):
        """Convert a Qdrant hit to standardized result."""
        content = hit.payload.get('text', '') or hit.payload.get('content', '')
        return cls(
            id=getattr(hit, 'id', 'unknown'),
            score=getattr(hit, 'score', 0.0),
            content=content,
            metadata={
                'doc_id': hit.payload.get('doc_id', 'unknown'),
                'page': hit.payload.get('page', 'unknown'),
                'type': hit.payload.get('type', 'unknown'),
            },
            raw=hit  # Store the original hit for backward compatibility
        )

class RetrievalService:
    """Unified retrieval service for vector and graph-based retrieval."""
    
    def __init__(self):
        """Initialize retrieval service with dependencies."""
        self._initialize_dependencies()
    
    @handle_errors(error_class=DatabaseError, reraise=True)
    def _initialize_dependencies(self):
        """Initialize required dependencies."""
        # Configure LlamaIndex to use our local models first (before any LlamaIndex components are used)
        try:
            from src.models.embedder import configure_llamaindex_for_local_models
            configure_llamaindex_for_local_models()
            logger.info("✅ Configured LlamaIndex to use local models")
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure LlamaIndex: {e}")
        
        # Import embedder
        from src.models.embedder import embed
        self.embed = embed
        
        # Import Qdrant client
        from qdrant_client import QdrantClient
        from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection = QDRANT_COLLECTION
        
        # Initialize query parser
        from ..utils import QueryParser
        self._query_parser = None
        
        logger.info("Initialized retrieval service dependencies")

        # Initialize cross-encoder placeholders (lazy load on first use)
        self._ce_tokenizer = None
        self._ce_model = None
    
    def _log_hits(self, prefix: str, hits: list, limit: int = 5):
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
    
    def bootstrap_jieba_from_qdrant(self):
        """Bootstrap jieba dictionary with all document IDs from Qdrant."""
        try:
            doc_ids = self.get_all_doc_ids()
            if doc_ids:
                # Initialize the query parser with all doc IDs
                from ..utils import QueryParser
                self._query_parser = QueryParser(doc_ids)
                logger.info(f"✅ Bootstrapped jieba with {len(doc_ids)} document IDs from Qdrant")
                return True
            else:
                logger.warning("No document IDs found in Qdrant for bootstrapping")
                return False
        except Exception as e:
            logger.error(f"Failed to bootstrap jieba from Qdrant: {e}")
            return False
    
    def _get_query_parser(self):
        """Get or create the query parser instance with known doc IDs."""
        if self._query_parser is None:
            # Get all unique document IDs from Qdrant for fuzzy matching
            from ..utils import QueryParser
            known_doc_ids = self.get_all_doc_ids()
            self._query_parser = QueryParser(known_doc_ids)
        return self._query_parser
    
    def parse_constraints_for_text(self, text: str):
        """Parse a free-form query into (cleaned_query, constraints) using the parser.
        
        Exposed for upstream callers (e.g., chat route) to decide gating based on constraints
        such as explicit doc_id or page references.
        """
        parser = self._get_query_parser()
        return parser.parse_query(text)
    
    def add_new_doc_id_to_parser(self, doc_id: str):
        """Add a new document ID to the query parser."""
        parser = self._get_query_parser()
        return parser.add_new_doc_id(doc_id)
    
    @handle_errors(error_class=DatabaseError, fallback_return=[])
    def get_all_doc_ids(self):
        """Retrieve all unique document IDs from the Qdrant collection."""
        # Use scroll to get all points and extract unique doc_ids
        doc_ids = set()
        limit = 1000
        offset = None
        
        while True:
            response = self.client.scroll(
                collection_name=self.collection,
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
    
    def retrieve(self, query: str, **kwargs) -> List[Any]:
        """Legacy retrieve function for backward compatibility.
        
        This function maintains the same interface as the original retrieve function
        but uses the new standardized retrieval service under the hood.
        """
        # Increase top_k for table queries
        default_top_k = kwargs.get('top_k', 8)
        
        # Check if this is a table query by looking for table-related keywords
        is_table_query = any(keyword in query.lower() for keyword in ['table', '表格', 'show me', 'complete', 'display', 'find', 'contents'])
        if is_table_query:
            # Increase limit for table queries
            adjusted_top_k = min(default_top_k * 3, 30)  # Up to 30 for table queries
            kwargs['top_k'] = adjusted_top_k
        
        # Create a retrieval request from the parameters
        request = RetrievalRequest(
            query=query,
            top_k=kwargs.get('top_k', 8),
            score_threshold=kwargs.get('score_threshold', 0.3),
            room_id=kwargs.get('room_id'),
            scope=kwargs.get('scope_filter'),
            constraints_override=kwargs.get('constraints_override'),
            search_query_override=kwargs.get('search_query_override'),
            prefer_chat_scope=kwargs.get('prefer_chat_scope', True),
            extra_filters=kwargs.get('extra_filters')
        )
        
        # Perform retrieval using the new standardized service
        results = self.retrieve_with_request(request)
        
        # Return the raw hits for backward compatibility
        return [result.raw for result in results]
    
    def retrieve_with_request(self, request: RetrievalRequest) -> List[RetrievalResult]:
        """Perform retrieval based on request parameters.
        
        Args:
            request: Retrieval request with parameters
            
        Returns:
            List of retrieval results
        """
        try:
            # Parse query for constraints
            parser = self._get_query_parser()
            cleaned_query, constraints = parser.parse_query(request.query)
            
            # Merge any caller-provided constraints
            if request.constraints_override:
                try:
                    # Merge doc_ids
                    if 'doc_ids' in request.constraints_override and request.constraints_override['doc_ids']:
                        base = set(constraints.get('doc_ids', []) or [])
                        base.update(request.constraints_override['doc_ids'])
                        constraints['doc_ids'] = list(base)
                    # Merge pages
                    if 'pages' in request.constraints_override and request.constraints_override['pages']:
                        base = set(constraints.get('pages', []) or [])
                        base.update(request.constraints_override['pages'])
                        constraints['pages'] = list(base)
                    # Merge content_types
                    if 'content_types' in request.constraints_override and request.constraints_override['content_types']:
                        base = set(constraints.get('content_types', []) or [])
                        base.update(request.constraints_override['content_types'])
                        constraints['content_types'] = list(base)
                except Exception as e:
                    logger.warning(f"Failed to merge constraints_override: {e}")
            
            logger.info(f"Original query: {request.query}")
            logger.info(f"Cleaned query: {cleaned_query}")
            logger.info(f"Extracted constraints: {constraints}")
            
            # Use cleaned query for embedding if we have semantic content, else use original
            search_query = request.search_query_override or (cleaned_query if cleaned_query.strip() else request.query)
            query_vector = self.embed(search_query)
            
            # Build filters
            filters = self._build_filters(request, constraints)
            
            # Perform vector search
            dense_hits = self._vector_search(
                query_vector, 
                filters, 
                request.top_k, 
                request.score_threshold,
                constraints
            )
            
            # Perform keyword search if needed
            keyword_hits = self._keyword_search(
                search_query,
                filters,
                request.top_k,
                constraints
            )
            
            # Perform fallback search if needed
            scroll_hits = []
            if not (dense_hits or keyword_hits) and 'doc_id' in filters:
                scroll_hits = self._fallback_search(
                    filters.get('doc_id'),
                    request.top_k
                )
            
            # Compute BM25 over candidate pool (dense + keyword + scroll)
            candidate_pool = []
            seen_ids = set()
            for lst in (dense_hits or []), (keyword_hits or []), (scroll_hits or []):
                for h in lst:
                    hid = getattr(h, 'id', None)
                    if hid is None or hid in seen_ids:
                        continue
                    candidate_pool.append(h)
                    seen_ids.add(hid)

            bm25_scores = self._bm25_scores(search_query, candidate_pool) if candidate_pool else {}

            # Merge and deduplicate over a larger pool, then fuse and cut to top_k
            pool_limit = max(len(candidate_pool), request.top_k * 5 or 50)
            baseline_pool = self._merge_and_deduplicate(
                dense_hits,
                keyword_hits,
                scroll_hits,
                pool_limit,
                request.score_threshold,
                constraints.get('content_types', [])
            )

            # RRF reordering using ranks from dense/keyword and BM25 ranks, then trim
            rrf_rank_map = self._rrf_scores(dense_hits, keyword_hits, bm25_scores)
            if rrf_rank_map and baseline_pool:
                baseline_pool.sort(key=lambda h: rrf_rank_map.get(getattr(h, 'id', None), -math.inf), reverse=True)
            final_hits = baseline_pool[: request.top_k]

            # Optional: For table queries, expand to include all row hits for selected tables
            is_table_query = any(ct in ['table_row', 'table_summary'] for ct in constraints.get('content_types', []))
            has_table_like_hit = any(((getattr(h, 'payload', {}) or {}).get('type') in ['table_chunk', 'table', 'row'] or (getattr(h, 'payload', {}) or {}).get('level') in ['table', 'row']) for h in final_hits)
            if (is_table_query or has_table_like_hit) and final_hits:
                try:
                    final_hits = self._expand_table_rows(final_hits, max_rows=200)
                except Exception as e:
                    logger.warning(f"Failed to expand table rows: {e}")

            # Cross-encoder rerank the final list for precision
            try:
                final_hits = self._cross_encoder_rerank(search_query, final_hits, top_n=len(final_hits))
            except Exception as e:
                logger.warning(f"Cross-encoder rerank skipped due to error: {e}")
            
            # Convert to standard format
            return [RetrievalResult.from_qdrant_hit(hit) for hit in final_hits]
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def _bm25_scores(self, query: str, hits: List[Any]) -> Dict[Any, float]:
        """Compute BM25 scores over a candidate pool of hits.

        Returns mapping: hit.id -> bm25 score
        """
        try:
            docs = []
            ids = []
            for h in hits:
                payload = getattr(h, 'payload', {}) or {}
                text = payload.get('text') or payload.get('content') or ''
                ids.append(getattr(h, 'id', None))
                docs.append(list(jieba.cut(text)))
            if not docs:
                return {}
            bm25 = BM25Okapi(docs)
            q_tokens = list(jieba.cut(query))
            scores = bm25.get_scores(q_tokens)
            return {hid: float(sc) for hid, sc in zip(ids, scores) if hid is not None}
        except Exception as e:
            logger.debug(f"BM25 scoring failed: {e}")
            return {}

    def _rrf_scores(self, dense_hits: List[Any], keyword_hits: List[Any], bm25_scores: Dict[Any, float], k: int = 60) -> Dict[Any, float]:
        """Compute Reciprocal Rank Fusion scores from dense, keyword, and BM25.

        Returns mapping: hit.id -> rrf score (higher is better)
        """
        try:
            id_to_rrf: Dict[Any, float] = {}

            def add_rank_contrib(lst: List[Any]):
                if not lst:
                    return
                # Sort by score descending
                ordered = sorted(lst, key=lambda h: getattr(h, 'score', 0.0), reverse=True)
                for idx, h in enumerate(ordered, start=1):
                    hid = getattr(h, 'id', None)
                    if hid is None:
                        continue
                    id_to_rrf[hid] = id_to_rrf.get(hid, 0.0) + 1.0 / (k + idx)

            add_rank_contrib(dense_hits or [])
            add_rank_contrib(keyword_hits or [])

            # BM25 by rank
            if bm25_scores:
                ordered_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
                for idx, (hid, _) in enumerate(ordered_bm25, start=1):
                    id_to_rrf[hid] = id_to_rrf.get(hid, 0.0) + 1.0 / (k + idx)

            return id_to_rrf
        except Exception as e:
            logger.debug(f"RRF scoring failed: {e}")
            return {}

    def _cross_encoder_rerank(self, query: str, hits: List[Any], top_n: int = 50) -> List[Any]:
        """Cross-encoder rerank of hits using a HF model.

        Returns hits reordered by cross-encoder scores; preserves original hit objects.
        """
        if not hits:
            return hits

        try:
            if self._ce_model is None or self._ce_tokenizer is None:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self._ce_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._ce_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._ce_model.to(device)
                self._ce_model.eval()

            # Prepare pairs
            pairs = []
            idx_map = []
            for i, h in enumerate(hits[:top_n]):
                payload = getattr(h, 'payload', {}) or {}
                text = payload.get('text') or payload.get('content')
                if not text:
                    continue
                pairs.append((query, text))
                idx_map.append(i)

            if not pairs:
                return hits

            device = next(self._ce_model.parameters()).device
            enc = self._ce_tokenizer(
                [q for q, _ in pairs],
                [t for _, t in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = self._ce_model(**enc).logits.squeeze(-1).detach().float().cpu().tolist()

            scored = list(zip(idx_map, logits))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Reorder hits: top_n by CE score first, then the rest in original order
            ordered_indices = [i for i, _ in scored]
            seen = set(ordered_indices)
            tail = [i for i in range(len(hits)) if i not in seen]
            new_order = ordered_indices + tail
            return [hits[i] for i in new_order]
        except Exception as e:
            logger.debug(f"Cross-encoder rerank failed: {e}")
            return hits

    def _expand_table_rows(self, hits: List[Any], max_rows: int = 200) -> List[Any]:
        """Given selected hits, if any correspond to table embeddings, fetch additional row-level hits
        from the same table to allow whole-table reconstruction.

        Args:
            hits: Current list of Qdrant hits
            max_rows: Maximum number of row-level points to fetch per table

        Returns:
            Extended list of hits with row-level entries appended (deduped by id)
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Collect unique (doc_id, table_id) pairs from table hits
        tables: set[tuple[str, str]] = set()
        for h in hits:
            payload = getattr(h, 'payload', {}) or {}
            level = payload.get('level') or payload.get('type')
            if level in ['table', 'row'] or ('table_id' in payload):
                doc_id = payload.get('doc_id')
                table_id = payload.get('table_id')
                if doc_id and table_id:
                    tables.add((doc_id, table_id))

        if not tables:
            return hits

        existing_ids = {getattr(h, 'id', None) for h in hits}
        extended: List[Any] = list(hits)

        for doc_id, table_id in tables:
            try:
                flt = Filter(must=[
                    FieldCondition(key='doc_id', match=MatchValue(value=doc_id)),
                    FieldCondition(key='table_id', match=MatchValue(value=table_id)),
                    FieldCondition(key='level', match=MatchValue(value='row')),
                ])
                points, _ = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=flt,
                    with_payload=True,
                    with_vectors=False,
                    limit=max_rows,
                )
                for p in points or []:
                    pid = getattr(p, 'id', None)
                    if pid and pid not in existing_ids:
                        extended.append(p)
                        existing_ids.add(pid)
            except Exception as e:
                logger.debug(f"Table rows expansion failed for {doc_id}#{table_id}: {e}")

        return extended
    
    def _build_filters(self, request: RetrievalRequest, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build filters from request and constraints.
        
        Args:
            request: Retrieval request
            constraints: Constraints extracted from query
            
        Returns:
            Dictionary of filters
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
        
        must_filters = []
        
        # Room scoping
        if request.room_id and request.prefer_chat_scope:
            try:
                must_filters.append(FieldCondition(key="room_id", match=MatchValue(value=request.room_id)))
                must_filters.append(FieldCondition(key="scope", match=MatchValue(value="chat")))
            except Exception:
                pass
                
        # Explicit scope filter override
        if request.scope_filter:
            try:
                must_filters.append(FieldCondition(key="scope", match=MatchValue(value=request.scope_filter)))
                if request.room_id:
                    must_filters.append(FieldCondition(key="room_id", match=MatchValue(value=request.room_id)))
            except Exception:
                pass
        
        # Arbitrary extra filters on payload keys
        if request.extra_filters:
            try:
                for k, v in request.extra_filters.items():
                    if isinstance(v, list):
                        must_filters.append(FieldCondition(key=k, match=MatchAny(any=v)))
                    else:
                        must_filters.append(FieldCondition(key=k, match=MatchValue(value=v)))
            except Exception:
                pass
        
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
        
        # Create filter object if we have constraints
        return Filter(must=must_filters) if must_filters else None
    
    def _vector_search(self, query_vector, filters, top_k, score_threshold, constraints):
        """Perform vector search with Qdrant.
        
        Args:
            query_vector: Query vector
            filters: Qdrant filter object
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            constraints: Query constraints
            
        Returns:
            List of Qdrant hits
        """
        # Increase limit and seed count to allow richer expansion
        is_table_query = any(ct in ['table_row', 'table_summary'] for ct in constraints.get('content_types', []))
        base_multiplier = 6  # previously 3
        table_multiplier = 8
        if is_table_query:
            initial_limit = min(max(top_k * table_multiplier, 24), 120)
        else:
            initial_limit = min(max(top_k * base_multiplier, 18), 80)
        
        # Check if we have doc_id constraints
        doc_id_constraints_applied = bool(constraints.get('doc_ids'))
        
        # Try with constraints first
        dense_hits = []
        try:
            # If we have doc_id constraints, be more permissive with threshold
            effective_threshold = min(score_threshold, 0.25) if doc_id_constraints_applied else score_threshold
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=filters,
                limit=initial_limit,
                score_threshold=effective_threshold,
            )
            dense_hits = response.points if hasattr(response, 'points') else []
            self._log_hits("Dense (constrained)", dense_hits)
        except Exception as e:
            logger.warning(f"Score threshold not supported, using manual filtering: {e}")
            try:
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=filters,
                    limit=initial_limit,
                )
                dense_hits = response.points if hasattr(response, 'points') else []
                dense_hits = [h for h in dense_hits if h.score >= (min(score_threshold, 0.25) if doc_id_constraints_applied else score_threshold)]
                self._log_hits("Dense (manual filter)", dense_hits)
            except Exception as e2:
                logger.warning(f"Constrained search failed: {e2}")
                # Graceful fallback: try without constraints
                if filters:
                    logger.info("Retrying dense search without constraints...")
                    try:
                        response = self.client.query_points(
                            collection_name=self.collection,
                            query=query_vector,
                            limit=initial_limit,
                            score_threshold=score_threshold
                        )
                        dense_hits = response.points if hasattr(response, 'points') else []
                        self._log_hits("Dense (no constraints)", dense_hits)
                    except Exception as e3:
                        logger.warning(f"Score threshold not supported in fallback: {e3}")
                        response = self.client.query_points(
                            collection_name=self.collection,
                            query=query_vector,
                            limit=initial_limit,
                        )
                        dense_hits = response.points if hasattr(response, 'points') else []
                        dense_hits = [h for h in dense_hits if h.score >= score_threshold]
                        self._log_hits("Dense (no constraints, manual filter)", dense_hits)
                else:
                    dense_hits = []
        
        # Graceful fallback for doc_id constraints
        if doc_id_constraints_applied and not dense_hits:
            logger.info("No results with doc_id constraints, trying without them...")
            try:
                from qdrant_client.http.models import Filter, FieldCondition
                # Remove doc_id constraints but keep other constraints
                if hasattr(filters, 'must'):
                    fallback_filters = Filter(must=[f for f in filters.must if not isinstance(f, FieldCondition) or f.key != "doc_id"])
                    
                    response = self.client.query_points(
                        collection_name=self.collection,
                        query=query_vector,
                        query_filter=fallback_filters,
                        limit=initial_limit,
                        score_threshold=score_threshold
                    )
                    fallback_hits = response.points if hasattr(response, 'points') else []
                    if fallback_hits:
                        logger.info(f"Found {len(fallback_hits)} results without doc_id constraints")
                        dense_hits = fallback_hits
            except Exception as e:
                logger.warning(f"Fallback search failed: {e}")
        
        return dense_hits
    
    def _keyword_search(self, search_query, filters, top_k, constraints):
        """Perform keyword search with Qdrant.
        
        Args:
            search_query: Search query text
            filters: Qdrant filter object
            top_k: Number of results to return
            constraints: Query constraints
            
        Returns:
            List of Qdrant hits
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText
        
        keyword_hits = []
        try:
            # Extract must conditions from filters if available
            must_conditions = []
            if hasattr(filters, 'must'):
                must_conditions = filters.must
            
            # Build keyword filter (with doc_id if present, but without hard content_type filtering)
            keyword_must = [FieldCondition(key="text", match=MatchText(text=search_query))]
            keyword_must.extend(must_conditions)
            
            # Create dummy vector for keyword search using actual embedding dimension
            dummy_vector = [0.0] * len(self.embed("test"))
            
            try:
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=dummy_vector,
                    query_filter=Filter(must=keyword_must),
                    limit=top_k * 3,
                )
                keyword_hits = response.points if hasattr(response, 'points') else []
                self._log_hits("Keyword", keyword_hits)
            except Exception as qe:
                # Attempt to auto-create text index and retry once
                logger.warning(f"Keyword search failed (attempting to create text index): {qe}")
                try:
                    from qdrant_client.http.models import PayloadSchemaType
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name="text",
                        field_schema=PayloadSchemaType.TEXT,
                    )
                    # Retry once after creating the index
                    response = self.client.query_points(
                        collection_name=self.collection,
                        query=dummy_vector,
                        query_filter=Filter(must=keyword_must),
                        limit=top_k * 3,
                    )
                    keyword_hits = response.points if hasattr(response, 'points') else []
                    self._log_hits("Keyword", keyword_hits)
                except Exception as ce:
                    logger.warning(f"Keyword search failed after index creation attempt: {ce}")
                    keyword_hits = []
        except Exception as e:
            logger.warning(f"Keyword search failed (text index missing?): {e}")
            keyword_hits = []
            
        return keyword_hits
    
    def _fallback_search(self, doc_ids, top_k):
        """Perform fallback search using scroll if other methods fail.
        
        Args:
            doc_ids: Document IDs to search for
            top_k: Number of results to return
            
        Returns:
            List of Qdrant hits
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
        
        scroll_hits = []
        try:
            # Create filter for doc_ids
            if isinstance(doc_ids, list):
                if len(doc_ids) == 1:
                    scroll_filter = FieldCondition(key="doc_id", match=MatchValue(value=doc_ids[0]))
                else:
                    scroll_filter = FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            else:
                scroll_filter = FieldCondition(key="doc_id", match=MatchValue(value=doc_ids))
            
            points, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[scroll_filter]),
                with_payload=True,
                with_vectors=False,
                limit=max(top_k, 3),
            )
            scroll_hits = points or []
            if scroll_hits:
                logger.info(f"Scroll fallback returned {len(scroll_hits)} points for doc_ids={doc_ids}")
                self._log_hits("Scroll", scroll_hits)
        except Exception as e:
            logger.warning(f"Scroll fallback failed: {e}")
        
        return scroll_hits
    
    def _merge_and_deduplicate(self, dense_hits, keyword_hits, scroll_hits, top_k, score_threshold, prefer_content_types):
        """Merge and deduplicate hits from different search methods.
        
        Args:
            dense_hits: Hits from vector search
            keyword_hits: Hits from keyword search
            scroll_hits: Hits from fallback search
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            prefer_content_types: Content types to prefer
            
        Returns:
            List of deduplicated hits
        """
        # Merge all hits
        all_hits = {}
        for h in dense_hits + keyword_hits + scroll_hits:
            # Use best score if duplicate
            if (prev := all_hits.get(h.id)) is None or h.score > prev.score:
                all_hits[h.id] = h
        
        # Sort by score desc and keep top_k unique (dedup by doc_id+page)
        seen_sources = set()
        final = []
        
        # For table queries, use table-aware deduplication
        is_table_query = any(ct in ['table_row', 'table_summary'] for ct in prefer_content_types or [])
        
        for hit in sorted(all_hits.values(), key=lambda x: x.score, reverse=True):
            doc_id = hit.payload.get("doc_id", "unknown")
            page = hit.payload.get("page", "unknown")
            
            # Table-aware deduplication
            if is_table_query:
                # For tables, include table_id in deduplication key to preserve multiple tables
                table_id = hit.payload.get("table_id", "unknown")
                key = f"{doc_id}:{page}:{table_id}"
                
                # Also check if this is a different table embedding type (row vs summary)
                level = hit.payload.get("level", "")
                type_key = f"{key}:{level}"
                
                if type_key in seen_sources:
                    continue
                seen_sources.add(type_key)

                # If we added any row hit, force-include one matching table summary (whole-table context)
                if level == 'row':
                    summary_key = f"{key}:table"
                    if summary_key not in seen_sources:
                        for cand in sorted(all_hits.values(), key=lambda x: x.score, reverse=True):
                            if cand.payload.get('doc_id') == doc_id and cand.payload.get('page') == page and cand.payload.get('table_id') == table_id and cand.payload.get('level') == 'table':
                                final.append(cand)
                                seen_sources.add(summary_key)
                                break
            else:
                # Standard deduplication for non-table queries
                key = f"{doc_id}:{page}"
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
                # Check both 'type' and 'level' fields for content type matching
                content_type = hit.payload.get('type', '')
                level = hit.payload.get('level', '')
                
                # Map table levels to content types for filtering
                if level == 'row' and 'table_row' in prefer_content_types:
                    return 0
                elif level == 'table' and 'table_summary' in prefer_content_types:
                    return 0
                elif content_type in prefer_content_types:
                    return 0
                else:
                    return 1
            final = sorted(final, key=type_rank)
        
        logger.info(f"🔍 dense={len(dense_hits)} keyword={len(keyword_hits)} → {len(final)} unique hits; top_score={top_score:.3f}")
        self._log_hits("Final", final)
        
        return final

# Global instance for easy import
retrieval_service = RetrievalService()