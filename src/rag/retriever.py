# src/RAG/retriever.py
from qdrant_client import QdrantClient
from ..config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from ..models.embedder import embed

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3):
    """
    Retrieve relevant documents with deduplication and score filtering.
    
    Args:
        query: Search query
        top_k: Number of results to return
        score_threshold: Minimum similarity score threshold
        
    Returns:
        List of deduplicated, high-quality hits
    """
    q_vec = embed(query)
    
    # Retrieve more results initially to allow for deduplication
    initial_limit = min(top_k * 3, 50)  # Get 3x more results for filtering
    
    try:
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=initial_limit,
            score_threshold=score_threshold  # Filter low-quality results
        )
    except Exception as e:
        # Fallback if score_threshold is not supported
        print(f"⚠️ Score threshold not supported, using manual filtering: {e}")
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=initial_limit
        )
        # Manual score filtering
        hits = [hit for hit in hits if hit.score >= score_threshold]
    
    # Deduplicate results by (doc_id, page) pairs
    seen_sources = set()
    unique_hits = []
    
    for hit in hits:
        doc_id = hit.payload.get('doc_id', 'unknown')
        page = hit.payload.get('page', 'unknown')
        source_key = f"{doc_id}:{page}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_hits.append(hit)
            
            # Stop when we have enough unique results
            if len(unique_hits) >= top_k:
                break
    
    print(f"🔍 Retrieved {len(hits)} initial results → {len(unique_hits)} unique sources (score ≥ {score_threshold})")
    
    return unique_hits
