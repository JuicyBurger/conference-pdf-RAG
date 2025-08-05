# src/rag/suggestions/catalog.py

import logging
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from ...config import QDRANT_URL, QDRANT_API_KEY, QDRANT_QA_DB
from ...models.embedder import embed

logger = logging.getLogger(__name__)

# Setup Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,
)

@dataclass
class QuestionSuggestion:
    """Represents a question suggestion with metadata"""
    question_text: str
    doc_id: str
    sections: List[Dict]  # [{page, chunk_id, ...}, ...]
    tags: Optional[List[str]] = None
    popularity_score: float = 0.0

def init_suggestion_collection(collection_name: str = QDRANT_QA_DB) -> None:
    """
    Initialize the Qdrant collection for storing question suggestions.
    """
    try:
        # Get vector size from the embedder model
        from ...models.embedder import model
        vector_size = model.get_sentence_embedding_dimension()
        
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )
        logger.info(f"✅ Suggestion collection '{collection_name}' initialized: vectors={vector_size}")
        
        # Create payload indexes for fast filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema="keyword",
        )
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="popularity_score",
            field_schema="float",
        )
        
        logger.info(f"🔑 Payload indexes created for 'doc_id' and 'popularity_score'")
        
    except Exception as e:
        logger.error(f"Failed to initialize suggestion collection: {e}")
        raise

def store_questions(
    suggestions: List[QuestionSuggestion], 
    collection_name: str = QDRANT_QA_DB
) -> None:
    """
    Store question suggestions in the Qdrant collection.
    """
    if not suggestions:
        logger.warning("No suggestions to store")
        return
    
    try:
        # Prepare points for batch insertion
        points = []
        for i, suggestion in enumerate(suggestions):
            # Create embedding for the question text
            vector = embed(suggestion.question_text)
            
            # Create point with metadata - use UUID for point ID
            point = PointStruct(
                id=str(uuid.uuid4()),  # Use proper UUID format
                vector=vector,
                payload={
                    "question_text": suggestion.question_text,
                    "doc_id": suggestion.doc_id,
                    "sections": suggestion.sections,
                    "tags": suggestion.tags or [],
                    "popularity_score": suggestion.popularity_score,
                }
            )
            points.append(point)
        
        # Batch insert
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"✅ Stored {len(suggestions)} question suggestions for doc_id: {suggestions[0].doc_id}")
        
    except Exception as e:
        logger.error(f"Failed to store suggestions: {e}")
        raise

def retrieve_suggestions(
    doc_id: str,
    topic: Optional[str] = None,
    k: int = 5,
    collection_name: str = QDRANT_QA_DB
) -> List[Dict]:
    """
    Retrieve question suggestions for a specific document.
    
    Args:
        doc_id: Document ID to get suggestions for
        topic: Optional topic/query for semantic search
        k: Number of suggestions to return
        collection_name: Qdrant collection name
    
    Returns:
        List of suggestion dictionaries with question_text and metadata
    """
    try:
        # Build filter for doc_id
        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
        
        if topic:
            # Semantic search with topic embedding
            topic_vector = embed(topic)
            search_result = client.search(
                collection_name=collection_name,
                query_vector=topic_vector,
                query_filter=doc_filter,
                limit=k,
                score_threshold=0.1  # Low threshold for suggestions
            )
        else:
            # Simple retrieval ordered by popularity score
            search_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=doc_filter,
                limit=k,
                order_by="popularity_score"  # Order by popularity when no topic
            )[0]  # scroll returns (points, next_page_offset)
        
        # Format results
        suggestions = []
        for point in search_result:
            suggestions.append({
                "question_text": point.payload["question_text"],
                "doc_id": point.payload["doc_id"],
                "sections": point.payload["sections"],
                "tags": point.payload.get("tags", []),
                "popularity_score": point.payload.get("popularity_score", 0.0),
                "relevance_score": getattr(point, 'score', None)  # For semantic search
            })
        
        logger.info(f"🔍 Retrieved {len(suggestions)} suggestions for doc_id: {doc_id}")
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to retrieve suggestions: {e}")
        return []

def get_all_doc_ids_with_suggestions(collection_name: str = QDRANT_QA_DB) -> List[str]:
    """
    Get all document IDs that have suggestions stored.
    """
    try:
        # Use scroll to get all points and extract unique doc_ids
        result = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Should be enough for most use cases
            with_payload=["doc_id"]
        )
        
        doc_ids = set()
        for point in result[0]:
            doc_ids.add(point.payload["doc_id"])
        
        return list(doc_ids)
        
    except Exception as e:
        logger.error(f"Failed to get doc_ids with suggestions: {e}")
        return []

def update_popularity_score(
    question_id: str, 
    increment: float = 1.0,
    collection_name: str = QDRANT_QA_DB
) -> bool:
    """
    Update the popularity score for a specific question.
    Note: This is a simplified implementation. For high-traffic scenarios,
    consider using a separate counter store like Redis.
    """
    try:
        # Get current point
        point = client.retrieve(
            collection_name=collection_name,
            ids=[question_id],
            with_payload=True
        )
        
        if not point:
            logger.warning(f"Question ID {question_id} not found")
            return False
        
        # Update popularity score
        current_score = point[0].payload.get("popularity_score", 0.0)
        new_score = current_score + increment
        
        # Update the point
        client.set_payload(
            collection_name=collection_name,
            payload={"popularity_score": new_score},
            points=[question_id]
        )
        
        logger.info(f"✅ Updated popularity score for {question_id}: {current_score} -> {new_score}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update popularity score: {e}")
        return False