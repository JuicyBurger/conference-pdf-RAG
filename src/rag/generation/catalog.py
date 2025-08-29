"""
Question suggestions catalog for Qdrant storage and retrieval.

This module handles the storage, retrieval, and management of question suggestions
in Qdrant for fast frontend access.
"""

import logging
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from .models import QuestionSuggestion
from ..utils import handle_errors, DatabaseError, setup_logger
from ...config import QDRANT_URL, QDRANT_API_KEY, QDRANT_QA_DB
from ...models.embedder import embed, model

logger = setup_logger(__name__)

# Setup Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,
)


@handle_errors(error_class=DatabaseError, reraise=True)
def init_suggestion_collection(collection_name: str = QDRANT_QA_DB) -> None:
    """
    Initialize the Qdrant collection for storing question suggestions.
    
    Args:
        collection_name: Name of the collection to initialize
    """
    try:
        # Get vector size from the embedder model
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
            field_schema="keyword"
        )
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="room_id",
            field_schema="keyword"
        )
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="popularity_score",
            field_schema="float"
        )
        
        logger.info(f"✅ Created payload indexes for '{collection_name}'")
        
    except Exception as e:
        logger.error(f"Failed to initialize suggestion collection: {e}")
        raise


@handle_errors(error_class=DatabaseError, reraise=False)
def store_questions(suggestions: List[QuestionSuggestion], 
                   collection_name: str = QDRANT_QA_DB) -> bool:
    """
    Store question suggestions in Qdrant.
    
    Args:
        suggestions: List of QuestionSuggestion objects to store
        collection_name: Name of the collection to store in
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not suggestions:
            logger.warning("No suggestions provided to store")
            return True
        
        points = []
        for suggestion in suggestions:
            # Generate embedding for the question
            vector = embed(suggestion.question)
            
            # Create point
            point = PointStruct(
                id=suggestion.suggestion_id,
                vector=vector,
                payload=suggestion.to_qdrant_payload()
            )
            points.append(point)
        
        # Upsert points to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"✅ Stored {len(suggestions)} question suggestions in '{collection_name}'")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store question suggestions: {e}")
        return False


@handle_errors(error_class=DatabaseError, fallback_return=[])
def retrieve_suggestions(question: str,
                        doc_id: Optional[str] = None,
                        room_id: Optional[str] = None,
                        limit: int = 5,
                        score_threshold: float = 0.3,
                        collection_name: str = QDRANT_QA_DB) -> List[Dict[str, Any]]:
    """
    Retrieve similar question suggestions from Qdrant.
    
    Args:
        question: Query question to find similar suggestions for
        doc_id: Optional document ID filter
        room_id: Optional room ID filter
        limit: Maximum number of suggestions to return
        score_threshold: Minimum similarity score threshold
        collection_name: Name of the collection to search
        
    Returns:
        List of suggestion dictionaries with scores
    """
    try:
        # Generate embedding for the query
        query_vector = embed(question)
        
        # Build filter conditions
        filter_conditions = []
        if doc_id:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )
        if room_id:
            filter_conditions.append(
                FieldCondition(key="room_id", match=MatchValue(value=room_id))
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search for similar suggestions
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        # Format results
        suggestions = []
        for scored_point in search_result:
            suggestion = {
                "question": scored_point.payload.get("question"),
                "doc_id": scored_point.payload.get("doc_id"),
                "score": scored_point.score,
                "suggestion_id": scored_point.payload.get("suggestion_id"),
                "popularity_score": scored_point.payload.get("popularity_score", 0.0),
                "page": scored_point.payload.get("page"),
                "room_id": scored_point.payload.get("room_id"),
                "created_at": scored_point.payload.get("created_at"),
                "metadata": scored_point.payload.get("metadata", {})
            }
            suggestions.append(suggestion)
        
        logger.debug(f"Retrieved {len(suggestions)} suggestions for query: {question[:50]}...")
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to retrieve suggestions: {e}")
        return []


@handle_errors(error_class=DatabaseError, fallback_return=[])
def retrieve_random_suggestions(doc_id: Optional[str] = None,
                               room_id: Optional[str] = None,
                               limit: int = 5,
                               collection_name: str = QDRANT_QA_DB) -> List[Dict[str, Any]]:
    """
    Retrieve random question suggestions from Qdrant.
    
    Args:
        doc_id: Optional document ID filter
        room_id: Optional room ID filter  
        limit: Maximum number of suggestions to return
        collection_name: Name of the collection to search
        
    Returns:
        List of random suggestion dictionaries
    """
    try:
        # Build filter conditions
        filter_conditions = []
        if doc_id:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )
        if room_id:
            filter_conditions.append(
                FieldCondition(key="room_id", match=MatchValue(value=room_id))
            )
        
        scroll_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Get random points using scroll
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit * 3,  # Get more than needed for randomization
            with_payload=True,
            with_vectors=False
        )
        
        # Shuffle and limit results
        import random
        random.shuffle(points)
        points = points[:limit]
        
        # Format results
        suggestions = []
        for point in points:
            suggestion = {
                "question": point.payload.get("question"),
                "doc_id": point.payload.get("doc_id"),
                "suggestion_id": point.payload.get("suggestion_id"),
                "popularity_score": point.payload.get("popularity_score", 0.0),
                "page": point.payload.get("page"),
                "room_id": point.payload.get("room_id"),
                "created_at": point.payload.get("created_at"),
                "metadata": point.payload.get("metadata", {})
            }
            suggestions.append(suggestion)
        
        logger.info(f"🎲 Retrieved {len(suggestions)} random suggestions")
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to retrieve random suggestions: {e}")
        return []


@handle_errors(error_class=DatabaseError, fallback_return=[])
def get_all_doc_ids_with_suggestions(collection_name: str = QDRANT_QA_DB) -> List[str]:
    """
    Get all unique document IDs that have suggestions stored.
    
    Args:
        collection_name: Name of the collection to search
        
    Returns:
        List of unique document IDs
    """
    try:
        # Scroll through all points to collect unique doc_ids
        doc_ids = set()
        offset = None
        
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            for point in points:
                doc_id = point.payload.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)
            
            offset = next_offset
            if offset is None:
                break
        
        doc_id_list = list(doc_ids)
        logger.info(f"Found {len(doc_id_list)} documents with suggestions")
        return doc_id_list
        
    except Exception as e:
        logger.error(f"Failed to get doc IDs with suggestions: {e}")
        return []


@handle_errors(error_class=DatabaseError, reraise=False)
def update_popularity_score(suggestion_id: str,
                           new_score: float,
                           collection_name: str = QDRANT_QA_DB) -> bool:
    """
    Update the popularity score of a suggestion.
    
    Args:
        suggestion_id: ID of the suggestion to update
        new_score: New popularity score
        collection_name: Name of the collection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update the point's payload
        client.set_payload(
            collection_name=collection_name,
            payload={"popularity_score": new_score},
            points=[suggestion_id]
        )
        
        logger.debug(f"Updated popularity score for suggestion {suggestion_id}: {new_score}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update popularity score: {e}")
        return False


# Legacy compatibility functions
def store_question_suggestions(questions: List[str],
                             doc_id: str,
                             collection_name: str = QDRANT_QA_DB) -> bool:
    """
    Legacy function for storing question suggestions.
    
    Args:
        questions: List of question strings
        doc_id: Document ID
        collection_name: Collection name
        
    Returns:
        True if successful, False otherwise
    """
    # Convert to QuestionSuggestion objects
    suggestions = []
    for question in questions:
        suggestion = QuestionSuggestion(
            question=question,
            doc_id=doc_id,
            suggestion_id=str(uuid.uuid4()),
            created_at=datetime.now()
        )
        suggestions.append(suggestion)
    
    return store_questions(suggestions, collection_name)
