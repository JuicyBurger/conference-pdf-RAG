"""
Suggestions API Routes

Handles question suggestions for chat interface.
"""

import logging
from flask import Blueprint, request, jsonify

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.API.utils.response import success_response, error_response
from src.rag.suggestions import retrieve_suggestions, get_all_doc_ids_with_suggestions

logger = logging.getLogger(__name__)

suggestions_bp = Blueprint('suggestions', __name__)

@suggestions_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """
    Get question suggestions for a document or across all documents.
    
    Query Parameters:
    - doc_id (optional): Document ID to get suggestions for
    - topic (optional): Topic/query for semantic search
    - k (optional): Number of suggestions to return (default: 5)
    
    Returns:
    {
        "doc_id": "document_id",
        "suggestions": ["question1", "question2", ...],
        "total": 5
    }
    """
    try:
        # Get query parameters
        doc_id = request.args.get('doc_id')
        topic = request.args.get('topic')
        k = int(request.args.get('k', 5))
        
        # Validate parameters
        if k <= 0 or k > 20:  # Reasonable limits
            return error_response("Parameter 'k' must be between 1 and 20", 400)
        
        if not doc_id:
            # If no doc_id provided, return available documents with suggestions
            doc_ids = get_all_doc_ids_with_suggestions()
            return success_response({
                "available_docs": doc_ids,
                "message": "Provide doc_id parameter to get suggestions for a specific document"
            })
        
        # Retrieve suggestions for the specific document
        suggestions_data = retrieve_suggestions(
            doc_id=doc_id,
            topic=topic,
            k=k
        )
        
        if not suggestions_data:
            return success_response({
                "doc_id": doc_id,
                "suggestions": [],
                "total": 0,
                "message": f"No suggestions found for document: {doc_id}"
            })
        
        # Extract just the question text for the response
        suggestions = [item["question_text"] for item in suggestions_data]
        
        response_data = {
            "doc_id": doc_id,
            "suggestions": suggestions,
            "total": len(suggestions)
        }
        
        # Add topic info if provided
        if topic:
            response_data["topic"] = topic
        
        logger.info(f"✅ Retrieved {len(suggestions)} suggestions for doc_id: {doc_id}")
        return success_response(response_data)
        
    except ValueError as e:
        logger.error(f"Invalid parameter in suggestions request: {e}")
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return error_response("Internal server error while retrieving suggestions", 500)

@suggestions_bp.route('/suggestions/docs', methods=['GET'])
def list_docs_with_suggestions():
    """
    List all document IDs that have suggestions available.
    
    Returns:
    {
        "doc_ids": ["doc1", "doc2", ...],
        "total": 5
    }
    """
    try:
        doc_ids = get_all_doc_ids_with_suggestions()
        
        return success_response({
            "doc_ids": doc_ids,
            "total": len(doc_ids)
        })
        
    except Exception as e:
        logger.error(f"Error listing docs with suggestions: {e}")
        return error_response("Internal server error while listing documents", 500)