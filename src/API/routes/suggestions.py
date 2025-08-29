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
from src.rag.generation import retrieve_suggestions, retrieve_random_suggestions, get_all_doc_ids_with_suggestions
from src.rag.generation import generate_suggestions_for_doc, batch_generate_suggestions
from src.API.services.chat_service import chat_service

logger = logging.getLogger(__name__)

suggestions_bp = Blueprint('suggestions', __name__)

def run_async(coro):
    """Helper to run async functions in Flask routes"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@suggestions_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """
    Get question suggestions for a document, chat room, or across all available suggestions.
    
    Query Parameters:
    - doc_id (optional): Document ID to get suggestions for
    - room_id (optional): Chat room ID to get suggestions for  
    - topic (optional): Topic/query for semantic search
    - k (optional): Number of suggestions to return (default: 5)
    
    Returns:
    {
        "doc_id": "document_id",  // or "room_id": "room-123" 
        "suggestions": ["question1", "question2", ...],
        "total": 5
    }
    """
    try:
        # Get query parameters
        doc_id = request.args.get('doc_id')
        room_id = request.args.get('room_id')
        topic = request.args.get('topic')
        k = int(request.args.get('k', 5))
        
        # Validate parameters
        if k <= 0 or k > 20:  # Reasonable limits
            return error_response("Parameter 'k' must be between 1 and 20", 400)
        
        if doc_id and room_id:
            return error_response("Provide either doc_id OR room_id, not both", 400)
        
        # If room_id provided, convert to room-based doc_id format
        if room_id:
            doc_id = f"room_{room_id}"
        
        if not doc_id:
            # If no doc_id/room_id provided, return random suggestions
            logger.info(f"🎲 No doc_id or room_id provided, retrieving {k} random suggestions")
            suggestions_data = retrieve_random_suggestions(k=k)
            
            if not suggestions_data:
                return success_response([])
            
            # Extract just the question text for the response
            suggestions = [{'id': item["id"], 'text': item["question_text"]} for item in suggestions_data]
            
            logger.info(f"✅ Retrieved {len(suggestions)} random suggestions")
            return success_response(suggestions)
        
        # Retrieve suggestions for the specific document/room
        suggestions_data = retrieve_suggestions(
            doc_id=doc_id,
            topic=topic,
            k=k
        )
        
        if not suggestions_data:
            identifier = room_id if room_id else doc_id
            identifier_type = "room" if room_id else "document"
            return success_response([])
        
        # Extract just the question text for the response
        suggestions = [{'id': item["id"], 'text': item["question_text"]} for item in suggestions_data]
        
        identifier = room_id if room_id else doc_id
        logger.info(f"✅ Retrieved {len(suggestions)} suggestions for {identifier}")
        return success_response(suggestions)
        
    except ValueError as e:
        logger.error(f"Invalid parameter in suggestions request: {e}")
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return error_response("Internal server error while retrieving suggestions", 500)

@suggestions_bp.route('/suggestions', methods=['POST'])
def generate_suggestions():
    """
    Generate question suggestions based on documents or chat room conversation context.
    
    Request Body (JSON or form-urlencoded):
    {
        "doc_id": "document_id",        // Single document (optional if room_id provided)
        "room_id": "room-123",         // Chat room ID to analyze (optional if doc_id provided)
        "num_questions": 8,            // Number of questions to generate (default: 8)
        "use_lightweight": true,       // Use lightweight generation (default: true)
        "auto_init_collection": true   // Auto-initialize collection (default: true)
    }
    
    Returns:
    {
        "status": "success",
        "data": {
            "doc_id": "document_id",  // or "room_id": "room-123"
            "success": true,
            "num_questions": 8,
            "message": "Successfully generated 8 suggestions"
        }
    }
    """
    try:
        # Handle both JSON and form-urlencoded content types
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
        else:
            # Handle form-urlencoded data
            data = {}
            for key in request.form:
                value = request.form[key]
                # Handle boolean values
                if key in ['use_lightweight', 'auto_init_collection']:
                    data[key] = value.lower() in ['true', '1', 'yes', 'on']
                # Handle integer values
                elif key in ['num_questions']:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        return error_response(f"Invalid value for {key}: must be an integer", 400)
                else:
                    data[key] = value
        
        if not data:
            return error_response("Request body is required", 400)
        
        # Validate input parameters
        doc_id = data.get('doc_id')
        room_id = data.get('room_id')
        num_questions = data.get('num_questions', 8)
        use_lightweight = data.get('use_lightweight', True)
        auto_init_collection = data.get('auto_init_collection', True)
        
        # Validate fields
        if doc_id and room_id:
            return error_response("Provide either doc_id OR room_id, not both", 400)
        
        # Validate num_questions
        if not isinstance(num_questions, int) or num_questions <= 0 or num_questions > 20:
            return error_response("num_questions must be an integer between 1 and 20", 400)
        
        # Handle room-based generation
        if room_id:
            logger.info(f"🔄 Generating suggestions for chat room: {room_id}")
            
            # Fetch chat history from the room (API as controller)
            try:
                chat_history = run_async(chat_service.get_chat_history(room_id, recency_k=20))
                
                if not chat_history:
                    return error_response(f"No chat history found for room: {room_id}", 404)
                
                # Build chat context string
                conversation_context = []
                for msg in chat_history[-10:]:  # Last 10 messages for context
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = msg["content"][:200]  # Limit length
                    conversation_context.append(f"{role}: {content}")
                
                chat_context = "\n".join(conversation_context)
                
                # Use room_id as doc_id with prefix for storage
                storage_doc_id = f"room_{room_id}"
                
                # Generate suggestions with chat context
                success = generate_suggestions_for_doc(
                    doc_id=storage_doc_id,
                    num_questions=num_questions,
                    auto_init_collection=auto_init_collection,
                    use_lightweight=use_lightweight,
                    chat_context=chat_context
                )
                
                if success:
                    return success_response({
                        "room_id": room_id,
                        "success": True,
                        "num_questions": num_questions,
                        "message": f"Successfully generated {num_questions} suggestions based on conversation context"
                    })
                else:
                    return error_response(f"Failed to generate suggestions for room: {room_id}", 500)
                    
            except Exception as e:
                logger.error(f"Error fetching chat history for room {room_id}: {e}")
                return error_response(f"Failed to access chat room: {room_id}", 500)
        
        # Handle document-based generation
        elif doc_id:
            logger.info(f"🔄 Generating suggestions for document: {doc_id}")
            
            success = generate_suggestions_for_doc(
                doc_id=doc_id,
                num_questions=num_questions,
                auto_init_collection=auto_init_collection,
                use_lightweight=use_lightweight,
                chat_context=None  # No chat context for document-based
            )
            
            if success:
                return success_response({
                    "doc_id": doc_id,
                    "success": True,
                    "num_questions": num_questions,
                    "use_lightweight": use_lightweight,
                    "message": f"Successfully generated {num_questions} suggestions for document"
                })
            else:
                return error_response(f"Failed to generate suggestions for document: {doc_id}", 500)
        
        # Handle case when neither doc_id nor room_id is provided
        else:
            return error_response("Either doc_id or room_id is required for generation", 400)
        
    except ValueError as e:
        logger.error(f"Invalid parameter in generate suggestions request: {e}")
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        return error_response("Internal server error while generating suggestions", 500)

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