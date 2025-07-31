"""
Chat Route Handlers

Handles chat messaging, history, and RAG integration.
"""

import asyncio
import logging
import sys
import os
from flask import Blueprint, request, jsonify
from datetime import datetime, timezone

# Add the API directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services and utilities
from src.API.services.chat_service import chat_service
from src.API.utils.response import success_response, error_response, validation_error_response, not_found_response

# Import existing RAG components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.rag.retriever import retrieve
from src.rag.generator import generate_answer
from src.models.reranker import rerank

logger = logging.getLogger(__name__)
chat_bp = Blueprint('chat', __name__)


def run_async(coro):
    """Helper to run async functions in Flask routes"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def build_context_prompt(user_message: str, chat_history: list, document_hits: list = None) -> str:
    """Build a comprehensive context prompt with chat history and documents"""
    
    # Build chat context from recent messages
    chat_context = ""
    if chat_history:
        chat_context = "Previous conversation:\n"
        for msg in chat_history[-10:]:  # Last 10 messages for context
            role = "User" if msg["role"] == "user" else "ai"
            chat_context += f"{role}: {msg['content']}\n"
        chat_context += "\n"
    
    # Build document context
    doc_context = ""
    if document_hits:
        doc_context = "Relevant documents:\n"
        for i, hit in enumerate(document_hits[:3], 1):  # Top 3 documents
            content = hit.payload.get('content', '')[:500]  # First 500 chars
            page = hit.payload.get('page', 'unknown')
            doc_context += f"Document {i} (Page {page}): {content}...\n"
        doc_context += "\n"
    
    # Combine everything
    full_prompt = f"""{chat_context}{doc_context}Current question: {user_message}

Please respond in Traditional Chinese, referencing both the conversation history and any relevant documents. If referencing documents, include page numbers."""
    
    return full_prompt


@chat_bp.route('/message', methods=['POST'])
def send_message():
    """Send a message and get AI response"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify(validation_error_response("body", "Request body is required")), 400
        
        room_id = data.get('room_id')
        message = data.get('message', '').strip()
        user_id = data.get('user_id')
        files = data.get('files', [])
        
        if not message:
            return jsonify(validation_error_response("message", "Message is required")), 400
        
        # Auto-create room if no room_id provided
        is_new_room = False
        if not room_id or room_id == "new":
            print(f"🆕 Creating new room for message: {message[:50]}...")
            room_data = run_async(chat_service.create_room(message, user_id))
            room_id = room_data["room_id"]
            is_new_room = True
            print(f"✅ Created room {room_id[:8]} with title: {room_data['room_title']}")
        else:
            # Check if room exists, if not create it
            existing_room = run_async(chat_service.get_room(room_id))
            if not existing_room:
                print(f"🆕 Room {room_id[:8]} doesn't exist, creating it...")
                room_data = run_async(chat_service.create_room(message, user_id))
                # Use the provided room_id instead of generated one
                room_data["room_id"] = room_id
                chat_service.rooms[room_id] = room_data
                is_new_room = True
        
        # Add user message to chat history (updated field names)
        user_msg_id = run_async(chat_service.add_message(room_id, "user", message, user_id, files))
        
        # Generate AI response using RAG with proper context
        try:
            # 1. Get chat history FIRST for context
            chat_history = run_async(chat_service.get_chat_history(room_id, recency_k=20))
            
            # 2. Retrieve relevant documents
            document_hits = retrieve(query=message, top_k=5, score_threshold=0.3)
            
            # 3. Build comprehensive context
            if document_hits:
                # Rerank for better relevance if we have multiple hits
                if len(document_hits) > 1:
                    document_hits = rerank(message, document_hits)
                
                # Generate answer with both document and chat context
                context_prompt = build_context_prompt(message, chat_history, document_hits)
                ai_response = generate_answer(context_prompt, document_hits)
            else:
                # Fallback: Use chat history for context even without documents
                if len(chat_history) > 1:  # More than just the current message
                    context_prompt = build_context_prompt(message, chat_history[:-1])  # Exclude the just-added message
                    # Create a dummy hit for the generator to work
                    dummy_hit = type('Hit', (), {
                        'payload': {'content': '基於對話歷史回答', 'page': 'chat_history'}
                    })()
                    ai_response = generate_answer(context_prompt, [dummy_hit])
                else:
                    ai_response = "你好！我可以幫你分析文件或回答問題。請告訴我你需要什麼幫助，或者上傳一些相關文件。"
            
            # Add AI response to chat history (updated field names)
            ai_msg_id = run_async(chat_service.add_message(room_id, "ai", ai_response, None, []))
            
            # Get updated conversation for response (includes the AI response we just added)
            recent_messages = run_async(chat_service.get_chat_history(room_id, recency_k=20))
            
            # Sort messages by timestamp ascending
            recent_messages.sort(key=lambda x: x["timestamp"])
            
            # Get room metadata for response
            room_data = run_async(chat_service.get_room(room_id))
            room_title = room_data["room_title"] if room_data else f"Chat Room {room_id[:8]}"
            created_at = room_data["created_at"] if room_data else (recent_messages[0]["timestamp"] if recent_messages else datetime.now(timezone.utc).isoformat())
            
            # Return response with new structure
            return jsonify({
                "status": "success",
                "data": {
                    "room_id": room_id,
                    "room_title": room_title,
                    "createdAt": created_at,
                    "updatedAt": recent_messages[-1]["timestamp"] if recent_messages else datetime.now(timezone.utc).isoformat(),
                    "messages": recent_messages
                }
            })
            
        except Exception as rag_error:
            logger.error(f"RAG processing error: {rag_error}")
            
            # Fallback response
            fallback_response = "抱歉，我在處理你的請求時遇到了問題。請再試一次。"
            ai_msg_id = run_async(chat_service.add_message(room_id, "ai", fallback_response, None, []))
            
            # Get updated conversation for response (includes the AI response we just added)
            recent_messages = run_async(chat_service.get_chat_history(room_id, recency_k=20))
            
            # Sort messages by timestamp ascending
            recent_messages.sort(key=lambda x: x["timestamp"])
            
            # Get room metadata for response
            room_data = run_async(chat_service.get_room(room_id))
            room_title = room_data["room_title"] if room_data else f"Chat Room {room_id[:8]}"
            created_at = room_data["created_at"] if room_data else (recent_messages[0]["timestamp"] if recent_messages else datetime.now(timezone.utc).isoformat())
            
            return jsonify({
                "status": "success",
                "data": {
                    "room_id": room_id,
                    "room_title": room_title,
                    "createdAt": created_at,
                    "updatedAt": recent_messages[-1]["timestamp"] if recent_messages else datetime.now(timezone.utc).isoformat(),
                    "messages": recent_messages
                }
            })
    
    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        return jsonify(error_response(f"Failed to process message: {str(e)}")), 500


@chat_bp.route('/histories/<room_id>', methods=['GET'])
def get_chat_history(room_id):
    """Get chat history for a room"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        limit = min(limit, 100)  # Cap at 100 messages
        
        # Get chat history
        messages = run_async(chat_service.get_chat_history(room_id, recency_k=limit))
        
        # Sort messages by timestamp ascending
        messages.sort(key=lambda x: x["timestamp"])
        
        # Get room metadata
        room_data = run_async(chat_service.get_room(room_id))
        room_title = room_data["room_title"] if room_data else f"Chat Room {room_id[:8]}"
        created_at = room_data["created_at"] if room_data else (messages[0]["timestamp"] if messages else datetime.now(timezone.utc).isoformat())
        
        return jsonify({
            "status": "success",
            "data": {
                "room_id": room_id,
                "room_title": room_title,
                "createdAt": created_at,
                "updatedAt": messages[-1]["timestamp"] if messages else datetime.now(timezone.utc).isoformat(),
                "messages": messages
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify(error_response(f"Failed to get chat history: {str(e)}")), 500


@chat_bp.route('/search/<room_id>', methods=['POST'])
def search_messages(room_id):
    """Search messages in a room using semantic search"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify(validation_error_response("body", "Request body is required")), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify(validation_error_response("query", "Search query is required")), 400
        
        # Get search parameters
        limit = data.get('limit', 20)
        limit = min(limit, 50)  # Cap at 50 results
        
        # Perform semantic search
        results = run_async(chat_service.search_messages(room_id, query, semantic_k=limit))
        
        return jsonify(success_response({
            "room_id": room_id,
            "query": query,
            "results": results,
            "result_count": len(results)
        }))
    
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        return jsonify(error_response(f"Failed to search messages: {str(e)}")), 500


@chat_bp.route('/rooms/<room_id>', methods=['DELETE'])
def delete_room(room_id):
    """Delete room and all its messages completely"""
    try:
        # Check if room exists first
        room_data = run_async(chat_service.get_room(room_id))
        if not room_data:
            return jsonify(not_found_response("Room not found")), 404
        
        # Delete the room and all its messages
        success = run_async(chat_service.delete_room(room_id))
        
        if success:
            return jsonify({
                "status": "success",
                "data": {
                    "room_id": room_id,
                    "room_title": room_data.get("room_title", "Unknown"),
                    "deleted": True
                }
            })
        else:
            return jsonify(error_response("Failed to delete room")), 500
    
    except Exception as e:
        logger.error(f"Error deleting room: {e}")
        return jsonify(error_response(f"Failed to delete room: {str(e)}")), 500



@chat_bp.route('/rooms', methods=['GET'])
def list_rooms():
    """List active chat rooms ordered by last updated"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Cap at 100 rooms
        
        # Get active rooms
        rooms = run_async(chat_service.list_active_rooms(limit=limit))
        
        return jsonify({
            "status": "success",
            "limit": limit,
            "total_count": len(rooms),
            "data": rooms
        })
    
    except Exception as e:
        logger.error(f"Error listing rooms: {e}")
        return jsonify(error_response(f"Failed to list rooms: {str(e)}")), 500 