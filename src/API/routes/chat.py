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
from src.API.services.room_service import room_service
from src.API.utils.response import success_response, error_response, validation_error_response, not_found_response
from src.API.utils.file_handler import validate_pdf_file, validate_file_size, FileValidationError

# Import existing RAG components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.rag.router import router

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


def build_context_prompt(user_message: str, chat_history: list, document_hits: list = None, pdf_summary: str = None) -> str:
    """Build a comprehensive context prompt with chat history, documents, and PDF summary"""
    
    # Build chat context from recent messages
    chat_context = ""
    if chat_history:
        chat_context = "Previous conversation:\n"
        for msg in chat_history[-10:]:  # Last 10 messages for context
            role = "User" if msg["role"] == "user" else "ai"
            chat_context += f"{role}: {msg['content']}\n"
        chat_context += "\n"
    
    # Build PDF summary context (NEW: Priority over document hits)
    pdf_context = ""
    if pdf_summary:
        pdf_context = f"Uploaded PDF Content:\n{pdf_summary}\n\n"
    
    # Build document context (from existing RAG system)
    doc_context = ""
    if document_hits:
        doc_context = "Relevant documents from knowledge base:\n"
        for i, hit in enumerate(document_hits[:4], 1):  # Top 4 documents
            # Support both 'content' and 'text' payload keys
            content = hit.payload.get('content') or hit.payload.get('text', '')
            page = hit.payload.get('page', 'unknown')
            doc_context += f"Document {i} (Page {page}): {content}...\n"
        doc_context += "\n"
    
    # Combine everything with PDF content getting priority
    if pdf_summary:
        full_prompt = f"""{chat_context}{pdf_context}{doc_context}Current question: {user_message}

Please respond in Traditional Chinese based on the uploaded PDF content first, then reference conversation history and any relevant documents from the knowledge base. Be specific and cite information from the PDF when relevant."""
    else:
        full_prompt = f"""{chat_context}{doc_context}Current question: {user_message}

Please respond in Traditional Chinese, referencing both the conversation history and any relevant documents. If referencing documents, include page numbers."""
    
    return full_prompt


@chat_bp.route('/message', methods=['POST'])
def send_message():
    """Send a message and get AI response with optional PDF upload"""
    try:
        # Check if this is multipart form data (with file) or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle multipart form data with file upload
            content = request.form.get('content', '').strip()
            room_id = request.form.get('room_id')
            # Normalize string placeholders to None so we auto-create a room
            if isinstance(room_id, str) and room_id.strip().lower() in {"null", "none", "undefined", ""}:
                room_id = None
            user_id = request.form.get('user_id')
            
            if not content:
                return jsonify(validation_error_response("content", "Content is required")), 400
            
            # Check for PDF file
            pdf_file = None
            if 'file' in request.files:
                pdf_file = request.files['file']
                if pdf_file.filename:
                    # Validate PDF file
                    try:
                        validate_pdf_file(pdf_file)
                        validate_file_size(pdf_file, max_size_mb=30)
                    except FileValidationError as e:
                        return jsonify(validation_error_response("file", str(e))), 400
        else:
            data = request.get_json()
            
            if not data:
                return jsonify(validation_error_response("body", "Request body is required")), 400
            
            room_id = data.get('room_id')
            # Normalize string placeholders to None so we auto-create a room
            if isinstance(room_id, str) and room_id.strip().lower() in {"null", "none", "undefined", ""}:
                room_id = None
            content = data.get('content', '').strip()
            user_id = data.get('user_id')
            
            if not content:
                return jsonify(validation_error_response("content", "Content is required")), 400
            
            pdf_file = None  # No file in JSON mode
        
        # Ensure room exists using room service
        room_id, _ = run_async(room_service.ensure_room_exists(room_id, content, user_id))
        
        # Process PDF file if provided using room service
        uploaded_files = []
        pdf_summary = None
        if pdf_file:
            pdf_summary, uploaded_files, pdf_error_message = room_service.process_pdf_file(pdf_file, room_id)
        
        # Add user message to chat history (with uploaded files)
        run_async(chat_service.add_message(room_id, "user", content, user_id, uploaded_files))
        
        # Check for PDF processing error first (skip RAG entirely if error exists)
        if 'pdf_error_message' in locals():
            # Return the PDF processing error as the AI response, skip RAG
            ai_response = pdf_error_message
            logger.info(f"🚨 PDF processing error detected, skipping RAG: {pdf_error_message[:100]}...")
        else:
            # Generate AI response using RAG with proper context (NEW: PDF summary integration)
            try:
                # 1. Get chat history FIRST for context
                chat_history = run_async(chat_service.get_chat_history(room_id, recency_k=20))
                logger.info(f"🔍 Chat history: {len(chat_history)} messages")
                
                # 2. Parse explicit constraints for doc_id/pages to preserve intent
                from src.rag.retrieval.retrieval_service import retrieval_service
                _cleaned, _constraints = retrieval_service.parse_constraints_for_text(content)
                has_explicit_constraints = bool(
                    (_constraints.get('doc_ids') and len(_constraints['doc_ids']) > 0) or
                    (_constraints.get('pages') and len(_constraints['pages']) > 0)
                )
                # 2b. Rewrite query using chat history for better retrieval (optional)
                try:
                    from src.config import RAG_DISABLE_QUERY_REWRITER
                except Exception:
                    RAG_DISABLE_QUERY_REWRITER = True
                if not RAG_DISABLE_QUERY_REWRITER:
                    from src.rag.utils import PromptRewriterLLM
                    rewriter = PromptRewriterLLM()
                    rewritten_query = rewriter.rewrite(content, chat_history)
                else:
                    rewritten_query = content
                logger.info(f"🔄 Original query: '{content}'")
                logger.info(f"🔄 Rewritten query: '{rewritten_query}'")
            
                # 3. Use router to select appropriate engine based on room configuration
                query_for_answer = rewritten_query
                logger.info("🧭 Using router to select appropriate RAG engine")
                ai_response = router.answer_legacy(query_for_answer, room_id, pdf_summary=pdf_summary)
             
            except TimeoutError as timeout_error:
                logger.error(f"RAG processing timeout: {timeout_error}")
                ai_response = "抱歉，處理時間過長。請稍後再試或嘗試更簡潔的問題。"
            except Exception as rag_error:
                logger.error(f"RAG processing error: {rag_error}")
                fallback_response = "抱歉，我在處理你的請求時遇到了問題。請再試一次。"
                ai_response = fallback_response
         
        # Add AI response to chat history
        run_async(chat_service.add_message(room_id, "ai", ai_response, None, []))
        
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