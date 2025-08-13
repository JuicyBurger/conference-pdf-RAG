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
from src.API.utils.file_handler import validate_pdf_file, validate_file_size, FileValidationError

# Import existing RAG components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.rag.router import router
from src.data.pdf_summarizer import summarize_pdf_content, extract_pdf_text_for_chat
from src.config import QDRANT_COLLECTION

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
                        validate_file_size(pdf_file, max_size_mb=50)
                    except FileValidationError as e:
                        return jsonify(validation_error_response("file", str(e))), 400
        else:
            # Handle JSON data (existing functionality)
            data = request.get_json()
            
            # Validate required fields
            if not data:
                return jsonify(validation_error_response("body", "Request body is required")), 400
            
            room_id = data.get('room_id')
            # Normalize string placeholders to None so we auto-create a room
            if isinstance(room_id, str) and room_id.strip().lower() in {"null", "none", "undefined", ""}:
                room_id = None
            content = data.get('content', '').strip()
            user_id = data.get('user_id')
            files = data.get('files', [])
            
            if not content:
                return jsonify(validation_error_response("content", "Content is required")), 400
            
            pdf_file = None  # No file in JSON mode
        
        # Auto-create room if no room_id provided
        is_new_room = False
        if not room_id or room_id == "new":
            print(f"🆕 Creating new room for content: {content[:50]}...")
            room_data = run_async(chat_service.create_room(content, user_id))
            room_id = room_data["room_id"]
            is_new_room = True
            print(f"✅ Created room {room_id[:8]} with title: {room_data['room_title']}")
        else:
            # Check if room exists, if not create it
            existing_room = run_async(chat_service.get_room(room_id))
            if not existing_room:
                print(f"🆕 Room {room_id[:8]} doesn't exist, creating it...")
                room_data = run_async(chat_service.create_room(content, user_id))
                # Use the provided room_id instead of generated one
                room_data["room_id"] = room_id
                chat_service.rooms[room_id] = room_data
                is_new_room = True
        
        # Process PDF file first if provided (NEW: Parse and summarize instead of ingesting)
        uploaded_files = []
        pdf_summary = None
        if pdf_file:
            print(f"📄 Processing PDF file: {pdf_file.filename}")
            
            # Ensure upload directory exists
            upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save PDF temporarily for processing
            import time
            timestamp = int(time.time())
            safe_filename = f"{timestamp}_{pdf_file.filename}"
            temp_path = os.path.join(upload_dir, safe_filename)
            
            # Save file temporarily for processing
            pdf_file.save(temp_path)
            print(f"💾 Saved PDF temporarily to: {temp_path}")
            
            try:
                # NEW: Summarize PDF content; also index to Qdrant scoped to room for local RAG
                print(f"🔄 Analyzing and summarizing PDF: {pdf_file.filename}")
                
                # Check file size to determine strategy
                file_size = os.path.getsize(temp_path)
                size_mb = file_size / (1024 * 1024)
                
                if size_mb > 5:  # Large files get chunked summarization
                    print(f"📊 Large PDF ({size_mb:.1f}MB), using chunked summarization...")
                    summary_result = summarize_pdf_content(
                        temp_path, 
                        max_pages=100,  # Allow more pages for chunked processing
                        summary_length="medium"
                    )
                    
                    if summary_result.get("error"):
                        # Check if it's a size threshold error
                        if "too large for chat processing" in summary_result["error"]:
                            error_msg = (
                                f"📄 PDF '{pdf_file.filename}' is too large for chat processing. "
                                f"Please use the RAG ingestion pipeline instead:\n\n"
                                f"1. Upload via /api/v1/upload endpoint\n"
                                f"2. Wait for processing to complete\n"
                                f"3. Then ask questions about the document\n\n"
                                f"Details: {summary_result['error']}"
                            )
                            raise Exception(error_msg)
                        else:
                            raise Exception(f"Summarization failed: {summary_result['error']}")
                    
                    pdf_summary = summary_result["summary"]
                    print(f"✅ Generated chunked summary: {summary_result['summary_chars']} chars from {summary_result['total_chars']} original chars")
                    
                elif size_mb > 1:  # Medium files get regular summarization
                    print(f"📄 Medium PDF ({size_mb:.1f}MB), generating summary...")
                    summary_result = summarize_pdf_content(
                        temp_path,
                        max_pages=50,
                        summary_length="medium"
                    )
                    
                    if summary_result.get("error"):
                        # Check if it's a size threshold error
                        if "too large for chat processing" in summary_result["error"]:
                            error_msg = (
                                f"📄 PDF '{pdf_file.filename}' is too large for chat processing. "
                                f"Please use the RAG ingestion pipeline instead:\n\n"
                                f"1. Upload via /api/v1/upload endpoint\n"
                                f"2. Wait for processing to complete\n"
                                f"3. Then ask questions about the document\n\n"
                                f"Details: {summary_result['error']}"
                            )
                            raise Exception(error_msg)
                        else:
                            raise Exception(f"Summarization failed: {summary_result['error']}")
                    
                    pdf_summary = summary_result["summary"]
                    print(f"✅ Generated summary: {summary_result['summary_chars']} chars from {summary_result['total_chars']} original chars")
                    
                else:  # Small files get full text extraction
                    print(f"📄 Small PDF ({size_mb:.1f}MB), extracting full text...")
                    extract_result = extract_pdf_text_for_chat(temp_path, max_chars=15000)  # Reduced limit
                    
                    if extract_result.get("error"):
                        raise Exception(f"Text extraction failed: {extract_result['error']}")
                    
                    pdf_summary = extract_result["text"]
                    print(f"✅ Extracted text: {extract_result['extracted_chars']} chars from {extract_result['page_count']} pages")
                
                # Add file metadata with summary info
                if size_mb > 5:
                    processing_type = "chunked_summary"
                elif size_mb > 1:
                    processing_type = "summary"
                else:
                    processing_type = "full_text"
                
                file_info = {
                    "filename": pdf_file.filename,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }
                uploaded_files.append(file_info)
                
                # Index PDF into Qdrant with room scoping for local RAG
                try:
                    from src.rag.indexing.indexer import index_pdf
                    from src.config import QDRANT_COLLECTION
                    extra_payload = {"room_id": room_id, "scope": "chat"}
                    _ = index_pdf(temp_path, collection_name=QDRANT_COLLECTION, doc_id=os.path.splitext(pdf_file.filename)[0], extra_payload=extra_payload)
                    print("✅ Indexed uploaded PDF into room-scoped Qdrant collection")
                except Exception as idx_err:
                    print(f"⚠️ Failed to index uploaded PDF into Qdrant: {idx_err}")
                
            except Exception as e:
                print(f"❌ Error processing PDF: {e}")
                pdf_summary = None
                # Store error message to return as AI response
                pdf_error_message = str(e)
                # Add minimal file info (no error details)
                file_info = {
                    "filename": pdf_file.filename,
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
                uploaded_files.append(file_info)
            
            finally:
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                    print(f"🗑️ Cleaned up temporary file: {temp_path}")
                except:
                    pass  # Ignore cleanup errors
        
        # Add user message to chat history (with uploaded files)
        user_msg_id = run_async(chat_service.add_message(room_id, "user", content, user_id, uploaded_files))
        
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
                from src.rag.retriever import parse_constraints_for_text
                _cleaned, _constraints = parse_constraints_for_text(content)
                has_explicit_constraints = bool(
                    (_constraints.get('doc_ids') and len(_constraints['doc_ids']) > 0) or
                    (_constraints.get('pages') and len(_constraints['pages']) > 0)
                )
                # 2b. Rewrite query using chat history for better retrieval
                from src.rag.llm_query_rewriter import PromptRewriterLLM
                rewriter = PromptRewriterLLM()
                rewritten_query = rewriter.rewrite(content, chat_history)
                logger.info(f"🔄 Original query: '{content}'")
                logger.info(f"🔄 Rewritten query: '{rewritten_query}'")
            
                # 3. Use router to generate answer (vector/graph/hybrid per room mode)
                query_for_answer = (
                    build_context_prompt(content, chat_history, None, pdf_summary)
                    if pdf_summary else rewritten_query
                )
                ai_response = router.answer(room_id, query_for_answer, pdf_summary=pdf_summary)
             
            except Exception as rag_error:
                logger.error(f"RAG processing error: {rag_error}")
                
                # Fallback response
                fallback_response = "抱歉，我在處理你的請求時遇到了問題。請再試一次。"
                ai_response = fallback_response
         
        # Add AI response to chat history
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