"""
Room Service for managing chat rooms and PDF processing.

This service handles room creation, validation, and PDF processing operations.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from werkzeug.datastructures import FileStorage

from .chat_service import chat_service
from ..utils.file_handler import validate_pdf_file, validate_file_size, FileValidationError
from src.rag.indexing.indexer import index_pdf
from src.config import QDRANT_COLLECTION
from src.data.pdf_summarizer import summarize_pdf_content, extract_pdf_text_for_chat

logger = logging.getLogger(__name__)


class RoomService:
    """Service for managing chat rooms and PDF processing."""
    
    def __init__(self):
        """Initialize the room service."""
        self.upload_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            'data', 
            'uploads'
        )
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def ensure_room_exists(self, room_id: Optional[str], content: str, user_id: Optional[str]) -> Tuple[str, bool]:
        """
        Ensure a room exists, creating it if necessary.
        
        Args:
            room_id: Optional room ID
            content: Initial content for room title
            user_id: Optional user ID
            
        Returns:
            Tuple of (room_id, is_new_room)
        """
        is_new_room = False
        
        # Auto-create room if no room_id provided
        if not room_id or room_id == "new":
            logger.info(f"🆕 Creating new room for content: {content[:50]}...")
            room_data = await chat_service.create_room(content, user_id)
            room_id = room_data["room_id"]
            is_new_room = True
            logger.info(f"✅ Created room {room_id[:8]} with title: {room_data['room_title']}")
        else:
            # Check if room exists, if not create it
            existing_room = await chat_service.get_room(room_id)
            if not existing_room:
                logger.info(f"🆕 Room {room_id[:8]} doesn't exist, creating it...")
                room_data = await chat_service.create_room(content, user_id)
                # Use the provided room_id instead of generated one
                room_data["room_id"] = room_id
                chat_service.rooms[room_id] = room_data
                is_new_room = True
        
        return room_id, is_new_room
    
    def process_pdf_file(self, pdf_file: FileStorage, room_id: str) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Process an uploaded PDF file.
        
        Args:
            pdf_file: Uploaded PDF file
            room_id: Room ID for scoping
            
        Returns:
            Tuple of (pdf_summary, uploaded_files, error_message)
        """
        uploaded_files = []
        pdf_summary = None
        error_message = None
        
        logger.info(f"📄 Processing PDF file: {pdf_file.filename}")
        
        # Save PDF temporarily for processing
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{pdf_file.filename}"
        temp_path = os.path.join(self.upload_dir, safe_filename)
        
        try:
            # Save file temporarily for processing
            pdf_file.save(temp_path)
            logger.info(f"💾 Saved PDF temporarily to: {temp_path}")
            
            # Process PDF based on size
            pdf_summary = self._process_pdf_by_size(temp_path, pdf_file.filename, room_id)
            
            # Add file metadata
            file_info = {
                "filename": pdf_file.filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            uploaded_files.append(file_info)
            
        except FileValidationError as e:
            error_message = str(e)
            logger.error(f"❌ PDF validation error: {error_message}")
        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ Error processing PDF: {error_message}")
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
                logger.info(f"🗑️ Cleaned up temporary file: {temp_path}")
            except:
                pass  # Ignore cleanup errors
        
        return pdf_summary, uploaded_files, error_message
    
    def _process_pdf_by_size(self, temp_path: str, filename: str, room_id: str) -> str:
        """
        Process PDF based on file size.
        
        Args:
            temp_path: Path to temporary PDF file
            filename: Original filename
            room_id: Room ID for scoping
            
        Returns:
            Processed PDF summary/text
        """
        # Check file size to determine strategy
        file_size = os.path.getsize(temp_path)
        size_mb = file_size / (1024 * 1024)
        
        if size_mb > 5:  # Large files get chunked summarization
            logger.info(f"📊 Large PDF ({size_mb:.1f}MB), using chunked summarization...")
            summary_result = summarize_pdf_content(
                temp_path, 
                max_pages=100,  # Allow more pages for chunked processing
                summary_length="medium"
            )
            
            if summary_result.get("error"):
                self._handle_summarization_error(summary_result["error"], filename)
            
            pdf_summary = summary_result["summary"]
            logger.info(f"✅ Generated chunked summary: {summary_result['summary_chars']} chars from {summary_result['total_chars']} original chars")
            
        elif size_mb > 1:  # Medium files get regular summarization
            logger.info(f"📄 Medium PDF ({size_mb:.1f}MB), generating summary...")
            summary_result = summarize_pdf_content(
                temp_path,
                max_pages=50,
                summary_length="medium"
            )
            
            if summary_result.get("error"):
                self._handle_summarization_error(summary_result["error"], filename)
            
            pdf_summary = summary_result["summary"]
            logger.info(f"✅ Generated summary: {summary_result['summary_chars']} chars from {summary_result['total_chars']} original chars")
            
        else:  # Small files get full text extraction
            logger.info(f"📄 Small PDF ({size_mb:.1f}MB), extracting full text...")
            extract_result = extract_pdf_text_for_chat(temp_path, max_chars=15000)  # Reduced limit
            
            if extract_result.get("error"):
                raise Exception(f"Text extraction failed: {extract_result['error']}")
            
            pdf_summary = extract_result["text"]
            logger.info(f"✅ Extracted text: {extract_result['extracted_chars']} chars from {extract_result['page_count']} pages")
        
        # Index PDF into Qdrant with room scoping for local RAG
        try:
            extra_payload = {"room_id": room_id, "scope": "chat"}
            _ = index_pdf(temp_path, collection_name=QDRANT_COLLECTION, doc_id=os.path.splitext(filename)[0], extra_payload=extra_payload)
            logger.info("✅ Indexed uploaded PDF into room-scoped Qdrant collection")
        except Exception as idx_err:
            logger.warning(f"⚠️ Failed to index uploaded PDF into Qdrant: {idx_err}")
        
        return pdf_summary
    
    def _handle_summarization_error(self, error: str, filename: str) -> None:
        """Handle summarization errors with appropriate messaging."""
        if "too large for chat processing" in error:
            error_msg = (
                f"📄 PDF '{filename}' is too large for chat processing. "
                f"Please use the RAG ingestion pipeline instead:\n\n"
                f"1. Upload via /api/v1/upload endpoint\n"
                f"2. Wait for processing to complete\n"
                f"3. Then ask questions about the document\n\n"
                f"Details: {error}"
            )
            raise Exception(error_msg)
        else:
            raise Exception(f"Summarization failed: {error}")


# Global instance
room_service = RoomService()
