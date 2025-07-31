"""
Upload Route Handlers

Handles PDF file upload and ingestion with progress tracking.
"""

import logging
import sys
import os
from flask import Blueprint, request, jsonify

# Add the API directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services and utilities
from src.API.services.ingestion import ingestion_service
from src.API.utils.response import success_response, error_response, validation_error_response, progress_response
from src.API.utils.file_handler import validate_multiple_files, FileValidationError

logger = logging.getLogger(__name__)
upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/pdf', methods=['POST'])
def upload_pdf():
    """Upload PDF files for ingestion"""
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify(validation_error_response("files", "No files provided")), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify(validation_error_response("files", "No files selected")), 400
        
        # Validate files
        try:
            validated_files = validate_multiple_files(files, max_files=10, max_size_mb=50)
        except FileValidationError as e:
            return jsonify(validation_error_response("files", str(e))), 400
        
        # Start ingestion process
        def progress_callback(task_id, message):
            """Progress callback for real-time updates"""
            logger.info(f"Task {task_id}: {message}")
        
        task_id = ingestion_service.start_ingestion(validated_files, progress_callback)
        
        # Get initial task status
        task_status = ingestion_service.get_ingestion_status(task_id)
        
        return jsonify(success_response({
            "task_id": task_id,
            "status": task_status,
            "files_uploaded": len(validated_files),
            "file_names": [f.filename for f in validated_files]
        }, "PDF upload started successfully")), 202  # 202 Accepted for async processing
    
    except Exception as e:
        logger.error(f"Error in upload_pdf: {e}")
        return jsonify(error_response(f"Failed to upload PDF: {str(e)}")), 500


@upload_bp.route('/status/<task_id>', methods=['GET'])
def get_upload_status(task_id):
    """Get upload/ingestion status"""
    try:
        task_status = ingestion_service.get_ingestion_status(task_id)
        
        if not task_status:
            return jsonify(error_response(
                f"Task not found: {task_id}",
                "TASK_NOT_FOUND"
            )), 404
        
        return jsonify(progress_response(task_id, task_status))
    
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        return jsonify(error_response(f"Failed to get status: {str(e)}")), 500


@upload_bp.route('/history', methods=['GET'])
def get_upload_history():
    """Get upload history"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        limit = min(limit, 100)  # Cap at 100 tasks
        
        # Get task history
        history = ingestion_service.list_ingestion_history(limit)
        
        return jsonify(success_response({
            "tasks": history,
            "task_count": len(history)
        }, f"Retrieved {len(history)} ingestion tasks"))
    
    except Exception as e:
        logger.error(f"Error getting upload history: {e}")
        return jsonify(error_response(f"Failed to get history: {str(e)}")), 500


@upload_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_upload(task_id):
    """Cancel an ongoing upload/ingestion (placeholder)"""
    try:
        # TODO: Implement task cancellation
        # For now, return not implemented
        return jsonify(error_response(
            "Task cancellation not yet implemented",
            "NOT_IMPLEMENTED"
        )), 501
    
    except Exception as e:
        logger.error(f"Error cancelling upload: {e}")
        return jsonify(error_response(f"Failed to cancel upload: {str(e)}")), 500 