"""
Status Route Handlers

Handles system status and progress tracking endpoints.
"""

import logging
import sys
import os
from flask import Blueprint, request, jsonify

# Add the API directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services and utilities
from src.API.services.ingestion import ingestion_service
from src.API.services.chat_service import chat_service
from src.API.utils.response import success_response, error_response

# Import system status info
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import QDRANT_URL, QDRANT_COLLECTION

logger = logging.getLogger(__name__)
status_bp = Blueprint('status', __name__)


@status_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """Get detailed progress for a task"""
    try:
        task_status = ingestion_service.get_ingestion_status(task_id)
        
        if not task_status:
            return jsonify(error_response(
                f"Task not found: {task_id}",
                "TASK_NOT_FOUND"
            )), 404
        
        return jsonify(success_response(task_status, "Progress retrieved successfully"))
    
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return jsonify(error_response(f"Failed to get progress: {str(e)}")), 500


@status_bp.route('/system', methods=['GET'])
def get_system_status():
    """Get overall system status"""
    try:
        # Check Qdrant connection
        qdrant_status = "unknown"
        document_count = 0
        chat_collection_status = "unknown"
        
        try:
            from qdrant_client import QdrantClient
            from src.config import QDRANT_API_KEY
            
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # Check document collection
            try:
                doc_collection = client.get_collection(QDRANT_COLLECTION)
                document_count = doc_collection.points_count
                qdrant_status = "healthy"
            except:
                qdrant_status = "no_documents"
            
            # Check chat collection
            try:
                chat_collection = client.get_collection(chat_service.CHAT_COLLECTION)
                chat_message_count = chat_collection.points_count
                chat_collection_status = "healthy"
            except:
                chat_collection_status = "not_initialized"
                chat_message_count = 0
        
        except Exception as e:
            qdrant_status = f"error: {str(e)}"
            chat_collection_status = f"error: {str(e)}"
            chat_message_count = 0
        
        # Get recent ingestion tasks
        recent_tasks = ingestion_service.list_ingestion_history(limit=5)
        active_tasks = [task for task in recent_tasks if task["status"] in ["pending", "processing"]]
        
        return jsonify(success_response({
            "service": "sinon-rag-api",
            "status": "healthy",
            "qdrant": {
                "url": QDRANT_URL,
                "status": qdrant_status,
                "document_collection": QDRANT_COLLECTION,
                "document_count": document_count,
                "chat_collection": chat_service.CHAT_COLLECTION,
                "chat_collection_status": chat_collection_status,
                "chat_message_count": chat_message_count
            },
            "ingestion": {
                "active_tasks": len(active_tasks),
                "recent_tasks": len(recent_tasks),
                "task_details": active_tasks
            },
            "chat": {
                "buffer_rooms": len(chat_service.buffer.buffers),
                "service_status": "running"
            }
        }, "System status retrieved successfully"))
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify(error_response(f"Failed to get system status: {str(e)}")), 500


@status_bp.route('/tasks', methods=['GET'])
def list_all_tasks():
    """List all recent tasks with their status"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)
        status_filter = request.args.get('status')  # pending, processing, completed, failed
        
        limit = min(limit, 100)  # Cap at 100 tasks
        
        # Get all tasks
        all_tasks = ingestion_service.list_ingestion_history(limit * 2)  # Get more to filter
        
        # Filter by status if requested
        if status_filter:
            filtered_tasks = [task for task in all_tasks if task["status"] == status_filter]
        else:
            filtered_tasks = all_tasks
        
        # Limit results
        final_tasks = filtered_tasks[:limit]
        
        # Calculate summary statistics
        stats = {
            "total": len(final_tasks),
            "pending": len([t for t in final_tasks if t["status"] == "pending"]),
            "processing": len([t for t in final_tasks if t["status"] == "processing"]),
            "completed": len([t for t in final_tasks if t["status"] == "completed"]),
            "failed": len([t for t in final_tasks if t["status"] == "failed"])
        }
        
        return jsonify(success_response({
            "tasks": final_tasks,
            "statistics": stats,
            "filters": {
                "status": status_filter,
                "limit": limit
            }
        }, f"Retrieved {len(final_tasks)} tasks"))
    
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return jsonify(error_response(f"Failed to list tasks: {str(e)}")), 500


@status_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with Qdrant and Neo4j indicators"""
    try:
        # Qdrant
        qdrant_ok = False
        try:
            from qdrant_client import QdrantClient
            from src.config import QDRANT_API_KEY
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False

        # Neo4j
        neo4j_ok = False
        try:
            from src.rag.graph.graph_store import get_driver
            driver = get_driver()
            with driver.session() as session:
                session.run("RETURN 1 as ok")
            neo4j_ok = True
        except Exception:
            neo4j_ok = False

        status = "healthy" if (qdrant_ok and neo4j_ok) else ("degraded" if (qdrant_ok or neo4j_ok) else "down")
        return jsonify(success_response({
            "status": status,
            "qdrant": qdrant_ok,
            "neo4j": neo4j_ok
        }, "Health status"))
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify(error_response(f"Health check failed: {str(e)}")), 500