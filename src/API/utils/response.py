"""
Response Utilities

Standardized response formatting for API endpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def success_response(data: Any = None, message: str = "Success") -> Dict:
    """Create a standardized success response"""
    return {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def error_response(message: str, error_code: str = "GENERAL_ERROR", details: Any = None) -> Dict:
    """Create a standardized error response"""
    return {
        "status": "error",
        "message": message,
        "error_code": error_code,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def validation_error_response(field: str, message: str) -> Dict:
    """Create a validation error response"""
    return error_response(
        message=f"Validation error: {message}",
        error_code="VALIDATION_ERROR",
        details={"field": field, "error": message}
    )


def not_found_response(resource: str, resource_id: str = None) -> Dict:
    """Create a not found error response"""
    message = f"{resource} not found"
    if resource_id:
        message += f": {resource_id}"
    
    return error_response(
        message=message,
        error_code="NOT_FOUND",
        details={"resource": resource, "resource_id": resource_id}
    )


def progress_response(task_id: str, progress_data: Dict) -> Dict:
    """Create a progress tracking response"""
    return {
        "status": "progress",
        "task_id": task_id,
        "data": progress_data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    } 