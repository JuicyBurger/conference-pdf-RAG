"""
File Handling Utilities

File upload validation and processing utilities.
"""

import os
import mimetypes
from typing import List, Tuple, Optional
from werkzeug.datastructures import FileStorage


class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass


def validate_pdf_file(file: FileStorage) -> bool:
    """Validate that uploaded file is a PDF"""
    if not file:
        raise FileValidationError("No file provided")
    
    if not file.filename:
        raise FileValidationError("File has no filename")
    
    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        raise FileValidationError("File must be a PDF")
    
    # Check MIME type
    file_mime = file.content_type
    if file_mime and file_mime not in ['application/pdf']:
        raise FileValidationError("Invalid file type. Only PDF files are allowed")
    
    return True


def validate_file_size(file: FileStorage, max_size_mb: int = 50) -> bool:
    """Validate file size"""
    if not file:
        raise FileValidationError("No file provided")
    
    # Get file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    
    if file_size > max_size_bytes:
        raise FileValidationError(f"File size exceeds {max_size_mb}MB limit")
    
    if file_size == 0:
        raise FileValidationError("File is empty")
    
    return True


def validate_multiple_files(files: List[FileStorage], max_files: int = 10, max_size_mb: int = 50) -> List[FileStorage]:
    """Validate multiple uploaded files"""
    if not files:
        raise FileValidationError("No files provided")
    
    if len(files) > max_files:
        raise FileValidationError(f"Too many files. Maximum {max_files} files allowed")
    
    validated_files = []
    total_size = 0
    
    for i, file in enumerate(files):
        try:
            validate_pdf_file(file)
            validate_file_size(file, max_size_mb)
            
            # Calculate total size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            total_size += file_size
            
            validated_files.append(file)
            
        except FileValidationError as e:
            raise FileValidationError(f"File {i+1} ({file.filename}): {str(e)}")
    
    # Check total size limit
    max_total_size = max_files * max_size_mb * 1024 * 1024
    if total_size > max_total_size:
        raise FileValidationError(f"Total file size exceeds limit of {max_files * max_size_mb}MB")
    
    return validated_files


def get_safe_filename(filename: str) -> str:
    """Generate a safe filename by removing dangerous characters"""
    if not filename:
        return "unnamed_file.pdf"
    
    # Keep only alphanumeric, dots, dashes, and underscores
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '.-_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    safe_filename = ''.join(safe_chars)
    
    # Ensure it ends with .pdf
    if not safe_filename.lower().endswith('.pdf'):
        safe_filename += '.pdf'
    
    return safe_filename


def estimate_processing_time(file_size_bytes: int) -> int:
    """Estimate processing time in seconds based on file size"""
    # Rough estimate: 1MB = 10 seconds processing time
    mb_size = file_size_bytes / (1024 * 1024)
    estimated_seconds = int(mb_size * 10)
    
    # Minimum 30 seconds, maximum 10 minutes
    return max(30, min(estimated_seconds, 600)) 