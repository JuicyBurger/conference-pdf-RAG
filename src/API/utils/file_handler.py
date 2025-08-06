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


 