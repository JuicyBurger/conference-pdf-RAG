"""
PDF Ingestion Service

Handles PDF upload processing with real-time progress tracking.
Integrates with existing RAG indexer for document processing.
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Optional, Callable
import threading
import time

# Import from our existing RAG system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.rag.indexing.indexer import index_pdf
from src.config import QDRANT_COLLECTION


class IngestionProgress:
    """Thread-safe progress tracking for ingestion tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, dict] = {}
        self.lock = threading.Lock()
    
    def create_task(self, task_id: str, total_files: int, file_names: list) -> dict:
        """Create a new ingestion task"""
        with self.lock:
            task = {
                "task_id": task_id,
                "status": "pending",  # pending, processing, completed, failed
                "progress": 0,  # 0-100 percentage
                "current_file": None,
                "files_processed": 0,
                "total_files": total_files,
                "file_names": file_names,
                "message": "Initializing...",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "estimated_completion": None,
                "error": None,
                "chunks_indexed": 0
            }
            self.tasks[task_id] = task
            return task.copy()
    
    def update_task(self, task_id: str, **updates) -> Optional[dict]:
        """Update task progress"""
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            self.tasks[task_id].update(updates)
            
            # Calculate estimated completion
            if updates.get("files_processed", 0) > 0 and updates.get("status") == "processing":
                task = self.tasks[task_id]
                elapsed = time.time() - time.mktime(
                    datetime.fromisoformat(task["started_at"].replace('Z', '+00:00')).timetuple()
                )
                avg_time_per_file = elapsed / task["files_processed"]
                remaining_files = task["total_files"] - task["files_processed"]
                estimated_seconds = remaining_files * avg_time_per_file
                
                estimated_completion = datetime.now(timezone.utc)
                estimated_completion = estimated_completion.replace(
                    second=estimated_completion.second + int(estimated_seconds)
                )
                self.tasks[task_id]["estimated_completion"] = estimated_completion.isoformat()
            
            return self.tasks[task_id].copy()
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task status"""
        with self.lock:
            return self.tasks.get(task_id, {}).copy() if task_id in self.tasks else None
    
    def complete_task(self, task_id: str, success: bool = True, error: str = None) -> Optional[dict]:
        """Mark task as completed or failed"""
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            self.tasks[task_id].update({
                "status": "completed" if success else "failed",
                "progress": 100 if success else self.tasks[task_id]["progress"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "message": "Completed successfully" if success else f"Failed: {error}",
                "error": error if not success else None
            })
            
            return self.tasks[task_id].copy()
    
    def list_tasks(self, limit: int = 50) -> list:
        """List recent tasks"""
        with self.lock:
            tasks = list(self.tasks.values())
            # Sort by started_at, newest first
            tasks.sort(key=lambda x: x["started_at"], reverse=True)
            return tasks[:limit]


class AsyncIngestionService:
    """Async PDF ingestion service with progress tracking"""
    
    def __init__(self, upload_dir: str = "data/uploads"):
        self.upload_dir = upload_dir
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent ingestions
        self.progress = IngestionProgress()
        os.makedirs(upload_dir, exist_ok=True)
    
    async def save_uploaded_files(self, files: list) -> tuple[list, str]:
        """Save uploaded files and return paths and task ID"""
        task_id = f"ingest-{uuid.uuid4()}"
        saved_paths = []
        
        for file in files:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # Save file
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: file.save(file_path)
            )
            saved_paths.append(file_path)
        
        return saved_paths, task_id
    
    def _process_single_pdf(self, file_path: str, task_id: str, file_index: int, progress_callback: Callable = None) -> int:
        """Process a single PDF file with progress updates"""
        try:
            filename = os.path.basename(file_path)
            
            # Update progress
            self.progress.update_task(
                task_id,
                status="processing",
                current_file=filename,
                message=f"Processing {filename}..."
            )
            
            if progress_callback:
                progress_callback(task_id, f"Starting {filename}")
            
            # Process the PDF using existing indexer
            chunks_indexed = index_pdf(file_path, collection_name=QDRANT_COLLECTION)
            
            # Update progress
            self.progress.update_task(
                task_id,
                files_processed=file_index + 1,
                chunks_indexed=self.progress.get_task(task_id).get("chunks_indexed", 0) + chunks_indexed,
                message=f"Completed {filename} ({chunks_indexed} chunks)"
            )
            
            if progress_callback:
                progress_callback(task_id, f"Completed {filename}")
            
            return chunks_indexed
            
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
            self.progress.update_task(task_id, message=error_msg)
            if progress_callback:
                progress_callback(task_id, error_msg)
            raise
    
    async def _ingest_files_async(self, file_paths: list, task_id: str, progress_callback: Callable = None):
        """Async wrapper for file ingestion"""
        try:
            loop = asyncio.get_event_loop()
            total_chunks = 0
            
            for i, file_path in enumerate(file_paths):
                # Calculate progress percentage
                progress_pct = int((i / len(file_paths)) * 100)
                self.progress.update_task(task_id, progress=progress_pct)
                
                # Process file
                chunks = await loop.run_in_executor(
                    self.executor,
                    self._process_single_pdf,
                    file_path, task_id, i, progress_callback
                )
                total_chunks += chunks
                
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass  # Ignore cleanup errors
            
            # Mark as completed
            self.progress.complete_task(task_id, success=True)
            final_task = self.progress.get_task(task_id)
            final_task["total_chunks"] = total_chunks
            
            if progress_callback:
                progress_callback(task_id, f"Ingestion completed! Indexed {total_chunks} chunks from {len(file_paths)} files.")
            
        except Exception as e:
            self.progress.complete_task(task_id, success=False, error=str(e))
            if progress_callback:
                progress_callback(task_id, f"Ingestion failed: {str(e)}")
    
    async def start_ingestion(self, files: list, progress_callback: Callable = None) -> str:
        """Start PDF ingestion process with progress tracking"""
        
        # Save files and create task
        file_paths, task_id = await self.save_uploaded_files(files)
        file_names = [os.path.basename(path) for path in file_paths]
        
        # Create progress tracking
        self.progress.create_task(task_id, len(file_paths), file_names)
        
        # Start ingestion in background
        asyncio.create_task(self._ingest_files_async(file_paths, task_id, progress_callback))
        
        return task_id
    
    def get_ingestion_status(self, task_id: str) -> Optional[dict]:
        """Get ingestion progress status"""
        return self.progress.get_task(task_id)
    
    def list_ingestion_history(self, limit: int = 50) -> list:
        """List recent ingestion tasks"""
        return self.progress.list_tasks(limit)


# Global service instance
ingestion_service = AsyncIngestionService() 