"""
PDF Ingestion Service

Handles PDF upload processing with real-time progress tracking.
Integrates with existing RAG indexer for document processing.
"""

import asyncio
import logging
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
from src.indexing import index_nodes_vector
from src.data.pdf_ingestor import build_page_nodes
from src.config import QDRANT_COLLECTION
from werkzeug.utils import secure_filename
from src.rag.graph.indexer import ingest_pdfs_to_graph, build_graph_from_nodes
from src.rag.utils import handle_errors, DatabaseError, setup_logger
from src.indexing import VectorIndexer
from src.rag.graph.indexer import extract_entities_relations_and_index
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from src.config import (
    QDRANT_COLLECTION,
    INDEX_BATCH_SIZE,
    GRAPH_BATCH_SIZE,
    TRAINING_CORPUS_ID,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_DATABASE,
    SKIP_GRAPH_EXTRACTION,
    GRAPH_WRITE_CHUNKS_TO_NEO4J,
)

import json
from math import ceil
from time import time

# Configure logging
logger = setup_logger(__name__)


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
        # Allow limited concurrency; user will pause LLM during embedding
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.progress = IngestionProgress()
        os.makedirs(upload_dir, exist_ok=True)
        # Use our centralized logger instead of creating a new one
        self.logger = logger
    
    def save_uploaded_files_sync(self, files: list) -> tuple[list, str]:
        """Save uploaded files synchronously and return file infos and task ID.

        Returns a list of dicts: { 'path': saved_path, 'original_filename': original, 'preferred_doc_id': original_without_ext }
        """
        task_id = f"ingest-{uuid.uuid4()}"
        saved_files_info = []
        for file in files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = file.filename or f"upload_{uuid.uuid4()}.pdf"
            # Save with a safe name on disk, but keep original filename for doc_id
            safe_name = secure_filename(original_filename) or f"upload_{uuid.uuid4()}.pdf"
            filename = f"{timestamp}_{safe_name}"
            file_path = os.path.join(self.upload_dir, filename)
            # Save file synchronously
            file.save(file_path)
            saved_files_info.append({
                "path": file_path,
                "original_filename": original_filename,
                "preferred_doc_id": os.path.splitext(original_filename)[0],
            })
        return saved_files_info, task_id
    
    @handle_errors(error_class=DatabaseError, reraise=True)
    def _process_single_pdf(self, file_path: str, task_id: str, file_index: int, progress_callback: Callable = None, original_filename: str | None = None, preferred_doc_id: str | None = None, room_id: str | None = None, scope: str | None = None) -> int:
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
            
            # Process the PDF using existing indexer (pass through preferred doc_id to preserve user-visible name)
            # Scope: if room_id provided, mark as chat-scoped; else global
            extra_payload = {}
            if room_id:
                extra_payload["room_id"] = room_id
            # default scope if provided, else leave unset
            if scope:
                extra_payload["scope"] = scope

            # Extract nodes and index using new system
            nodes = build_page_nodes(file_path)

            # Optionally write chunks to Neo4j before vector indexing
            if GRAPH_WRITE_CHUNKS_TO_NEO4J:
                try:
                    self.logger.info(f"[Ingestion] Writing chunks to Neo4j for doc_id={preferred_doc_id}")
                    # Respect training corpus scoping when available
                    corpus_id = extra_payload.get("corpus_id") if isinstance(extra_payload, dict) else None
                    _ = build_graph_from_nodes(nodes, preferred_doc_id, corpus_id=corpus_id)
                except Exception as e:
                    self.logger.warning(f"[Ingestion] Neo4j chunk ingest skipped due to error: {e}")

            result = index_nodes_vector(
                nodes=nodes,
                doc_id=preferred_doc_id,
                collection_name=QDRANT_COLLECTION,
                extra_payload=extra_payload or None,
            )
            chunks_indexed = result.indexed_count if result.success else 0
            
            # Update progress
            self.progress.update_task(
                task_id,
                files_processed=file_index + 1,
                chunks_indexed=self.progress.get_task(task_id).get("chunks_indexed", 0) + chunks_indexed,
                message=f"Completed {original_filename or filename} ({chunks_indexed} chunks)"
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
    
    @handle_errors(error_class=DatabaseError, reraise=False)
    async def _ingest_files_async(self, files_info: list, task_id: str, progress_callback: Callable = None):
        """Async wrapper for file ingestion"""
        try:
            loop = asyncio.get_event_loop()
            total_chunks = 0

            for i, info in enumerate(files_info):
                file_path = info["path"]
                original_filename = info["original_filename"]
                preferred_doc_id = info["preferred_doc_id"]
                # Calculate progress percentage
                progress_pct = int((i / len(files_info)) * 100)
                self.progress.update_task(task_id, progress=progress_pct)

                # Process file
                chunks = await loop.run_in_executor(
                    self.executor,
                    self._process_single_pdf,
                    file_path, task_id, i, progress_callback, original_filename, preferred_doc_id
                )
                total_chunks += chunks

                # Clean up uploaded file after processing
                try:
                    os.remove(file_path)
                except Exception as cleanup_err:
                    # Non-fatal cleanup errors
                    self.logger.debug(f"Cleanup failed for {file_path}: {cleanup_err}")
            
            # Add document IDs to query parser for future queries
            try:
                from src.rag.retrieval.retrieval_service import retrieval_service
                for info in files_info:
                    doc_id = info["preferred_doc_id"]
                    retrieval_service.add_new_doc_id_to_parser(doc_id)
                    self.logger.info(f"Added '{doc_id}' to query parser vocabulary")
            except Exception as e:
                self.logger.warning(f"Failed to add document IDs to query parser: {e}")
            
            # Mark as completed
            self.progress.complete_task(task_id, success=True)
            final_task = self.progress.get_task(task_id)
            final_task["total_chunks"] = total_chunks
            
            if progress_callback:
                progress_callback(task_id, f"Ingestion completed! Indexed {total_chunks} chunks from {len(files_info)} files.")
            
        except Exception as e:
            self.progress.complete_task(task_id, success=False, error=str(e))
            if progress_callback:
                progress_callback(task_id, f"Ingestion failed: {str(e)}")
    
    def _run_background(self, files_info: list, task_id: str, progress_callback: Callable = None):
        """Run the async ingest in a dedicated event loop thread"""
        try:
            asyncio.run(self._ingest_files_async(files_info, task_id, progress_callback))
        except Exception as e:
            # Ensure task is marked failed if top-level error occurs
            self.progress.complete_task(task_id, success=False, error=str(e))
            if progress_callback:
                progress_callback(task_id, f"Ingestion failed: {str(e)}")

    def start_ingestion(self, files: list, progress_callback: Callable = None) -> str:
        """Start PDF ingestion process with progress tracking (synchronous kickoff)."""
        # Save files synchronously and create task
        files_info, task_id = self.save_uploaded_files_sync(files)
        file_names = [info["original_filename"] for info in files_info]

        # Create progress tracking
        self.progress.create_task(task_id, len(files_info), file_names)

        # Start background thread to run async ingestion
        threading.Thread(
            target=self._run_background,
            args=(files_info, task_id, progress_callback),
            daemon=True
        ).start()

        return task_id

    @handle_errors(error_class=DatabaseError, fallback_return="")
    def start_training_ingestion(self, files: list, training_room_id: str | None, progress_callback: Callable = None, *, skip_graph: bool | None = None, index_batch_size: int | None = None, graph_batch_size: int | None = None) -> str:
        """Start GraphRAG training ingestion with progress tracking.
        Optional overrides: skip_graph, index_batch_size, graph_batch_size
        """
        # Save files synchronously and create task
        files_info, task_id = self.save_uploaded_files_sync(files)
        file_names = [info["original_filename"] for info in files_info]

        # Mark as processing
        self.progress.create_task(task_id, len(files_info), file_names)

        # Resolve effective settings
        effective_skip_graph = SKIP_GRAPH_EXTRACTION if skip_graph is None else bool(skip_graph)
        effective_index_bs = INDEX_BATCH_SIZE if index_batch_size is None else int(index_batch_size)
        effective_graph_bs = GRAPH_BATCH_SIZE if graph_batch_size is None else int(graph_batch_size)

        def _run():
            try:
                total_files = len(files_info)
                processed = 0
                for i, info in enumerate(files_info):
                    self.progress.update_task(task_id, status="processing", current_file=info["original_filename"])
                    if progress_callback:
                        progress_callback(task_id, f"Training ingest {info['original_filename']}")
                    try:
                        # Prepared artifacts path
                        from src.config import PREPARED_DIR
                        # Prefer cache folder named after original filename (no extension)
                        original_doc_id = info["preferred_doc_id"]
                        saved_doc_id = os.path.splitext(os.path.basename(info["path"]))[0]
                        base_dir_original = os.path.join(PREPARED_DIR, original_doc_id)
                        base_dir_saved = os.path.join(PREPARED_DIR, saved_doc_id)
                        # Choose base_dir: if original cache exists use it, else saved name
                        if os.path.exists(os.path.join(base_dir_original, "nodes_all.json")):
                            doc_id = original_doc_id
                            base_dir = base_dir_original
                        else:
                            doc_id = saved_doc_id
                            base_dir = base_dir_saved
                        # Build prepared if missing; write under original name for stable caching
                        if not os.path.exists(os.path.join(base_dir, "nodes_all.json")):
                            print(f"🔧 Building prepared data for '{original_doc_id}'...")
                            from src.data.pdf_ingestor import build_page_nodes
                            build_page_nodes(info["path"], save_prepared=True, doc_id_override=original_doc_id)
                            doc_id = original_doc_id
                            base_dir = base_dir_original
                        else:
                            print(f"✅ Using existing prepared data for '{original_doc_id}'")
                        # Load nodes for batch processing
                        import json
                        with open(os.path.join(base_dir, "nodes_all.json"), "r", encoding="utf-8") as f:
                            nodes = json.load(f)
                        print(f"📊 Loaded {len(nodes)} nodes for batch processing")
                        print(f"🔢 Batch configuration: INDEX_BATCH_SIZE={effective_index_bs}, GRAPH_BATCH_SIZE={effective_graph_bs}")
                        print(f"📦 Estimated batches: {ceil(len(nodes) / effective_index_bs)} index batches, {ceil(len(nodes) / effective_graph_bs)} graph batches")
                        
                        # Write chunks to Neo4j (if enabled)
                        if GRAPH_WRITE_CHUNKS_TO_NEO4J:
                            try:
                                print(f"🔗 Writing chunks to Neo4j for doc_id={doc_id}")
                                from src.rag.graph.indexer import build_graph_from_nodes
                                chunk_count = build_graph_from_nodes(nodes, doc_id, corpus_id=TRAINING_CORPUS_ID)
                                print(f"✅ Wrote {chunk_count} chunks to Neo4j")
                            except Exception as e:
                                print(f"⚠️ Neo4j chunk ingest failed: {e}")
                        
                        # KG (optional)
                        if effective_skip_graph:
                            erc = {"entities": 0, "relations": 0, "communities": 0, "qdrant_vectors": 0}
                        else:
                            # Temporarily patch batch size for this call
                            global GRAPH_BATCH_SIZE
                            old_gbs = GRAPH_BATCH_SIZE
                            GRAPH_BATCH_SIZE = effective_graph_bs
                            try:
                                erc = self._batch_graph_extract(nodes, doc_id, TRAINING_CORPUS_ID, base_dir, task_id)
                            finally:
                                GRAPH_BATCH_SIZE = old_gbs
                        # Qdrant indexing
                        extra_payload = {"corpus_id": TRAINING_CORPUS_ID, "scope": "graph"}
                        global INDEX_BATCH_SIZE
                        old_ibs = INDEX_BATCH_SIZE
                        INDEX_BATCH_SIZE = effective_index_bs
                        try:
                            vec_count = self._batch_index_nodes(nodes, doc_id, extra_payload, base_dir, task_id)
                        finally:
                            INDEX_BATCH_SIZE = old_ibs
                        self.progress.update_task(task_id, message=f"Neo4j E={erc.get('entities')} R={erc.get('relations')} / Qdrant {vec_count}")
                    finally:
                        try:
                            os.remove(info["path"])
                        except Exception:
                            pass
                    processed += 1
                    self.progress.update_task(task_id, files_processed=processed, progress=int(processed * 100 / total_files))
                self.progress.complete_task(task_id, success=True)
            except Exception as e:
                self.progress.complete_task(task_id, success=False, error=str(e))
                if progress_callback:
                    progress_callback(task_id, f"Training ingestion failed: {e}")

        threading.Thread(target=_run, daemon=True).start()
        return task_id
    
    def get_ingestion_status(self, task_id: str) -> Optional[dict]:
        """Get ingestion progress status"""
        return self.progress.get_task(task_id)
    
    def list_ingestion_history(self, limit: int = 50) -> list:
        """List recent ingestion tasks"""
        return self.progress.list_tasks(limit)

    def _batch_index_nodes(self, nodes: list, doc_id: str, extra_payload: dict | None, base_dir: str, task_id: str) -> int:
        indexer = VectorIndexer(QDRANT_COLLECTION)
        # Proactively ensure payload indexes (including TEXT on 'text') before batching
        try:
            indexer._ensure_indexes_exist()
        except Exception as e:
            self.progress.update_task(task_id, message=f"Warning: failed ensuring payload indexes: {e}")
        total = 0
        total_batches = ceil(len(nodes) / INDEX_BATCH_SIZE) if nodes else 0
        print(f"🚀 Starting Qdrant indexing: {len(nodes)} nodes in {total_batches} batches of {INDEX_BATCH_SIZE}")
        ckpt_path = os.path.join(base_dir, ".index_checkpoint.json")
        start_batch = 0
        try:
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    start_batch = int((json.load(f) or {}).get("next_batch", 0))
        except Exception:
            start_batch = 0
        for b in range(start_batch, total_batches):
            t0 = time()
            batch = nodes[b*INDEX_BATCH_SIZE:(b+1)*INDEX_BATCH_SIZE]
            print(f"📦 Processing Qdrant batch {b+1}/{total_batches} ({len(batch)} nodes)...")
            self.progress.update_task(task_id, message=f"Indexing batch {b+1}/{total_batches} ({len(batch)} nodes)...")
            res = indexer.index_nodes(batch, doc_id, extra_payload)
            dt = time() - t0
            total += res.indexed_count if res.success else 0
            self.progress.update_task(task_id, message=f"Indexed batch {b+1}/{total_batches} in {dt:.1f}s (+{res.indexed_count if res.success else 0})")
            try:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"next_batch": b+1}, f)
            except Exception:
                pass
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        except Exception:
            pass
        return total

    def _batch_graph_extract(self, nodes: list, doc_id: str, corpus_id: str, base_dir: str, task_id: str) -> dict:
        if SKIP_GRAPH_EXTRACTION:
            self.progress.update_task(task_id, message="Skipping KG extraction (SKIP_GRAPH_EXTRACTION=1)")
            return {"entities": 0, "relations": 0, "communities": 0, "qdrant_vectors": 0}
        print(f"🧠 Starting KG extraction: {len(nodes)} nodes in {ceil(len(nodes) / GRAPH_BATCH_SIZE)} batches of {GRAPH_BATCH_SIZE}")
        graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database=NEO4J_DATABASE,
        )
        texts = [n.get("text", "") for n in nodes if n.get("text")]  
        total_batches = ceil(len(texts) / GRAPH_BATCH_SIZE) if texts else 0
        summary = {"entities": 0, "relations": 0, "communities": 0, "qdrant_vectors": 0}
        ckpt_path = os.path.join(base_dir, ".graph_checkpoint.json")
        start_batch = 0
        try:
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    start_batch = int((json.load(f) or {}).get("next_batch", 0))
        except Exception:
            start_batch = 0
        for b in range(start_batch, total_batches):
            t0 = time()
            chunk_texts = texts[b*GRAPH_BATCH_SIZE:(b+1)*GRAPH_BATCH_SIZE]
            print(f"🧠 Processing KG batch {b+1}/{total_batches} ({len(chunk_texts)} texts)...")
            self.progress.update_task(task_id, message=f"KG extract batch {b+1}/{total_batches} ({len(chunk_texts)} texts)...")
            stats = extract_entities_relations_and_index(chunk_texts, doc_id, corpus_id, graph_store)
            dt = time() - t0
            for k in ("entities", "relations", "communities", "qdrant_vectors"):
                try:
                    summary[k] = (summary.get(k, 0) or 0) + (stats.get(k, 0) or 0)
                except Exception:
                    pass
            self.progress.update_task(task_id, message=f"KG batch {b+1}/{total_batches} done in {dt:.1f}s")
            try:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"next_batch": b+1}, f)
            except Exception:
                pass
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        except Exception:
            pass
        return summary


# Global service instance
ingestion_service = AsyncIngestionService() 