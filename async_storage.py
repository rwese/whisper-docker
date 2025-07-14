"""
File-based async task storage for stateless processing
"""

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from transcription_core import TranscriptionService


@dataclass
class TaskMetadata:
    id: str
    filename: str
    file_hash: str
    status: str  # pending, processing, completed, failed
    model: str
    language: Optional[str]
    output_format: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    ttl_expires_at: str = None
    file_size: int = 0
    error_message: Optional[str] = None
    processing_duration: Optional[float] = None


class AsyncStorage:
    """File-based storage for async transcription tasks"""

    def __init__(self, storage_path: str = "storage", ttl_days: int = 7):
        self.storage_path = Path(storage_path)
        self.ttl_days = ttl_days

        # Create directory structure
        self.tasks_dir = self.storage_path / "tasks"
        self.files_dir = self.storage_path / "files"
        self.results_dir = self.storage_path / "results"

        for dir_path in [self.tasks_dir, self.files_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="async-transcription-"
        )
        self.processing_queue = asyncio.Queue()

        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        # Cache for transcription services
        self.transcription_services = {}
        self.transcription_services_lock = threading.Lock()

    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()

    def _get_task_path(self, task_id: str) -> Path:
        """Get path to task metadata file"""
        return self.tasks_dir / f"{task_id}.json"

    def _get_file_path(self, file_hash: str) -> Path:
        """Get path to stored file"""
        return self.files_dir / file_hash

    def _get_result_path(self, task_id: str) -> Path:
        """Get path to result file"""
        return self.results_dir / f"{task_id}.json"

    def _load_task(self, task_id: str) -> Optional[TaskMetadata]:
        """Load task metadata from disk"""
        task_path = self._get_task_path(task_id)
        if not task_path.exists():
            return None

        try:
            with open(task_path, "r") as f:
                data = json.load(f)
            return TaskMetadata(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _save_task(self, task: TaskMetadata):
        """Save task metadata to disk"""
        task_path = self._get_task_path(task.id)
        with open(task_path, "w") as f:
            json.dump(asdict(task), f, indent=2)

    def _find_task_by_hash(self, file_hash: str) -> Optional[str]:
        """Find existing task ID for a file hash"""
        for task_file in self.tasks_dir.glob("*.json"):
            try:
                with open(task_file, "r") as f:
                    data = json.load(f)
                if data.get("file_hash") == file_hash:
                    return data.get("id")
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def create_task(
        self,
        filename: str,
        file_content: bytes,
        model: str,
        language: Optional[str] = None,
        output_format: str = "json",
    ) -> Dict:
        """
        Create a new transcription task

        Returns dict with task_id, filename, file_hash, created_at
        """
        file_hash = self._calculate_file_hash(file_content)

        # Check if file already exists and has a pending/completed task
        existing_task_id = self._find_task_by_hash(file_hash)
        if existing_task_id:
            existing_task = self._load_task(existing_task_id)
            if existing_task and existing_task.status in [
                "pending",
                "processing",
                "completed",
            ]:
                return {
                    "task_id": existing_task.id,
                    "filename": existing_task.filename,
                    "file_hash": file_hash,
                    "created_at": existing_task.created_at,
                    "duplicate": True,
                }

        # Create new task
        task_id = str(uuid.uuid4())
        now = datetime.utcnow()
        ttl_expires = now + timedelta(days=self.ttl_days)

        task = TaskMetadata(
            id=task_id,
            filename=filename,
            file_hash=file_hash,
            status="pending",
            model=model,
            language=language,
            output_format=output_format,
            created_at=now.isoformat(),
            ttl_expires_at=ttl_expires.isoformat(),
            file_size=len(file_content),
        )

        # Store file if it doesn't exist
        file_path = self._get_file_path(file_hash)
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(file_content)

        # Save task metadata
        self._save_task(task)

        # Queue for processing
        asyncio.create_task(self._queue_task(task_id))

        return {
            "task_id": task_id,
            "filename": filename,
            "file_hash": file_hash,
            "created_at": task.created_at,
            "duplicate": False,
        }

    async def _queue_task(self, task_id: str):
        """Add task to processing queue"""
        await self.processing_queue.put(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status and metadata"""
        task = self._load_task(task_id)
        if not task:
            return None

        # Calculate processing duration if completed
        duration_seconds = None
        if task.started_at and task.completed_at:
            start = datetime.fromisoformat(task.started_at)
            end = datetime.fromisoformat(task.completed_at)
            duration_seconds = (end - start).total_seconds()

        return {
            "task_id": task.id,
            "filename": task.filename,
            "file_hash": task.file_hash,
            "status": task.status,
            "model": task.model,
            "language": task.language,
            "output_format": task.output_format,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "ttl_expires_at": task.ttl_expires_at,
            "file_size": task.file_size,
            "error_message": task.error_message,
            "processing_duration_seconds": duration_seconds,
        }

    def get_task_result(self, task_id: str) -> Optional[Dict]:
        """Get transcription result for completed task"""
        task = self._load_task(task_id)
        if not task or task.status != "completed":
            return None

        result_path = self._get_result_path(task_id)
        if not result_path.exists():
            return None

        try:
            with open(result_path, "r") as f:
                result = json.load(f)

            # Add metadata
            result["task_metadata"] = self.get_task_status(task_id)
            return result
        except (json.JSONDecodeError, IOError):
            return None

    def _get_transcription_service(self, model_name: str) -> TranscriptionService:
        """Get or create transcription service for model (thread-safe)"""
        with self.transcription_services_lock:
            if model_name not in self.transcription_services:
                self.transcription_services[model_name] = TranscriptionService(
                    model_name
                )
            return self.transcription_services[model_name]

    def _process_task(self, task_id: str):
        """Process a single transcription task"""
        task = self._load_task(task_id)
        if not task or task.status != "pending":
            return

        try:
            # Update status to processing
            task.status = "processing"
            task.started_at = datetime.utcnow().isoformat()
            self._save_task(task)

            # Load file content
            file_path = self._get_file_path(task.file_hash)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {task.file_hash}")

            with open(file_path, "rb") as f:
                file_content = f.read()

            # Get transcription service
            service = self._get_transcription_service(task.model)

            # Perform transcription
            result = service.transcribe_buffer(
                file_content, filename=task.filename, language=task.language
            )

            # Add character count
            result["character_count"] = len(result["text"])

            # Save result
            result_path = self._get_result_path(task_id)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Update task status
            task.status = "completed"
            task.completed_at = datetime.utcnow().isoformat()

            # Calculate processing duration
            if task.started_at:
                start = datetime.fromisoformat(task.started_at)
                end = datetime.fromisoformat(task.completed_at)
                task.processing_duration = (end - start).total_seconds()

            self._save_task(task)

        except Exception as e:
            # Update task with error
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow().isoformat()
            self._save_task(task)

    async def start_processing_loop(self):
        """Start the async processing loop"""
        while True:
            try:
                # Wait for task in queue
                task_id = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )

                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self._process_task, task_id)

            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def _cleanup_worker(self):
        """Background worker to clean up expired tasks"""
        while True:
            try:
                self._cleanup_expired_tasks()
                # Run cleanup every hour
                time.sleep(3600)
            except Exception as e:
                print(f"Error in cleanup worker: {e}")
                time.sleep(60)  # Retry after 1 minute on error

    def _cleanup_expired_tasks(self):
        """Remove expired tasks and files"""
        now = datetime.utcnow()

        # Find expired tasks
        expired_tasks = []
        for task_file in self.tasks_dir.glob("*.json"):
            try:
                with open(task_file, "r") as f:
                    data = json.load(f)

                ttl_expires = datetime.fromisoformat(data.get("ttl_expires_at", ""))
                if now > ttl_expires:
                    expired_tasks.append(data)

            except (json.JSONDecodeError, ValueError, KeyError):
                # Invalid task file, remove it
                task_file.unlink(missing_ok=True)

        # Clean up expired tasks
        for task_data in expired_tasks:
            task_id = task_data.get("id")
            file_hash = task_data.get("file_hash")

            if task_id:
                # Remove task file
                self._get_task_path(task_id).unlink(missing_ok=True)

                # Remove result file
                self._get_result_path(task_id).unlink(missing_ok=True)

            # Check if file is still referenced by other tasks
            if file_hash and not self._find_task_by_hash(file_hash):
                # No other tasks reference this file, remove it
                self._get_file_path(file_hash).unlink(missing_ok=True)

    def list_tasks(self, limit: int = 100) -> List[Dict]:
        """List recent tasks (for debugging/admin)"""
        tasks = []
        for task_file in sorted(
            self.tasks_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
        ):
            if len(tasks) >= limit:
                break

            try:
                with open(task_file, "r") as f:
                    data = json.load(f)
                tasks.append(data)
            except (json.JSONDecodeError, IOError):
                continue

        return tasks


# Global storage instance
storage = AsyncStorage()
