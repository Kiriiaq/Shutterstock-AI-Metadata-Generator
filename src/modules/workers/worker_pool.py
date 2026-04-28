"""
WorkerPool - Multithreaded/multiprocess image processing pipeline
"""

import hashlib
import logging
import multiprocessing
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 9


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    A processing task for the worker pool
    """

    task_id: str
    task_type: str  # 'read_metadata', 'ai_analyze', 'write_metadata', 'validate', 'export'
    file_path: Path
    priority: TaskPriority = TaskPriority.NORMAL

    # Task parameters
    params: Dict[str, Any] = field(default_factory=dict)

    # State tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    # Callbacks
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value

    @property
    def duration_ms(self) -> Optional[int]:
        """Get task duration in milliseconds"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None


@dataclass
class BatchResult:
    """
    Results from a batch operation
    """

    batch_id: str
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    results: List[Tuple[str, Any]] = field(default_factory=list)  # (file_path, result)
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (file_path, error)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class WorkerPool:
    """
    Thread pool for I/O-bound tasks (metadata reading/writing, file operations)
    """

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize worker pool

        Args:
            max_workers: Maximum concurrent workers. Default: CPU count * 2 for threads
            use_processes: Use processes instead of threads (for CPU-bound tasks)
        """
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers = cpu_count * 2 if not use_processes else cpu_count

        self.max_workers = max_workers
        self.use_processes = use_processes

        # Task queue with priority
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Active tasks tracking
        self._active_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}

        # Executor
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None

        # Control flags
        self._running = False
        self._lock = threading.Lock()

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

        # Task handlers
        self._handlers: Dict[str, Callable] = {}

        logger.info(f"WorkerPool initialized with {max_workers} workers (processes={use_processes})")

    def register_handler(self, task_type: str, handler: Callable):
        """
        Register a handler function for a task type

        Args:
            task_type: Task type name
            handler: Function that takes (file_path, params) and returns result
        """
        self._handlers[task_type] = handler
        logger.debug(f"Registered handler for task type: {task_type}")

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        Set callback for progress updates

        Args:
            callback: Function(completed, total, current_file)
        """
        self._progress_callback = callback

    def start(self):
        """Start the worker pool"""
        if self._running:
            return

        self._running = True

        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info("WorkerPool started")

    def stop(self, wait: bool = True):
        """
        Stop the worker pool

        Args:
            wait: Wait for pending tasks to complete
        """
        self._running = False

        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

        logger.info("WorkerPool stopped")

    def submit_task(self, task: Task) -> str:
        """
        Submit a task for processing

        Args:
            task: Task to submit

        Returns:
            Task ID
        """
        if not task.task_id:
            task.task_id = str(uuid.uuid4())

        with self._lock:
            self._active_tasks[task.task_id] = task
            # Priority queue uses (priority, counter, task) to handle equal priorities
            self._task_queue.put((task.priority.value, time.time(), task))

        return task.task_id

    def submit_batch(
        self,
        file_paths: List[Path],
        task_type: str,
        params: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """
        Submit a batch of files for processing

        Args:
            file_paths: List of file paths
            task_type: Type of task to run
            params: Parameters for all tasks
            priority: Task priority

        Returns:
            Batch ID
        """
        batch_id = str(uuid.uuid4())
        params = params or {}
        params["batch_id"] = batch_id

        for file_path in file_paths:
            task = Task(
                task_id=f"{batch_id}_{file_path.name}",
                task_type=task_type,
                file_path=file_path,
                priority=priority,
                params=params.copy(),
            )
            self.submit_task(task)

        logger.info(f"Submitted batch {batch_id} with {len(file_paths)} tasks")
        return batch_id

    def process_queue(self, timeout: Optional[float] = None) -> BatchResult:
        """
        Process all tasks in the queue

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            BatchResult with all results
        """
        if not self._executor:
            self.start()

        batch_id = str(uuid.uuid4())
        batch_result = BatchResult(batch_id=batch_id, total_tasks=self._task_queue.qsize())

        start_time = time.time()
        futures: Dict[Future, Task] = {}

        while not self._task_queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Processing timeout reached")
                break

            try:
                _, _, task = self._task_queue.get_nowait()
            except queue.Empty:
                break

            if task.task_type not in self._handlers:
                logger.error(f"No handler for task type: {task.task_type}")
                task.status = TaskStatus.FAILED
                task.error = f"No handler registered for: {task.task_type}"
                batch_result.failed_tasks += 1
                batch_result.errors.append((str(task.file_path), task.error))
                continue

            task.status = TaskStatus.QUEUED
            handler = self._handlers[task.task_type]

            future = self._executor.submit(self._execute_task, handler, task.file_path, task.params)
            futures[future] = task

        # Collect results
        for future in as_completed(futures):
            task = futures[future]
            task.completed_at = datetime.now()

            try:
                result = future.result()
                task.result = result
                task.status = TaskStatus.COMPLETED
                batch_result.completed_tasks += 1
                batch_result.results.append((str(task.file_path), result))

                if task.on_complete:
                    task.on_complete(task)

            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                batch_result.failed_tasks += 1
                batch_result.errors.append((str(task.file_path), str(e)))
                logger.error(f"Task failed for {task.file_path}: {e}")

                if task.on_error:
                    task.on_error(task, e)

            # Update progress
            if self._progress_callback:
                completed = batch_result.completed_tasks + batch_result.failed_tasks
                self._progress_callback(completed, batch_result.total_tasks, str(task.file_path))

            # Move to completed
            with self._lock:
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                self._completed_tasks[task.task_id] = task

        batch_result.end_time = datetime.now()
        return batch_result

    def _execute_task(self, handler: Callable, file_path: Path, params: Dict[str, Any]) -> Any:
        """Execute a single task (runs in worker thread/process)"""
        return handler(file_path, params)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        with self._lock:
            if task_id in self._active_tasks:
                return self._active_tasks[task_id].status
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id].status
        return None

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task"""
        with self._lock:
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id].result
        return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task

        Returns:
            True if cancelled, False if not found or already running
        """
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    return True
        return False

    def clear_completed(self):
        """Clear completed tasks from memory"""
        with self._lock:
            self._completed_tasks.clear()

    @property
    def pending_count(self) -> int:
        """Number of pending tasks"""
        return self._task_queue.qsize()

    @property
    def active_count(self) -> int:
        """Number of active/running tasks"""
        with self._lock:
            return sum(1 for t in self._active_tasks.values() if t.status == TaskStatus.RUNNING)


class ProcessingPipeline:
    """
    Multi-stage processing pipeline for image workflows
    Chains multiple operations together
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize processing pipeline

        Args:
            max_workers: Maximum concurrent workers per stage
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()

        # Pipeline stages
        self._stages: List[Tuple[str, Callable]] = []

        # Worker pools for each stage
        self._pools: Dict[str, WorkerPool] = {}

        # Progress tracking
        self._progress_callback: Optional[Callable] = None
        self._current_stage: int = 0
        self._total_stages: int = 0

    def add_stage(self, name: str, handler: Callable[[Path, Dict[str, Any]], Any], use_processes: bool = False):
        """
        Add a processing stage

        Args:
            name: Stage name
            handler: Handler function(file_path, params) -> result
            use_processes: Use process pool instead of thread pool
        """
        self._stages.append((name, handler))

        # Create pool for this stage
        pool = WorkerPool(max_workers=self.max_workers, use_processes=use_processes)
        pool.register_handler(name, handler)
        self._pools[name] = pool

        logger.info(f"Added pipeline stage: {name}")

    def set_progress_callback(self, callback: Callable[[str, int, int, int, int], None]):
        """
        Set callback for progress updates

        Args:
            callback: Function(stage_name, stage_num, total_stages, completed, total)
        """
        self._progress_callback = callback

    def process(
        self, file_paths: List[Path], initial_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, BatchResult]:
        """
        Process files through all pipeline stages

        Args:
            file_paths: Files to process
            initial_params: Initial parameters passed to first stage

        Returns:
            Dict of stage_name -> BatchResult
        """
        if not self._stages:
            raise ValueError("Pipeline has no stages defined")

        results: Dict[str, BatchResult] = {}
        current_files = list(file_paths)
        params = initial_params or {}

        self._total_stages = len(self._stages)

        for stage_idx, (stage_name, handler) in enumerate(self._stages):
            self._current_stage = stage_idx + 1

            logger.info(f"Starting pipeline stage {self._current_stage}/{self._total_stages}: {stage_name}")

            pool = self._pools[stage_name]
            pool.start()

            # Setup progress callback for this stage
            if self._progress_callback:

                def stage_progress(completed, total, current_file):
                    self._progress_callback(stage_name, self._current_stage, self._total_stages, completed, total)

                pool.set_progress_callback(stage_progress)

            # Submit all files
            for file_path in current_files:
                task = Task(
                    task_id=f"{stage_name}_{file_path.name}_{uuid.uuid4().hex[:8]}",
                    task_type=stage_name,
                    file_path=file_path,
                    params=params.copy(),
                )
                pool.submit_task(task)

            # Process and collect results
            batch_result = pool.process_queue()
            results[stage_name] = batch_result

            # Filter files for next stage (only successful ones)
            current_files = [Path(file_path) for file_path, _ in batch_result.results]

            # Merge results into params for next stage
            params["previous_stage"] = stage_name
            params["previous_results"] = {str(fp): result for fp, result in batch_result.results}

            pool.stop()

            logger.info(
                f"Stage {stage_name} complete: {batch_result.completed_tasks}/{batch_result.total_tasks} succeeded"
            )

            # Stop pipeline if all files failed
            if not current_files:
                logger.warning(f"Pipeline stopped: no files passed stage {stage_name}")
                break

        return results

    def stop(self):
        """Stop all worker pools"""
        for pool in self._pools.values():
            pool.stop()


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file

    Args:
        file_path: Path to file
        chunk_size: Read chunk size

    Returns:
        Hex string of file hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def collect_image_files(
    directory: Path,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    exclude_folders: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """
    Collect all image files from a directory with filtering options

    Args:
        directory: Source directory
        recursive: Search subdirectories
        extensions: File extensions to include (default: common image formats)
        exclude_extensions: File extensions to exclude
        exclude_folders: Folder names to exclude (e.g., ['_backup', 'thumbs', 'cache'])
        exclude_patterns: Glob patterns to exclude (e.g., ['*_thumb.*', '*.bak'])

    Returns:
        List of image file paths
    """
    import fnmatch

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".eps"]

    if exclude_extensions is None:
        exclude_extensions = []

    if exclude_folders is None:
        exclude_folders = []

    if exclude_patterns is None:
        exclude_patterns = []

    # Normalize exclude folders to lowercase for comparison
    exclude_folders_lower = [f.lower() for f in exclude_folders]

    directory = Path(directory)
    files = []

    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        # Skip if in exclude list
        if ext.lower() in [e.lower() for e in exclude_extensions]:
            continue

        for file_path in directory.glob(f"{pattern}{ext}"):
            files.append(file_path)
        for file_path in directory.glob(f"{pattern}{ext.upper()}"):
            files.append(file_path)

    # Filter out excluded folders
    if exclude_folders_lower:
        filtered_files = []
        for file_path in files:
            # Check if any parent folder is in exclude list
            skip = False
            for parent in file_path.parents:
                if parent.name.lower() in exclude_folders_lower:
                    skip = True
                    break
            if not skip:
                filtered_files.append(file_path)
        files = filtered_files

    # Filter out excluded patterns
    if exclude_patterns:
        filtered_files = []
        for file_path in files:
            skip = False
            for pat in exclude_patterns:
                if fnmatch.fnmatch(file_path.name.lower(), pat.lower()):
                    skip = True
                    break
            if not skip:
                filtered_files.append(file_path)
        files = filtered_files

    # Remove duplicates and sort
    files = sorted(set(files))

    logger.info(f"Found {len(files)} image files in {directory}")
    return files


# Default stopwords for keyword cleaning
DEFAULT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "here",
    "there",
    "then",
    "once",
    "image",
    "photo",
    "picture",
    "stock",
    "shutterstock",
    "photography",
}


def clean_keywords_advanced(
    keywords: List[str],
    stopwords: Optional[set] = None,
    blacklist: Optional[set] = None,
    min_length: int = 2,
    max_length: int = 64,
    max_keywords: int = 50,
    remove_duplicates: bool = True,
    lowercase: bool = True,
) -> List[str]:
    """
    Advanced keyword cleaning with stopwords and blacklist support

    Args:
        keywords: List of keywords to clean
        stopwords: Set of words to remove (default: DEFAULT_STOPWORDS)
        blacklist: Set of forbidden words to remove
        min_length: Minimum keyword length
        max_length: Maximum keyword length
        max_keywords: Maximum number of keywords to return
        remove_duplicates: Remove duplicate keywords
        lowercase: Convert to lowercase

    Returns:
        Cleaned list of keywords
    """
    import re

    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    if blacklist is None:
        blacklist = set()

    # Combine stopwords and blacklist
    excluded_words = stopwords | blacklist

    cleaned = []
    seen = set()

    for kw in keywords:
        # Normalize
        if lowercase:
            kw = kw.lower()
        kw = kw.strip()

        # Remove special characters except hyphen and space
        kw = re.sub(r"[^\w\s-]", "", kw)

        # Normalize whitespace
        kw = re.sub(r"\s+", " ", kw).strip()

        # Skip if too short or too long
        if len(kw) < min_length or len(kw) > max_length:
            continue

        # Skip if in stopwords or blacklist
        if kw in excluded_words:
            continue

        # Skip duplicates
        if remove_duplicates:
            if kw in seen:
                continue
            seen.add(kw)

        cleaned.append(kw)

        # Stop if we have enough
        if len(cleaned) >= max_keywords:
            break

    return cleaned
