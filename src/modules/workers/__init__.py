"""
Worker pool for multithreaded/multiprocess image processing
"""

from .worker_pool import WorkerPool, ProcessingPipeline

__all__ = ["WorkerPool", "ProcessingPipeline"]
