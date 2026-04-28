"""
Worker pool for multithreaded/multiprocess image processing
"""

from .worker_pool import ProcessingPipeline, WorkerPool

__all__ = ["WorkerPool", "ProcessingPipeline"]
