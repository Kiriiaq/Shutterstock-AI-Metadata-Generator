"""
Core module - Configuration, logging, and database functionality.
"""

from .logger import setup_logger, get_logger, app_logger
from .params import ShutterstockParams, ParamMeta, PARAMS_META
from .config_manager import ConfigManager
from ..modules.storage.database import Database, ActionType, AuditLog, MetadataHistory

__all__ = [
    "setup_logger",
    "get_logger",
    "app_logger",
    "ShutterstockParams",
    "ParamMeta",
    "PARAMS_META",
    "ConfigManager",
    "Database",
    "ActionType",
    "AuditLog",
    "MetadataHistory",
]
