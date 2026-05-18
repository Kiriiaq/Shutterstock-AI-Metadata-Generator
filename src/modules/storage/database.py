"""
SQLite database layer for audit logging and metadata history
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of auditable actions"""

    METADATA_READ = "metadata_read"
    METADATA_WRITE = "metadata_write"
    AI_ANALYSIS = "ai_analysis"
    VALIDATION = "validation"
    EXPORT_CSV = "export_csv"
    UPLOAD_FTP = "upload_ftp"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    ERROR = "error"


@dataclass
class AuditLog:
    """
    Audit log entry for tracking all operations
    """

    id: Optional[int] = None
    timestamp: datetime = None
    action_type: ActionType = ActionType.METADATA_READ
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    batch_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MetadataHistory:
    """
    Historical metadata record for tracking changes
    """

    id: Optional[int] = None
    file_path: str = ""
    file_hash: str = ""
    timestamp: datetime = None
    metadata_type: str = "iptc"  # 'iptc', 'xmp', 'shutterstock'
    metadata_json: str = ""
    source: str = "ai_generated"  # 'ai_generated', 'user_input', 'imported', 'existing'
    version: int = 1

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Database:
    """
    SQLite database manager for Shutterstock AI v2.0
    Thread-safe with connection pooling
    """

    SCHEMA_VERSION = 1

    CREATE_TABLES_SQL = """
    -- Audit log table
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        action_type TEXT NOT NULL,
        file_path TEXT,
        file_hash TEXT,
        details TEXT,
        success INTEGER NOT NULL DEFAULT 1,
        error_message TEXT,
        duration_ms INTEGER,
        batch_id TEXT
    );

    -- Metadata history table
    CREATE TABLE IF NOT EXISTS metadata_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        metadata_type TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        source TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1
    );

    -- Processing queue table
    CREATE TABLE IF NOT EXISTS processing_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT UNIQUE NOT NULL,
        file_path TEXT NOT NULL,
        operations TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 5,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        result TEXT,
        error TEXT
    );

    -- Settings table
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Batch tracking table
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT UNIQUE NOT NULL,
        name TEXT,
        source_folder TEXT NOT NULL,
        output_folder TEXT,
        total_files INTEGER NOT NULL DEFAULT 0,
        processed_files INTEGER NOT NULL DEFAULT 0,
        failed_files INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        options TEXT
    );

    -- File status tracking
    CREATE TABLE IF NOT EXISTS file_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
        file_hash TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        last_modified TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        has_metadata INTEGER NOT NULL DEFAULT 0,
        has_ai_analysis INTEGER NOT NULL DEFAULT 0,
        last_processed TEXT,
        batch_id TEXT
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_file_path ON audit_log(file_path);
    CREATE INDEX IF NOT EXISTS idx_audit_batch_id ON audit_log(batch_id);
    CREATE INDEX IF NOT EXISTS idx_history_file_path ON metadata_history(file_path);
    CREATE INDEX IF NOT EXISTS idx_history_file_hash ON metadata_history(file_hash);
    CREATE INDEX IF NOT EXISTS idx_queue_status ON processing_queue(status);
    CREATE INDEX IF NOT EXISTS idx_file_status_path ON file_status(file_path);
    CREATE INDEX IF NOT EXISTS idx_file_status_hash ON file_status(file_hash);

    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database

        Args:
            db_path: Path to SQLite database file. Default: user data directory
        """
        if db_path is None:
            # Default to user's app data directory
            app_data = Path.home() / ".shutterstock_ai"
            app_data.mkdir(exist_ok=True)
            db_path = app_data / "shutterstock_ai.db"

        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize database
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create tables
        cursor.executescript(self.CREATE_TABLES_SQL)

        # Check and update schema version
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()

        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))
        elif row[0] < self.SCHEMA_VERSION:
            self._migrate_schema(row[0], self.SCHEMA_VERSION)

        conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    def _migrate_schema(self, from_version: int, to_version: int):
        """Migrate database schema between versions"""
        logger.info(f"Migrating database from v{from_version} to v{to_version}")
        # Add migrations here as needed
        pass

    # ==================== Audit Log Methods ====================

    def log_action(
        self,
        action_type: ActionType,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        batch_id: Optional[str] = None,
    ) -> int:
        """
        Log an action to the audit log

        Returns:
            ID of the created log entry
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO audit_log
            (timestamp, action_type, file_path, file_hash, details, success, error_message, duration_ms, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                action_type.value,
                file_path,
                file_hash,
                json.dumps(details) if details else None,
                1 if success else 0,
                error_message,
                duration_ms,
                batch_id,
            ),
        )

        conn.commit()
        return cursor.lastrowid

    def get_audit_logs(
        self,
        file_path: Optional[str] = None,
        action_type: Optional[ActionType] = None,
        batch_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Query audit logs with filters

        Returns:
            List of AuditLog entries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)

        if action_type:
            query += " AND action_type = ?"
            params.append(action_type.value)

        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)

        logs = []
        for row in cursor.fetchall():
            logs.append(
                AuditLog(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    action_type=ActionType(row["action_type"]),
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    details=json.loads(row["details"]) if row["details"] else None,
                    success=bool(row["success"]),
                    error_message=row["error_message"],
                    duration_ms=row["duration_ms"],
                    batch_id=row["batch_id"],
                )
            )

        return logs

    # ==================== Metadata History Methods ====================

    def save_metadata_history(
        self, file_path: str, file_hash: str, metadata_type: str, metadata: Dict[str, Any], source: str = "ai_generated"
    ) -> int:
        """
        Save metadata to history

        Returns:
            ID of the created history entry
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get current version for this file
        cursor.execute(
            """
            SELECT MAX(version) FROM metadata_history
            WHERE file_path = ? AND metadata_type = ?
        """,
            (file_path, metadata_type),
        )

        row = cursor.fetchone()
        version = (row[0] or 0) + 1

        cursor.execute(
            """
            INSERT INTO metadata_history
            (file_path, file_hash, timestamp, metadata_type, metadata_json, source, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                file_path,
                file_hash,
                datetime.now().isoformat(),
                metadata_type,
                json.dumps(metadata, ensure_ascii=False),
                source,
                version,
            ),
        )

        conn.commit()
        return cursor.lastrowid

    def get_metadata_history(self, file_path: str, metadata_type: Optional[str] = None) -> List[MetadataHistory]:
        """
        Get metadata history for a file

        Returns:
            List of MetadataHistory entries ordered by version desc
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM metadata_history WHERE file_path = ?"
        params = [file_path]

        if metadata_type:
            query += " AND metadata_type = ?"
            params.append(metadata_type)

        query += " ORDER BY version DESC"

        cursor.execute(query, params)

        history = []
        for row in cursor.fetchall():
            history.append(
                MetadataHistory(
                    id=row["id"],
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata_type=row["metadata_type"],
                    metadata_json=row["metadata_json"],
                    source=row["source"],
                    version=row["version"],
                )
            )

        return history

    def get_latest_metadata(self, file_path: str, metadata_type: str = "shutterstock") -> Optional[Dict[str, Any]]:
        """
        Get the most recent metadata for a file

        Returns:
            Metadata dict or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT metadata_json FROM metadata_history
            WHERE file_path = ? AND metadata_type = ?
            ORDER BY version DESC LIMIT 1
        """,
            (file_path, metadata_type),
        )

        row = cursor.fetchone()
        if row:
            return json.loads(row["metadata_json"])
        return None

    # ==================== Settings Methods ====================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row:
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]
        return default

    def set_setting(self, key: str, value: Any):
        """Set a setting value"""
        conn = self._get_connection()
        cursor = conn.cursor()

        value_str = json.dumps(value) if not isinstance(value, str) else value

        cursor.execute(
            """
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """,
            (key, value_str, datetime.now().isoformat()),
        )

        conn.commit()

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT key, value FROM settings")

        settings = {}
        for row in cursor.fetchall():
            try:
                settings[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                settings[row["key"]] = row["value"]

        return settings

    # ==================== Batch Tracking Methods ====================

    def create_batch(
        self,
        batch_id: str,
        source_folder: str,
        total_files: int,
        name: Optional[str] = None,
        output_folder: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a new batch record"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO batches
            (batch_id, name, source_folder, output_folder, total_files, created_at, status, options)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
            (
                batch_id,
                name or batch_id,
                source_folder,
                output_folder,
                total_files,
                datetime.now().isoformat(),
                json.dumps(options) if options else None,
            ),
        )

        conn.commit()
        return cursor.lastrowid

    def update_batch_progress(self, batch_id: str, processed: int, failed: int = 0, status: Optional[str] = None):
        """Update batch progress"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                """
                UPDATE batches
                SET processed_files = ?, failed_files = ?, status = ?
                WHERE batch_id = ?
            """,
                (processed, failed, status, batch_id),
            )
        else:
            cursor.execute(
                """
                UPDATE batches
                SET processed_files = ?, failed_files = ?
                WHERE batch_id = ?
            """,
                (processed, failed, batch_id),
            )

        conn.commit()

    def complete_batch(self, batch_id: str, status: str = "completed"):
        """Mark batch as completed"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE batches
            SET status = ?, completed_at = ?
            WHERE batch_id = ?
        """,
            (status, datetime.now().isoformat(), batch_id),
        )

        conn.commit()

    # ==================== File Status Methods ====================

    def set_file_flags(
        self,
        file_path: str,
        has_metadata: Optional[bool] = None,
        has_ai_analysis: Optional[bool] = None,
    ) -> None:
        """Update boolean flags on file_status without touching hash/size/last_modified.

        Creates a minimal placeholder row if none exists, then UPDATEs only the
        requested flags. Safe to call from pipelines that don't track full
        identity (e.g. AI analysis pipeline).
        """
        if has_metadata is None and has_ai_analysis is None:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        cursor.execute(
            """
            INSERT OR IGNORE INTO file_status
                (file_path, file_hash, file_size, last_modified, last_processed)
            VALUES (?, '', 0, ?, ?)
            """,
            (file_path, now, now),
        )

        updates: List[str] = []
        params: List[Any] = []
        if has_metadata is not None:
            updates.append("has_metadata = ?")
            params.append(1 if has_metadata else 0)
        if has_ai_analysis is not None:
            updates.append("has_ai_analysis = ?")
            params.append(1 if has_ai_analysis else 0)
        updates.append("last_processed = ?")
        params.append(now)
        params.append(file_path)

        cursor.execute(
            f"UPDATE file_status SET {', '.join(updates)} WHERE file_path = ?",
            params,
        )
        conn.commit()

    def update_file_status(
        self,
        file_path: str,
        file_hash: str,
        file_size: int,
        last_modified: datetime,
        status: str = "pending",
        has_metadata: bool = False,
        has_ai_analysis: bool = False,
        batch_id: Optional[str] = None,
    ):
        """Update or create file status record"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO file_status
            (file_path, file_hash, file_size, last_modified, status, has_metadata, has_ai_analysis, last_processed, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                file_path,
                file_hash,
                file_size,
                last_modified.isoformat(),
                status,
                1 if has_metadata else 0,
                1 if has_ai_analysis else 0,
                datetime.now().isoformat(),
                batch_id,
            ),
        )

        conn.commit()

    def get_file_status(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific file"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM file_status WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_pending_files(self, batch_id: Optional[str] = None, limit: int = 100) -> List[str]:
        """Get list of pending files"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if batch_id:
            cursor.execute(
                """
                SELECT file_path FROM file_status
                WHERE status = 'pending' AND batch_id = ?
                LIMIT ?
            """,
                (batch_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT file_path FROM file_status
                WHERE status = 'pending'
                LIMIT ?
            """,
                (limit,),
            )

        return [row["file_path"] for row in cursor.fetchall()]

    # ==================== Statistics Methods ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Total files processed
        cursor.execute("SELECT COUNT(*) FROM file_status WHERE status = 'completed'")
        stats["total_processed"] = cursor.fetchone()[0]

        # Total with AI analysis
        cursor.execute("SELECT COUNT(*) FROM file_status WHERE has_ai_analysis = 1")
        stats["with_ai_analysis"] = cursor.fetchone()[0]

        # Total with metadata
        cursor.execute("SELECT COUNT(*) FROM file_status WHERE has_metadata = 1")
        stats["with_metadata"] = cursor.fetchone()[0]

        # Recent errors
        cursor.execute("""
            SELECT COUNT(*) FROM audit_log
            WHERE success = 0 AND timestamp >= datetime('now', '-24 hours')
        """)
        stats["recent_errors"] = cursor.fetchone()[0]

        # Batches summary
        cursor.execute("SELECT COUNT(*), SUM(processed_files), SUM(failed_files) FROM batches")
        row = cursor.fetchone()
        stats["total_batches"] = row[0]
        stats["total_batch_processed"] = row[1] or 0
        stats["total_batch_failed"] = row[2] or 0

        return stats

    def close(self):
        """Close database connection"""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def vacuum(self):
        """Optimize database"""
        conn = self._get_connection()
        conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def export_audit_log(self, output_path: Path, format: str = "json") -> int:
        """
        Export audit log to file

        Args:
            output_path: Output file path
            format: 'json' or 'csv'

        Returns:
            Number of records exported
        """
        logs = self.get_audit_logs(limit=10000)

        if format == "json":
            data = []
            for log in logs:
                data.append(
                    {
                        "id": log.id,
                        "timestamp": log.timestamp.isoformat(),
                        "action_type": log.action_type.value,
                        "file_path": log.file_path,
                        "success": log.success,
                        "error_message": log.error_message,
                        "duration_ms": log.duration_ms,
                        "batch_id": log.batch_id,
                        "details": log.details,
                    }
                )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["ID", "Timestamp", "Action", "File Path", "Success", "Error", "Duration (ms)", "Batch ID"]
                )
                for log in logs:
                    writer.writerow(
                        [
                            log.id,
                            log.timestamp.isoformat(),
                            log.action_type.value,
                            log.file_path,
                            log.success,
                            log.error_message,
                            log.duration_ms,
                            log.batch_id,
                        ]
                    )

        return len(logs)

    def clear_audit_log(self) -> int:
        """Supprime toutes les entrées de la table ``audit_log``.

        Phase G+3 (2026-05-19) — exposé pour le bouton « Vider »
        ajouté à droite d'« Exporter… » dans le panneau Historique
        du workspace et dans la modale Audit. Action destructive,
        l'appelant doit demander confirmation à l'utilisateur AVANT
        d'appeler cette méthode.

        Returns:
            Nombre d'entrées supprimées (count avant le DELETE).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_log")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM audit_log")
        conn.commit()
        return int(count)
