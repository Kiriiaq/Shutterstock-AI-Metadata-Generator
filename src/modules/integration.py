"""
Integration module - Connects v2.0 modules with the existing application
Provides a unified API for all v2.0 functionality
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .engines.iptc_engine import IPTCEngine, IPTCTemplate
from .engines.metadata_reader import ExifToolNotFoundError, MetadataReader
from .engines.metadata_writer import MetadataWriter
from .models.metadata_models import (
    ImageMetadata,
    IPTCFields,
    ShutterstockMetadata,
    ValidationResult,
)
from .storage.database import ActionType, AuditLog, Database
from .workers.worker_pool import (
    BatchResult,
    Task,
    WorkerPool,
    collect_image_files,
    compute_file_hash,
)

logger = logging.getLogger(__name__)


class ShutterstockAIv2:
    """
    Main integration class for Shutterstock AI v2.0
    Provides unified API for all v2.0 functionality
    """

    def __init__(self, db_path: Optional[Path] = None, exiftool_path: Optional[str] = None, max_workers: int = 4):
        """
        Initialize Shutterstock AI v2.0

        Args:
            db_path: Path to SQLite database
            exiftool_path: Path to ExifTool executable
            max_workers: Maximum concurrent workers for processing
        """
        # Initialize database
        self.database = Database(db_path)

        # Load settings from database
        self._settings = self.database.get_all_settings()

        # Initialize engines
        try:
            self.metadata_reader = MetadataReader(exiftool_path or self._settings.get("exiftool_path"))
            self.metadata_writer = MetadataWriter(
                exiftool_path or self._settings.get("exiftool_path"),
                create_backup=self._settings.get("create_backup", True),
            )
            self._exiftool_available = True
        except ExifToolNotFoundError:
            logger.warning("ExifTool not found - metadata read/write disabled")
            self.metadata_reader = None
            self.metadata_writer = None
            self._exiftool_available = False

        self.iptc_engine = IPTCEngine()

        # Initialize worker pool
        self.max_workers = max_workers
        self.worker_pool = WorkerPool(max_workers=max_workers)

        # Register task handlers
        self._register_handlers()

        # Callbacks
        self._progress_callback: Optional[Callable] = None
        self._status_callback: Optional[Callable] = None

    def _register_handlers(self):
        """Register task handlers for worker pool"""
        self.worker_pool.register_handler("read_metadata", self._handle_read_metadata)
        self.worker_pool.register_handler("write_metadata", self._handle_write_metadata)
        self.worker_pool.register_handler("validate", self._handle_validate)

    @property
    def exiftool_available(self) -> bool:
        """Check if ExifTool is available"""
        return self._exiftool_available

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: (completed, total, current_file)"""
        self._progress_callback = callback
        self.worker_pool.set_progress_callback(callback)

    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status messages"""
        self._status_callback = callback

    def _update_status(self, message: str):
        """Update status via callback"""
        if self._status_callback:
            self._status_callback(message)
        logger.info(message)

    # ==================== Metadata Operations ====================

    def read_metadata(self, file_path: Path) -> Optional[ImageMetadata]:
        """
        Read metadata from a single image file

        Args:
            file_path: Path to image file

        Returns:
            ImageMetadata object or None if failed
        """
        if not self.metadata_reader:
            logger.error("MetadataReader not available (ExifTool missing)")
            return None

        start_time = time.time()

        try:
            metadata = self.metadata_reader.read(file_path)

            # Log to database
            duration_ms = int((time.time() - start_time) * 1000)
            self.database.log_action(
                ActionType.METADATA_READ,
                file_path=str(file_path),
                success=True,
                duration_ms=duration_ms,
                details={"sources": [s.value for s in metadata.metadata_sources]},
            )

            return metadata

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.database.log_action(
                ActionType.METADATA_READ,
                file_path=str(file_path),
                success=False,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            logger.error(f"Failed to read metadata from {file_path}: {e}")
            return None

    def write_metadata(
        self,
        file_path: Path,
        iptc: Optional[IPTCFields] = None,
        shutterstock: Optional[ShutterstockMetadata] = None,
        xmp: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write metadata to an image file

        Args:
            file_path: Path to image file
            iptc: IPTC fields to write
            shutterstock: Shutterstock metadata to write
            xmp: XMP data to write

        Returns:
            True if successful
        """
        if not self.metadata_writer:
            logger.error("MetadataWriter not available (ExifTool missing)")
            return False

        start_time = time.time()

        try:
            file_hash = compute_file_hash(file_path)

            if shutterstock:
                # Convert Shutterstock to IPTC and write
                iptc = self.iptc_engine.create_iptc_from_shutterstock(shutterstock)

            if iptc:
                self.metadata_writer.write_iptc(file_path, iptc)

                # Save to history
                self.database.save_metadata_history(
                    file_path=str(file_path),
                    file_hash=file_hash,
                    metadata_type="iptc",
                    metadata=iptc.to_dict(),
                    source="user_input",
                )

            if xmp:
                self.metadata_writer.write_xmp(file_path, xmp)

            # Log success
            duration_ms = int((time.time() - start_time) * 1000)
            self.database.log_action(
                ActionType.METADATA_WRITE,
                file_path=str(file_path),
                file_hash=file_hash,
                success=True,
                duration_ms=duration_ms,
            )

            return True

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.database.log_action(
                ActionType.METADATA_WRITE,
                file_path=str(file_path),
                success=False,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            logger.error(f"Failed to write metadata to {file_path}: {e}")
            return False

    def write_shutterstock_metadata(self, file_path: Path, metadata: ShutterstockMetadata) -> bool:
        """
        Write Shutterstock-formatted metadata to image

        Args:
            file_path: Path to image file
            metadata: ShutterstockMetadata object

        Returns:
            True if successful
        """
        if not self.metadata_writer:
            return False

        start_time = time.time()

        try:
            write_iptc = self._settings.get("write_iptc", True)
            write_xmp = self._settings.get("write_xmp", True)

            self.metadata_writer.write_shutterstock_metadata(
                file_path, metadata, write_iptc=write_iptc, write_xmp=write_xmp
            )

            # Save to history
            file_hash = compute_file_hash(file_path)
            self.database.save_metadata_history(
                file_path=str(file_path),
                file_hash=file_hash,
                metadata_type="shutterstock",
                metadata=metadata.to_csv_row(),
                source="ai_generated",
            )

            duration_ms = int((time.time() - start_time) * 1000)
            self.database.log_action(
                ActionType.METADATA_WRITE,
                file_path=str(file_path),
                file_hash=file_hash,
                success=True,
                duration_ms=duration_ms,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to write Shutterstock metadata: {e}")
            return False

    # ==================== Batch Processing ====================

    def process_folder(
        self, folder_path: Path, operations: List[str], recursive: bool = True, batch_name: Optional[str] = None
    ) -> BatchResult:
        """
        Process all images in a folder

        Args:
            folder_path: Source folder
            operations: Operations to perform ('read_metadata', 'write_metadata', 'validate')
            recursive: Include subdirectories
            batch_name: Optional batch name

        Returns:
            BatchResult with processing results
        """
        folder_path = Path(folder_path)

        # Collect files
        self._update_status(f"Scanning folder: {folder_path}")
        files = collect_image_files(folder_path, recursive=recursive)

        if not files:
            logger.warning(f"No image files found in {folder_path}")
            return BatchResult(batch_id="empty", total_tasks=0)

        # Create batch in database
        import uuid

        batch_id = str(uuid.uuid4())

        self.database.create_batch(
            batch_id=batch_id,
            source_folder=str(folder_path),
            total_files=len(files),
            name=batch_name,
            options={"operations": operations, "recursive": recursive},
        )

        self.database.log_action(
            ActionType.BATCH_START,
            details={"batch_id": batch_id, "files": len(files), "operations": operations},
            batch_id=batch_id,
        )

        self._update_status(f"Processing {len(files)} files...")

        # Start worker pool
        self.worker_pool.start()

        # Submit tasks for each operation
        for operation in operations:
            for file_path in files:
                task = Task(
                    task_id=f"{batch_id}_{operation}_{file_path.name}",
                    task_type=operation,
                    file_path=file_path,
                    params={"batch_id": batch_id},
                )
                self.worker_pool.submit_task(task)

        # Process queue
        result = self.worker_pool.process_queue()
        result.batch_id = batch_id

        # Update batch status
        self.database.update_batch_progress(
            batch_id, processed=result.completed_tasks, failed=result.failed_tasks, status="completed"
        )

        self.database.log_action(
            ActionType.BATCH_END,
            details={
                "batch_id": batch_id,
                "completed": result.completed_tasks,
                "failed": result.failed_tasks,
                "duration": result.duration_seconds,
            },
            batch_id=batch_id,
        )

        self.worker_pool.stop()

        self._update_status(
            f"Completed: {result.completed_tasks}/{result.total_tasks} files ({result.failed_tasks} failed)"
        )

        return result

    def read_metadata_batch(self, file_paths: List[Path]) -> List[Tuple[Path, Optional[ImageMetadata]]]:
        """
        Read metadata from multiple files

        Args:
            file_paths: List of file paths

        Returns:
            List of (path, metadata) tuples
        """
        if not self.metadata_reader:
            return [(p, None) for p in file_paths]

        results = self.metadata_reader.read_batch(file_paths)
        return [(p, m) for p, m, e in results]

    # ==================== Task Handlers ====================

    def _handle_read_metadata(self, file_path: Path, params: Dict[str, Any]) -> Optional[ImageMetadata]:
        """Handler for read_metadata task"""
        return self.read_metadata(file_path)

    def _handle_write_metadata(self, file_path: Path, params: Dict[str, Any]) -> bool:
        """Handler for write_metadata task"""
        iptc = params.get("iptc")
        xmp = params.get("xmp")
        shutterstock = params.get("shutterstock")

        if shutterstock:
            return self.write_shutterstock_metadata(file_path, shutterstock)
        else:
            return self.write_metadata(file_path, iptc=iptc, xmp=xmp)

    def _handle_validate(self, file_path: Path, params: Dict[str, Any]) -> ValidationResult:
        """Handler for validate task"""
        return self.validate_image(file_path)

    # ==================== Validation ====================

    def validate_image(self, file_path: Path) -> ValidationResult:
        """
        Validate an image and its metadata

        Args:
            file_path: Path to image file

        Returns:
            ValidationResult object
        """
        result = ValidationResult(is_valid=True)
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            result.is_valid = False
            result.errors.append(f"File not found: {file_path}")
            return result

        # Check file size
        stat = file_path.stat()
        max_size = 50 * 1024 * 1024  # 50 MB for JPEG

        if stat.st_size > max_size:
            result.errors.append(f"File too large: {stat.st_size / (1024 * 1024):.1f} MB (max 50 MB)")
            result.is_valid = False

        # Check resolution
        if self.metadata_reader:
            try:
                quick_info = self.metadata_reader.get_quick_info(file_path)
                width = quick_info.get("ImageWidth", 0)
                height = quick_info.get("ImageHeight", 0)
                megapixels = (width * height) / 1_000_000

                min_mp = self._settings.get("min_resolution_mp", 4.0)
                if megapixels < min_mp:
                    result.errors.append(f"Resolution too low: {megapixels:.1f} MP (min {min_mp} MP)")
                    result.is_valid = False

                # Add to completeness score
                result.completeness_score = 50  # Base score for valid file

                # Check metadata
                metadata = self.read_metadata(file_path)
                if metadata:
                    if metadata.has_iptc:
                        result.completeness_score += 25
                    if metadata.iptc.keywords and len(metadata.iptc.keywords) >= 7:
                        result.completeness_score += 15
                    if metadata.iptc.caption:
                        result.completeness_score += 10

                    # Validate IPTC
                    if metadata.iptc:
                        is_valid, errors, warnings = self.iptc_engine.validate_iptc(metadata.iptc)
                        result.warnings.extend(warnings)
                        if not is_valid:
                            result.errors.extend(errors)

            except Exception as e:
                result.warnings.append(f"Could not read image info: {e}")

        # Log validation
        self.database.log_action(
            ActionType.VALIDATION,
            file_path=str(file_path),
            success=result.is_valid,
            details={"errors": result.errors, "warnings": result.warnings, "score": result.completeness_score},
        )

        return result

    def validate_shutterstock_metadata(self, metadata: ShutterstockMetadata) -> ValidationResult:
        """
        Validate Shutterstock metadata

        Args:
            metadata: ShutterstockMetadata object

        Returns:
            ValidationResult object
        """
        result = ValidationResult(is_valid=metadata.is_valid)
        result.errors = metadata.validation_errors.copy()

        # Additional quality checks
        if metadata.title:
            # Check for common issues
            if metadata.title.isupper():
                result.warnings.append("Title is all uppercase - consider title case")
            if len(metadata.title) < 20:
                result.warnings.append("Title is very short - add more detail")

        if metadata.keywords:
            if len(set(metadata.keywords)) != len(metadata.keywords):
                result.warnings.append("Duplicate keywords found")

            # Check keyword quality
            short_keywords = [k for k in metadata.keywords if len(k) < 3]
            if short_keywords:
                result.warnings.append(f"Very short keywords: {', '.join(short_keywords)}")

        # Calculate scores
        result.completeness_score = self._calculate_completeness_score(metadata)
        result.quality_score = self._calculate_quality_score(metadata)
        result.seo_score = self._calculate_seo_score(metadata)

        return result

    def _calculate_completeness_score(self, metadata: ShutterstockMetadata) -> float:
        """Calculate completeness score 0-100"""
        score = 0

        if metadata.title:
            score += 25
        if metadata.description:
            score += 25
        if len(metadata.keywords) >= 7:
            score += 30
        if metadata.categories:
            score += 20

        return score

    def _calculate_quality_score(self, metadata: ShutterstockMetadata) -> float:
        """Calculate quality score 0-100"""
        score = 50  # Base score

        # Title quality
        if metadata.title:
            if 30 <= len(metadata.title) <= 150:
                score += 15
            if not metadata.title.isupper():
                score += 5

        # Keywords quality
        if metadata.keywords:
            if len(metadata.keywords) >= 20:
                score += 15
            if len(set(metadata.keywords)) == len(metadata.keywords):
                score += 10

        # Description quality
        if metadata.description and len(metadata.description) >= 50:
            score += 5

        return min(100, score)

    def _calculate_seo_score(self, metadata: ShutterstockMetadata) -> float:
        """Calculate SEO score 0-100"""
        score = 50

        # Keyword count matters for SEO
        if metadata.keywords:
            kw_count = len(metadata.keywords)
            if kw_count >= 30:
                score += 20
            elif kw_count >= 20:
                score += 15
            elif kw_count >= 10:
                score += 10

        # Categories help with discoverability
        if len(metadata.categories) == 2:
            score += 10

        # Title with good keywords
        if metadata.title and metadata.keywords:
            title_lower = metadata.title.lower()
            matches = sum(1 for kw in metadata.keywords if kw.lower() in title_lower)
            if matches >= 2:
                score += 20
            elif matches >= 1:
                score += 10

        return min(100, score)

    # ==================== History & Audit ====================

    def get_metadata_history(self, file_path: Path) -> List[Dict[str, Any]]:
        """Get metadata history for a file"""
        history = self.database.get_metadata_history(str(file_path))
        return [
            {
                "timestamp": h.timestamp,
                "type": h.metadata_type,
                "source": h.source,
                "version": h.version,
                "metadata": h.metadata_json,
            }
            for h in history
        ]

    def get_audit_logs(self, file_path: Optional[Path] = None, limit: int = 100) -> List[AuditLog]:
        """Get audit logs"""
        return self.database.get_audit_logs(file_path=str(file_path) if file_path else None, limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.database.get_statistics()

    # ==================== Templates ====================

    def get_templates(self) -> List[str]:
        """Get available IPTC templates"""
        return self.iptc_engine.list_templates()

    def get_template(self, name: str) -> Optional[IPTCTemplate]:
        """Get a specific template"""
        return self.iptc_engine.get_template(name)

    def apply_template(self, iptc: IPTCFields, template_name: str) -> IPTCFields:
        """Apply a template to IPTC fields"""
        template = self.iptc_engine.get_template(template_name)
        if template:
            return template.apply_to(iptc)
        return iptc

    # ==================== Settings ====================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: Any):
        """Set a setting value"""
        self._settings[key] = value
        self.database.set_setting(key, value)

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings"""
        return self._settings.copy()

    # ==================== AI vs Existing Comparison ====================

    def compare_ai_with_existing(self, file_path: Path, ai_metadata: ShutterstockMetadata) -> Dict[str, Any]:
        """
        Compare AI-generated metadata with existing metadata in file

        Args:
            file_path: Path to image file
            ai_metadata: AI-generated ShutterstockMetadata

        Returns:
            Comparison result with conflicts and suggestions
        """
        result = {
            "file_path": str(file_path),
            "has_existing": False,
            "conflicts": [],
            "matches": [],
            "ai_only": [],
            "existing_only": [],
            "recommendation": "use_ai",
            "merge_suggestion": None,
        }

        # Read existing metadata
        existing = self.read_metadata(file_path)
        if not existing or not existing.has_iptc:
            result["recommendation"] = "use_ai"
            result["ai_only"] = ["title", "description", "keywords", "categories"]
            return result

        result["has_existing"] = True
        existing_iptc = existing.iptc

        # Compare title
        if existing_iptc.headline or existing_iptc.object_name:
            existing_title = existing_iptc.headline or existing_iptc.object_name
            if ai_metadata.title:
                if existing_title.lower().strip() == ai_metadata.title.lower().strip():
                    result["matches"].append({"field": "title", "value": existing_title})
                else:
                    result["conflicts"].append(
                        {
                            "field": "title",
                            "existing": existing_title,
                            "ai": ai_metadata.title,
                            "similarity": self._calculate_similarity(existing_title, ai_metadata.title),
                        }
                    )
        else:
            if ai_metadata.title:
                result["ai_only"].append("title")

        # Compare description
        if existing_iptc.caption:
            if ai_metadata.description:
                if existing_iptc.caption.lower().strip() == ai_metadata.description.lower().strip():
                    result["matches"].append({"field": "description", "value": existing_iptc.caption})
                else:
                    result["conflicts"].append(
                        {
                            "field": "description",
                            "existing": existing_iptc.caption,
                            "ai": ai_metadata.description,
                            "similarity": self._calculate_similarity(existing_iptc.caption, ai_metadata.description),
                        }
                    )
        else:
            if ai_metadata.description:
                result["ai_only"].append("description")

        # Compare keywords
        existing_keywords = set(k.lower() for k in (existing_iptc.keywords or []))
        ai_keywords = set(k.lower() for k in (ai_metadata.keywords or []))

        common_keywords = existing_keywords & ai_keywords
        existing_only_kw = existing_keywords - ai_keywords
        ai_only_kw = ai_keywords - existing_keywords

        if existing_keywords and ai_keywords:
            overlap_ratio = len(common_keywords) / max(len(existing_keywords), len(ai_keywords))

            result["keywords_comparison"] = {
                "existing_count": len(existing_keywords),
                "ai_count": len(ai_keywords),
                "common_count": len(common_keywords),
                "overlap_ratio": round(overlap_ratio, 2),
                "common": list(common_keywords),
                "existing_only": list(existing_only_kw),
                "ai_only": list(ai_only_kw),
            }

            if overlap_ratio < 0.3:
                result["conflicts"].append(
                    {
                        "field": "keywords",
                        "existing_count": len(existing_keywords),
                        "ai_count": len(ai_keywords),
                        "overlap": round(overlap_ratio * 100, 1),
                        "message": "Low keyword overlap - significant difference",
                    }
                )
        elif ai_keywords:
            result["ai_only"].append("keywords")
        elif existing_keywords:
            result["existing_only"].append("keywords")

        # Generate recommendation
        conflict_count = len(result["conflicts"])
        if conflict_count == 0:
            result["recommendation"] = "use_ai"
            result["recommendation_reason"] = "No conflicts found, AI metadata can be used"
        elif conflict_count <= 2:
            result["recommendation"] = "merge"
            result["recommendation_reason"] = f"{conflict_count} minor conflicts - consider merging"
        else:
            result["recommendation"] = "review"
            result["recommendation_reason"] = f"{conflict_count} conflicts - manual review recommended"

        # Generate merge suggestion
        result["merge_suggestion"] = self._generate_merge_suggestion(existing_iptc, ai_metadata, result)

        return result

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple similarity ratio between two strings"""
        if not str1 or not str2:
            return 0.0

        # Simple word overlap similarity
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return round(intersection / union, 2) if union > 0 else 0.0

    def _generate_merge_suggestion(
        self, existing: IPTCFields, ai: ShutterstockMetadata, comparison: Dict[str, Any]
    ) -> IPTCFields:
        """Generate a merged IPTC suggestion from existing and AI metadata"""
        merged = IPTCFields()

        # Title: prefer longer/more descriptive
        existing_title = existing.headline or existing.object_name or ""
        ai_title = ai.title or ""

        if len(ai_title) > len(existing_title) * 1.2:  # AI is significantly longer
            merged.headline = ai_title[:256]
            merged.object_name = ai_title[:64]
        else:
            merged.headline = existing_title[:256] if existing_title else ai_title[:256]
            merged.object_name = existing_title[:64] if existing_title else ai_title[:64]

        # Description: prefer longer
        if len(ai.description or "") > len(existing.caption or "") * 1.2:
            merged.caption = ai.description
        else:
            merged.caption = existing.caption or ai.description

        # Keywords: merge both sets
        all_keywords = list(set((existing.keywords or []) + (ai.keywords or [])))
        merged.keywords = all_keywords[:50]  # Shutterstock max

        # Keep existing creator info
        merged.byline = existing.byline
        merged.byline_title = existing.byline_title
        merged.credit = existing.credit
        merged.source = existing.source
        merged.copyright_notice = existing.copyright_notice

        # Keep existing location
        merged.city = existing.city
        merged.sublocation = existing.sublocation
        merged.province_state = existing.province_state
        merged.country_code = existing.country_code
        merged.country_name = existing.country_name

        # Categories from AI
        if ai.categories:
            merged.supplemental_categories = ai.categories[:2]

        return merged

    # ==================== Metadata Diff ====================

    def get_metadata_diff(
        self, file_path: Path, new_metadata: Optional[IPTCFields] = None, version1: int = None, version2: int = None
    ) -> Dict[str, Any]:
        """
        Get diff between metadata versions or current vs new

        Args:
            file_path: Path to image file
            new_metadata: New metadata to compare with current (optional)
            version1: First version number from history (optional)
            version2: Second version number from history (optional)

        Returns:
            Diff result with changes
        """
        result = {
            "file_path": str(file_path),
            "changes": [],
            "additions": [],
            "removals": [],
            "unchanged": [],
            "summary": "",
        }

        # Get current metadata from file
        current = self.read_metadata(file_path)
        current_iptc = current.iptc if current else IPTCFields()

        if new_metadata:
            # Compare current with new
            result = self._compare_iptc_fields(current_iptc, new_metadata)
            result["comparison_type"] = "current_vs_new"
        elif version1 is not None and version2 is not None:
            # Compare two history versions
            history = self.database.get_metadata_history(str(file_path), "iptc")

            v1_data = None
            v2_data = None

            import json as json_module

            for h in history:
                if h.version == version1:
                    v1_data = IPTCFields.from_dict(json_module.loads(h.metadata_json))
                if h.version == version2:
                    v2_data = IPTCFields.from_dict(json_module.loads(h.metadata_json))

            if v1_data and v2_data:
                result = self._compare_iptc_fields(v1_data, v2_data)
                result["comparison_type"] = f"version_{version1}_vs_{version2}"
        else:
            # Compare current with latest history
            history = self.database.get_metadata_history(str(file_path), "iptc")
            if history:
                import json

                latest = IPTCFields.from_dict(json.loads(history[0].metadata_json))
                result = self._compare_iptc_fields(latest, current_iptc)
                result["comparison_type"] = "history_vs_current"

        # Generate summary
        total_changes = (
            len(result.get("changes", [])) + len(result.get("additions", [])) + len(result.get("removals", []))
        )
        if total_changes == 0:
            result["summary"] = "No changes detected"
        else:
            result["summary"] = (
                f"{total_changes} changes: {len(result.get('changes', []))} modified, {len(result.get('additions', []))} added, {len(result.get('removals', []))} removed"
            )

        return result

    def _compare_iptc_fields(self, old: IPTCFields, new: IPTCFields) -> Dict[str, Any]:
        """Compare two IPTCFields objects and return differences"""
        result = {"changes": [], "additions": [], "removals": [], "unchanged": []}

        # Fields to compare
        fields = [
            ("object_name", "Title"),
            ("headline", "Headline"),
            ("caption", "Description"),
            ("byline", "Creator"),
            ("byline_title", "Creator Title"),
            ("credit", "Credit"),
            ("source", "Source"),
            ("copyright_notice", "Copyright"),
            ("city", "City"),
            ("sublocation", "Sublocation"),
            ("province_state", "State/Province"),
            ("country_code", "Country Code"),
            ("country_name", "Country"),
            ("category", "Category"),
            ("special_instructions", "Instructions"),
            ("transmission_reference", "Job ID"),
            ("urgency", "Urgency"),
        ]

        for field_name, display_name in fields:
            old_val = getattr(old, field_name, None)
            new_val = getattr(new, field_name, None)

            # Normalize None and empty strings
            old_val = old_val if old_val else None
            new_val = new_val if new_val else None

            if old_val == new_val:
                if old_val is not None:
                    result["unchanged"].append({"field": field_name, "display_name": display_name, "value": old_val})
            elif old_val is None and new_val is not None:
                result["additions"].append({"field": field_name, "display_name": display_name, "new_value": new_val})
            elif old_val is not None and new_val is None:
                result["removals"].append({"field": field_name, "display_name": display_name, "old_value": old_val})
            else:
                result["changes"].append(
                    {"field": field_name, "display_name": display_name, "old_value": old_val, "new_value": new_val}
                )

        # Compare keywords specially
        old_keywords = set(old.keywords or [])
        new_keywords = set(new.keywords or [])

        added_keywords = new_keywords - old_keywords
        removed_keywords = old_keywords - new_keywords

        if added_keywords or removed_keywords:
            result["changes"].append(
                {
                    "field": "keywords",
                    "display_name": "Keywords",
                    "added": list(added_keywords),
                    "removed": list(removed_keywords),
                    "old_count": len(old_keywords),
                    "new_count": len(new_keywords),
                }
            )
        elif old_keywords:
            result["unchanged"].append({"field": "keywords", "display_name": "Keywords", "count": len(old_keywords)})

        # Compare supplemental categories
        old_cats = set(old.supplemental_categories or [])
        new_cats = set(new.supplemental_categories or [])

        if old_cats != new_cats:
            result["changes"].append(
                {
                    "field": "supplemental_categories",
                    "display_name": "Categories",
                    "old_value": list(old_cats),
                    "new_value": list(new_cats),
                }
            )

        return result

    def format_diff_for_display(self, diff: Dict[str, Any]) -> str:
        """Format a diff result for text display"""
        lines = []
        lines.append("=== Metadata Diff ===")
        lines.append(f"File: {diff.get('file_path', 'Unknown')}")
        lines.append(f"Type: {diff.get('comparison_type', 'Unknown')}")
        lines.append("")

        if diff.get("changes"):
            lines.append("MODIFIED:")
            for change in diff["changes"]:
                if change["field"] == "keywords":
                    lines.append(f"  {change['display_name']}:")
                    lines.append(f"    + Added: {', '.join(change.get('added', []))}")
                    lines.append(f"    - Removed: {', '.join(change.get('removed', []))}")
                else:
                    lines.append(f"  {change['display_name']}:")
                    lines.append(f"    - {change.get('old_value', '')}")
                    lines.append(f"    + {change.get('new_value', '')}")

        if diff.get("additions"):
            lines.append("")
            lines.append("ADDED:")
            for add in diff["additions"]:
                lines.append(f"  + {add['display_name']}: {add['new_value']}")

        if diff.get("removals"):
            lines.append("")
            lines.append("REMOVED:")
            for rem in diff["removals"]:
                lines.append(f"  - {rem['display_name']}: {rem['old_value']}")

        lines.append("")
        lines.append(f"Summary: {diff.get('summary', '')}")

        return "\n".join(lines)

    # ==================== AI Pipeline ====================

    def init_ai(self, ollama_url: str = None, model: str = None, timeout: int = None):
        """
        Initialize AI components

        Args:
            ollama_url: Ollama server URL
            model: Vision model to use
            timeout: Request timeout in seconds
        """
        from .ai.ollama_client import OllamaClient
        from .ai.prompt_templates import Platform
        from .ai.vision_analyzer import VisionAnalyzer

        url = ollama_url or self._settings.get("ollama_url", "http://localhost:11434")
        timeout = timeout or int(self._settings.get("ollama_timeout", 120))
        model = model or self._settings.get("ollama_model")

        self.ollama_client = OllamaClient(base_url=url, timeout=timeout)
        self.vision_analyzer = VisionAnalyzer(
            client=self.ollama_client, model=model, platform=Platform.SHUTTERSTOCK, timeout=timeout
        )

        # Register AI task handler
        self.worker_pool.register_handler("ai_analyze", self._handle_ai_analyze)

        logger.info(f"AI initialized: {url} with model {model}")

    def check_ai_status(self) -> Dict[str, Any]:
        """
        Check AI availability and status

        Returns:
            Status dictionary
        """
        if not hasattr(self, "ollama_client"):
            return {
                "available": False,
                "status": "not_initialized",
                "message": "AI not initialized. Call init_ai() first.",
            }

        try:
            connected = self.ollama_client.check_connection()
            info = self.ollama_client.get_status_info()

            return {
                "available": connected,
                "status": info["status"],
                "url": info["url"],
                "version": info.get("version", "unknown"),
                "current_model": info.get("current_model"),
                "vision_models": len(self.ollama_client.list_vision_models()),
                "message": "Online" if connected else "Offline",
            }
        except Exception as e:
            return {"available": False, "status": "error", "message": str(e)}

    def analyze_image_ai(self, file_path: Path, skip_if_has_metadata: bool = False) -> Dict[str, Any]:
        """
        Analyze a single image using AI

        Args:
            file_path: Path to image file
            skip_if_has_metadata: Skip if image has existing metadata

        Returns:
            Analysis result dictionary
        """
        if not hasattr(self, "vision_analyzer"):
            self.init_ai()

        file_path = Path(file_path)
        start_time = time.time()

        # Log action
        batch_id = f"ai_single_{int(time.time())}"

        try:
            # Get existing metadata if needed
            existing = None
            if skip_if_has_metadata and self.metadata_reader:
                existing = self.read_metadata(file_path)
                if existing and existing.has_iptc:
                    if existing.iptc.headline and existing.iptc.keywords:
                        self.database.log_action(
                            ActionType.AI_ANALYSIS,
                            file_path=str(file_path),
                            success=True,
                            batch_id=batch_id,
                            details={"message": "Skipped - already has metadata"},
                        )
                        return {
                            "success": True,
                            "skipped": True,
                            "reason": "Already has metadata",
                            "file_path": str(file_path),
                        }

            # Analyze with AI
            result = self.vision_analyzer.analyze_image(
                file_path,
                skip_if_has_metadata=False,  # Already checked above
            )

            elapsed = int((time.time() - start_time) * 1000)

            if result.is_successful:
                # Log success
                self.database.log_action(
                    ActionType.AI_ANALYSIS,
                    file_path=str(file_path),
                    success=True,
                    duration_ms=elapsed,
                    batch_id=batch_id,
                    details={"model": result.model_used, "keywords_count": len(result.keywords)},
                )

                # Update file status
                self.database.set_file_flags(
                    str(file_path),
                    has_ai_analysis=True,
                )

                return {
                    "success": True,
                    "file_path": str(file_path),
                    "title": result.title,
                    "description": result.description,
                    "keywords": result.keywords,
                    "categories": result.categories,
                    "editorial": result.editorial,
                    "model_used": result.model_used,
                    "processing_time_ms": elapsed,
                    "tokens_per_second": result.tokens_per_second,
                }
            else:
                self.database.log_action(
                    ActionType.AI_ANALYSIS,
                    file_path=str(file_path),
                    success=False,
                    duration_ms=elapsed,
                    batch_id=batch_id,
                    error_message=result.error,
                )

                return {"success": False, "file_path": str(file_path), "error": result.error}

        except Exception as e:
            logger.error(f"AI analysis failed for {file_path}: {e}")
            self.database.log_action(
                ActionType.AI_ANALYSIS,
                file_path=str(file_path),
                success=False,
                error_message=str(e),
                batch_id=batch_id,
            )
            return {"success": False, "file_path": str(file_path), "error": str(e)}

    def analyze_batch_ai(
        self,
        file_paths: List[Path],
        skip_if_has_metadata: bool = True,
        write_metadata: bool = False,
        on_progress: Callable[[int, int, str], None] = None,
        on_result: Callable[[Dict], None] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple images using AI

        Args:
            file_paths: List of image paths
            skip_if_has_metadata: Skip images with existing metadata
            write_metadata: Write AI results to files
            on_progress: Progress callback (completed, total, current_file)
            on_result: Callback for each result

        Returns:
            Batch result summary
        """
        if not hasattr(self, "vision_analyzer"):
            self.init_ai()

        batch_id = f"ai_batch_{int(time.time())}"
        start_time = time.time()

        # Create batch record. AI batches are file-list driven (no source
        # folder), so we pass empty string for the schema-required column.
        self.database.create_batch(
            batch_id=batch_id,
            source_folder="",
            total_files=len(file_paths),
        )

        results = []
        completed = 0
        failed = 0
        skipped = 0

        def progress_callback(progress):
            if on_progress:
                on_progress(progress.completed, progress.total, progress.current_file)

        def result_callback(result):
            nonlocal completed, failed, skipped

            if result.status.value == "completed":
                completed += 1

                # Optionally write metadata
                if write_metadata and result.is_successful:
                    self._write_ai_result(result, batch_id)

            elif result.status.value == "failed":
                failed += 1
            elif result.status.value == "skipped":
                skipped += 1

            # Convert to dict and callback
            result_dict = result.to_dict()
            results.append(result_dict)

            if on_result:
                on_result(result_dict)

            # Update batch progress. The DB schema only persists processed
            # vs failed; skipped count is rolled into processed (not-failed).
            self.database.update_batch_progress(
                batch_id,
                processed=completed + skipped,
                failed=failed,
            )

        # Run batch analysis
        self.vision_analyzer.analyze_batch(
            file_paths,
            skip_if_has_metadata=skip_if_has_metadata,
            on_progress=progress_callback,
            on_result=result_callback,
        )

        elapsed = int((time.time() - start_time) * 1000)

        # Complete batch. Counts have already been persisted by the
        # update_batch_progress calls in result_callback above.
        self.database.complete_batch(batch_id, status="completed")

        return {
            "batch_id": batch_id,
            "total": len(file_paths),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "duration_ms": elapsed,
            "success_rate": (completed / len(file_paths) * 100) if file_paths else 0,
            "results": results,
        }

    def _write_ai_result(self, result, batch_id: str):
        """Write AI analysis result to image metadata"""
        if not self.metadata_writer or not result.is_successful:
            return

        try:
            # Create IPTC fields from result
            iptc = IPTCFields(
                headline=result.title,
                object_name=result.title[:64] if result.title else None,
                caption=result.description,
                keywords=result.keywords,
                supplemental_categories=result.categories,
            )

            # Apply defaults
            iptc.byline = self._settings.get("default_byline")
            iptc.copyright_notice = self._settings.get("default_copyright")

            # Write to file
            success = self.metadata_writer.write_metadata_auto(
                result.file_path,
                iptc=iptc,
                write_iptc=self._settings.get("write_iptc", True),
                write_xmp=self._settings.get("write_xmp", True),
            )

            if success:
                self.database.log_action(
                    ActionType.METADATA_WRITE,
                    file_path=str(result.file_path),
                    success=True,
                    batch_id=batch_id,
                    details={"message": "AI metadata written"},
                )

                self.database.set_file_flags(
                    str(result.file_path),
                    has_metadata=True,
                )

        except Exception as e:
            logger.error(f"Failed to write AI metadata: {e}")

    def _handle_ai_analyze(self, task: Task) -> Any:
        """Handler for AI analysis tasks"""
        file_path = Path(task.data.get("file_path"))
        skip_if_has = task.data.get("skip_if_has_metadata", True)

        return self.analyze_image_ai(file_path, skip_if_has)

    def process_folder_ai(
        self,
        folder: Path,
        recursive: bool = True,
        skip_if_has_metadata: bool = True,
        write_metadata: bool = False,
        on_progress: Callable = None,
        on_result: Callable = None,
    ) -> Dict[str, Any]:
        """
        Process all images in a folder with AI

        Args:
            folder: Folder path
            recursive: Include subfolders
            skip_if_has_metadata: Skip images with metadata
            write_metadata: Write results to files
            on_progress: Progress callback
            on_result: Result callback

        Returns:
            Processing summary
        """
        # Collect files
        files = collect_image_files(folder, recursive=recursive)

        if not files:
            return {"success": False, "error": f"No images found in {folder}", "total": 0}

        # Process batch
        return self.analyze_batch_ai(
            files,
            skip_if_has_metadata=skip_if_has_metadata,
            write_metadata=write_metadata,
            on_progress=on_progress,
            on_result=on_result,
        )

    # ==================== Cleanup ====================

    def close(self):
        """Clean up resources"""
        if self.worker_pool:
            self.worker_pool.stop()
        if self.database:
            self.database.close()
