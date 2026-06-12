"""
Integration module - Connects v2.0 modules with the existing application
Provides a unified API for all v2.0 functionality
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .engines.iptc_engine import IPTCEngine
from .engines.metadata_reader import ExifToolNotFoundError, MetadataReader
from .engines.metadata_writer import MetadataWriter
from .models.metadata_models import (
    ImageMetadata,
    IPTCFields,
    ShutterstockMetadata,
    ValidationResult,
)
from .storage.database import ActionType, AuditLog, Database
from .workers.worker_pool import WorkerPool, compute_file_hash

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

        # Load license (always succeeds — falls back to Community on
        # any error). Stored as a plain attribute; the UI reads via the
        # ``license`` property below.
        from .licensing import load_license
        self._license = load_license()

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

        # Callbacks
        self._progress_callback: Optional[Callable] = None
        self._status_callback: Optional[Callable] = None

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

    # ==================== Ollama model management ====================

    def list_vision_models(self, *, refresh: bool = False) -> List[str]:
        """Return the names of vision-capable models installed locally.

        Lazy-initialises the OllamaClient if needed so the UI can call
        this without a prior ``init_ai()``. Empty list on any error.
        """
        try:
            if not hasattr(self, "ollama_client"):
                self.init_ai()
            if refresh:
                self.ollama_client.list_models(refresh=True)
            return [m.name for m in self.ollama_client.list_vision_models()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_vision_models failed: %s", exc)
            return []

    def preload_model(self, model_name: str) -> Tuple[bool, str]:
        """Load *model_name* into Ollama memory (warms up VRAM/RAM).

        Persists the choice in settings (``ollama_model``) so the
        next session re-loads the same model automatically.

        Returns ``(success, message)`` — never raises.
        """
        if not model_name:
            return False, "Aucun modèle indiqué."
        try:
            if not hasattr(self, "ollama_client"):
                self.init_ai(model=model_name)
            else:
                # Make sure the analyzer points at the requested model
                # so subsequent ``analyze_image`` calls use it.
                if hasattr(self, "vision_analyzer"):
                    self.vision_analyzer.model = model_name

            ok = self.ollama_client.load_model(model_name)
            if not ok:
                return False, f"Échec chargement modèle {model_name}"
            # Persist preference for next session.
            try:
                self.set_setting("ollama_model", model_name)
            except Exception:  # noqa: BLE001
                logger.debug("ollama_model setting persist failed", exc_info=True)
            return True, f"Modèle chargé : {model_name}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("preload_model failed: %s", exc)
            return False, str(exc)

    def get_current_model(self) -> Optional[str]:
        """Name of the model currently warm in Ollama memory, or None."""
        try:
            if not hasattr(self, "ollama_client"):
                return None
            return self.ollama_client.current_model
        except Exception:  # noqa: BLE001
            return None

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

    def _has_complete_iptc(self, file_path: Path) -> bool:
        """True when the file already carries a usable IPTC block.

        Same criterion as the single-image path in ``analyze_image_ai``:
        headline + keywords present. Never raises — unreadable files
        simply go through the AI pass.
        """
        try:
            existing = self.read_metadata(file_path)
        except Exception:  # noqa: BLE001
            return False
        return bool(existing and existing.has_iptc and existing.iptc.headline and existing.iptc.keywords)

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
        total_files = len(file_paths)

        # Pre-filter files that already carry usable IPTC. This must live
        # here: VisionAnalyzer.analyze_image only honours
        # skip_if_has_metadata when existing_metadata is supplied, and the
        # batch path never supplies it — without this pass the checkbox
        # « Ignorer si méta » would be a no-op (audit B-01).
        to_analyze: List[Path] = [Path(p) for p in file_paths]
        if skip_if_has_metadata and self.metadata_reader is not None:
            to_analyze = []
            for idx, path in enumerate((Path(p) for p in file_paths), start=1):
                if self._has_complete_iptc(path):
                    skipped += 1
                    self.database.log_action(
                        ActionType.AI_ANALYSIS,
                        file_path=str(path),
                        success=True,
                        batch_id=batch_id,
                        details={"message": "Skipped - already has metadata"},
                    )
                    result_dict = {
                        "file_path": str(path),
                        "status": "skipped",
                        "title": None,
                        "description": None,
                        "keywords": [],
                        "categories": [],
                        "editorial": False,
                        "error": "Already has metadata",
                        "processing_time_ms": 0,
                        "model_used": "",
                    }
                    results.append(result_dict)
                    if on_result:
                        on_result(result_dict)
                    if on_progress:
                        on_progress(skipped, total_files, path.name)
                    self.database.update_batch_progress(batch_id, processed=skipped, failed=0)
                else:
                    to_analyze.append(path)
        pre_skipped = skipped

        def progress_callback(progress):
            if on_progress:
                on_progress(progress.completed + pre_skipped, total_files, progress.current_file)

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

        # Run batch analysis on the remaining files. The skip decision was
        # already made above, so the analyzer-level flag stays off.
        if to_analyze:
            self.vision_analyzer.analyze_batch(
                to_analyze,
                skip_if_has_metadata=False,
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

    # ==================== Expert Report (AI-optional) ====================

    def build_expert_report(
        self,
        file_path: Path,
        *,
        use_ai: bool = False,
        ai_result: Optional[Dict[str, Any]] = None,
    ):
        """Build a multi-platform expert report for one image.

        Works without AI by default — designed for low-power machines
        where Ollama isn't installed or shouldn't be invoked. The
        heuristic pass reads existing IPTC metadata + does a cheap
        PIL probe; the result is a fully-populated
        :class:`ExpertMetadataReport` ready for CSV export or display.

        ``use_ai=True`` requests an AI pass on top of the heuristic
        baseline. If Ollama isn't available, we silently fall back to
        heuristic-only — the user-facing posture of this app is "AI
        is a nice-to-have, never a hard requirement".

        Args:
            file_path: Image to analyse.
            use_ai: If True, attempt to enrich with vision-model output.
            ai_result: Pre-computed AI dict (skips the Ollama call,
                useful for tests and for re-running the export without
                spending tokens).
        """
        # Lazy imports — keep the analysis subpackage out of the
        # cold-start path of the GUI.
        from .analysis.expert_report import (
            build_expert_report,
            enrich_with_ai_result,
        )
        from .analysis.platform_compliance import check_platform_compliance

        file_path = Path(file_path)

        # Compliance is computed first so the report has the size/MP
        # context even if the IPTC read fails.
        compliance = check_platform_compliance(file_path)

        # Try to read existing IPTC; absence is fine (heuristic
        # builder degrades gracefully).
        image_metadata = None
        if self.metadata_reader is not None:
            try:
                image_metadata = self.metadata_reader.read(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("read_metadata failed in build_expert_report: %s", exc)

        report = build_expert_report(
            file_path,
            iptc=image_metadata.iptc if image_metadata else None,
            image_metadata=image_metadata,
            compliance=compliance,
        )

        # AI overlay — only when explicitly requested AND a result is
        # either provided or obtainable from the vision analyzer.
        if use_ai or ai_result is not None:
            payload = ai_result
            if payload is None and hasattr(self, "vision_analyzer"):
                try:
                    ai = self.analyze_image_ai(file_path)
                    if ai.get("success"):
                        payload = ai
                except Exception as exc:  # noqa: BLE001
                    logger.warning("AI enrichment failed, falling back: %s", exc)
                    payload = None
            if payload:
                report = enrich_with_ai_result(report, payload)

        return report

    def build_expert_reports_batch(
        self,
        file_paths: List[Path],
        *,
        use_ai: bool = False,
        on_progress: Callable[[int, int, str], None] = None,
    ) -> List[Any]:
        """Build reports for a batch of files (sequential, low-RAM)."""
        reports = []
        total = len(file_paths)
        for idx, path in enumerate(file_paths, start=1):
            try:
                report = self.build_expert_report(path, use_ai=use_ai)
                reports.append(report)
            except Exception as exc:  # noqa: BLE001
                logger.error("build_expert_report failed for %s: %s", path, exc)
            if on_progress:
                try:
                    on_progress(idx, total, Path(path).name)
                except Exception:  # noqa: BLE001
                    pass
        return reports

    def export_double_csv(
        self,
        reports: List[Any],
        output_dir: Path,
        *,
        basename: str = "metadata",
    ):
        """Write Adobe + Shutterstock CSVs from a list of reports.

        Returns an :class:`ExportResult` with both file paths.
        """
        from .export.csv_exporter import export_double_csv as _export

        return _export(reports, Path(output_dir), basename=basename)

    def export_batch(
        self,
        paths: List[Path],
        output_dir: Path,
        *,
        platform: str = "both",
        write_iptc: bool = False,
        use_ai: bool = False,
        basename: str = "metadata",
        ftp_config: Optional[Any] = None,
        on_progress: Optional[Callable[[Any], None]] = None,
    ):
        """End-to-end batch export: heuristic reports → CSV → IPTC → FTP.

        Args:
            paths: Images to process.
            output_dir: Destination folder for the CSVs.
            platform: ``"adobe"``, ``"shutterstock"`` or ``"both"``.
            write_iptc: If True, writes the report back into each
                file's IPTC. Requires ExifTool.
            use_ai: If True AND vision_analyzer is available, runs
                Ollama enrichment per file. Default False to keep the
                pipeline cheap on low-power machines.
            basename: Prefix for output CSV names.
            ftp_config: Optional FtpConfig — pushes the produced CSVs
                to the contributor portal after export.
            on_progress: Optional callback fired at every per-file
                lifecycle transition.

        Returns:
            BatchExportResult — never raises on per-file errors.
        """
        from .export.batch import Platform as _Platform
        from .export.batch import run_export_batch

        try:
            platform_enum = _Platform(platform)
        except ValueError:
            platform_enum = _Platform.BOTH

        ai_runner = None
        if use_ai and hasattr(self, "vision_analyzer"):
            def _ai_runner(p: Path) -> Dict[str, Any]:
                res = self.vision_analyzer.analyze_image(p)
                if not res.is_successful:
                    return {}
                return {
                    "title": res.title,
                    "description": res.description,
                    "keywords": res.keywords,
                    "categories": res.categories,
                }
            ai_runner = _ai_runner

        return run_export_batch(
            paths,
            Path(output_dir),
            platform=platform_enum,
            write_iptc=write_iptc and self._exiftool_available,
            use_ai=use_ai,
            basename=basename,
            iptc_writer=self.metadata_writer if write_iptc and self._exiftool_available else None,
            ai_runner=ai_runner,
            metadata_reader=self.metadata_reader,
            ftp_config=ftp_config,
            on_progress=on_progress,
        )

    def test_ftp_connection(self, ftp_config: Any) -> Tuple[bool, str]:
        """Probe the FTP endpoint — used by the UI's « Tester » button."""
        from .export.ftp_uploader import test_connection
        return test_connection(ftp_config)

    # ==================== Licence ====================

    @property
    def license(self):  # noqa: A003 — public surface, intentional
        """Currently active license — always a valid object.

        Falls back to ``License.community()`` when no file is installed
        or when verification fails. The UI uses
        ``api.license.is_pro()`` and ``api.license.has_feature(name)``
        for gating.
        """
        return self._license

    def activate_license(self, payload_or_text: Any) -> Tuple[bool, str]:
        """Install a license payload pasted by the user.

        Args:
            payload_or_text: A dict (parsed JSON) OR a string (raw text
                pasted from the customer email). The string is JSON-
                parsed first.

        Returns:
            ``(success, message)``. On success, the license file is
            written to ``~/.shutterstock_ai/license.json`` and
            ``self.license`` is refreshed.
        """
        import json as _json

        from .licensing import (
            DEFAULT_LICENSE_PATH,
            load_license,
            verify_license_payload,
        )

        if isinstance(payload_or_text, str):
            text = payload_or_text.strip()
            if not text:
                return False, "Clé vide."
            try:
                payload = _json.loads(text)
            except _json.JSONDecodeError as exc:
                return False, f"JSON invalide : {exc}"
        elif isinstance(payload_or_text, dict):
            payload = payload_or_text
        else:
            return False, "Type de clé non supporté."

        if not verify_license_payload(payload):
            return False, "Signature de la clé invalide."

        # Persist + reload via load_license to get the canonical
        # validation path (expiration, tier enum coercion).
        try:
            DEFAULT_LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_LICENSE_PATH.write_text(
                _json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            return False, f"Écriture impossible : {exc}"

        self._license = load_license()
        if not self._license.is_pro():
            return False, "La clé est valide mais ne donne pas accès au Pro."
        return True, f"Licence activée : {self._license.tier.value} ({self._license.email})"

    def deactivate_license(self) -> Tuple[bool, str]:
        """Remove the local license file → back to Community."""
        from .licensing import DEFAULT_LICENSE_PATH, load_license

        try:
            if DEFAULT_LICENSE_PATH.exists():
                DEFAULT_LICENSE_PATH.unlink()
        except OSError as exc:
            return False, f"Suppression impossible : {exc}"
        self._license = load_license()
        return True, "Licence retirée — mode Community actif."

    # ==================== Community quota (data export) =============

    # Settings key tracking the number of data exports a Community user
    # has run. Stored as int in the ``settings`` SQLite table so the
    # count survives app restarts — a session-scoped counter would let
    # the user reset it by closing the window, defeating the trial.
    _EXPORT_COUNTER_KEY = "community_exports_used"

    def export_quota_remaining(self) -> int:
        """How many data exports the user may still run for free.

        Returns:
            * ``-1`` if the licence unlocks ``data_export`` (= unlimited).
            * Otherwise ``COMMUNITY_EXPORT_QUOTA`` minus the persisted
              counter, clamped to ``[0, QUOTA]``.

        The UI calls this **before** an export to decide whether to run
        it or show the upsell.
        """
        from .licensing import COMMUNITY_EXPORT_QUOTA

        if self._license.has_feature("data_export"):
            return -1
        used = int(self.get_setting(self._EXPORT_COUNTER_KEY, 0) or 0)
        remaining = COMMUNITY_EXPORT_QUOTA - used
        if remaining < 0:
            return 0
        if remaining > COMMUNITY_EXPORT_QUOTA:
            return COMMUNITY_EXPORT_QUOTA
        return remaining

    def consume_export_quota(self) -> int:
        """Increment the Community export counter, return new remaining.

        No-op for licensed users (returns ``-1``). The UI invokes it
        once per successful export so the next one sees the updated
        count.
        """
        from .licensing import COMMUNITY_EXPORT_QUOTA

        if self._license.has_feature("data_export"):
            return -1
        used = int(self.get_setting(self._EXPORT_COUNTER_KEY, 0) or 0)
        used += 1
        self.set_setting(self._EXPORT_COUNTER_KEY, used)
        remaining = COMMUNITY_EXPORT_QUOTA - used
        return max(0, remaining)

    def reset_export_quota(self) -> None:
        """Reset the Community export counter (admin / test helper)."""
        self.set_setting(self._EXPORT_COUNTER_KEY, 0)

    # ==================== Cleanup ====================

    def close(self):
        """Clean up resources"""
        if self.worker_pool:
            self.worker_pool.stop()
        if self.database:
            self.database.close()
