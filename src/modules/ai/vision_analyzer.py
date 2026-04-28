"""
Vision Analyzer - Image analysis using Ollama vision models
Generates metadata from images using AI
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .ollama_client import OllamaClient, OllamaError
from .prompt_templates import Platform, PromptTemplates, PromptType

logger = logging.getLogger(__name__)


class AnalysisStatus(Enum):
    """Status of image analysis"""

    PENDING = "pending"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AnalysisResult:
    """Result of image analysis"""

    file_path: Path
    status: AnalysisStatus
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    editorial: bool = False
    error: Optional[str] = None
    processing_time_ms: int = 0
    model_used: str = ""
    tokens_per_second: float = 0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": str(self.file_path),
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "categories": self.categories,
            "editorial": self.editorial,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }

    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful"""
        return self.status == AnalysisStatus.COMPLETED and self.title is not None


@dataclass
class BatchProgress:
    """Progress tracking for batch analysis"""

    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    current_file: str = ""
    start_time: float = 0
    elapsed_ms: int = 0

    @property
    def processed(self) -> int:
        return self.completed + self.failed + self.skipped

    @property
    def remaining(self) -> int:
        return self.total - self.processed

    @property
    def progress_percent(self) -> float:
        if self.total == 0:
            return 0
        return (self.processed / self.total) * 100

    @property
    def success_rate(self) -> float:
        if self.processed == 0:
            return 0
        return (self.completed / self.processed) * 100


class VisionAnalyzer:
    """
    Image analyzer using Ollama vision models
    Generates metadata from images
    """

    # Default vision models (in order of preference)
    DEFAULT_MODELS = [
        "llama3.2-vision:11b",
        "llama3.2-vision:latest",
        "llava:13b",
        "llava:7b",
        "llava:latest",
        "moondream:latest",
    ]

    def __init__(
        self,
        client: OllamaClient = None,
        model: str = None,
        platform: Platform = Platform.SHUTTERSTOCK,
        timeout: int = 120,
    ):
        """
        Initialize vision analyzer

        Args:
            client: OllamaClient instance (creates one if not provided)
            model: Vision model to use
            platform: Target platform for metadata
            timeout: Request timeout in seconds
        """
        self.client = client or OllamaClient(timeout=timeout)
        self.model = model
        self.platform = platform
        self.templates = PromptTemplates(platform)
        self.timeout = timeout

        self._on_progress: Optional[Callable[[BatchProgress], None]] = None
        self._cancel_requested = False

    @property
    def is_available(self) -> bool:
        """Check if analyzer is ready"""
        return self.client.check_connection() and self._has_vision_model()

    def _has_vision_model(self) -> bool:
        """Check if a vision model is available"""
        if self.model and self.client.is_model_available(self.model):
            return True
        # Check for any vision model
        return len(self.client.list_vision_models()) > 0

    def get_available_models(self) -> List[str]:
        """Get list of available vision models"""
        return [m.name for m in self.client.list_vision_models()]

    def select_model(self, model: str = None) -> str:
        """
        Select and set the vision model to use

        Args:
            model: Preferred model name

        Returns:
            Selected model name

        Raises:
            OllamaError: If no vision model available
        """
        if model and self.client.is_model_available(model):
            self.model = model
            return model

        # Try default models in order
        for default in self.DEFAULT_MODELS:
            if self.client.is_model_available(default):
                self.model = default
                return default

        # Use any available vision model
        vision_models = self.client.list_vision_models()
        if vision_models:
            self.model = vision_models[0].name
            return self.model

        raise OllamaError("No vision model available")

    def analyze_image(
        self,
        image_path: Path,
        prompt_type: PromptType = PromptType.FULL,
        skip_if_has_metadata: bool = False,
        existing_metadata: Dict = None,
    ) -> AnalysisResult:
        """
        Analyze a single image

        Args:
            image_path: Path to image file
            prompt_type: Type of metadata to generate
            skip_if_has_metadata: Skip if image already has metadata
            existing_metadata: Existing metadata to consider

        Returns:
            AnalysisResult object
        """
        image_path = Path(image_path)
        start_time = time.time()

        result = AnalysisResult(file_path=image_path, status=AnalysisStatus.PENDING)

        # Validate file
        if not image_path.exists():
            result.status = AnalysisStatus.FAILED
            result.error = "File not found"
            return result

        # Check for existing metadata
        if skip_if_has_metadata and existing_metadata:
            if existing_metadata.get("title") and existing_metadata.get("keywords"):
                result.status = AnalysisStatus.SKIPPED
                result.error = "Already has metadata"
                return result

        # Ensure model is selected
        if not self.model:
            try:
                self.select_model()
            except OllamaError as e:
                result.status = AnalysisStatus.FAILED
                result.error = str(e)
                return result

        result.model_used = self.model
        result.status = AnalysisStatus.ANALYZING

        try:
            # Get prompt
            prompt = self.templates.get_prompt(prompt_type)

            # Analyze with vision model
            response = self.client.analyze_image(model=self.model, image_path=image_path, prompt=prompt)

            # Parse response
            parsed = self.templates.parse_response(response.response)

            result.title = parsed.get("title")
            result.description = parsed.get("description")
            result.keywords = parsed.get("keywords", [])
            result.categories = parsed.get("categories", [])
            result.editorial = parsed.get("editorial", False)
            result.raw_response = response.response
            result.tokens_per_second = response.tokens_per_second

            # Validate result
            if result.title or result.keywords:
                result.status = AnalysisStatus.COMPLETED
            else:
                result.status = AnalysisStatus.FAILED
                result.error = "Failed to parse response"

        except Exception as e:
            result.status = AnalysisStatus.FAILED
            result.error = str(e)
            logger.error(f"Analysis failed for {image_path}: {e}")

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    def analyze_batch(
        self,
        image_paths: List[Path],
        prompt_type: PromptType = PromptType.FULL,
        max_workers: int = 1,
        skip_if_has_metadata: bool = False,
        on_progress: Callable[[BatchProgress], None] = None,
        on_result: Callable[[AnalysisResult], None] = None,
    ) -> List[AnalysisResult]:
        """
        Analyze multiple images

        Args:
            image_paths: List of image paths
            prompt_type: Type of metadata to generate
            max_workers: Number of parallel workers (1 for sequential)
            skip_if_has_metadata: Skip images with existing metadata
            on_progress: Progress callback
            on_result: Callback for each result

        Returns:
            List of AnalysisResult objects
        """
        self._cancel_requested = False
        self._on_progress = on_progress

        progress = BatchProgress(total=len(image_paths), start_time=time.time())

        results = []

        if max_workers <= 1:
            # Sequential processing
            for path in image_paths:
                if self._cancel_requested:
                    break

                progress.current_file = path.name

                result = self.analyze_image(path, prompt_type=prompt_type, skip_if_has_metadata=skip_if_has_metadata)

                results.append(result)
                self._update_progress(progress, result)

                if on_result:
                    on_result(result)

        else:
            # Parallel processing (limited for API)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.analyze_image, path, prompt_type, skip_if_has_metadata): path
                    for path in image_paths
                }

                for future in as_completed(futures):
                    if self._cancel_requested:
                        break

                    path = futures[future]
                    progress.current_file = path.name

                    try:
                        result = future.result()
                    except Exception as e:
                        result = AnalysisResult(file_path=path, status=AnalysisStatus.FAILED, error=str(e))

                    results.append(result)
                    self._update_progress(progress, result)

                    if on_result:
                        on_result(result)

        return results

    def _update_progress(self, progress: BatchProgress, result: AnalysisResult):
        """Update progress after processing an image"""
        if result.status == AnalysisStatus.COMPLETED:
            progress.completed += 1
        elif result.status == AnalysisStatus.FAILED:
            progress.failed += 1
        elif result.status == AnalysisStatus.SKIPPED:
            progress.skipped += 1

        progress.elapsed_ms = int((time.time() - progress.start_time) * 1000)

        if self._on_progress:
            try:
                self._on_progress(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def cancel(self):
        """Cancel ongoing batch analysis"""
        self._cancel_requested = True

    def check_editorial(self, image_path: Path) -> Dict[str, Any]:
        """
        Check if image contains editorial content

        Args:
            image_path: Path to image

        Returns:
            Dict with 'editorial' bool and 'reason' string
        """
        if not self.model:
            self.select_model()

        prompt = self.templates.get_editorial_check_prompt()

        try:
            response = self.client.analyze_image(model=self.model, image_path=image_path, prompt=prompt)

            # Parse response
            lines = response.response.strip().split("\n")
            editorial = False
            reason = ""

            for line in lines:
                line = line.strip()
                if line.upper().startswith("EDITORIAL:"):
                    value = line[10:].strip().upper()
                    editorial = value.startswith("YES")
                elif line.upper().startswith("REASON:"):
                    reason = line[7:].strip()

            return {"editorial": editorial, "reason": reason, "raw_response": response.response}

        except Exception as e:
            return {"editorial": False, "reason": f"Check failed: {e}", "error": str(e)}

    def regenerate_field(self, image_path: Path, field: str, current_value: Any = None) -> Optional[Any]:
        """
        Regenerate a specific metadata field

        Args:
            image_path: Path to image
            field: Field to regenerate (title, description, keywords, categories)
            current_value: Current value for context

        Returns:
            New value for the field
        """
        prompt_map = {
            "title": PromptType.TITLE_ONLY,
            "description": PromptType.DESCRIPTION_ONLY,
            "keywords": PromptType.KEYWORDS_ONLY,
            "categories": PromptType.CATEGORIES_ONLY,
        }

        prompt_type = prompt_map.get(field.lower(), PromptType.FULL)
        result = self.analyze_image(image_path, prompt_type=prompt_type)

        if result.is_successful:
            return getattr(result, field, None)

        return None
