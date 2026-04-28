"""
Ollama Client - API client for local Ollama server
Handles connection, model management, and image analysis
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama errors"""

    pass


class OllamaConnectionError(OllamaError):
    """Connection to Ollama server failed"""

    pass


class OllamaModelError(OllamaError):
    """Model-related error"""

    pass


class OllamaTimeoutError(OllamaError):
    """Request timeout"""

    pass


class OllamaStatus(Enum):
    """Ollama server status"""

    UNKNOWN = "unknown"
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about an Ollama model"""

    name: str
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes"""
        return self.size / (1024**3) if self.size else 0

    @property
    def is_vision(self) -> bool:
        """Check if model supports vision"""
        vision_models = ["llama3.2-vision", "llava", "moondream", "bakllava"]
        return any(v in self.name.lower() for v in vision_models)


@dataclass
class GenerateResponse:
    """Response from generate endpoint"""

    response: str
    model: str
    done: bool
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second"""
        if self.eval_duration > 0:
            return self.eval_count / (self.eval_duration / 1e9)
        return 0


class OllamaClient:
    """
    Client for Ollama API
    Handles connection, model management, and generation
    """

    DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self, base_url: str = None, timeout: int = 120, on_status_change: Callable[[OllamaStatus], None] = None
    ):
        """
        Initialize Ollama client

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            on_status_change: Callback when status changes
        """
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.on_status_change = on_status_change

        self._status = OllamaStatus.UNKNOWN
        self._current_model: Optional[str] = None
        self._models_cache: List[ModelInfo] = []
        self._lock = Lock()
        self._last_check = 0

    @property
    def status(self) -> OllamaStatus:
        """Current server status"""
        return self._status

    @status.setter
    def status(self, value: OllamaStatus):
        """Set status and trigger callback"""
        if self._status != value:
            self._status = value
            if self.on_status_change:
                try:
                    self.on_status_change(value)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")

    @property
    def current_model(self) -> Optional[str]:
        """Currently loaded model"""
        return self._current_model

    # ==================== Connection & Status ====================

    def check_connection(self) -> bool:
        """
        Check if Ollama server is reachable

        Returns:
            True if server is online
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)

            if response.status_code == 200:
                self.status = OllamaStatus.ONLINE
                self._last_check = time.time()
                return True
            else:
                self.status = OllamaStatus.ERROR
                return False

        except requests.exceptions.ConnectionError:
            self.status = OllamaStatus.OFFLINE
            return False
        except requests.exceptions.Timeout:
            self.status = OllamaStatus.OFFLINE
            return False
        except Exception as e:
            logger.error(f"Connection check error: {e}")
            self.status = OllamaStatus.ERROR
            return False

    def get_status_info(self) -> Dict[str, Any]:
        """
        Get detailed status information

        Returns:
            Status dictionary with details
        """
        info = {
            "status": self.status.value,
            "url": self.base_url,
            "current_model": self._current_model,
            "models_available": len(self._models_cache),
            "last_check": self._last_check,
        }

        # Try to get server version
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=3)
            if response.status_code == 200:
                info["version"] = response.json().get("version", "unknown")
        except (requests.RequestException, ValueError):
            info["version"] = "unknown"

        return info

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection with a simple prompt

        Returns:
            Test result with timing
        """
        result = {"success": False, "message": "", "response_time_ms": 0, "model": None}

        if not self.check_connection():
            result["message"] = f"Cannot connect to Ollama at {self.base_url}"
            return result

        # Get first available model
        models = self.list_models()
        if not models:
            result["message"] = "No models available"
            return result

        # Prefer vision model
        test_model = None
        for m in models:
            if m.is_vision:
                test_model = m.name
                break
        if not test_model:
            test_model = models[0].name

        # Simple test prompt
        start_time = time.time()
        try:
            response = self.generate(model=test_model, prompt="Say 'OK' if you are working.", stream=False)

            elapsed = (time.time() - start_time) * 1000
            result["success"] = True
            result["message"] = response.response.strip()
            result["response_time_ms"] = int(elapsed)
            result["model"] = test_model

        except Exception as e:
            result["message"] = f"Test failed: {e}"

        return result

    # ==================== Model Management ====================

    def list_models(self, refresh: bool = False) -> List[ModelInfo]:
        """
        List available models

        Args:
            refresh: Force refresh from server

        Returns:
            List of ModelInfo objects
        """
        if self._models_cache and not refresh:
            return self._models_cache

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = []

                for m in data.get("models", []):
                    model = ModelInfo(
                        name=m.get("name", ""),
                        size=m.get("size", 0),
                        digest=m.get("digest", ""),
                        modified_at=m.get("modified_at", ""),
                        details=m.get("details", {}),
                    )
                    models.append(model)

                self._models_cache = models
                self.status = OllamaStatus.ONLINE
                return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            self.status = OllamaStatus.ERROR

        return []

    def list_vision_models(self) -> List[ModelInfo]:
        """
        List only vision-capable models

        Returns:
            List of vision models
        """
        return [m for m in self.list_models() if m.is_vision]

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get info for a specific model

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo or None
        """
        for m in self.list_models():
            if m.name == model_name:
                return m
        return None

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally"""
        return any(m.name == model_name for m in self.list_models())

    def load_model(self, model_name: str) -> bool:
        """
        Pre-load a model into memory

        Args:
            model_name: Model to load

        Returns:
            True if successful
        """
        if not self.is_model_available(model_name):
            logger.error(f"Model not available: {model_name}")
            return False

        try:
            # Send empty generate to load model
            self.status = OllamaStatus.BUSY

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "prompt": "", "stream": False},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                self._current_model = model_name
                self.status = OllamaStatus.ONLINE
                logger.info(f"Model loaded: {model_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.status = OllamaStatus.ERROR

        return False

    def unload_model(self) -> bool:
        """
        Unload current model from memory

        Returns:
            True if successful
        """
        if not self._current_model:
            return True

        try:
            # Ollama automatically unloads after timeout
            # Force by loading with keep_alive=0
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self._current_model, "prompt": "", "keep_alive": 0, "stream": False},
                timeout=30,
            )

            self._current_model = None
            self.status = OllamaStatus.ONLINE
            return True

        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False

    # ==================== Generation ====================

    def generate(
        self,
        model: str,
        prompt: str,
        images: List[str] = None,
        stream: bool = False,
        options: Dict[str, Any] = None,
        on_token: Callable[[str], None] = None,
    ) -> GenerateResponse:
        """
        Generate response from model

        Args:
            model: Model name
            prompt: Text prompt
            images: List of base64-encoded images
            stream: Enable streaming
            options: Model options (temperature, etc.)
            on_token: Callback for streaming tokens

        Returns:
            GenerateResponse object
        """
        payload = {"model": model, "prompt": prompt, "stream": stream}

        if images:
            payload["images"] = images

        if options:
            payload["options"] = options

        self.status = OllamaStatus.BUSY

        try:
            if stream and on_token:
                return self._generate_stream(payload, on_token)
            else:
                return self._generate_sync(payload)

        except requests.exceptions.Timeout:
            self.status = OllamaStatus.ERROR
            raise OllamaTimeoutError(f"Request timed out after {self.timeout}s")
        except Exception as e:
            self.status = OllamaStatus.ERROR
            raise OllamaError(f"Generation failed: {e}")
        finally:
            if self.status == OllamaStatus.BUSY:
                self.status = OllamaStatus.ONLINE

    def _generate_sync(self, payload: Dict) -> GenerateResponse:
        """Synchronous generation"""
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)

        if response.status_code != 200:
            raise OllamaError(f"API error: {response.status_code}")

        data = response.json()
        self._current_model = payload["model"]
        self.status = OllamaStatus.ONLINE

        return GenerateResponse(
            response=data.get("response", ""),
            model=data.get("model", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0),
        )

    def _generate_stream(self, payload: Dict, on_token: Callable[[str], None]) -> GenerateResponse:
        """Streaming generation"""
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout, stream=True)

        if response.status_code != 200:
            raise OllamaError(f"API error: {response.status_code}")

        full_response = ""
        final_data = {}

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                full_response += token

                if on_token:
                    on_token(token)

                if data.get("done", False):
                    final_data = data

        self._current_model = payload["model"]
        self.status = OllamaStatus.ONLINE

        return GenerateResponse(
            response=full_response,
            model=final_data.get("model", payload["model"]),
            done=True,
            total_duration=final_data.get("total_duration", 0),
            load_duration=final_data.get("load_duration", 0),
            prompt_eval_count=final_data.get("prompt_eval_count", 0),
            eval_count=final_data.get("eval_count", 0),
            eval_duration=final_data.get("eval_duration", 0),
        )

    # ==================== Image Helpers ====================

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """
        Encode image to base64 for API

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(
        self, model: str, image_path: Path, prompt: str, options: Dict[str, Any] = None
    ) -> GenerateResponse:
        """
        Analyze an image with a prompt

        Args:
            model: Vision model name
            image_path: Path to image
            prompt: Analysis prompt
            options: Model options

        Returns:
            GenerateResponse
        """
        image_b64 = self.encode_image(image_path)

        return self.generate(model=model, prompt=prompt, images=[image_b64], options=options)
