"""
AI module for Ollama/LLaMA integration
Provides vision-based metadata generation for images
"""

from .ollama_client import OllamaClient, OllamaStatus, OllamaError
from .vision_analyzer import VisionAnalyzer, AnalysisResult
from .prompt_templates import PromptTemplates, PromptType

__all__ = [
    "OllamaClient",
    "OllamaStatus",
    "OllamaError",
    "VisionAnalyzer",
    "AnalysisResult",
    "PromptTemplates",
    "PromptType"
]
