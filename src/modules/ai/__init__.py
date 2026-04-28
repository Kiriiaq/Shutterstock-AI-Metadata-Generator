"""
AI module for Ollama/LLaMA integration
Provides vision-based metadata generation for images
"""

from .ollama_client import OllamaClient, OllamaError, OllamaStatus
from .prompt_templates import PromptTemplates, PromptType
from .vision_analyzer import AnalysisResult, VisionAnalyzer

__all__ = [
    "OllamaClient",
    "OllamaStatus",
    "OllamaError",
    "VisionAnalyzer",
    "AnalysisResult",
    "PromptTemplates",
    "PromptType",
]
