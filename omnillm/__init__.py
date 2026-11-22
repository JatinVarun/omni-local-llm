from .adapters.openai_adapter import OpenAICompatibleAdapter
from .core.config import ConfigError
from .core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidRequestError,
    OmniLLMError,
    ToolLoopLimitError,
    UnsupportedFeatureError,
)
from .core.hardware import (
    FitRecommendation,
    HardwareProfile,
    QuantFitResult,
    QuantFitStatus,
    detect_hardware,
    estimate_model_memory,
    recommend_model_fit,
    suggest_models_for_hardware,
)
from .core.manager import LocalLLMManager
from .core.session import ChatSession
from .core.tools import ToolRegistry, function_to_tool_schema, tool
from .core.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelTarget,
    Usage,
)
from .harness import BenchmarkHarness, EvalHarness, NeedleHarness

__version__ = "0.3.0"
__all__ = [
    "BackendNotFoundError",
    "BackendUnavailableError",
    "BenchmarkHarness",
    "ChatChunk",
    "ChatRequest",
    "ChatResponse",
    "ChatSession",
    "ConfigError",
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EvalHarness",
    "FitRecommendation",
    "HardwareProfile",
    "InvalidRequestError",
    "LocalLLMManager",
    "ModelTarget",
    "NeedleHarness",
    "OmniLLMError",
    "OpenAICompatibleAdapter",
    "QuantFitResult",
    "QuantFitStatus",
    "ToolLoopLimitError",
    "ToolRegistry",
    "UnsupportedFeatureError",
    "Usage",
    "detect_hardware",
    "estimate_model_memory",
    "function_to_tool_schema",
    "recommend_model_fit",
    "suggest_models_for_hardware",
    "tool",
]
