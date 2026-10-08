"""Generation module for prompt construction, LLM clients, and streaming."""

from app.generation.prompt import PromptBuilder, is_comparison_query
from app.generation.llm import (
    LLMClient,
    MockLLM,
    OpenAICompatibleClient,
    get_llm_client,
    LLMError,
    LLMUnavailableError,
    LLMTimeoutError,
    LLMAuthenticationError,
)

__all__ = [
    "PromptBuilder",
    "is_comparison_query",
    "LLMClient",
    "MockLLM",
    "OpenAICompatibleClient",
    "get_llm_client",
    "LLMError",
    "LLMUnavailableError",
    "LLMTimeoutError",
    "LLMAuthenticationError",
]
