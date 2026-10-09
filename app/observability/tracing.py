import functools
import logging
from typing import Optional, List, Dict, Any, Callable
from langsmith import traceable

logger = logging.getLogger(__name__)


def trace_component(
    name: Optional[str] = None,
    run_type: str = "chain",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator for tracing a component in the RAG pipeline.
    Uses LangSmith traceable decorator with fallback to transparent execution.

    Args:
        name: Human-readable span name in LangSmith UI (e.g. 'HybridRetriever.retrieve')
        run_type: Span category ('chain', 'retriever', 'llm', 'prompt', 'parser')
        tags: Optional tags for filtering in LangSmith UI
        metadata: Optional key-value metadata to attach to the trace
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__
        try:
            return traceable(
                name=span_name,
                run_type=run_type,
                tags=tags,
                metadata=metadata,
            )(fn)
        except Exception as e:
            logger.debug(f"Failed to attach LangSmith tracer to {span_name}: {e}")
            return fn

    return decorator
