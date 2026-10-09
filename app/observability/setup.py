import os
import logging
from typing import Optional
from langsmith import Client

from app.config import get_settings

logger = logging.getLogger(__name__)

_langsmith_client: Optional[Client] = None


def setup_observability() -> bool:
    """
    Configure environment variables for LangSmith tracing based on app settings.
    Returns True if tracing is actively configured with an API key, False otherwise.
    """
    settings = get_settings()

    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        logger.info(
            f"LangSmith observability enabled! Project: '{settings.langchain_project}', Endpoint: '{settings.langchain_endpoint}'"
        )
        return True
    else:
        # Safe offline mode: ensure tracing won't attempt unauthenticated network calls
        if not settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.debug("LangSmith observability is disabled or running in safe offline mode (no API key configured).")
        return False


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently active with valid configuration."""
    settings = get_settings()
    return bool(settings.langchain_tracing_v2 and settings.langchain_api_key)


def get_langsmith_client() -> Optional[Client]:
    """Return an authenticated LangSmith Client if configured, or None in offline mode."""
    global _langsmith_client
    if not is_tracing_enabled():
        return None

    if _langsmith_client is None:
        try:
            settings = get_settings()
            _langsmith_client = Client(
                api_key=settings.langchain_api_key,
                api_url=settings.langchain_endpoint,
            )
        except Exception as e:
            logger.warning(f"Could not initialize LangSmith Client: {e}")
            return None

    return _langsmith_client
