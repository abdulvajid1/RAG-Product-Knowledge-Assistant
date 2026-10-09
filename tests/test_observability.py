import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from app.config import get_settings
from app.observability import (
    setup_observability,
    is_tracing_enabled,
    get_langsmith_client,
    trace_component,
)
from app.evaluation.langsmith_eval import (
    retrieval_evaluator,
    answer_correctness_evaluator,
    ExperimentRunner,
)


def test_observability_setup_defaults(monkeypatch):
    """Verify that observability safely defaults to disabled/offline mode without API key."""
    from app.config import Settings
    offline_settings = Settings(LANGCHAIN_TRACING_V2=False, LANGCHAIN_API_KEY="")
    monkeypatch.setattr("app.observability.setup.get_settings", lambda: offline_settings)
    monkeypatch.setattr("app.config.get_settings", lambda: offline_settings)
    monkeypatch.setattr("app.observability.setup._langsmith_client", None)
    assert setup_observability() is False
    assert is_tracing_enabled() is False
    assert get_langsmith_client() is None


def test_trace_component_sync_passthrough():
    """Verify that @trace_component wraps sync functions transparently."""
    @trace_component(name="test_sync_fn", run_type="chain")
    def compute(x: int, y: int) -> int:
        return x * y + 10

    result = compute(5, 4)
    assert result == 30


@pytest.mark.asyncio
async def test_trace_component_async_passthrough():
    """Verify that @trace_component wraps async functions transparently."""
    @trace_component(name="test_async_fn", run_type="llm")
    async def generate_mock(prompt: str) -> str:
        await asyncio.sleep(0.001)
        return f"Echo: {prompt}"

    result = await generate_mock("test query")
    assert result == "Echo: test query"


@pytest.mark.asyncio
async def test_trace_component_async_stream_passthrough():
    """Verify that @trace_component wraps async generators (streaming tokens) transparently."""
    @trace_component(name="test_stream_fn", run_type="llm")
    async def stream_tokens():
        yield "token1 "
        yield "token2 "
        yield "token3"

    tokens = []
    async for t in stream_tokens():
        tokens.append(t)

    assert "".join(tokens) == "token1 token2 token3"


def test_eval_dataset_schema_compatibility():
    """Verify that eval_questions.json is formatted correctly for LangSmith dataset sync."""
    dataset_path = Path("data/eval/eval_questions.json")
    assert dataset_path.exists()

    runner = ExperimentRunner(dataset_path=dataset_path)
    questions = runner.load_dataset()
    assert len(questions) >= 20

    for q in questions:
        assert "id" in q and q["id"]
        assert "question" in q and q["question"]
        assert "expected_sources" in q
        assert "expected_answer" in q
        assert "answerable" in q
        assert isinstance(q["answerable"], bool)


def test_langsmith_evaluators_logic():
    """Verify that retrieval and answer correctness evaluators compute scores properly."""
    # Mock Run and Example objects
    mock_run = MagicMock()
    mock_run.outputs = {
        "sources": [
            {"product_id": "P001", "document": "solarmax_550.pdf", "page": 4},
        ],
        "answer": "The maximum operating temperature is 85°C.",
    }

    mock_example = MagicMock()
    mock_example.outputs = {
        "expected_sources": [
            {"product_id": "P001", "document": "solarmax_550.pdf", "page": 4},
        ],
        "expected_answer": "85°C",
        "answerable": True,
    }

    ret_eval = retrieval_evaluator(mock_run, mock_example)
    assert ret_eval["key"] == "retrieval_quality"
    assert ret_eval["score"] == 1.0  # MRR=1.0

    ans_eval = answer_correctness_evaluator(mock_run, mock_example)
    assert ans_eval["key"] == "answer_groundedness"
    assert ans_eval["score"] == 1.0  # Correct answer
