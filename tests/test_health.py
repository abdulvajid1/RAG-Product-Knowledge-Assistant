import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from app.config import get_settings


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify GET /health returns 200 with component statuses per Spec 9.1."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        components = data["components"]
        assert "vector_store" in components
        assert "embedder" in components
        assert "llm" in components
        assert "status" in components["vector_store"]
        assert "status" in components["embedder"]
        assert "status" in components["llm"]


@pytest.mark.asyncio
async def test_frontend_served():
    """Verify GET / serves HTML content and static JS references the streaming endpoint per Spec 12."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "app.js" in response.text

        # Verify static JS file is served and references /ask/stream
        js_res = await client.get("/static/app.js")
        assert js_res.status_code == 200
        assert "/ask/stream" in js_res.text


@pytest.mark.asyncio
async def test_empty_query_validation():
    """Verify empty or whitespace-only query yields 422 with clean JSON error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Empty string
        res1 = await client.post("/ask", json={"query": ""})
        assert res1.status_code == 422
        body1 = res1.json()
        assert "error" in body1
        assert body1["error"]["code"] == "invalid_input"

        # Whitespace-only string
        res2 = await client.post("/ask", json={"query": "   \n\t  "})
        assert res2.status_code == 422
        body2 = res2.json()
        assert "error" in body2
        assert body2["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_overlength_query_validation():
    """Verify query exceeding MAX_QUERY_CHARS yields 422 with clean JSON error."""
    settings = get_settings()
    overlength_query = "x" * (settings.max_query_chars + 10)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ask", json={"query": overlength_query})
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "invalid_input"


def test_settings_load_defaults():
    """Verify config settings load default parameters as specified."""
    settings = get_settings()
    assert settings.top_k == 5
    assert settings.score_threshold == 0.35
    assert settings.max_context_chars == 8000
    assert settings.chunk_size_tokens == 500
    assert settings.chunk_overlap_tokens == 60
    assert settings.ocr_min_text_chars == 30
    assert settings.ocr_low_conf_word == 30.0
    assert settings.ocr_low_conf_chunk == 60.0
    assert settings.retrieval_mode in ["baseline", "hybrid"]

