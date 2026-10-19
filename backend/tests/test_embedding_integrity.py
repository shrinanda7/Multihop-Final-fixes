import json
import math
import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock

from backend.core.llm_client import (
    OllamaClient,
    EmbeddingError,
    validate_embedding,
    get_expected_embedding_dim,
    ollama_client,
    LLMCallLog,
)
from backend.search.local_search import local_search
from backend.pipeline.neo4j_store import neo4j_store

@pytest.fixture
def test_client():
    """Create a fresh OllamaClient instance for isolated testing."""
    client = OllamaClient()
    client._mock_mode = False
    return client

# ---------------------------------------------------------------------------
# 1. Successful Embeddings & Live / Valid Vector Validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_embedding_validation():
    """Verify that a valid embedding vector passes all integrity checks."""
    valid_vec = [0.1] * 768
    validate_embedding(valid_vec, expected_dim=768)

    # Test with live client if Ollama is running and has embed model
    health = ollama_client.check_health()
    if health.get("status") == "connected" and health.get("has_embed_model"):
        vec, log = await ollama_client.get_embedding("landmark research query")
        assert log.status == "success"
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) and math.isfinite(x) for x in vec)

# ---------------------------------------------------------------------------
# 2. Connection Exceptions: No Pseudo-Embeddings & Strict Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_exception_raises_and_prevents_pseudo_embeddings(test_client, monkeypatch):
    """When embedding client encounters connection errors, raise EmbeddingError and NEVER return hash pseudo-embeddings."""
    async def mock_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused by Ollama daemon")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # 1. Default raises EmbeddingError
    with pytest.raises(EmbeddingError) as exc_info:
        await test_client.get_embedding("query text", raise_on_error=True)
    assert "Failed to generate embedding from Ollama service" in str(exc_info.value)

    # 2. raise_on_error=False returns empty vector and status='error', NOT a 64-dim pseudo-vector
    vec, log = await test_client.get_embedding("query text", raise_on_error=False)
    assert vec == []
    assert len(vec) == 0
    assert log.status == "error"
    assert "Connection refused" in log.error_message

# ---------------------------------------------------------------------------
# 3. HTTP Failures (404, 500, etc.)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_http_failure_handling(test_client, monkeypatch):
    """When the server responds with non-200 HTTP status, handle failure explicitly."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"error": "model \'unknown-model\' not found"}'

    async def mock_post(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Must raise EmbeddingError with HTTP details
    with pytest.raises(EmbeddingError) as exc_info:
        await test_client.get_embedding("query text", model="unknown-model")
    assert "HTTP 404" in str(exc_info.value)

    # With raise_on_error=False
    vec, log = await test_client.get_embedding("query text", model="unknown-model", raise_on_error=False)
    assert vec == []
    assert log.status == "failed"
    assert "HTTP 404" in log.error_message

# ---------------------------------------------------------------------------
# 4. Malformed Responses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_malformed_response_json(test_client, monkeypatch):
    """When response is missing 'embedding' field or contains invalid JSON, report explicit failure."""
    # Case A: Missing 'embedding' key
    mock_resp_missing = MagicMock()
    mock_resp_missing.status_code = 200
    mock_resp_missing.json.return_value = {"other_field": 123}

    async def mock_post_missing(*args, **kwargs):
        return mock_resp_missing

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_missing)

    with pytest.raises(EmbeddingError) as exc_info:
        await test_client.get_embedding("query text")
    assert "missing 'embedding' field" in str(exc_info.value)

    # Case B: Malformed JSON body
    mock_resp_bad_json = MagicMock()
    mock_resp_bad_json.status_code = 200
    mock_resp_bad_json.json.side_effect = ValueError("Invalid JSON syntax")

    async def mock_post_bad_json(*args, **kwargs):
        return mock_resp_bad_json

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_bad_json)

    with pytest.raises(EmbeddingError) as exc_info_json:
        await test_client.get_embedding("query text")
    assert "Malformed JSON" in str(exc_info_json.value)

# ---------------------------------------------------------------------------
# 5. Invalid Vectors: Non-finite, Non-numeric, Wrong Dimension, Empty
# ---------------------------------------------------------------------------

def test_validate_embedding_checks():
    """Unit tests for validate_embedding vector assertions."""
    # Empty vector
    with pytest.raises(EmbeddingError, match="vector is empty"):
        validate_embedding([])

    # Non-list type
    with pytest.raises(EmbeddingError, match="must be a list"):
        validate_embedding("invalid_string")

    # Non-finite values (NaN / Inf)
    with pytest.raises(EmbeddingError, match="non-finite"):
        validate_embedding([0.1, float("nan"), 0.3])
    with pytest.raises(EmbeddingError, match="non-finite"):
        validate_embedding([0.1, float("inf"), 0.3])

    # Non-numeric elements
    with pytest.raises(EmbeddingError, match="non-finite or non-numeric"):
        validate_embedding([0.1, "not-a-number", 0.3])

    # Dimensional consistency check
    with pytest.raises(EmbeddingError, match="dimension mismatch: expected 768, got 64"):
        validate_embedding([0.1] * 64, expected_dim=768)

@pytest.mark.asyncio
async def test_embedding_api_dimensional_mismatch(test_client, monkeypatch):
    """When API returns a vector of mismatched dimension for the configured model, raise EmbeddingError."""
    # Return 64-dim vector for nomic-embed-text (which expects 768)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embedding": [0.05] * 64}

    async def mock_post(*args, **kwargs):
        return mock_resp

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(EmbeddingError) as exc_info:
        await test_client.get_embedding("query text", model="nomic-embed-text")
    assert "dimension mismatch" in str(exc_info.value)

# ---------------------------------------------------------------------------
# 6. Intentional Test Mock Embeddings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_intentional_mock_embeddings(test_client):
    """When mock mode is intentionally enabled for testing, return valid, finite, dimensionally consistent vectors."""
    test_client.set_mock_mode(True, dim=768)

    vec, log = await test_client.get_embedding("Multi-Head Attention Transformer")
    assert log.status == "mock"
    assert len(vec) == 768
    assert all(isinstance(x, float) and math.isfinite(x) for x in vec)

    # Verify unit normalization
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3

    # Turning mock mode off returns to live client
    test_client.set_mock_mode(False)
    assert test_client._mock_mode is False

# ---------------------------------------------------------------------------
# 7. Caller Retrieval Handling: No Silent Pseudo-Vectors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_search_handles_embedding_failure_without_pseudo_vectors(monkeypatch):
    """Confirm that local search does not crash or use pseudo-embeddings when embedding fails."""
    paper_id = "test_paper_local_embed_fail"
    neo4j_store.save_entities_and_relations(
        paper_id=paper_id,
        entities=[
            {"id": "e_attn", "name": "Attention", "type": "Method", "description": "Self-attention mechanism"}
        ],
        relations=[],
        chunks=[{"chunk_id": "c1", "content": "Attention is all you need", "section": "1. Intro", "page": 1}]
    )

    # Force get_embedding to fail
    async def mock_fail_embedding(*args, **kwargs):
        raise EmbeddingError("Ollama daemon is offline")

    monkeypatch.setattr(ollama_client, "get_embedding", mock_fail_embedding)

    result = await local_search.search(
        question="What is the Attention mechanism?",
        paper_id=paper_id,
        paper_title="Attention Paper"
    )

    assert result["mode"] == "local"
    assert "answer" in result
    assert "trace" in result
    # Check trace includes the FALLBACK step indicating vector matching was unavailable
    actions = [s.get("action") for s in result["trace"]["steps"]]
    assert "FALLBACK" in actions

# ---------------------------------------------------------------------------
# 8. Exactly One Call-Log Entry Invariant on Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exactly_one_call_log_entry_on_failure(test_client, monkeypatch):
    """Confirm that every failed embedding request produces exactly one call_log entry."""
    # Subtest 1: Connection exception (raise_on_error=True)
    async def mock_network_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_network_error)

    test_client.call_logs.clear()
    with pytest.raises(EmbeddingError):
        await test_client.get_embedding("test query", raise_on_error=True)
    assert len(test_client.call_logs) == 1
    assert test_client.call_logs[0].status == "error"

    # Subtest 2: Connection exception (raise_on_error=False)
    test_client.call_logs.clear()
    vec, log = await test_client.get_embedding("test query", raise_on_error=False)
    assert len(test_client.call_logs) == 1
    assert log.status == "error"
    assert vec == []

    # Subtest 3: HTTP 500 error
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    mock_resp_500.text = "Internal Server Error"

    async def mock_http_500(*args, **kwargs):
        return mock_resp_500

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_http_500)

    test_client.call_logs.clear()
    with pytest.raises(EmbeddingError):
        await test_client.get_embedding("test query", raise_on_error=True)
    assert len(test_client.call_logs) == 1
    assert test_client.call_logs[0].status == "failed"

    # Subtest 4: Malformed response / dimensional mismatch
    mock_resp_bad = MagicMock()
    mock_resp_bad.status_code = 200
    mock_resp_bad.json.return_value = {"embedding": []}  # empty vector

    async def mock_http_bad_vector(*args, **kwargs):
        return mock_resp_bad

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_http_bad_vector)

    test_client.call_logs.clear()
    with pytest.raises(EmbeddingError):
        await test_client.get_embedding("test query", raise_on_error=True)
    assert len(test_client.call_logs) == 1
    assert test_client.call_logs[0].status == "failed"

# ---------------------------------------------------------------------------
# 9. Empty Report Embeddings Excluded from Cache and Retrieval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_report_embeddings_excluded_from_cache_and_retrieval(tmp_path, monkeypatch):
    """Empty report embeddings must not be persisted as valid vectors and must not be loaded as valid vectors."""
    from backend.pipeline.report_generator import CommunityReportGenerator
    from backend.core.clustering import HierarchicalCommunity

    generator = CommunityReportGenerator()
    generator.cache_dir = tmp_path

    comm = HierarchicalCommunity(
        community_id="comm_test_no_embed",
        level=1,
        title="Community Without Embeddings",
        member_node_ids=["n1"]
    )
    comm_dict = {comm.id: comm}
    nodes = [{"id": "n1", "name": "Node 1", "type": "Concept", "description": "Desc"}]
    edges = []

    # Force embedding generation to fail
    async def mock_fail_embed(*args, **kwargs):
        return [], LLMCallLog(call_type="embedding", model="nomic-embed-text", status="failed")

    monkeypatch.setattr(ollama_client, "get_embedding", mock_fail_embed)

    # Mock generate to succeed
    async def mock_generate(*args, **kwargs):
        return "Executive Summary\n\nDetailed content", LLMCallLog(call_type="completion", model="llama3.1:8b", status="success")

    monkeypatch.setattr(ollama_client, "generate", mock_generate)

    # 1. Run generation
    updated = await generator.generate_reports_for_hierarchy(comm_dict, nodes, edges, use_cache=False)
    assert updated[comm.id].embedding == []

    # 2. Check cached file on disk: "embedding" key must not be persisted with empty vector
    cache_file = tmp_path / f"{comm.id}_report.json"
    assert cache_file.exists()
    with open(cache_file, "r") as f:
        cache_data = json.load(f)
    assert "embedding" not in cache_data

    # 3. If cache file manually contains empty or invalid embedding, reloading sanitizes it
    cache_data["embedding"] = []
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    comm2 = HierarchicalCommunity(
        community_id="comm_test_no_embed",
        level=1,
        title="Community Without Embeddings",
        member_node_ids=["n1"]
    )
    reloaded = await generator.generate_reports_for_hierarchy({comm2.id: comm2}, nodes, edges, use_cache=True)
    assert reloaded[comm2.id].embedding == []
