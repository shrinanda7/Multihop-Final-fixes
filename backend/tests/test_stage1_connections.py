"""Stage 1 Verification Tests: Configuration and Service Connections."""
import pytest
from backend.config import settings
from backend.core.llm_client import ollama_client
from backend.pipeline.neo4j_store import neo4j_store

def test_settings_loaded():
    """Verify that settings are loaded with correct defaults and toggles."""
    assert settings.OLLAMA_BASE_URL.startswith("http")
    assert settings.OLLAMA_SMALL_MODEL != ""
    assert settings.OLLAMA_LARGE_MODEL != ""
    # Credentials are intentionally not shipped with the repository. An unset URI is valid for offline tests.
    assert (not settings.NEO4J_URI) or settings.NEO4J_URI.startswith(("neo4j", "bolt"))
    assert settings.ENABLE_TARJAN_COMPRESSION is True
    assert settings.ENABLE_BRIDGE_AWARE_PRUNING is True
    assert settings.ENABLE_BUDGET_AWARE_TRAVERSAL is True
    assert settings.GIANT_SCC_THRESHOLD > 0.0
    assert settings.PAPERS_DIR.exists()

def test_ollama_health_check():
    """Verify Ollama health check responds gracefully without unhandled exceptions."""
    status = ollama_client.check_health()
    assert "status" in status
    assert "base_url" in status
    # status can be 'connected', 'unreachable', or 'error' - must not raise
    assert status["status"] in ("connected", "unreachable", "error")

def test_neo4j_health_check_and_fallback():
    """Verify Neo4j health check and fallback capability."""
    status = neo4j_store.check_health()
    assert "status" in status
    assert "uri" in status
    # Must report connection status or offline with fallback
    assert status["status"] in ("connected", "offline", "error")

    # In-memory store should always function even if external Neo4j is offline
    neo4j_store.save_entities_and_relations(
        paper_id="test_paper",
        entities=[{"id": "e1", "name": "Transformer", "type": "Architecture"}],
        relations=[{"source": "e1", "target": "e1", "type": "SELF_LOOP"}],
        chunks=[{"chunk_id": "c1", "content": "Test content", "section": "Intro", "page": 1}]
    )
    graph = neo4j_store.get_paper_graph("test_paper")
    assert len(graph["nodes"]) >= 1
    assert graph["nodes"][0]["name"] == "Transformer"
