"""Regression tests for provenance-preserving multi-paper graph retrieval."""
import pytest

from backend.search.multi_paper_search import multi_paper_search
from backend.pipeline.neo4j_store import neo4j_store
from backend.core.llm_client import ollama_client, LLMCallLog


@pytest.mark.asyncio
async def test_multi_paper_query_builds_cross_paper_links_and_cites_evidence(monkeypatch):
    graphs = {
        "paper_a": {
            "nodes": [
                {"id": "rag", "name": "Retrieval Augmented Generation", "type": "Method", "description": "Combines retrieval with generation", "source_chunk_ids": ["a1"]},
                {"id": "dense", "name": "Dense Retrieval", "type": "Method", "description": "Retrieves semantically similar passages", "source_chunk_ids": ["a2"]},
            ],
            "edges": [{"source": "rag", "target": "dense", "type": "USES", "weight": 1.0, "source_chunk_ids": ["a1"]}],
        },
        "paper_b": {
            "nodes": [
                {"id": "rag2", "name": "Retrieval Augmented Generation", "type": "Method", "description": "Uses a retriever and generator", "source_chunk_ids": ["b1"]},
                {"id": "graph", "name": "Graph Retrieval", "type": "Method", "description": "Uses graph paths for evidence", "source_chunk_ids": ["b2"]},
            ],
            "edges": [{"source": "rag2", "target": "graph", "type": "USES", "weight": 1.0, "source_chunk_ids": ["b1"]}],
        },
    }
    chunks = {
        "paper_a": [
            {"chunk_id": "a1", "content": "Retrieval augmented generation combines external evidence with a language model.", "section": "Introduction", "page": 1},
            {"chunk_id": "a2", "content": "Dense retrieval selects semantically similar passages using vector representations.", "section": "Method", "page": 3},
        ],
        "paper_b": [
            {"chunk_id": "b1", "content": "Retrieval augmented generation can combine graph retrieval with evidence traversal.", "section": "Overview", "page": 2},
            {"chunk_id": "b2", "content": "Graph retrieval follows connected entity paths to support multi-hop questions.", "section": "Method", "page": 4},
        ],
    }
    monkeypatch.setattr(neo4j_store, "get_paper_graph", lambda pid, compressed=False: graphs.get(pid, {"nodes": [], "edges": []}))
    monkeypatch.setattr(neo4j_store, "get_chunks_by_ids", lambda pid, ids=None: [c for c in chunks.get(pid, []) if not ids or c["chunk_id"] in ids])

    async def fake_generate(prompt, **kwargs):
        assert "[E1]" in prompt
        assert "paper_a" in prompt or "Paper A" in prompt
        assert "paper_b" in prompt or "Paper B" in prompt
        return "The papers connect retrieval-augmented generation with dense and graph retrieval [E1, E2].", LLMCallLog(call_type="generate", model="test", total_tokens=42, status="success")

    monkeypatch.setattr(ollama_client, "generate", fake_generate)
    result = await multi_paper_search.search(
        question="How does retrieval augmented generation connect dense retrieval and graph retrieval for multi-hop evidence?",
        paper_ids=["paper_a", "paper_b"],
        paper_titles={"paper_a": "Paper A", "paper_b": "Paper B"},
    )
    assert result["status"] == "ok"
    assert result["cross_paper_links"] >= 2
    assert {item["paper_id"] for item in result["evidence"]} == {"paper_a", "paper_b"}
    assert "[E1" in result["answer"]
    assert any(step["action"] == "SELECT" for step in result["trace"]["steps"])


@pytest.mark.asyncio
async def test_multi_paper_query_reports_missing_papers_without_fabricating_graph(monkeypatch):
    monkeypatch.setattr(neo4j_store, "get_paper_graph", lambda pid, compressed=False: {"nodes": [], "edges": []})
    result = await multi_paper_search.search("compare methods", ["not_indexed"])
    assert result["status"] == "no_indexed_graph"
    assert result["missing_paper_ids"] == ["not_indexed"]
    assert result["evidence"] == []
    assert result["total_llm_calls"] == 0
