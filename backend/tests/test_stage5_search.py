"""Stage 5 Verification Tests: Search Modes (Static, Dynamic, Ours, Local) & Traversal Traces."""
import pytest
from backend.search.query_router import query_router
from backend.pipeline.neo4j_store import neo4j_store

@pytest.fixture(autouse=True)
def setup_test_graph():
    """Seed test graph for search evaluations."""
    paper_id = "test_paper_stage5"
    nodes = [
        {"id": "transformer", "name": "Transformer", "type": "Architecture", "description": "Sequence-to-sequence model using self-attention", "source_chunk_ids": ["c1"]},
        {"id": "attention", "name": "Multi-Head Attention", "type": "Method", "description": "Parallel attention heads capturing diverse representation subspaces", "source_chunk_ids": ["c1"]},
        {"id": "bleu", "name": "BLEU Score", "type": "Metric", "description": "Bilingual Evaluation Understudy metric for translation quality", "source_chunk_ids": ["c2"]},
        {"id": "wmt", "name": "WMT 2014 En-De", "type": "Dataset", "description": "Machine translation benchmark dataset", "source_chunk_ids": ["c2"]},
    ]
    edges = [
        {"source": "transformer", "target": "attention", "type": "USES", "weight": 1.0, "source_chunk_ids": ["c1"]},
        {"source": "attention", "target": "transformer", "type": "BELONGS_TO", "weight": 1.0, "source_chunk_ids": ["c1"]},
        {"source": "transformer", "target": "wmt", "type": "EVALUATED_ON", "weight": 1.0, "source_chunk_ids": ["c2"]},
        {"source": "wmt", "target": "bleu", "type": "MEASURED_BY", "weight": 1.0, "source_chunk_ids": ["c2"]}
    ]
    chunks = [
        {"chunk_id": "c1", "section": "3.1 Multi-Head Attention", "page": 4, "content": "Multi-Head Attention allows the model to jointly attend to information."},
        {"chunk_id": "c2", "section": "5.1 Machine Translation", "page": 7, "content": "On the WMT 2014 English-to-German task, the Transformer establishes a new state of the art BLEU score of 28.4."}
    ]
    neo4j_store.save_entities_and_relations(paper_id, nodes, edges, chunks)
    return paper_id

@pytest.mark.asyncio
async def test_static_search():
    res = await query_router.execute_query(
        question="How does Multi-Head Attention improve performance?",
        paper_id="test_paper_stage5",
        paper_title="Attention Is All You Need",
        mode="static"
    )
    assert res["mode"] == "static"
    assert "answer" in res
    assert "trace" in res
    assert len(res["trace"]["steps"]) > 0
    assert res["total_llm_calls"] >= 1

@pytest.mark.asyncio
async def test_dynamic_search():
    res = await query_router.execute_query(
        question="What dataset was used for evaluation and what was the metric?",
        paper_id="test_paper_stage5",
        paper_title="Attention Is All You Need",
        mode="dynamic"
    )
    assert res["mode"] == "dynamic"
    assert "answer" in res
    assert "trace" in res
    assert len(res["trace"]["steps"]) > 0

@pytest.mark.asyncio
async def test_ours_search_with_bridge_and_budget():
    res = await query_router.execute_query(
        question="Describe the core architecture and translation benchmarks.",
        paper_id="test_paper_stage5",
        paper_title="Attention Is All You Need",
        mode="ours",
        llm_budget=6
    )
    assert res["mode"] == "ours"
    assert "answer" in res
    assert "compressed_graph" in res
    assert "original_graph" in res
    assert "trace" in res
    # Verify trace contains actions
    actions = [s["action"] for s in res["trace"]["steps"]]
    assert "VISIT" in actions

@pytest.mark.asyncio
async def test_local_search():
    res = await query_router.execute_query(
        question="What is the BLEU score on WMT 2014?",
        paper_id="test_paper_stage5",
        paper_title="Attention Is All You Need",
        mode="local"
    )
    assert res["mode"] == "local"
    assert "answer" in res
    assert "trace" in res

@pytest.mark.asyncio
@pytest.mark.parametrize("search_mode", ["ours", "static", "dynamic", "local", "auto"])
async def test_missing_graph_returns_no_indexed_graph(search_mode):
    """Confirm that any query mode on an unindexed paper returns explicit no_indexed_graph status with empty topologies."""
    unindexed_id = "missing_paper_unindexed_001"
    res = await query_router.execute_query(
        question="What are the key contributions?",
        paper_id=unindexed_id,
        paper_title="Unindexed Paper Title",
        mode=search_mode
    )

    assert res["status"] == "no_indexed_graph"
    assert res["paper_id"] == unindexed_id
    assert "No indexed knowledge graph found" in res["answer"]
    assert res["original_graph"] == {"nodes": [], "edges": []}
    assert res["compressed_graph"]["nodes"] == []
    assert res["compressed_graph"]["edges"] == []
    assert res["total_llm_calls"] == 0
    assert res["total_tokens"] == 0

@pytest.mark.asyncio
async def test_missing_graph_causes_no_persistence_writes_or_answer_generation_calls(monkeypatch):
    """Confirm that a missing graph does NOT trigger persistence writes (Neo4j or in-memory) or answer-generation LLM calls."""
    from backend.core.llm_client import ollama_client

    unindexed_id = "missing_paper_unindexed_002"

    # Spy/guard against unexpected write calls
    save_called = False
    def guard_save(*args, **kwargs):
        nonlocal save_called
        save_called = True
        raise AssertionError("neo4j_store.save_entities_and_relations should NOT be called for missing graph!")

    # Spy/guard against LLM generation calls
    llm_called = False
    async def guard_generate(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("ollama_client.generate should NOT be called when no indexed graph exists!")

    monkeypatch.setattr(neo4j_store, "save_entities_and_relations", guard_save)
    monkeypatch.setattr(ollama_client, "generate", guard_generate)

    # Initial check: no nodes in persistence
    assert len(neo4j_store.get_paper_graph(unindexed_id)["nodes"]) == 0
    assert unindexed_id not in neo4j_store._memory_nodes
    assert unindexed_id not in neo4j_store._memory_edges

    res = await query_router.execute_query(
        question="Explain the core mechanism of this unindexed paper.",
        paper_id=unindexed_id,
        paper_title="Unindexed Paper Title",
        mode="ours"
    )

    # Verify status
    assert res["status"] == "no_indexed_graph"

    # Verify neither save nor LLM was called
    assert not save_called, "save_entities_and_relations was called unexpectedly"
    assert not llm_called, "ollama_client.generate was called unexpectedly"

    # Verify no fabricated data was saved in Neo4j or in-memory persistence
    assert unindexed_id not in neo4j_store._memory_nodes
    assert unindexed_id not in neo4j_store._memory_edges
    post_check = neo4j_store.get_paper_graph(unindexed_id)
    assert len(post_check["nodes"]) == 0
    assert len(post_check["edges"]) == 0

@pytest.mark.asyncio
async def test_valid_indexed_paper_query_execution():
    """Confirm that a valid, indexed paper follows the normal query retrieval pipeline."""
    res = await query_router.execute_query(
        question="How does Multi-Head Attention relate to the Transformer architecture?",
        paper_id="test_paper_stage5",
        paper_title="Attention Is All You Need",
        mode="ours"
    )
    assert res.get("status") != "no_indexed_graph"
    assert "answer" in res
    assert len(res["answer"]) > 0
    assert "original_graph" in res
    assert len(res["original_graph"]["nodes"]) > 0
    assert "compressed_graph" in res
    assert "trace" in res
    assert len(res["trace"]["steps"]) > 0


