"""Focused Regression Tests: Persistence Outcomes, Storage Modes, Transactions & Ingestion Propagation."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from backend.pipeline.neo4j_store import Neo4jStore, PersistenceResult, PersistenceError
from backend.storage.neo4j_store import neo4j_store as storage_neo4j_store
from backend.pipeline.ingestion_manager import IngestionManager, JobStatus, PaperMetadata

SAMPLE_ENTITIES = [
    {"id": "e_gcn", "name": "GCN", "type": "Architecture", "description": "Graph Convolutional Network", "source_chunk_ids": ["c1"]}
]
SAMPLE_RELATIONS = [
    {"source": "e_gcn", "target": "e_gcn", "type": "SELF_CONNECT", "weight": 1.0}
]
SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "section": "1. Intro", "page": 1, "content": "Sample content"}
]

@pytest.fixture
def clean_store():
    """Create a fresh Neo4jStore instance with clean in-memory state."""
    store = Neo4jStore()
    store._memory_nodes.clear()
    store._memory_edges.clear()
    yield store
    store.close()

# ---------------------------------------------------------------------------
# 1. Neo4j Unavailable Tests
# ---------------------------------------------------------------------------

def test_neo4j_unavailable_with_fallback(clean_store, monkeypatch):
    """When Neo4j is offline and memory fallback is allowed, explicitly return in_memory mode and durable=False."""
    monkeypatch.setattr(clean_store, "get_driver", lambda: None)
    clean_store._connected = False

    result = clean_store.save_entities_and_relations(
        paper_id="paper_offline_1",
        entities=SAMPLE_ENTITIES,
        relations=SAMPLE_RELATIONS,
        chunks=SAMPLE_CHUNKS,
        allow_memory_fallback=True
    )

    # Must distinguish in-memory from durable Neo4j
    assert result.success is True
    assert result.storage_mode == "in_memory"
    assert result.durable is False
    assert result["storage_mode"] == "in_memory"
    assert result["durable"] is False

    # get_paper_graph must also reflect in_memory mode
    graph = clean_store.get_paper_graph("paper_offline_1")
    assert graph["storage_mode"] == "in_memory"
    assert graph["durable"] is False
    assert len(graph["nodes"]) == 1

def test_neo4j_unavailable_without_fallback(clean_store, monkeypatch):
    """When Neo4j is offline and memory fallback is disabled, return failed persistence."""
    monkeypatch.setattr(clean_store, "get_driver", lambda: None)
    clean_store._connected = False

    result = clean_store.save_entities_and_relations(
        paper_id="paper_offline_2",
        entities=SAMPLE_ENTITIES,
        relations=SAMPLE_RELATIONS,
        chunks=SAMPLE_CHUNKS,
        allow_memory_fallback=False
    )

    assert result.success is False
    assert result.storage_mode == "failed"
    assert result.durable is False
    assert "unavailable" in result.error.lower()
    # No data leaked to memory
    assert "paper_offline_2" not in clean_store._memory_nodes

def test_intentional_in_memory_mode(clean_store):
    """Intentional in-memory mode for testing or local development is explicitly exposed."""
    clean_store.set_in_memory_mode(True)

    health = clean_store.check_health()
    assert health["storage_mode"] == "in_memory"
    assert health["durable"] is False
    assert health["status"] == "in_memory_mode"

    result = clean_store.save_entities_and_relations(
        paper_id="paper_intentional_mem",
        entities=SAMPLE_ENTITIES,
        relations=SAMPLE_RELATIONS,
        chunks=SAMPLE_CHUNKS
    )
    assert result.success is True
    assert result.storage_mode == "in_memory"
    assert result.durable is False

# ---------------------------------------------------------------------------
# 2. Write Failure & Transactional Rollback Tests
# ---------------------------------------------------------------------------

def test_neo4j_write_failure_reports_failed_and_does_not_leak_memory(clean_store, monkeypatch):
    """When a Neo4j write transaction fails, report failure and do NOT silently populate in-memory fallback."""
    mock_session = MagicMock()
    mock_session.execute_write.side_effect = RuntimeError("Disk full / Transaction aborted")

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    monkeypatch.setattr(clean_store, "get_driver", lambda: mock_driver)
    clean_store._connected = True

    result = clean_store.save_entities_and_relations(
        paper_id="paper_fail_1",
        entities=SAMPLE_ENTITIES,
        relations=SAMPLE_RELATIONS,
        chunks=SAMPLE_CHUNKS
    )

    assert result.success is False
    assert result.storage_mode == "failed"
    assert result.durable is False
    assert "Disk full" in result.error

    # In-memory store must NOT have cached data as successful
    assert "paper_fail_1" not in clean_store._memory_nodes
    assert "paper_fail_1" not in clean_store._memory_edges

def test_neo4j_write_failure_raises_when_requested(clean_store, monkeypatch):
    """When raise_on_failure is enabled, raise PersistenceError."""
    mock_session = MagicMock()
    mock_session.execute_write.side_effect = RuntimeError("Constraint violation")

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    monkeypatch.setattr(clean_store, "get_driver", lambda: mock_driver)
    clean_store._connected = True

    with pytest.raises(PersistenceError) as exc_info:
        clean_store.save_entities_and_relations(
            paper_id="paper_fail_2",
            entities=SAMPLE_ENTITIES,
            relations=SAMPLE_RELATIONS,
            chunks=SAMPLE_CHUNKS,
            raise_on_failure=True
        )
    assert "Constraint violation" in str(exc_info.value)

# ---------------------------------------------------------------------------
# 3. Successful Persistence & Transaction Execution Tests
# ---------------------------------------------------------------------------

def test_neo4j_successful_transactional_persistence(clean_store, monkeypatch):
    """When Neo4j transaction succeeds, report durable Neo4j persistence."""
    captured_units = []

    mock_session = MagicMock()
    def fake_execute_write(unit, *args, **kwargs):
        captured_units.append(unit)
        fake_tx = MagicMock()
        return unit(fake_tx)

    mock_session.execute_write.side_effect = fake_execute_write

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    monkeypatch.setattr(clean_store, "get_driver", lambda: mock_driver)
    clean_store._connected = True

    result = clean_store.save_entities_and_relations(
        paper_id="paper_success_1",
        entities=SAMPLE_ENTITIES,
        relations=SAMPLE_RELATIONS,
        chunks=SAMPLE_CHUNKS
    )

    assert result.success is True
    assert result.storage_mode == "neo4j"
    assert result.durable is True
    assert len(captured_units) == 1, "Must execute writes within an atomic transaction unit"
    assert "paper_success_1" in clean_store._memory_nodes

def test_backend_storage_module_reexport():
    """Verify backend.storage.neo4j_store exports match backend.pipeline.neo4j_store."""
    assert storage_neo4j_store is not None
    assert hasattr(storage_neo4j_store, "save_entities_and_relations")
    assert hasattr(storage_neo4j_store, "get_paper_graph")


def test_neo4j_relationship_source_chunk_ids_persistence(clean_store, monkeypatch):
    """Verify that relationship source-chunk provenance is included in the persistence query and passed through correctly."""
    executed_queries = []

    fake_tx = MagicMock()
    def fake_run(query, **kwargs):
        executed_queries.append((query, kwargs))

    fake_tx.run.side_effect = fake_run

    mock_session = MagicMock()
    def fake_execute_write(unit, *args, **kwargs):
        return unit(fake_tx)

    mock_session.execute_write.side_effect = fake_execute_write

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    monkeypatch.setattr(clean_store, "get_driver", lambda: mock_driver)
    clean_store._connected = True

    relations_with_chunks = [
        {
            "source": "e_gcn",
            "target": "e_transformer",
            "type": "COMPARED_WITH",
            "description": "Comparison on benchmark",
            "weight": 1.5,
            "source_chunk_ids": ["c1", "c2"],
        },
        {
            "source": "e_gcn",
            "target": "e_bert",
            "type": "EVALUATED_ON",
            "description": "No source chunk ids provided",
            "weight": 1.0,
            # source_chunk_ids omitted
        },
        {
            "source": "e_bert",
            "target": "e_transformer",
            "type": "BASED_ON",
            "description": "Null source chunk ids",
            "weight": 0.8,
            "source_chunk_ids": None,
        },
    ]

    result = clean_store.save_entities_and_relations(
        paper_id="paper_rel_provenance",
        entities=SAMPLE_ENTITIES,
        relations=relations_with_chunks,
        chunks=SAMPLE_CHUNKS,
    )

    assert result.success is True
    assert result.storage_mode == "neo4j"
    assert result.durable is True

    # Find the relation query execution
    rel_runs = [
        (q, kwargs)
        for q, kwargs in executed_queries
        if "MERGE (source)-[rel:RELATED_TO" in q
    ]
    assert len(rel_runs) == 1, "Relationship query must be executed"

    query_str, params = rel_runs[0]
    # Verify the Cypher query persists source_chunk_ids with coalesce fallback to empty list
    assert "rel.source_chunk_ids = coalesce(r.source_chunk_ids, [])" in query_str
    # Verify relations payload was passed through
    assert params["relations"] == relations_with_chunks
    assert params["paper_id"] == "paper_rel_provenance"


def test_neo4j_get_paper_graph_entity_chunk_links(clean_store, monkeypatch):
    """Verify that get_paper_graph executes the MENTIONED_IN provenance query and returns entity_chunk_links."""
    executed_queries = []

    mock_session = MagicMock()

    def fake_run(query, **kwargs):
        executed_queries.append((query, kwargs))
        mock_result = MagicMock()
        if "MATCH (n:Entity" in query:
            mock_result.__iter__.return_value = [
                {"props": {"id": "e_gcn", "name": "GCN", "type": "Architecture"}}
            ]
        elif "MATCH (s:Entity" in query:
            mock_result.__iter__.return_value = [
                {"source": "e_gcn", "target": "e_gcn", "props": {"type": "SELF_CONNECT", "weight": 1.0}}
            ]
        elif "[:MENTIONED_IN]" in query:
            mock_result.__iter__.return_value = [
                {"entity_id": "e_gcn", "chunk_id": "c1"},
                {"entity_id": "e_gcn", "chunk_id": "c2"},
            ]
        else:
            mock_result.__iter__.return_value = []
        return mock_result

    mock_session.run.side_effect = fake_run

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    monkeypatch.setattr(clean_store, "get_driver", lambda: mock_driver)
    clean_store._connected = True

    graph = clean_store.get_paper_graph("paper_provenance_links", compressed=False)

    # Verify existing fields
    assert graph["storage_mode"] == "neo4j"
    assert graph["durable"] is True
    assert graph["compressed"] is False
    assert len(graph["nodes"]) == 1
    assert len(graph["edges"]) == 1

    # Verify provenance query was executed
    link_queries = [
        (q, kwargs)
        for q, kwargs in executed_queries
        if "MENTIONED_IN" in q
    ]
    assert len(link_queries) == 1, "MENTIONED_IN provenance query must be executed"
    assert link_queries[0][1]["paper_id"] == "paper_provenance_links"

    # Verify entity_chunk_links in response
    assert "entity_chunk_links" in graph
    assert graph["entity_chunk_links"] == [
        {"entity_id": "e_gcn", "chunk_id": "c1"},
        {"entity_id": "e_gcn", "chunk_id": "c2"},
    ]


# ---------------------------------------------------------------------------
# 4. Ingestion Status Propagation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingestion_fails_when_neo4j_write_fails(monkeypatch):
    """Ingestion must report FAILED and durable=False when Neo4j write transaction fails."""
    im = IngestionManager()

    # Mock pipeline steps up to storage
    monkeypatch.setattr("backend.pipeline.pdf_extractor.pdf_extractor.extract_from_pdf", lambda p: [{"title": "Intro", "text": "Content"}])
    monkeypatch.setattr("backend.pipeline.chunker.chunker.chunk_sections", lambda pid, s: [{"chunk_id": "c1", "section": "Intro", "page": 1, "content": "Content"}])
    async def mock_extract(c):
        return {"entities": SAMPLE_ENTITIES, "relations": SAMPLE_RELATIONS}
    monkeypatch.setattr("backend.pipeline.extractor.extractor.extract_from_chunk", mock_extract)

    # Force neo4j_store to return failure
    failed_result = PersistenceResult(success=False, storage_mode="failed", durable=False, error="Neo4j connection dropped")
    monkeypatch.setattr("backend.pipeline.ingestion_manager.neo4j_store.save_entities_and_relations", lambda *args, **kwargs: failed_result)

    job_id = "test_job_fail_persist"
    im.jobs[job_id] = JobStatus(job_id=job_id, paper_id="p_fail", paper_title="Test Fail Paper")

    fake_pdf = Path("fake_paper.pdf")
    await im._run_pipeline(job_id, fake_pdf, "p_fail", "Test Fail Paper")

    status = im.get_job_status(job_id)
    assert status.state == "FAILED"
    assert status.durable is False
    assert status.storage_mode == "failed"
    assert "Persistence failure" in status.message

@pytest.mark.asyncio
async def test_ingestion_cannot_report_durable_success_when_in_memory_only(monkeypatch):
    """When only in-memory storage succeeds, ingestion cannot report durable success."""
    im = IngestionManager()

    monkeypatch.setattr("backend.pipeline.pdf_extractor.pdf_extractor.extract_from_pdf", lambda p: [{"title": "Intro", "text": "Content"}])
    monkeypatch.setattr("backend.pipeline.chunker.chunker.chunk_sections", lambda pid, s: [{"chunk_id": "c1", "section": "Intro", "page": 1, "content": "Content"}])
    async def mock_extract(c):
        return {"entities": SAMPLE_ENTITIES, "relations": SAMPLE_RELATIONS}
    monkeypatch.setattr("backend.pipeline.extractor.extractor.extract_from_chunk", mock_extract)

    in_mem_result = PersistenceResult(success=True, storage_mode="in_memory", durable=False, message="Stored in memory only")
    monkeypatch.setattr("backend.pipeline.ingestion_manager.neo4j_store.save_entities_and_relations", lambda *args, **kwargs: in_mem_result)

    # Case A: require_durable=True must fail
    job_id_durable = "test_job_require_durable"
    im.jobs[job_id_durable] = JobStatus(job_id=job_id_durable, paper_id="p_in_mem_req", paper_title="Req Durable Paper")
    await im._run_pipeline(job_id_durable, Path("fake.pdf"), "p_in_mem_req", "Req Durable Paper", require_durable=True)

    status_req = im.get_job_status(job_id_durable)
    assert status_req.state == "FAILED"
    assert status_req.durable is False
    assert status_req.storage_mode == "in_memory"

    # Case B: require_durable=False marks in_memory with durable=False
    job_id_fallback = "test_job_fallback"
    im.jobs[job_id_fallback] = JobStatus(job_id=job_id_fallback, paper_id="p_in_mem_fall", paper_title="Fallback Paper")
    await im._run_pipeline(job_id_fallback, Path("fake.pdf"), "p_in_mem_fall", "Fallback Paper", require_durable=False)

    status_fall = im.get_job_status(job_id_fallback)
    assert status_fall.durable is False
    assert status_fall.storage_mode == "in_memory"
    assert "not durably stored" in status_fall.message

@pytest.mark.asyncio
async def test_ingestion_reports_durable_success_when_neo4j_succeeds(monkeypatch):
    """When Neo4j succeeds, ingestion reports durable=True and storage_mode='neo4j'."""
    im = IngestionManager()

    monkeypatch.setattr("backend.pipeline.pdf_extractor.pdf_extractor.extract_from_pdf", lambda p: [{"title": "Intro", "text": "Content"}])
    monkeypatch.setattr("backend.pipeline.chunker.chunker.chunk_sections", lambda pid, s: [{"chunk_id": "c1", "section": "Intro", "page": 1, "content": "Content"}])
    async def mock_extract(c):
        return {"entities": SAMPLE_ENTITIES, "relations": SAMPLE_RELATIONS}
    monkeypatch.setattr("backend.pipeline.extractor.extractor.extract_from_chunk", mock_extract)

    durable_result = PersistenceResult(success=True, storage_mode="neo4j", durable=True, message="Durably saved")
    monkeypatch.setattr("backend.pipeline.ingestion_manager.neo4j_store.save_entities_and_relations", lambda *args, **kwargs: durable_result)

    job_id = "test_job_durable_ok"
    im.jobs[job_id] = JobStatus(job_id=job_id, paper_id="p_durable", paper_title="Durable Paper")
    await im._run_pipeline(job_id, Path("fake.pdf"), "p_durable", "Durable Paper", require_durable=True)

    status = im.get_job_status(job_id)
    assert status.state == "COMPLETED"
    assert status.durable is True
    assert status.storage_mode == "neo4j"
    assert "durably stored" in status.message
