"""Stage 2 Verification Tests: PDF Extraction, Section-Aware Chunking, Entity Extraction & Merging."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from backend.config import settings
from backend.pipeline.pdf_extractor import pdf_extractor, ExtractedSection
from backend.pipeline.chunker import chunker, SectionAwareChunker
from backend.pipeline.extractor import extractor
from backend.pipeline.entity_merging import entity_merger
from backend.pipeline.neo4j_store import neo4j_store

@pytest.mark.asyncio
async def test_pdf_extraction_and_chunking():
    """Verify that a landmark paper PDF is extracted with sections and correctly chunked."""
    test_pdf = settings.PAPERS_DIR / "1609.02907_GCN.pdf"
    assert test_pdf.exists(), "Sample landmark PDF must exist in data/papers"

    sections = pdf_extractor.extract_from_pdf(test_pdf)
    assert len(sections) > 0, "Should extract at least one section"
    assert any("title" in s and len(s["text"]) > 20 for s in sections)

    chunks = chunker.chunk_sections("gcn_paper", sections)
    assert len(chunks) > 0
    assert chunks[0]["chunk_id"].startswith("gcn_paper_chunk_")
    assert "section" in chunks[0]
    assert "page" in chunks[0]
    assert "content" in chunks[0]

@pytest.mark.asyncio
async def test_entity_extraction_and_merging():
    """Verify entity & relationship extraction and merging."""
    sample_chunk = {
        "chunk_id": "test_chunk_0001",
        "section": "Model Architecture",
        "page": 2,
        "content": "We propose the Graph Convolutional Network (GCN) that leverages spectral graph convolutions. "
                   "GCN outperforms DeepWalk and node2vec on citation datasets like Cora and Citeseer. "
                   "The Transformer model uses Self-Attention mechanisms."
    }

    result = await extractor.extract_from_chunk(sample_chunk)
    assert "entities" in result
    assert "relations" in result

    entities = result["entities"]
    relations = result["relations"]

    assert len(entities) >= 2, f"Should extract multiple entities, got {len(entities)}"

    # Check entity format
    for ent in entities:
        assert "id" in ent
        assert "name" in ent
        assert "type" in ent
        assert "source_chunk_ids" in ent
        assert "test_chunk_0001" in ent["source_chunk_ids"]

    # Check relation format (must be directed)
    for rel in relations:
        assert "source" in rel
        assert "target" in rel
        assert "type" in rel
        assert rel["source"] != rel["target"]

    # Test merging
    raw_entities = entities + [
        {"id": "gcn", "name": "GCN", "type": "Architecture", "description": "GCN description", "source_chunk_ids": ["test_chunk_0002"]},
        {"id": "cora", "name": "Cora", "type": "Dataset", "description": "Benchmark dataset", "source_chunk_ids": ["test_chunk_0002"]}
    ]
    raw_relations = relations + [
        {"source": "gcn", "target": "cora", "type": "EVALUATED_ON", "description": "Tested on Cora", "weight": 1.0, "source_chunk_ids": ["test_chunk_0002"]}
    ]

    merged_ents, merged_rels = entity_merger.merge_graph(raw_entities, raw_relations)
    assert len(merged_ents) > 0
    # GCN should be normalized and chunk references merged
    gcn_nodes = [e for e in merged_ents if "graph_convolutional_network" in e["id"] or "gcn" in e["id"]]
    assert len(gcn_nodes) >= 1
    assert len(gcn_nodes[0]["source_chunk_ids"]) >= 1

    # Persist in store with isolated test paper ID
    import uuid
    test_pid = f"test_paper_stage2_{uuid.uuid4().hex[:8]}"
    neo4j_store.save_entities_and_relations(test_pid, merged_ents, merged_rels, [sample_chunk])
    graph = neo4j_store.get_paper_graph(test_pid)
    assert len(graph["nodes"]) == len(merged_ents)

@pytest.mark.asyncio
async def test_fallback_extraction_does_not_invent_directed_edges(monkeypatch):
    """When LLM extraction fails or is unavailable, fallback must extract entities but never invent directed relationships."""
    from backend.core.llm_client import LLMCallLog

    # Force LLM generation failure to trigger fallback
    async def mock_failed_generate(*args, **kwargs):
        return "", LLMCallLog(call_type="completion", model="test-model", status="error", error_message="LLM offline")

    monkeypatch.setattr(extractor.client, "generate", mock_failed_generate)

    chunk = {
        "chunk_id": "chunk_fallback_01",
        "section": "Methods",
        "page": 3,
        "content": "We evaluate Transformer and BERT architectures with Self-Attention and Multi-Head Attention mechanisms on SQuAD."
    }

    result = await extractor.extract_from_chunk(chunk)

    # Origin and status must explicitly indicate fallback
    assert result["origin"] == "fallback"
    assert result["status"] == "fallback"

    # Entities should be extracted heuristically with chunk provenance
    entities = result["entities"]
    assert len(entities) >= 3
    for ent in entities:
        assert ent["source_chunk_ids"] == ["chunk_fallback_01"]

    # Relations must be strictly empty (no synthetic ASSOCIATED_WITH or invented co-occurrence edges)
    relations = result["relations"]
    assert relations == []

@pytest.mark.asyncio
async def test_valid_llm_extraction_retains_relations_and_source_chunk_ids(monkeypatch):
    """Valid LLM extraction returns origin='llm' and retains directed relations with source chunk IDs."""
    from backend.core.llm_client import LLMCallLog
    import json

    llm_payload = {
        "entities": [
            {"name": "Transformer", "type": "Architecture", "description": "Sequence-to-sequence model"},
            {"name": "Self-Attention", "type": "Method", "description": "Attention mechanism"}
        ],
        "relationships": [
            {
                "source": "Transformer",
                "target": "Self-Attention",
                "type": "USES",
                "description": "Transformer relies on Self-Attention",
                "weight": 0.95
            }
        ]
    }

    async def mock_successful_generate(*args, **kwargs):
        return json.dumps(llm_payload), LLMCallLog(call_type="completion", model="test-model", status="success")

    monkeypatch.setattr(extractor.client, "generate", mock_successful_generate)

    chunk = {
        "chunk_id": "chunk_llm_01",
        "section": "Introduction",
        "page": 1,
        "content": "The Transformer model uses Self-Attention."
    }

    result = await extractor.extract_from_chunk(chunk)

    assert result["origin"] == "llm"
    assert result["status"] == "success"

    # Check entities
    assert len(result["entities"]) == 2
    for ent in result["entities"]:
        assert ent["source_chunk_ids"] == ["chunk_llm_01"]

    # Check relations
    assert len(result["relations"]) == 1
    rel = result["relations"][0]
    assert rel["source"] == "transformer"
    assert rel["target"] == "self_attention"
    assert rel["type"] == "USES"
    assert rel["weight"] == 0.95
    assert rel["source_chunk_ids"] == ["chunk_llm_01"]

def test_chunker_invalid_configurations():
    """Verify that SectionAwareChunker raises ValueError on invalid chunk_size or overlap."""
    # chunk_size must be positive integer
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        SectionAwareChunker(chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        SectionAwareChunker(chunk_size=-10, overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        SectionAwareChunker(chunk_size="100", overlap=0)  # type: ignore

    # overlap must be non-negative integer
    with pytest.raises(ValueError, match="overlap must be a non-negative integer"):
        SectionAwareChunker(chunk_size=100, overlap=-1)

    with pytest.raises(ValueError, match="overlap must be a non-negative integer"):
        SectionAwareChunker(chunk_size=100, overlap="10")  # type: ignore

    # overlap must be strictly less than chunk_size
    with pytest.raises(ValueError, match="overlap .* must be strictly less than chunk_size"):
        SectionAwareChunker(chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="overlap .* must be strictly less than chunk_size"):
        SectionAwareChunker(chunk_size=100, overlap=150)

def test_chunker_normal_chunking_with_overlap():
    """Verify that multi-chunk sections are cleanly split with exact configured overlap and boundary respect."""
    custom_chunker = SectionAwareChunker(chunk_size=10, overlap=3)

    # 25 words:
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)

    sections = [
        {"title": "Section 1", "page": 1, "text": text},
        {"title": "Section 2", "page": 2, "text": "short text only five words"}
    ]

    chunks = custom_chunker.chunk_sections("test_paper", sections)

    # Section 1 splits:
    # chunk 0: words 0..10 (10 words) -> step = 10 - 3 = 7
    # chunk 1: words 7..17 (10 words) -> step = 7
    # chunk 2: words 14..24 (10 words) -> step = 7
    # chunk 3: words 21..25 (4 words)
    # Section 2:
    # chunk 4: 5 words <= 10 (1 chunk)
    assert len(chunks) == 5

    # Check IDs and sections
    assert chunks[0]["chunk_id"] == "test_paper_chunk_0000"
    assert chunks[0]["section"] == "Section 1"
    assert chunks[0]["word_count"] == 10
    assert chunks[0]["content"] == " ".join(words[0:10])

    assert chunks[1]["chunk_id"] == "test_paper_chunk_0001"
    assert chunks[1]["section"] == "Section 1"
    assert chunks[1]["word_count"] == 10
    assert chunks[1]["content"] == " ".join(words[7:17])

    # Check that overlap words between chunk 0 and chunk 1 are words 7, 8, 9 (3 words)
    chunk0_words = chunks[0]["content"].split()
    chunk1_words = chunks[1]["content"].split()
    assert chunk0_words[-3:] == chunk1_words[:3]

    assert chunks[2]["chunk_id"] == "test_paper_chunk_0002"
    assert chunks[2]["section"] == "Section 1"
    assert chunks[2]["word_count"] == 10
    assert chunks[2]["content"] == " ".join(words[14:24])

    assert chunks[3]["chunk_id"] == "test_paper_chunk_0003"
    assert chunks[3]["section"] == "Section 1"
    assert chunks[3]["word_count"] == 4
    assert chunks[3]["content"] == " ".join(words[21:25])

    # Section 2 should start fresh without carrying over words from Section 1
    assert chunks[4]["chunk_id"] == "test_paper_chunk_0004"
    assert chunks[4]["section"] == "Section 2"
    assert chunks[4]["page"] == 2
    assert chunks[4]["word_count"] == 5
    assert chunks[4]["content"] == "short text only five words"

def test_pdf_extractor_multi_page_section_span(monkeypatch, tmp_path):
    """A section whose body text spans multiple pages receives the correct first and last contributing page."""
    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    # Page 1: Heading and first lines
    p1 = MagicMock()
    p1.extract_text.return_value = "1. Introduction\nText starting on page one.\nMore line on page one."
    # Page 2: Continuation text
    p2 = MagicMock()
    p2.extract_text.return_value = "Continuation on page two.\nAdditional arguments."
    # Page 3: Concluding lines of Introduction
    p3 = MagicMock()
    p3.extract_text.return_value = "Concluding remarks of introduction."

    mock_reader = MagicMock()
    mock_reader.pages = [p1, p2, p3]
    monkeypatch.setattr("backend.pipeline.pdf_extractor.PdfReader", lambda path: mock_reader)

    sections = pdf_extractor.extract_from_pdf(fake_pdf)
    intro_sec = [s for s in sections if "Introduction" in s["title"]]
    assert len(intro_sec) == 1
    assert intro_sec[0]["page"] == 1
    assert intro_sec[0]["end_page"] == 3
    assert "Text starting on page one" in intro_sec[0]["text"]
    assert "Concluding remarks of introduction" in intro_sec[0]["text"]

def test_pdf_extractor_heading_at_start_of_new_page_resets_tracking(monkeypatch, tmp_path):
    """A heading at the start of a new page resets page tracking correctly."""
    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    p1 = MagicMock()
    p1.extract_text.return_value = "1. Introduction\nIntroductory discussion on page one."
    p2 = MagicMock()
    p2.extract_text.return_value = "2. Related Work\nPrior literature analysis on page two."
    p3 = MagicMock()
    p3.extract_text.return_value = "3. Methodology\nProposed algorithm description on page three."

    mock_reader = MagicMock()
    mock_reader.pages = [p1, p2, p3]
    monkeypatch.setattr("backend.pipeline.pdf_extractor.PdfReader", lambda path: mock_reader)

    sections = pdf_extractor.extract_from_pdf(fake_pdf)
    assert len(sections) == 3

    intro = sections[0]
    assert "Introduction" in intro["title"]
    assert intro["page"] == 1
    assert intro["end_page"] == 1

    related = sections[1]
    assert "Related Work" in related["title"]
    assert related["page"] == 2
    assert related["end_page"] == 2

    method = sections[2]
    assert "Methodology" in method["title"]
    assert method["page"] == 3
    assert method["end_page"] == 3

def test_pdf_extractor_empty_page_does_not_extend_end_page(monkeypatch, tmp_path):
    """An empty page does not incorrectly extend a section's end_page."""
    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    p1 = MagicMock()
    p1.extract_text.return_value = "1. Introduction\nContent strictly on page one."
    p2 = MagicMock()
    p2.extract_text.return_value = "   \n\n  "  # blank/empty page
    p3 = MagicMock()
    p3.extract_text.return_value = None  # None/unextractable page

    mock_reader = MagicMock()
    mock_reader.pages = [p1, p2, p3]
    monkeypatch.setattr("backend.pipeline.pdf_extractor.PdfReader", lambda path: mock_reader)

    sections = pdf_extractor.extract_from_pdf(fake_pdf)
    intro_sec = [s for s in sections if "Introduction" in s["title"]]
    assert len(intro_sec) == 1
    # End page must remain 1 because pages 2 and 3 contained no text lines
    assert intro_sec[0]["page"] == 1
    assert intro_sec[0]["end_page"] == 1

def test_pdf_extractor_no_extractable_text_returns_empty_list(monkeypatch, tmp_path):
    """A PDF with no extractable text returns an empty list without crashing."""
    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    p1 = MagicMock()
    p1.extract_text.return_value = ""
    p2 = MagicMock()
    p2.extract_text.return_value = None
    p3 = MagicMock()
    p3.extract_text.return_value = "   \n   \n"

    mock_reader = MagicMock()
    mock_reader.pages = [p1, p2, p3]
    monkeypatch.setattr("backend.pipeline.pdf_extractor.PdfReader", lambda path: mock_reader)

    sections = pdf_extractor.extract_from_pdf(fake_pdf)
    assert sections == []

def test_pdf_extractor_section_dict_fields_retained(monkeypatch, tmp_path):
    """Existing section dictionaries retain title, page, end_page, and text."""
    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    p1 = MagicMock()
    p1.extract_text.return_value = "1. Abstract\nSummary text."
    mock_reader = MagicMock()
    mock_reader.pages = [p1]
    monkeypatch.setattr("backend.pipeline.pdf_extractor.PdfReader", lambda path: mock_reader)

    sections = pdf_extractor.extract_from_pdf(fake_pdf)
    assert len(sections) == 1
    sec = sections[0]
    assert set(sec.keys()) == {"title", "page", "end_page", "text"}
    assert isinstance(sec["title"], str)
    assert isinstance(sec["page"], int)
    assert isinstance(sec["end_page"], int)
    assert isinstance(sec["text"], str)

    # Also test ExtractedSection class directly
    sec_obj = ExtractedSection("Test Sec", 3, "Some content", end_page=5)
    d = sec_obj.to_dict()
    assert d == {"title": "Test Sec", "page": 3, "end_page": 5, "text": "Some content"}

    default_obj = ExtractedSection("Default End", 2, "Text")
    assert default_obj.to_dict() == {"title": "Default End", "page": 2, "end_page": 2, "text": "Text"}

def test_chunker_end_page_propagation():
    """Verify that chunk_sections correctly propagates end_page for single-chunk, multi-chunk, and legacy sections."""
    custom_chunker = SectionAwareChunker(chunk_size=5, overlap=2)

    sections = [
        # 1. Single-chunk section with explicit page and end_page
        {
            "title": "Single Chunk Span",
            "page": 2,
            "end_page": 4,
            "text": "short text four words"
        },
        # 2. Multi-chunk section with explicit page and end_page
        {
            "title": "Multi Chunk Span",
            "page": 5,
            "end_page": 8,
            "text": "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        },
        # 3. Legacy section with page but no end_page
        {
            "title": "Legacy Section",
            "page": 9,
            "text": "legacy word1 word2 word3 word4 word5 word6 word7"
        }
    ]

    chunks = custom_chunker.chunk_sections("paper_test", sections)

    # Section 1: 4 words <= chunk_size (5) -> 1 chunk
    c0 = chunks[0]
    assert c0["chunk_id"] == "paper_test_chunk_0000"
    assert c0["section"] == "Single Chunk Span"
    assert c0["page"] == 2
    assert c0["end_page"] == 4
    assert c0["content"] == "short text four words"
    assert c0["word_count"] == 4

    # Section 2: 10 words > chunk_size (5) with overlap=2 (step=3)
    # chunk 1: words 0..5 (5 words)
    # chunk 2: words 3..8 (5 words)
    # chunk 3: words 6..10 (4 words)
    # All must retain page=5 and end_page=8
    for idx in [1, 2, 3]:
        c = chunks[idx]
        assert c["section"] == "Multi Chunk Span"
        assert c["page"] == 5
        assert c["end_page"] == 8

    assert chunks[1]["chunk_id"] == "paper_test_chunk_0001"
    assert chunks[1]["content"] == "word1 word2 word3 word4 word5"
    assert chunks[1]["word_count"] == 5

    assert chunks[2]["chunk_id"] == "paper_test_chunk_0002"
    assert chunks[2]["content"] == "word4 word5 word6 word7 word8"
    assert chunks[2]["word_count"] == 5

    assert chunks[3]["chunk_id"] == "paper_test_chunk_0003"
    assert chunks[3]["content"] == "word7 word8 word9 word10"
    assert chunks[3]["word_count"] == 4

    # Section 3: 7 words > chunk_size (5) with overlap=2, legacy dictionary (no end_page)
    # chunk 4: words 0..5 -> page=9, end_page=9
    # chunk 5: words 3..7 -> page=9, end_page=9
    c4 = chunks[4]
    assert c4["chunk_id"] == "paper_test_chunk_0004"
    assert c4["section"] == "Legacy Section"
    assert c4["page"] == 9
    assert c4["end_page"] == 9
    assert c4["content"] == "legacy word1 word2 word3 word4"
    assert c4["word_count"] == 5

    c5 = chunks[5]
    assert c5["chunk_id"] == "paper_test_chunk_0005"
    assert c5["section"] == "Legacy Section"
    assert c5["page"] == 9
    assert c5["end_page"] == 9
    assert c5["content"] == "word3 word4 word5 word6 word7"
    assert c5["word_count"] == 5

    assert len(chunks) == 6


def test_entity_merging_canonical_alias_normalization():
    """Verify that hyphenated and space-delimited aliases resolve to the same canonical entity."""
    raw_entities = [
        {
            "id": "mha_hyphenated",
            "name": "multi-head self-attention",
            "type": "Architecture",
            "description": "Hyphenated mention",
            "source_chunk_ids": ["chunk_001"],
        },
        {
            "id": "mha_spaced",
            "name": "multi head self attention",
            "type": "Architecture",
            "description": "Spaced mention",
            "source_chunk_ids": ["chunk_002"],
        },
    ]
    raw_relations = []

    merged_ents, _ = entity_merger.merge_graph(raw_entities, raw_relations)

    assert len(merged_ents) == 1
    canonical_entity = merged_ents[0]
    assert canonical_entity["name"] == "Multi-Head Attention"
    assert canonical_entity["id"] == "multi_head_attention"
    assert set(canonical_entity["source_chunk_ids"]) == {"chunk_001", "chunk_002"}


@pytest.mark.asyncio
async def test_ingestion_processes_all_chunks_beyond_limit(monkeypatch):
    """Verify that IngestionManager extracts from every chunk when supplied with >= 26 chunks."""
    from backend.pipeline.ingestion_manager import IngestionManager, JobStatus
    from backend.pipeline.neo4j_store import PersistenceResult

    im = IngestionManager()

    # Supply 30 chunks
    num_test_chunks = 30
    mock_chunks = [
        {
            "chunk_id": f"chunk_{i:04d}",
            "section": "Section",
            "page": i + 1,
            "content": f"Content for chunk {i}",
            "word_count": 4,
        }
        for i in range(num_test_chunks)
    ]

    monkeypatch.setattr(
        "backend.pipeline.pdf_extractor.pdf_extractor.extract_from_pdf",
        lambda p: [{"title": "Section", "page": 1, "text": "Content"}],
    )
    monkeypatch.setattr(
        "backend.pipeline.chunker.chunker.chunk_sections",
        lambda pid, s: mock_chunks,
    )

    extracted_chunk_ids = []

    async def mock_extract_from_chunk(chunk):
        extracted_chunk_ids.append(chunk["chunk_id"])
        return {
            "entities": [
                {
                    "id": f"ent_{chunk['chunk_id']}",
                    "name": f"Entity {chunk['chunk_id']}",
                    "type": "Concept",
                    "description": "Desc",
                    "source_chunk_ids": [chunk["chunk_id"]],
                }
            ],
            "relations": [],
        }

    monkeypatch.setattr(
        "backend.pipeline.extractor.extractor.extract_from_chunk",
        mock_extract_from_chunk,
    )

    # Mock storage so persistence completes without requiring external database
    mock_persist_res = PersistenceResult(
        success=True, storage_mode="in_memory", durable=False, message="Success"
    )
    monkeypatch.setattr(
        "backend.pipeline.ingestion_manager.neo4j_store.save_entities_and_relations",
        lambda *args, **kwargs: mock_persist_res,
    )

    job_id = "test_job_all_chunks"
    im.jobs[job_id] = JobStatus(
        job_id=job_id,
        paper_id="paper_30_chunks",
        paper_title="Test 30 Chunks Paper",
    )

    await im._run_pipeline(
        job_id,
        Path("dummy.pdf"),
        "paper_30_chunks",
        "Test 30 Chunks Paper",
        require_durable=False,
    )

    # Verify extraction was invoked for every single chunk (all 30 >= 26)
    assert len(extracted_chunk_ids) == num_test_chunks
    expected_ids = [c["chunk_id"] for c in mock_chunks]
    assert extracted_chunk_ids == expected_ids

    status = im.get_job_status(job_id)
    assert status.num_chunks == num_test_chunks
    assert status.state == "COMPLETED"
