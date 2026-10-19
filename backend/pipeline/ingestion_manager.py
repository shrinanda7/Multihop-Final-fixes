"""Background Ingestion Job Manager with fine-grained status and progress states."""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.config import settings
from backend.pipeline.pdf_extractor import pdf_extractor
from backend.pipeline.chunker import chunker
from backend.pipeline.extractor import extractor
from backend.pipeline.entity_merging import entity_merger
from backend.pipeline.neo4j_store import neo4j_store

logger = logging.getLogger("graphrag.ingestion")

class JobStatus(BaseModel):
    job_id: str
    paper_id: str
    paper_title: str
    state: str = "QUEUED"  # QUEUED, EXTRACTING_PDF, CHUNKING, EXTRACTING_ENTITIES, MERGING_ENTITIES, STORING_GRAPH, COMPLETED, FAILED
    progress: int = 0      # 0 to 100
    message: str = "Job created"
    num_chunks: int = 0
    num_entities: int = 0
    num_relations: int = 0
    error: Optional[str] = None
    storage_mode: Optional[str] = None  # "neo4j", "in_memory", "failed"
    durable: bool = False

class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    filename: str
    file_size_kb: int
    is_preloaded: bool
    status: str = "ready" # pending, processing, ready, failed
    num_entities: int = 0
    num_relations: int = 0
    num_communities: int = 0
    storage_mode: Optional[str] = None  # "neo4j", "in_memory", "failed"
    durable: bool = False

class IngestionManager:
    """Orchestrates document ingestion background tasks and status tracking."""

    def __init__(self):
        self.jobs: Dict[str, JobStatus] = {}
        self.papers: Dict[str, PaperMetadata] = {}
        self._deleted_papers = set()
        self._init_preloaded_papers()

    def _init_preloaded_papers(self):
        """Register the 10 preloaded landmark research papers in the catalog."""
        if not settings.PAPERS_DIR.exists():
            return

        for pdf_file in settings.PAPERS_DIR.glob("*.pdf"):
            paper_id = pdf_file.stem
            if paper_id in self._deleted_papers or paper_id in self.papers:
                continue
            clean_title = pdf_file.stem.split("_", 1)[-1].replace("_", " ")
            size_kb = pdf_file.stat().st_size // 1024
            self.papers[paper_id] = PaperMetadata(
                paper_id=paper_id,
                title=clean_title,
                filename=pdf_file.name,
                file_size_kb=size_kb,
                is_preloaded=True,
                status="ready"
            )

    def get_all_papers(self) -> List[PaperMetadata]:
        return list(self.papers.values())

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        return self.jobs.get(job_id)

    def delete_paper(self, paper_id: str) -> bool:
        """Purge paper from catalog, disk PDF files, and Neo4j graph storage."""
        self._deleted_papers.add(paper_id)
        removed = self.papers.pop(paper_id, None)
        try:
            neo4j_store.clear_paper_graph(paper_id)
        except Exception as e:
            logger.warning(f"Error purging paper {paper_id} from Neo4j: {e}")

        # Delete any matching uploaded PDFs from PAPERS_DIR
        for f in settings.PAPERS_DIR.glob(f"*{paper_id}*.pdf"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        return removed is not None or True

    async def start_ingestion(
        self,
        pdf_path: Path,
        paper_id: Optional[str] = None,
        title: Optional[str] = None,
        require_durable: bool = False
    ) -> str:
        """Start async ingestion job."""
        job_id = str(uuid.uuid4())[:8]
        pid = paper_id or pdf_path.stem
        ptype = title or pdf_path.stem.replace("_", " ")

        status = JobStatus(job_id=job_id, paper_id=pid, paper_title=ptype, state="QUEUED", progress=5)
        self.jobs[job_id] = status

        # Spawn background task
        asyncio.create_task(self._run_pipeline(job_id, pdf_path, pid, ptype, require_durable=require_durable))
        return job_id

    async def _run_pipeline(
        self,
        job_id: str,
        pdf_path: Path,
        paper_id: str,
        title: str,
        require_durable: bool = False
    ):
        status = self.jobs[job_id]
        try:
            # 1. Extract PDF
            status.state = "EXTRACTING_PDF"
            status.progress = 15
            status.message = "Extracting structured text and section headings from PDF..."
            sections = pdf_extractor.extract_from_pdf(pdf_path)

            # 2. Chunking
            status.state = "CHUNKING"
            status.progress = 30
            status.message = f"Partitioning {len(sections)} sections into semantic chunks..."
            chunks = chunker.chunk_sections(paper_id, sections)
            status.num_chunks = len(chunks)

            # 3. Entity & Relation Extraction
            status.state = "EXTRACTING_ENTITIES"
            status.progress = 50
            status.message = f"Extracting entities & directed relations from {len(chunks)} chunks..."

            raw_entities: List[Dict[str, Any]] = []
            raw_relations: List[Dict[str, Any]] = []

            # Prioritize key structural chunks for deep LLM extraction to avoid API rate limits
            # (e.g. up to 4 key chunks covering abstract, model architecture, methodology)
            key_chunk_indices = set()
            for idx, c in enumerate(chunks):
                sec = c.get("section", "").lower()
                if any(k in sec for k in ("abstract", "intro", "model", "architecture", "method", "approach")):
                    key_chunk_indices.add(idx)
                if len(key_chunk_indices) >= 4:
                    break

            if not key_chunk_indices:
                key_chunk_indices = set(range(min(len(chunks), 4)))

            for idx, c in enumerate(chunks):
                if idx in key_chunk_indices:
                    extracted = await extractor.extract_from_chunk(c)
                    raw_entities.extend(extracted.get("entities", []))
                    raw_relations.extend(extracted.get("relations", []))
                    await asyncio.sleep(0.15)
                else:
                    # High-precision domain entity harvesting without LLM rate-limit overhead
                    fallback_res = extractor._rule_based_fallback(c)
                    raw_entities.extend(fallback_res.get("entities", []))

            # 4. Entity Merging & Deduplication
            status.state = "MERGING_ENTITIES"
            status.progress = 75
            status.message = "Canonical entity resolution and alias merging..."
            merged_ents, merged_rels = entity_merger.merge_graph(raw_entities, raw_relations)
            status.num_entities = len(merged_ents)
            status.num_relations = len(merged_rels)

            # 5. Graph Persistence (Neo4j / fallback)
            status.state = "STORING_GRAPH"
            status.progress = 90
            status.message = f"Persisting {len(merged_ents)} entities and {len(merged_rels)} relations to graph..."
            persist_res = neo4j_store.save_entities_and_relations(
                paper_id,
                merged_ents,
                merged_rels,
                chunks,
                allow_memory_fallback=not require_durable
            )

            # Explicit check: Failed persistence must NOT report success
            if not persist_res.success or persist_res.storage_mode == "failed":
                status.state = "FAILED"
                status.progress = 90
                status.error = persist_res.error or "Graph persistence failed"
                status.message = f"Persistence failure: {persist_res.error or 'Failed to persist graph'}"
                status.storage_mode = "failed"
                status.durable = False
                logger.error(f"Ingestion job {job_id} failed persistence: {persist_res.error}")
                return

            if persist_res.storage_mode == "in_memory":
                # In-memory storage only: cannot report durable success
                if require_durable:
                    status.state = "FAILED"
                    status.progress = 90
                    status.error = "Durable Neo4j persistence required, but only in-memory storage was available"
                    status.message = "Ingestion failed: durable Neo4j persistence required but unavailable."
                    status.storage_mode = "in_memory"
                    status.durable = False
                    return

                status.state = "COMPLETED"
                status.progress = 100
                status.storage_mode = "in_memory"
                status.durable = False
                status.message = (
                    f"Ingested into in-memory storage only (Neo4j offline/unavailable - not durably stored). "
                    f"Graph ready with {len(merged_ents)} entities and {len(merged_rels)} relations."
                )

                file_size = (pdf_path.stat().st_size // 1024) if pdf_path.exists() else 0
                self.papers[paper_id] = PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    filename=pdf_path.name,
                    file_size_kb=file_size,
                    is_preloaded=False,
                    status="ready",
                    num_entities=len(merged_ents),
                    num_relations=len(merged_rels),
                    storage_mode="in_memory",
                    durable=False
                )
            else:
                # Durable Neo4j persistence
                status.state = "COMPLETED"
                status.progress = 100
                status.storage_mode = "neo4j"
                status.durable = True
                status.message = (
                    f"Successfully ingested paper! Graph durably stored in Neo4j with "
                    f"{len(merged_ents)} entities and {len(merged_rels)} relations."
                )

                file_size = (pdf_path.stat().st_size // 1024) if pdf_path.exists() else 0
                self.papers[paper_id] = PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    filename=pdf_path.name,
                    file_size_kb=file_size,
                    is_preloaded=False,
                    status="ready",
                    num_entities=len(merged_ents),
                    num_relations=len(merged_rels),
                    storage_mode="neo4j",
                    durable=True
                )
        except Exception as e:
            logger.error(f"Ingestion job {job_id} failed: {e}", exc_info=True)
            status.state = "FAILED"
            status.error = str(e)
            status.message = f"Ingestion error: {e}"
            status.storage_mode = "failed"
            status.durable = False

ingestion_manager = IngestionManager()
