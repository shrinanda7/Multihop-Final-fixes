# Knowledge-Graph RAG for Research Papers: Architecture & Execution Plan

## 1. Overview & Research Contributions
This system is an end-to-end Knowledge-Graph RAG pipeline designed for research paper analysis and publication-grade evaluation. It implements our custom pipeline (independent of Microsoft GraphRAG) with three novel algorithmic contributions configured behind ablation toggles:

1. **Tarjan-Based Graph Compression (Iterative Implementation)**:
   - Identifies Strongly Connected Components (SCCs), bridges, and articulation points using iterative Tarjan algorithms (tested against NetworkX).
   - Collapses cyclic components into semantic supernodes while preserving complete node/edge membership mappings.
   - Preserves critical bridges and articulation points as unmerged "connector" elements.
   - Monitors and reports SCC size distribution to detect and mitigate giant-component collapse anomalies.
   - Dual-graph representation: stores both original graph and compressed quotient graph.

2. **Bridge-Aware Pruning**:
   - During hierarchical dynamic community traversal, communities rated as irrelevant that are connected via topological bridges or articulation points to relevant communities are retained as low-priority exploration candidates rather than dropped blindly.

3. **Budget-Aware Traversal & Topological Ordering**:
   - Best-first community expansion prioritized by relevance rating within a strict token/report budget constraint.
   - Evidence aggregation ordered by the condensation DAG topological ordering.

---

## 2. Technology Stack & Directory Structure
- **Backend**: Python 3.10+, FastAPI, Uvicorn, SSE (Server-Sent Events) streaming.
- **LLM / Embeddings**: Ollama (dual-model setup: small model for rapid community relevance ratings; larger model for information extraction, community summaries, map-reduce answering, and evaluation judge).
- **Graph Store**: Neo4j (labeled property graph, per-paper namespaces, chunk node references, dual original/compressed layers).
- **Frontend**: React 19 + Vite + TypeScript, Tailwind CSS, Cytoscape.js for interactive knowledge graph visualization with live traversal playback.
- **Testing & Verification**: Pytest with test matrices for graph algorithms, NetworkX comparisons, schema validations, and mock fallback execution.

```text
├── backend/
│   ├── config.py                 # Central configuration for all endpoints, models, ablation flags
│   ├── main.py                   # FastAPI server & route registration
│   ├── core/
│   │   ├── tarjan.py             # Custom iterative Tarjan SCC, bridge, articulation point algorithms
│   │   ├── compression.py        # Graph compression, supernode mapping, giant SCC safeguards
│   │   ├── clustering.py         # Hierarchical Leiden community detection & hierarchy builder
│   │   └── llm_client.py         # Ollama client with token counting, latency logging, retry logic
│   ├── pipeline/
│   │   ├── pdf_extractor.py      # PDF text & structure extraction (sections, page numbers)
│   │   ├── chunker.py            # Section-aware chunking
│   │   ├── extractor.py          # Entity & directed relationship extraction with strict JSON schema
│   │   ├── entity_merging.py     # Canonical entity resolution & synonym reconciliation
│   │   ├── neo4j_store.py        # Neo4j graph driver, schema constraints, paper namespaces
│   │   ├── report_generator.py   # Hierarchical bottom-up community summarization
│   │   └── ingestion_manager.py  # Background ingestion job runner with SSE status events
│   ├── search/
│   │   ├── router.py             # Query classifier (local entity-level vs. global community-level)
│   │   ├── local_search.py       # Entity vector embedding + k-hop expansion + chunk retrieval
│   │   ├── static_search.py      # Fixed Level-1 community map-reduce baseline
│   │   ├── dynamic_search.py     # Reference dynamic hierarchical top-down traversal
│   │   ├── ours_search.py        # Dynamic + Bridge-Aware Pruning + Budget-Aware Expansion
│   │   └── trace.py              # Structured execution trace & telemetry emitter
│   ├── evaluation/
│   │   ├── benchmark_runner.py   # 50-question comparative benchmark over 10 papers
│   │   ├── judge.py              # LLM judge (randomized order, comprehensiveness, diversity, empowerment)
│   │   └── metrics_export.py     # Output CSVs, statistical tables, and publication plots
│   └── tests/
│       ├── test_tarjan.py        # Tarjan vs NetworkX equivalence unit tests
│       ├── test_extraction.py    # Schema adherence & JSON parser resilience tests
│       ├── test_clustering.py    # Leiden community hierarchy tests
│       └── test_search.py        # Search mode traversal & trace verification
├── src/                          # React + Vite Frontend
│   ├── components/
│   │   ├── UploadPanel.tsx       # Drag-and-drop PDF upload & background progress monitor
│   │   ├── ChatView.tsx          # Multi-paper selection, mode toggle, streamed responses, citations
│   │   ├── GraphView.tsx         # Cytoscape.js interactive visualization (original vs compressed)
│   │   ├── TraversalControls.tsx # Live traversal animation, state color coding, step replay slider
│   │   └── TelemetryCard.tsx     # Tokens, LLM calls, latency, reports used breakdown
│   ├── services/
│   │   └── api.ts                # FastAPI SSE & REST client
│   └── types/                    # Shared TypeScript interfaces
├── data/
│   └── papers/                   # Storage for 10 research papers
├── PLAN.md                       # Master execution plan (this document)
└── PROGRESS.md                   # Stage tracking and checkpoint log
```

---

## 3. Stages of Implementation

### Stage 0: Configuration & Architecture Confirmation
- Collect system configurations (Ollama URL, model choices, Neo4j credentials, papers directory).
- Generate `PLAN.md` and initialize `PROGRESS.md`.
- Validate environment prerequisites.

### Stage 1: Scaffolding, Core Config & Connections
- Central configuration in `backend/config.py` with environment variable loading and ablation flags:
  - `ENABLE_TARJAN_COMPRESSION: bool = True`
  - `ENABLE_BRIDGE_AWARE_PRUNING: bool = True`
  - `ENABLE_BUDGET_AWARE_TRAVERSAL: bool = True`
- Establish Ollama client wrapper with token accounting, latency metrics, and connection health check.
- Establish Neo4j client wrapper with connection pooling, index setup, and paper namespace isolation.
- Graceful degradation: test and handle unreachable services with informative diagnostics.

### Stage 2: Document Ingestion, Chunking & Entity Extraction
- PDF parsing maintaining section headings, hierarchy, and page numbering.
- Section-aware chunking preserving semantic boundaries and chunk IDs.
- Entity & directed relation extraction using strict Pydantic schemas and JSON repair/retry loops.
- Entity deduplication and canonical resolution.
- Ingestion into Neo4j with bidirectional pointers between entities, relations, and source text chunks.
- Background task executor with fine-grained status tracking for UI polling/streaming.

### Stage 3: Custom Iterative Tarjan Graph Compression
- Custom iterative Tarjan's Strongly Connected Components (SCC) algorithm (non-recursive to avoid stack limits on deep graphs).
- Custom iterative Tarjan bridge and articulation point detection.
- Comprehensive verification suite in `pytest` asserting exact parity against `networkx.strongly_connected_components`, `networkx.bridges`, and `networkx.articulation_points`.
- SCC size distribution profiling: automated check for "giant component" anomalies with configurable mitigation heuristics (threshold cutoffs, weak edge pruning).
- Quotient graph constructor: supernodes with complete member sets and preserve bridges/articulation nodes as protected connector elements.
- Dual storage of original and compressed graphs in Neo4j.

### Stage 4: Hierarchical Leiden Community Detection & Reports
- Hierarchical community detection (Leiden / Louvain algorithm) on the compressed graph representation.
- Multi-level community hierarchy indexing (Level 0, 1, ..., Root).
- Bottom-up LLM community report generation using the larger model (summary, key findings, member entities, confidence score).
- Vector embedding of community reports and source chunks stored for fast hybrid retrieval.
- Multi-tier caching layer for summaries, ratings, and intermediate graph computations.

### Stage 5: Search Engines & Structured Telemetry
- Unified search trace schema:
  `{ community_id, depth, rating, decision ("visited" | "expanded" | "pruned" | "bridge_kept" | "selected"), reasoning }`
- **Static Search**: Map-reduce over Level-1 community reports with the larger model.
- **Dynamic Search**: Top-down hierarchy traversal; small LLM scores relevance [0-10]; prune unpromising subtrees; expand promising children; final map-reduce over selected reports.
- **Ours (Novel)**: Dynamic search + Bridge-Aware Pruning (preserve low-scoring communities if topologically bridging to high-scoring components) + Budget-Aware Best-First Traversal (topological sorting + token budget cutoffs).
- **Local Search**: Vector similarity against entities/chunks + k-hop sub-graph expansion.
- **Query Router**: Automatic classification between local factoid vs. global thematic queries.
- Verifiable citation generation directly linking report sections and source chunk quotes.

### Stage 6: FastAPI Backend & Streaming API
- REST endpoints:
  - `POST /api/papers/upload` & `GET /api/papers` & `GET /api/jobs/{job_id}`
  - `POST /api/ask` (supports static, dynamic, ours, local, auto)
  - `GET /api/query/stream/{query_id}`: Server-Sent Events (SSE) emitting token chunks and live traversal trace steps.
  - `GET /api/graph/{paper_id}?mode=original|compressed`: Cytoscape-formatted node and edge payloads with connector annotations.
  - `GET /api/health`: Ollama and Neo4j status diagnostics.

### Stage 7: Cytoscape.js Frontend & Interactive UI
- Modern, high-contrast, responsive React interface.
- **Upload & Paper Explorer**: Drag-and-drop PDF upload, progress bars, paper management, preloaded papers indicator.
- **Query & Streamed Chat**: Mode selector with ablation toggle inspection, real-time Markdown answers, interactive citation drawer.
- **Cytoscape Knowledge Graph Viewer**:
  - Seamless toggle between Original Graph and Compressed Graph.
  - Supernodes expandable to view internal constituent entities.
  - Distinct visual styling for connector nodes (articulation points) and bridge edges.
  - Live traversal playback: nodes animated and colored by status (`visited`, `expanded`, `pruned`, `bridge_kept`, `selected`).
  - Interactive replay scrub slider to step through search decisions sequentially.
  - Hover inspect cards for community reports and LLM relevance rating justifications.
- **Telemetry Sidecard**: Live breakdown of prompt tokens, completion tokens, latency, and LLM call counts.

### Stage 8: Scientific Evaluation & Ablation Suite
- 50 curated research questions across 10 preloaded papers.
- Automated runner executing:
  1. Static Baseline
  2. Reference Dynamic
  3. Ours (Full System)
  4. Ablation A: Ours without Bridge-Aware Pruning
  5. Ablation B: Ours without Tarjan Compression
  6. Ablation C: Ours without Budget-Aware Traversal
- LLM-as-a-Judge protocol with randomized pairwise / multi-candidate presentation (neutralizing position bias).
- Evaluation dimensions: Comprehensiveness, Diversity, Empowerment, Direct Citation Quality.
- Resource profiling: Compression ratio, token consumption, execution latency, LLM call counts.
- Formatted output tables, CSV logs, and comparative charts.
