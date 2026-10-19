# Tarjan-RAG: evaluation readiness checklist

This repository has targeted algorithm fixes, but it is **not yet certified end-to-end**. Use this checklist on the machine/environment that has access to the required packages, Neo4j and the configured LLM provider.

## 1. Configure safely

- Copy `.env.example` to `.env` locally and set valid credentials; never commit or share `.env`.
- Rotate credentials that were present in any previously shared archive.
- Keep `NEO4J_QUERY_API_ENABLED=false` except during trusted local debugging.
- Verify `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, and the selected LLM endpoint/model.

## 2. Install dependencies

Python (prefer a clean virtual environment):

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Frontend (Node version compatible with the Vite version in `package-lock.json`):

```bash
npm ci
```

## 3. Run automated checks

```bash
python -m compileall -q backend run_qa_trainer.py scripts/preload_papers.py
python -m pytest -q
npm run lint
npm run build
```

Focused graph and multi-paper regression tests: `python -m pytest -q backend/tests/test_multi_paper_search.py backend/tests/test_stage3_tarjan.py backend/tests/test_quotient_traversal.py` (19 passed in the available environment). The complete test suite is still being checked; frontend type-check/build and live integration with Neo4j/Ollama have not been verified here because package-registry access and live services are unavailable.

## 4. Run integration smoke tests

1. Start Neo4j and verify credentials/database connectivity.
2. Start the configured LLM provider and verify the selected model is available.
3. Start the Python API and TypeScript frontend using the documented project commands.
4. Ingest one known PDF; confirm graph nodes, relationships, and original chunks/provenance are persisted.
5. Ask a question with a known answer and confirm the returned answer cites only recovered source chunks.
6. Query a paper with no indexed graph and confirm the API returns `no_indexed_graph` rather than generating a fabricated answer.
7. Call `POST /api/query/multi-paper` with at least two ingested paper IDs. Confirm `cross_paper_links` only reflects exact normalized entity-name matches, selected evidence includes multiple `paper_id` values, and the answer cites `[E1]`-style evidence IDs.
8. Repeat with a graph containing a multi-node SCC, an articulation point, disconnected components, and parallel edges.

## 5. Before reporting research results

- Do not cite `data/evaluation/historical_unvalidated_benchmark.json`; it is retained only for provenance and contains unvalidated pre-fix metrics.
- Run fresh evaluations with the exact dataset, model/provider/model version, prompts, settings, random seed/order, and hardware recorded.
- Report attempted, successful and failed samples separately; report missing judge scores as missing.
- Compute graph preservation/path survival from the actual original and selected/compressed graphs, not from configuration flags.
- Verify that every ablation changes the intended algorithmic behavior. The bridge-aware and budget-aware quotient-selection toggles now have regression tests.
- Compare against baselines using the same questions, indexed papers, LLM/model settings and evaluation rubric. Include multiple runs or confidence intervals where feasible.
- Manually audit a sample of citations against original PDF pages/chunks. An LLM judge score alone is not evidence of citation correctness.

Do not describe the repository as fully evaluation-ready until the full test suite, frontend build, live integration smoke tests and fresh benchmark run have all passed.
