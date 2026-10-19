# Tarjan-RAG evaluation runbook (Windows / PowerShell)

## What is implemented

- Existing single-paper modes remain available through the existing app/API.
- Multi-paper `ours` queries in the React UI now proxy to the Python research backend at `http://127.0.0.1:8001` by default.
- The Python `POST /api/query/multi-paper` endpoint merges namespaced paper graphs, creates cross-paper links only for exact normalized non-generic entity-name matches, performs Tarjan SCC quotient compression and query-guided directed path selection, retrieves original chunks by paper and chunk ID, and requests a grounded synthesis with `[E1]` evidence citations.
- The multi-paper route accepts 2–12 paper IDs through the UI proxy; the Python endpoint itself also supports one ID for debugging.

## 1. Environment setup

Use a clean Python virtual environment. In PowerShell from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally. Configure Neo4j credentials and the Ollama URL/model names. Never commit or share `.env`; rotate credentials that appeared in any earlier archive.

Install the frontend dependencies with Node.js compatible with the lockfile:

```powershell
npm ci
```

## 2. Start required services

Start Neo4j and ensure the credentials in `.env` can connect. For LLM generation, start Ollama and pull the configured models, for example:

```powershell
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

In a separate terminal, start the Python backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

In another terminal, start the frontend/Express app:

```powershell
npm run dev
```

If Python is running elsewhere, set `PYTHON_RAG_API_URL` before starting the Express server.

## 3. Ingest papers into the Python backend

The React/Express paper catalog and Python/Neo4j research store are separate data paths. For multi-paper Tarjan-RAG, **each selected paper must be ingested through the Python API**, not just uploaded to the React app.

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8001/api/upload" `
  -F "file=@data/papers/1609.02907_GCN.pdf" `
  -F "title=Semi-Supervised Classification with Graph Convolutional Networks"
```

Poll the returned job ID until the state is `COMPLETED`, then repeat for the second paper. If the job reports `storage_mode=in_memory`, the graph is not durable and disappears when the backend restarts; use a working Neo4j connection for research evaluation.

## 4. Run a cross-paper query

You can use the UI by selecting at least two papers and choosing `ours`, or call the API directly:

```powershell
$body = @{
  paper_ids = @("1609.02907_GCN", "1706.02216_GraphSAGE")
  paper_titles = @{
    "1609.02907_GCN" = "Semi-Supervised Classification with Graph Convolutional Networks"
    "1706.02216_GraphSAGE" = "Inductive Representation Learning on Large Graphs"
  }
  question = "How do the papers use neighborhood aggregation, and what evidence supports the comparison?"
  max_evidence_chunks = 10
  max_context_chars = 24000
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/query/multi-paper" -Method Post -ContentType "application/json" -Body $body
```

Inspect `status`, `missing_paper_ids`, `cross_paper_links`, `selected_node_ids`, `evidence`, and `llm_status`. The answer must cite the returned evidence IDs (`[E1]`, etc.); manually verify each citation against the original PDF page/chunk.

## 5. Validation commands

```powershell
python -m compileall -q backend run_qa_trainer.py scripts/preload_papers.py
python -m pytest -q
npm run lint
npm run build
```

The multi-paper regression tests use mocked stores and an LLM. Passing them validates orchestration/provenance logic but does **not** substitute for live Neo4j/Ollama integration. Do not submit benchmark claims until all commands pass and a fresh, reproducible benchmark has been run with the exact model, prompts, dataset, and configuration recorded.
