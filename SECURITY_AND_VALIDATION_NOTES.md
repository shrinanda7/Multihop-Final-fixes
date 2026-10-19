# Security and validation notes

## Changes in this patch

- Removed the included `.env` from this deliverable. Do not commit or share real environment files.
- Removed hard-coded Neo4j credential fallbacks from `src/neo4jService.ts`, `backend/config.py`, and `run_qa_trainer.py`. Configure `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and optionally `NEO4J_DATABASE` through your local environment.
- Restricted Python PDF uploads to PDF-suffixed files with a PDF signature and a 50 MiB maximum, and replaced user-controlled destination paths with server-generated filenames. Preloaded IDs are validated before path/glob use.
- Added Express upload count/field limits and PDF signature validation.
- Disabled the raw Cypher console unless `NEO4J_QUERY_API_ENABLED=true`; when enabled, the route rejects write/admin operations and multi-statement input. Enable only in a trusted local environment; this endpoint does not implement user authentication.
- Replaced wildcard credentialed CORS with an environment-configurable origin allowlist (`CORS_ALLOW_ORIGINS`).
- Removed fabricated default LLM-judge scores and synthetic graph-preservation/compression values from new benchmark runs. Missing measurements remain `null`. Historical benchmark artifacts may contain old fabricated values and must not be cited as validated results.
- Added missing Python runtime/test dependencies to `requirements.txt`.

## Required actions outside this repository

1. Rotate any API keys and database credentials that were present in the uploaded `.env`. Removing the file from this patch does not revoke the credentials.
2. Review the current Neo4j instance's audit/activity logs if those credentials were active.
3. Keep `.env` local; populate `.env.example` with variable names and non-secret examples only.
4. Do not expose the Cypher console to a public network. The current endpoint is intended only for trusted debugging.

## Validation status

- Python compile check: passed.
- Targeted Tarjan and quotient-traversal tests: 14 passed.
- Targeted graph and multi-paper regression tests: 19 passed.
- Optional dependency fallbacks now allow offline development without Neo4j installed and use NetworkX Louvain clustering when `python-louvain` is unavailable. For a real deployment, install all declared dependencies from `requirements.txt` and configure Neo4j.
- Full Python test suite: not yet confirmed passing in this environment.
- Frontend type-check/build: not verified because `node_modules` is absent in the execution environment. Run `npm ci`, then `npm run lint` and `npm run build`.
- No live Neo4j, LLM-provider, or end-to-end browser tests were run.

## Multi-paper retrieval implementation

`POST /api/query/multi-paper` accepts 1–12 indexed paper IDs. The backend namespaces node IDs by paper, creates cross-paper links only for exact normalized entity-name matches (excluding generic labels), runs Tarjan SCC quotient compression and query-guided directed path selection, retrieves original source chunks, and prompts the configured LLM to answer with evidence IDs. The API reports missing/unindexed papers and missing LLM responses explicitly. This endpoint has mocked regression tests; live Neo4j/LLM end-to-end validation remains required.
