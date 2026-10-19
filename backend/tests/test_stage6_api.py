"""Stage 6 & 8 Verification Tests: FastAPI Endpoints, Streaming API, and Benchmark Suite."""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_health_and_papers_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        papers_res = await client.get("/api/papers")
        assert papers_res.status_code == 200
        papers = papers_res.json()
        assert len(papers) >= 10, "Should list at least the 10 preloaded landmark papers"

@pytest.mark.asyncio
async def test_graph_and_config_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Config GET & POST
        cfg_res = await client.get("/api/config")
        assert cfg_res.status_code == 200
        cfg = cfg_res.json()
        assert "ENABLE_TARJAN_COMPRESSION" in cfg

        up_res = await client.post("/api/config", json={"TOTAL_LLM_BUDGET": 14})
        assert up_res.status_code == 200
        assert up_res.json()["TOTAL_LLM_BUDGET"] == 14

        # Graph GET
        g_res = await client.get("/api/graph/1609.02907_GCN")
        assert g_res.status_code == 200
        data = g_res.json()
        assert "original_graph" in data
        assert "compressed_graph" in data
        assert "articulation_points" in data["compressed_graph"]
        assert "bridges" in data["compressed_graph"]

@pytest.mark.asyncio
async def test_query_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        q_res = await client.post("/api/query", json={
            "paper_id": "1609.02907_GCN",
            "paper_title": "Semi-Supervised Classification with GCN",
            "question": "What is the primary contribution of the paper?",
            "mode": "ours"
        })
        assert q_res.status_code == 200
        data = q_res.json()
        assert "answer" in data
        assert "trace" in data
        assert "compressed_graph" in data

@pytest.mark.asyncio
async def test_benchmark_questions_and_run():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Questions list
        q_res = await client.get("/api/benchmark/questions?limit=50")
        assert q_res.status_code == 200
        data = q_res.json()
        assert data["total"] == 50

        # Fast benchmark run on 1 question
        bench_res = await client.post("/api/benchmark/run?num_questions=1&randomize_order=false")
        assert bench_res.status_code == 200
        bench_data = bench_res.json()
        assert "summary" in bench_data
        assert "records" in bench_data
        assert len(bench_data["records"]) == 6 # 6 configurations evaluated

        # CSV Export
        csv_res = await client.get("/api/benchmark/export.csv")
        assert csv_res.status_code == 200
        assert "question_id" in csv_res.text
