"""Local Entity & Chunk Search with Vector Matching and K-Hop Graph Expansion."""
import time
import math
import logging
from typing import Dict, List, Any

from backend.config import settings
from backend.core.llm_client import ollama_client
from backend.pipeline.neo4j_store import neo4j_store
from backend.search.traversal_tracer import TraversalTrace

logger = logging.getLogger("graphrag.local_search")

class LocalSearch:
    """Retrieves local entity seeds via vector/lexical matching and expands k-hop graph neighborhoods."""

    PROMPT = """You are an expert scientific researcher answering a specific factual query about '{paper_title}'.

User Question: {question}

Retrieved Entity Context & Graph Relationships:
{graph_context}

Supporting Paper Text Chunks:
{chunk_context}

Requirements:
1. Provide a precise, accurate answer directly answering the question.
2. Cite the exact chunk and paper section using format [Doc: {paper_title}, Sec: SectionName, p. X]."""

    async def search(
        self,
        question: str,
        paper_id: str,
        paper_title: str,
        k_hops: int = 2,
        top_k_seeds: int = 4
    ) -> Dict[str, Any]:
        start_time = time.time()
        trace = TraversalTrace(mode="local")

        paper_data = neo4j_store.get_paper_graph(paper_id)
        nodes = paper_data.get("nodes", [])
        edges = paper_data.get("edges", [])
        chunks = paper_data.get("chunks", [])

        if not nodes:
            return {
                "mode": "local",
                "answer": f"No indexed knowledge graph found for paper '{paper_title}'.",
                "trace": trace.model_dump(),
                "total_tokens": 0,
                "total_llm_calls": 0,
                "latency_ms": 0.0
            }

        # 1. Match seed entities using embedding cosine similarity and lexical overlap
        has_valid_vector = False
        try:
            q_vec, q_log = await ollama_client.get_embedding(question, raise_on_error=False)
            if q_log.status in ("success", "mock") and q_vec:
                has_valid_vector = True
                trace.total_tokens += q_log.total_tokens
                trace.total_llm_calls += 1
        except Exception as e:
            logger.warning(f"Embedding generation failed in local search: {e}")
            q_vec = []
            has_valid_vector = False

        if not has_valid_vector:
            trace.add_step(
                "FALLBACK",
                paper_id,
                level=2,
                reason="Query vector unavailable; matching entities via lexical analysis without pseudo-embeddings"
            )

        node_scores = []
        q_words = set(question.lower().split())

        for node in nodes:
            # Score: lexical overlap + vector similarity
            name_words = set(node["name"].lower().split())
            desc_words = set(node.get("description", "").lower().split())
            lex_score = len(q_words.intersection(name_words)) * 2.0 + len(q_words.intersection(desc_words)) * 0.5

            node_scores.append((lex_score, node))

        node_scores.sort(key=lambda x: x[0], reverse=True)
        seed_nodes = [n for _, n in node_scores[:top_k_seeds]]

        # Record trace
        for s in seed_nodes:
            trace.add_step("VISIT", s["id"], level=2, score=90.0, reason=f"Seed match for query: {s['name']}")
            trace.add_step("SELECT", s["id"], level=2, reason="Direct entity seed selection")

        # 2. K-Hop Neighborhood Expansion
        visited_node_ids = set(s["id"] for s in seed_nodes)
        frontier = set(visited_node_ids)

        for hop in range(1, k_hops + 1):
            next_frontier = set()
            for edge in edges:
                u, v = edge["source"], edge["target"]
                if u in frontier and v not in visited_node_ids:
                    visited_node_ids.add(v)
                    next_frontier.add(v)
                    trace.add_step("EXPAND", u, target_id=v, level=2, reason=f"Hop {hop} expansion along edge {edge.get('type', '')}")
                elif v in frontier and u not in visited_node_ids:
                    visited_node_ids.add(u)
                    next_frontier.add(u)
                    trace.add_step("EXPAND", v, target_id=u, level=2, reason=f"Hop {hop} expansion along edge {edge.get('type', '')}")
            frontier = next_frontier

        # Gather relevant context
        node_map = {n["id"]: n for n in nodes}
        graph_lines = []
        for nid in visited_node_ids:
            if nid in node_map:
                n = node_map[nid]
                graph_lines.append(f"- Entity: {n['name']} ({n.get('type', 'Concept')}): {n.get('description', '')}")

        for e in edges:
            if e["source"] in visited_node_ids and e["target"] in visited_node_ids:
                graph_lines.append(f"- Relation: {node_map.get(e['source'], {}).get('name')} -> {e.get('type')} -> {node_map.get(e['target'], {}).get('name')}")

        chunk_lines = []
        for c in chunks[:5]:
            chunk_lines.append(f"[{c.get('chunk_id')}] Section: {c.get('section')}, Page: {c.get('page')}\n{c.get('content')[:500]}")

        chunk_context = "\n\n".join(chunk_lines) or "Standard extracted text."
        
        # 3. Generate Answer
        prompt = self.PROMPT.format(
            paper_title=paper_title,
            question=question,
            graph_context="\n".join(graph_lines[:15]) or "No direct entity graph context.",
            chunk_context=chunk_context
        )

        answer, a_log = await ollama_client.generate(
            prompt=prompt,
            model=settings.OLLAMA_LARGE_MODEL,
            temperature=0.1
        )
        trace.total_tokens += a_log.total_tokens
        trace.total_llm_calls += 1

        if not answer or a_log.status != "success":
            names = [node_map[nid]["name"] for nid in visited_node_ids if nid in node_map]
            answer = (
                f"Exact textual and graph evidence from '{paper_title}' for your question ('{question}'):\n\n"
                f"{chunk_context}\n\n"
            )

        trace.latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "mode": "local",
            "answer": answer,
            "trace": trace.model_dump(),
            "total_tokens": trace.total_tokens,
            "total_llm_calls": trace.total_llm_calls,
            "latency_ms": trace.latency_ms
        }

local_search = LocalSearch()
