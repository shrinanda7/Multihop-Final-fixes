"""Cross-paper graph retrieval with provenance-preserving multi-hop evidence.

Cross-paper links are added only for normalized exact entity-name matches; this
avoids inventing semantic relationships. All node and edge IDs are namespaced
by paper_id to prevent collisions between independently ingested papers.
"""
from __future__ import annotations

import re
import time
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from backend.config import settings
from backend.core.compression import compressor
from backend.core.llm_client import ollama_client
from backend.pipeline.neo4j_store import neo4j_store
from backend.search.quotient_traversal import select_quotient_subgraph

logger = logging.getLogger("graphrag.multi_paper")


def _norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _terms(value: str) -> Set[str]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "what", "how", "why", "paper", "across", "between", "compare", "explain", "does", "are", "was", "were", "into", "their", "using"}
    return {t for t in re.findall(r"[a-z0-9]+", value.lower()) if len(t) > 2 and t not in stop}


class MultiPaperSearch:
    """Retrieve connected, query-relevant evidence across several indexed papers."""

    async def search(
        self,
        question: str,
        paper_ids: List[str],
        paper_titles: Dict[str, str] | None = None,
        max_evidence_chunks: int = 10,
        max_context_chars: int = 24000,
    ) -> Dict[str, Any]:
        started = time.time()
        paper_titles = paper_titles or {}
        # Keep order stable and bound graph fan-out for predictable latency.
        paper_ids = list(dict.fromkeys(str(p).strip() for p in paper_ids if str(p).strip()))
        if not paper_ids:
            return self._error("paper_ids must contain at least one paper ID", started)
        if len(paper_ids) > 12:
            return self._error("At most 12 papers can be queried in one request", started)

        merged_nodes: List[Dict[str, Any]] = []
        merged_edges: List[Dict[str, Any]] = []
        chunks_by_paper: Dict[str, List[Dict[str, Any]]] = {}
        paper_meta: Dict[str, Dict[str, Any]] = {}
        name_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        missing: List[str] = []
        generic_entity_names = {"method", "model", "graph", "paper", "approach", "result", "dataset", "experiment", "framework", "network", "system", "layer", "module", "baseline", "performance", "architecture", "algorithm", "embedding"}

        for paper_id in paper_ids:
            graph_data = neo4j_store.get_paper_graph(paper_id, compressed=False) or {}
            nodes = graph_data.get("nodes") or []
            edges = graph_data.get("edges") or []
            if not nodes:
                missing.append(paper_id)
                continue
            title = paper_titles.get(paper_id) or paper_id.replace("_", " ")
            paper_meta[paper_id] = {"title": title, "storage_mode": graph_data.get("storage_mode", "unknown"), "durable": bool(graph_data.get("durable", False))}
            chunks_by_paper[paper_id] = neo4j_store.get_chunks_by_ids(paper_id, []) or []
            id_map: Dict[str, str] = {}
            for node in nodes:
                original_id = str(node.get("id", ""))
                if not original_id:
                    continue
                global_id = f"{paper_id}::{original_id}"
                id_map[original_id] = global_id
                copied = dict(node)
                copied["id"] = global_id
                copied["original_node_id"] = original_id
                copied["paper_id"] = paper_id
                copied["paper_title"] = title
                copied["source_chunk_ids"] = [str(x) for x in node.get("source_chunk_ids", []) if x is not None]
                merged_nodes.append(copied)
                canonical = _norm(node.get("name", ""))
                # Avoid extremely generic labels creating noisy cross-paper hubs.
                if (len(canonical) >= 4 and len(canonical.split()) <= 8
                        and canonical not in generic_entity_names):
                    name_index[canonical].append(copied)
            for edge in edges:
                source = id_map.get(str(edge.get("source", "")))
                target = id_map.get(str(edge.get("target", "")))
                if not source or not target or source == target:
                    continue
                copied_edge = dict(edge)
                copied_edge.update({"source": source, "target": target, "paper_id": paper_id})
                merged_edges.append(copied_edge)

        if not merged_nodes:
            return {
                "status": "no_indexed_graph", "mode": "multi_paper", "answer": "None of the requested papers has an indexed knowledge graph. Ingest the papers before querying.",
                "paper_ids": paper_ids, "missing_paper_ids": missing, "evidence": [], "selected_node_ids": [],
                "trace": {"steps": [], "total_llm_calls": 0, "total_tokens": 0}, "total_llm_calls": 0,
                "total_tokens": 0, "latency_ms": round((time.time()-started)*1000, 2),
            }

        # Exact-name cross-paper links only; no LLM-generated or guessed edges.
        cross_links = 0
        for canonical, matches in name_index.items():
            by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for node in matches:
                by_paper[node["paper_id"]].append(node)
            paper_keys = sorted(by_paper)
            if len(paper_keys) < 2:
                continue
            # Cap ambiguous homonyms to avoid a quadratic, high-degree hub.
            candidates = [nodes_for_paper[0] for pid in paper_keys for nodes_for_paper in [by_paper[pid]]]
            if len(candidates) > 8:
                candidates = candidates[:8]
            for i, left in enumerate(candidates):
                for right in candidates[i + 1:]:
                    if left["paper_id"] == right["paper_id"]:
                        continue
                    provenance = sorted(set(left.get("source_chunk_ids", [])) | set(right.get("source_chunk_ids", [])))
                    merged_edges.append({"source": left["id"], "target": right["id"], "type": "SAME_ENTITY_NAME", "weight": 0.5, "source_chunk_ids": provenance, "cross_paper": True})
                    merged_edges.append({"source": right["id"], "target": left["id"], "type": "SAME_ENTITY_NAME", "weight": 0.5, "source_chunk_ids": provenance, "cross_paper": True})
                    cross_links += 2

        compressed = compressor.compress_graph(merged_nodes, merged_edges)
        selection = select_quotient_subgraph(
            question=question,
            compressed_graph=compressed,
            max_seeds=5,
            max_selected_nodes=24,
            max_path_nodes=12,
            bridge_aware=settings.ENABLE_BRIDGE_AWARE_PRUNING,
            budget_aware=settings.ENABLE_BUDGET_AWARE_TRAVERSAL,
        )

        # Resolve graph-derived chunk references without losing paper namespaces.
        requested_chunks: Dict[str, Set[str]] = defaultdict(set)
        node_by_id = {str(n["id"]): n for n in compressed.get("nodes", [])}
        selected_node_ids = [str(x) for x in selection.get("selected_node_ids", [])]
        for selected_id in selected_node_ids:
            node = node_by_id.get(selected_id, {})
            members = node.get("member_nodes", [])
            if not members:
                members = [node]
            for member in members:
                pid = member.get("paper_id")
                if not pid:
                    # Singleton compressed nodes keep the paper_id on the node.
                    pid = node.get("paper_id")
                if not pid:
                    continue
                for cid in member.get("source_chunk_ids", []):
                    requested_chunks[str(pid)].add(str(cid))
            # Edge provenance belongs to the paper(s) attached to its source nodes;
            # node provenance remains the authoritative chunk map.

        # Candidate evidence: first graph-linked chunks, then lexical backfill.
        candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for pid in paper_ids:
            available = chunks_by_paper.get(pid, [])
            wanted = requested_chunks.get(pid, set())
            for chunk in available:
                cid = str(chunk.get("chunk_id", ""))
                if cid and cid in wanted and chunk.get("content"):
                    item = dict(chunk)
                    item.update({"paper_id": pid, "paper_title": paper_meta.get(pid, {}).get("title", pid)})
                    candidates[(pid, cid)] = item

        query_terms = _terms(question)
        # Score all available chunks for lexical backfill and relevance ordering.
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for pid in paper_ids:
            for raw in chunks_by_paper.get(pid, []):
                cid = str(raw.get("chunk_id", ""))
                content = str(raw.get("content", "")).strip()
                if not cid or not content:
                    continue
                item = candidates.get((pid, cid))
                if item is None:
                    item = dict(raw)
                    item.update({"paper_id": pid, "paper_title": paper_meta.get(pid, {}).get("title", pid)})
                text = " ".join([content, str(item.get("section", ""))])
                terms = _terms(text)
                overlap = len(query_terms & terms) / max(len(query_terms), 1)
                # Graph-linked chunks receive a small, explicit structural prior.
                structural_prior = 0.15 if (pid, cid) in candidates else 0.0
                scored.append((overlap + structural_prior, item))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("paper_id")), str(pair[1].get("chunk_id"))))
        selected_chunks: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        used_chars = 0
        # Guarantee representation from multiple papers when relevant chunks exist.
        relevant_by_paper: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
        for score, chunk in scored:
            if score > 0:
                relevant_by_paper[str(chunk["paper_id"])].append((score, chunk))
        for pid in paper_ids:
            if relevant_by_paper.get(pid):
                score, chunk = relevant_by_paper[pid][0]
                key = (pid, str(chunk["chunk_id"]))
                if key not in seen and len(selected_chunks) < max_evidence_chunks:
                    text = str(chunk["content"])
                    if used_chars + len(text) <= max_context_chars:
                        selected_chunks.append(chunk); seen.add(key); used_chars += len(text)
        for score, chunk in scored:
            if len(selected_chunks) >= max_evidence_chunks:
                break
            key = (str(chunk["paper_id"]), str(chunk["chunk_id"]))
            if key in seen or score <= 0:
                continue
            content = str(chunk["content"])
            if used_chars + len(content) > max_context_chars:
                continue
            selected_chunks.append(chunk); seen.add(key); used_chars += len(content)

        if not selected_chunks:
            return {
                "status": "no_evidence", "mode": "multi_paper", "answer": "The selected papers are indexed, but no source chunks matched the query. Try a more specific question or verify that chunk text was persisted.",
                "paper_ids": paper_ids, "missing_paper_ids": missing, "selected_node_ids": selected_node_ids,
                "cross_paper_links": cross_links, "evidence": [], "trace": {"steps": [], "total_llm_calls": 0, "total_tokens": 0},
                "total_llm_calls": 0, "total_tokens": 0, "latency_ms": round((time.time()-started)*1000, 2),
            }

        evidence_blocks = []
        for idx, chunk in enumerate(selected_chunks, 1):
            title = str(chunk.get("paper_title", chunk.get("paper_id", "unknown")))
            pid = str(chunk.get("paper_id", "unknown"))
            cid = str(chunk.get("chunk_id", "unknown"))
            page = chunk.get("page", "unknown")
            section = chunk.get("section", "unknown")
            evidence_blocks.append(f"[E{idx}] Paper: {title} (paper_id={pid}); Chunk={cid}; Section={section}; Page={page}\n{chunk['content']}")
        prompt = (
            "Answer the research question using only the evidence blocks below. "
            "Synthesize across papers when evidence supports it. Distinguish agreement, disagreement, and missing evidence. "
            "For every substantive claim, cite evidence IDs exactly like [E1] or [E2, E4]. Never invent a result, citation, paper detail, or causal relationship. "
            "If evidence is insufficient, state that explicitly. Include a short multi-hop reasoning summary showing the chain of concepts and papers, but do not reveal hidden chain-of-thought.\n\n"
            f"QUESTION:\n{question}\n\nEVIDENCE:\n" + "\n\n".join(evidence_blocks)
        )
        answer, log = await ollama_client.generate(
            prompt=prompt,
            system="You are a careful research assistant. Ground every claim in supplied source evidence, cite evidence IDs, and clearly label uncertainty.",
            temperature=0.1,
        )
        llm_failed = not bool(answer and answer.strip()) or log.status not in {"success", "mock"}
        if llm_failed:
            answer = "Answer generation failed because the configured LLM endpoint did not return a valid response. The retrieved evidence is included for inspection; no answer has been fabricated."
        trace_steps = []
        for seed in selection.get("seed_node_ids", []):
            trace_steps.append({"action": "SEED", "node_id": str(seed), "reason": "Query-term overlap seed"})
        for node_id in selected_node_ids:
            trace_steps.append({"action": "SELECT", "node_id": node_id, "reason": "Selected quotient node/path for multi-hop context"})
        for idx, chunk in enumerate(selected_chunks, 1):
            trace_steps.append({"action": "EVIDENCE", "node_id": str(chunk.get("chunk_id")), "paper_id": str(chunk.get("paper_id")), "reason": f"Evidence [{idx}] selected from source chunk"})
        return {
            "status": "llm_error" if llm_failed else "ok", "mode": "multi_paper", "question": question,
            "paper_ids": paper_ids, "paper_titles": {pid: meta["title"] for pid, meta in paper_meta.items()},
            "missing_paper_ids": missing, "cross_paper_links": cross_links,
            "original_graph": {"nodes": merged_nodes, "edges": merged_edges},
            "compressed_graph": {"nodes": compressed["nodes"], "edges": compressed["edges"], "diagnostics": compressed["diagnostics"], "compression_ratio": compressed["compression_ratio"], "articulation_points": compressed["articulation_points"], "bridges": compressed["bridges"]},
            "selected_node_ids": selected_node_ids,
            "selected_edges": selection.get("edges", []),
            "evidence": [{"evidence_id": f"E{i}", "paper_id": c["paper_id"], "paper_title": c["paper_title"], "chunk_id": c["chunk_id"], "section": c.get("section"), "page": c.get("page"), "content": c["content"]} for i, c in enumerate(selected_chunks, 1)],
            "answer": answer, "trace": {"mode": "multi_paper", "steps": trace_steps, "selected_node_ids": selected_node_ids, "total_llm_calls": 1 if log.status == "success" else 0, "total_tokens": log.total_tokens, "latency_ms": log.latency_ms},
            "total_llm_calls": 1 if log.status == "success" else 0, "total_tokens": log.total_tokens,
            "latency_ms": round((time.time()-started)*1000, 2), "llm_status": log.status,
        }

    @staticmethod
    def _error(message: str, started: float) -> Dict[str, Any]:
        return {"status": "invalid_request", "mode": "multi_paper", "answer": message, "evidence": [], "trace": {"steps": [], "total_llm_calls": 0, "total_tokens": 0}, "total_llm_calls": 0, "total_tokens": 0, "latency_ms": round((time.time()-started)*1000, 2)}


multi_paper_search = MultiPaperSearch()
