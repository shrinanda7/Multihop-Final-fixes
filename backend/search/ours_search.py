"""OUR Search Method: Tarjan Quotient Graph with Grounded Chunk Retrieval and Budget-Aware Traversal."""
import os
import time
import re
import logging
from typing import Dict, List, Set, Any, Optional
from .quotient_traversal import select_quotient_subgraph
from backend.pipeline.neo4j_store import neo4j_store

from backend.config import settings
from backend.core.llm_client import ollama_client
from backend.core.clustering import HierarchicalCommunity
from backend.search.traversal_tracer import TraversalTrace

logger = logging.getLogger("graphrag.ours_search")

class OursSearch:
    """Novel method: Tarjan-compressed traversal with bridge-aware topological preservation and adaptive call budgeting."""

    RATING_PROMPT = """Rate the relevance of this scientific concept cluster to the user query (0 to 100).
User Question: {question}

Cluster: {title} (Level {level})
Protected Connector Status: {connector_status}
Member Entities: {members}
Summary:
{summary}

Return format:
Rating: <number>
Reason: <one sentence>"""

    REDUCE_PROMPT = """You are a senior AI research scientist synthesizing findings from our bridge-aware knowledge graph traversal of '{paper_title}'.

User Question: {question}

Grounded Evidence and Topology Findings:
{findings}

Requirements:
1. Provide a comprehensive, mathematically sound, and empirical answer.
2. Ground all claims with specific citations: [Source: Doc={paper_title}; Chunk=chunk_id; Section=SectionName; Page=page].
3. Detail how topological bridges and core components interact to support the paper's claims."""

    async def search(
        self,
        question: str,
        paper_id: str,
        paper_title: str,
        communities: Dict[str, HierarchicalCommunity],
        compressed_graph: Dict[str, Any],
        relevance_threshold: float = settings.RELEVANCE_THRESHOLD,
        llm_budget: int = settings.TOTAL_LLM_BUDGET,
    ) -> Dict[str, Any]:
        start_time = time.time()
        trace = TraversalTrace(mode="ours")

        def finish(
            answer: str,
            selection: Dict[str, Any],
            selected_chunks: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            trace.latency_ms = round((time.time() - start_time) * 1000.0, 2)
            return {
                "mode": "ours",
                "answer": answer,
                "trace": trace.model_dump(),
                "selected_reports": selected_reports,
                "selected_node_ids": selection.get("selected_node_ids", []),
                "selected_chunk_ids": [
                    str(c.get("chunk_id", "")) for c in selected_chunks
                ],
                "evidence_chunk_count": len(selected_chunks),
                "evidence_token_estimate": sum(
                    self._estimate_tokens(self._format_chunk(c, paper_title))
                    for c in selected_chunks
                ),
                "total_tokens": trace.total_tokens,
                "total_llm_calls": trace.total_llm_calls,
                "latency_ms": trace.latency_ms,
                "bridge_kept_count": len(
                    selection.get("connector_component_ids", [])
                ),
            }

        # 1. Select a connected, query-relevant subgraph.
        selection = select_quotient_subgraph(
            question=question,
            compressed_graph=compressed_graph,
            bridge_aware=settings.ENABLE_BRIDGE_AWARE_PRUNING,
            budget_aware=settings.ENABLE_BUDGET_AWARE_TRAVERSAL,
        )

        selected_ids = {
            str(node_id)
            for node_id in selection.get("selected_node_ids", [])
        }
        connector_ids = {
            str(node_id)
            for node_id in selection.get("connector_component_ids", [])
        }

        graph_nodes = compressed_graph.get("nodes", [])
        node_by_id = {str(n.get("id")): n for n in graph_nodes}

        # Preserve the existing selected_reports response field.
        selected_reports = [
            comm.id
            for comm in communities.values()
            if selected_ids.intersection(
                str(node_id) for node_id in comm.member_node_ids
            )
        ]

        for seed_id in selection.get("seed_node_ids", []):
            trace.add_step(
                "SEED", str(seed_id), level=0,
                reason="Query-relevant quotient-graph seed selected.",
            )

        for node_id in selection.get("selected_node_ids", []):
            trace.add_step(
                "SELECT", str(node_id), level=0,
                score=selection.get("scores", {}).get(node_id, 0.0) or 0.0,
                reason="Selected by query-guided quotient-graph traversal.",
            )

        for connector_id in connector_ids:
            trace.add_step(
                "BRIDGE_PRESERVE", str(connector_id), level=0,
                reason="Retained as a traversal connector.",
            )

        # Record selected quotient nodes as visited in the traversal trace.
        # This is instrumentation only; it does not change graph selection.
        seed_ids = {
            str(node_id)
            for node_id in selection.get("seed_node_ids", [])
        }
        selected_scores = selection.get("scores", {})

        for node_id in sorted(selected_ids):
            trace.add_step(
                "VISIT",
                node_id,
                level=1,
                score=selected_scores.get(node_id, 0),
                reason=(
                    "Visited selected seed node."
                    if node_id in seed_ids
                    else "Visited selected connector or path node."
                ),
            )

        # 2. Collect chunk provenance from selected quotient nodes and edges.
        chunk_ids: List[str] = []
        seen_chunk_ids: Set[str] = set()

        def add_chunk_ids(values):
            if not isinstance(values, (list, tuple, set)):
                return
            for value in values:
                if value is None:
                    continue
                chunk_id = str(value)
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    chunk_ids.append(chunk_id)

        for node_id in selected_ids:
            node = node_by_id.get(str(node_id), {})
            add_chunk_ids(node.get("source_chunk_ids", []))

            # Some compressed nodes retain their original member records.
            for member in node.get("member_nodes", []):
                if isinstance(member, dict):
                    add_chunk_ids(member.get("source_chunk_ids", []))

        # Provenance fallback: map member_node_ids through entity_chunk_links
        try:
            raw_paper_graph = neo4j_store.get_paper_graph(paper_id, compressed=False)
            entity_chunk_links = raw_paper_graph.get("entity_chunk_links", [])
            if entity_chunk_links:
                selected_member_ids: Set[str] = set()
                for node_id in selected_ids:
                    node = node_by_id.get(str(node_id), {})
                    selected_member_ids.add(str(node_id))
                    for mid in node.get("member_node_ids", []):
                        selected_member_ids.add(str(mid))
                    for member in node.get("member_nodes", []):
                        if isinstance(member, dict) and "id" in member:
                            selected_member_ids.add(str(member["id"]))
                        elif isinstance(member, str):
                            selected_member_ids.add(str(member))

                for link in entity_chunk_links:
                    if str(link.get("entity_id")) in selected_member_ids:
                        cid = link.get("chunk_id")
                        if cid is not None:
                            add_chunk_ids([cid])
        except Exception:
            logger.warning("Failed to map entity_chunk_links fallback for paper %s", paper_id)

        selected_edges = selection.get("selected_edges", [])
        if not selected_edges:
            selected_edges = [
                edge for edge in compressed_graph.get("edges", [])
                if str(edge.get("source")) in selected_ids
                and str(edge.get("target")) in selected_ids
            ]

        for edge in selected_edges:
            add_chunk_ids(edge.get("source_chunk_ids", []))

        if not selected_ids:
            trace.add_step(
                "CHUNK_FALLBACK",
                "source_chunks",
                level=0,
                reason=(
                    "No quotient nodes matched the query; falling back to "
                    "lexical retrieval over this paper's original chunks."
                ),
            )

            try:
                all_chunks = neo4j_store.get_chunks_by_ids(paper_id, [])
            except Exception:
                logger.exception("Fallback source-chunk retrieval failed.")
                all_chunks = []

            query_terms = {
                term for term in re.findall(r"[a-zA-Z0-9]+", question.lower())
                if len(term) > 2
            }

            ranked_chunks = []
            for chunk in all_chunks:
                content = str(chunk.get("content", "")).strip()
                if not content:
                    continue

                content_terms = set(
                    re.findall(r"[a-zA-Z0-9]+", content.lower())
                )
                overlap = len(query_terms & content_terms)
                if overlap:
                    ranked_chunks.append((overlap, chunk))

            ranked_chunks.sort(
                key=lambda item: (
                    -item[0],
                    str(item[1].get("chunk_id", "")),
                )
            )

            fallback_chunks = [chunk for _, chunk in ranked_chunks[:6]]

            if not fallback_chunks:
                trace.add_step(
                    "NO_EVIDENCE",
                    "source_chunks",
                    level=0,
                    reason="No query-matching source chunks were found.",
                )
                return finish(
                    "No query-relevant source chunks were found. "
                    "Check whether the paper's original chunks are available.",
                    selection,
                    [],
                )

            # Populate the normal evidence pipeline with original chunks.
            chunk_ids = [
                str(chunk["chunk_id"])
                for chunk in fallback_chunks
                if chunk.get("chunk_id") is not None
            ]
            selected_ids = set()
            selection["selected_node_ids"] = []

        # 3. Recover original source text. Never invent missing citations.
        try:
            chunks = neo4j_store.get_chunks_by_ids(paper_id, chunk_ids)
        except Exception:
            logger.exception("Original evidence-chunk retrieval failed.")
            chunks = []

        chunks_by_id = {
            str(chunk.get("chunk_id")): chunk
            for chunk in chunks
            if chunk.get("chunk_id") is not None and chunk.get("content")
        }

        # Restore traversal order, not arbitrary database order.
        ordered_chunks = [
            chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id
        ]

        if not ordered_chunks:
            trace.add_step(
                "NO_EVIDENCE", "source_chunks", level=0,
                reason=(
                    "No original text chunks could be recovered for the "
                    "selected traversal. Answer generation was skipped."
                ),
            )
            return finish(
                "The graph traversal selected relevant nodes, but I couldn't "
                "recover their original source text. I won't generate a "
                "research answer or fabricate citations without that evidence.",
                selection,
                [],
            )

        # 4. Enforce a context-token budget before synthesis.
        context_budget = max(
            0,
            int(os.getenv(
                "OURS_CONTEXT_TOKEN_BUDGET",
                str(getattr(settings, "OURS_CONTEXT_TOKEN_BUDGET", 6000)),
            )),
        )
        strict_rules = (
            "\n\nSTRICT EVIDENCE RULES:\n"
            "- Answer using only the supplied evidence chunks.\n"
            "- Cite a chunk only using its exact [Source: ...] label.\n"
            "- Do not invent page numbers, sections, quotations, or citations.\n"
            "- If the evidence is insufficient, explicitly say what is missing.\n"
            "- Distinguish evidence from inference.\n"
        )
        fixed_prompt = self.REDUCE_PROMPT.format(
            question=question,
            paper_title=paper_title,
            findings="",
        )
        fixed_tokens = self._estimate_tokens(fixed_prompt + strict_rules)
        evidence_budget = max(0, context_budget - fixed_tokens)

        selected_chunks = []
        findings = []
        used_evidence_tokens = 0

        for chunk in ordered_chunks:
            formatted = self._format_chunk(chunk, paper_title)
            cost = self._estimate_tokens(formatted)

            if used_evidence_tokens + cost > evidence_budget:
                trace.add_step(
                    "BUDGET_PRUNE",
                    str(chunk.get("chunk_id", "unknown")),
                    level=0,
                    reason=(
                        "Evidence chunk omitted because it exceeded the "
                        "remaining estimated context-token budget."
                    ),
                )
                continue

            selected_chunks.append(chunk)
            findings.append(formatted)
            used_evidence_tokens += cost

        if not selected_chunks:
            trace.add_step(
                "NO_EVIDENCE", "context_budget", level=0,
                reason="No evidence chunk fits within the configured budget.",
            )
            return finish(
                "The retrieved evidence does not fit within the configured "
                "context budget. Increase OURS_CONTEXT_TOKEN_BUDGET to "
                "continue; no unsupported answer was generated.",
                selection,
                [],
            )

        if llm_budget < 1:
            return finish(
                "Evidence was retrieved, but the LLM call budget is zero. "
                "No answer-generation call was made.\n\n"
                + "\n\n".join(findings),
                selection,
                selected_chunks,
            )

        findings_text = "\n\n".join(findings)
        reduce_prompt = self.REDUCE_PROMPT.format(
            question=question,
            paper_title=paper_title,
            findings=findings_text,
        )

        # Do not make the model invent bibliographic metadata.
        reduce_prompt += strict_rules

        # The budget above is estimated; check the assembled prompt too.
        estimated_prompt_tokens = self._estimate_tokens(reduce_prompt)
        if estimated_prompt_tokens > context_budget:
            return finish(
                "The complete synthesis prompt exceeds the configured "
                "estimated context budget. No LLM call was made.",
                selection,
                selected_chunks,
            )

        try:
            answer, log = await ollama_client.generate(
                prompt=reduce_prompt,
                model=settings.OLLAMA_LARGE_MODEL,
                temperature=0.2,
            )

            trace.total_tokens += getattr(log, "total_tokens", 0) or 0
            trace.total_llm_calls += 1

            if not answer or getattr(log, "status", None) != "success":
                answer = (
                    "Answer generation failed. The following original "
                    "evidence was retrieved:\n\n" + findings_text
                )

        except Exception:
            logger.exception("Tarjan-RAG answer synthesis failed.")
            answer = (
                "Answer generation failed. The following original evidence "
                "was retrieved:\n\n" + findings_text
            )

        trace.add_step(
            "SYNTHESIZE", "evidence_chunks", level=0,
            reason=(
                f"Synthesized from {len(selected_chunks)} original chunks; "
                f"estimated prompt tokens={estimated_prompt_tokens}, "
                f"configured budget={context_budget}."
            ),
        )

        return finish(answer, selection, selected_chunks)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative approximation; not a substitute for model tokenization."""
        return (len(text) + 2) // 3

    @staticmethod
    def _format_chunk(chunk: Dict[str, Any], paper_title: str) -> str:
        chunk_id = str(chunk.get("chunk_id", "unknown"))
        section = str(chunk.get("section") or "unknown")
        page = chunk.get("page")
        page_text = str(page) if page is not None else "unknown"
        content = str(chunk.get("content", "")).strip()

        return (
            f"[Source: Doc={paper_title}; Chunk={chunk_id}; "
            f"Section={section}; Page={page_text}]\n{content}"
        )

    async def _rate_cluster(
        self,
        question: str,
        comm: HierarchicalCommunity,
        has_connector: bool,
        trace: TraversalTrace
    ) -> tuple[float, str]:
        """Rate cluster relevance with small LLM."""
        prompt = self.RATING_PROMPT.format(
            question=question,
            title=comm.title,
            level=comm.level,
            connector_status="YES (Contains Critical Graph Articulation Point)" if has_connector else "NO",
            members=", ".join(comm.member_node_ids[:5]),
            summary=comm.summary or comm.report[:400]
        )

        res, log = await ollama_client.generate(
            prompt=prompt,
            model=settings.OLLAMA_SMALL_MODEL,
            temperature=0.1
        )
        trace.total_tokens += log.total_tokens
        trace.total_llm_calls += 1

        score = 80.0
        reason = "Aligned with query entities."
        if res:
            m = re.search(r"Rating:\s*(\d+)", res, re.IGNORECASE)
            if m:
                score = float(m.group(1))
            r_m = re.search(r"Reason:\s*(.*)", res, re.IGNORECASE)
            if r_m:
                reason = r_m.group(1).strip()
        else:
            q_terms = set(question.lower().split())
            comm_terms = set((comm.title + " " + comm.summary).lower().split())
            overlap = len(q_terms.intersection(comm_terms))
            score = min(95.0, 45.0 + overlap * 20.0)
            reason = f"Keyword overlap: {overlap} terms."

        return score, reason

ours_search = OursSearch()
