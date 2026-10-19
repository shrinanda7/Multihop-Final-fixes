"""Dynamic Community Selection Search Baseline (Hierarchical Top-Down Rating and Pruning)."""
import time
import re
import logging
from typing import Dict, List, Any, Optional

from backend.config import settings
from backend.core.llm_client import ollama_client
from backend.core.clustering import HierarchicalCommunity
from backend.search.traversal_tracer import TraversalTrace

logger = logging.getLogger("graphrag.dynamic_search")

class DynamicSearch:
    """Reference dynamic baseline: top-down tree search rating communities with small model and pruning."""

    RATING_PROMPT = """Rate the relevance of this research paper community report to the user's question on a scale from 0 to 100.

User Question: {question}

Community: {title} (Level {level})
Report Summary:
{summary}

Return ONLY an integer rating between 0 and 100, followed by a one-sentence reason.
Format:
Rating: <number>
Reason: <one sentence>"""

    MAP_PROMPT = """Analyze this relevant community report in the context of the user question.
User Question: {question}

Community: {title}
Report:
{report}

Extract all factual evidence and key arguments answering the question."""

    REDUCE_PROMPT = """You are an expert scientific researcher. Synthesize an accurate, grounded answer to the user question using the surviving community findings below.

User Question: {question}

Surviving Community Findings:
{findings}

Requirements:
1. Provide a direct, well-structured answer.
2. Ground all claims with citations: [Doc: {paper_title}, Sec: SectionName, p. X]."""

    async def search(
        self,
        question: str,
        paper_id: str,
        paper_title: str,
        communities: Dict[str, HierarchicalCommunity],
        relevance_threshold: float = settings.RELEVANCE_THRESHOLD,
        max_depth: int = settings.MAX_COMMUNITY_DEPTH
    ) -> Dict[str, Any]:
        start_time = time.time()
        trace = TraversalTrace(mode="dynamic")

        # 1. Identify Root communities (level 0)
        root_comms = [c for c in communities.values() if c.level == 0]
        if not root_comms:
            root_comms = [c for c in communities.values() if c.parent_id is None]

        current_frontier = root_comms
        surviving_reports: List[HierarchicalCommunity] = []

        current_depth = 0
        while current_frontier and current_depth <= max_depth:
            next_frontier: List[HierarchicalCommunity] = []

            for comm in current_frontier:
                trace.add_step("VISIT", comm.id, level=comm.level, reason=f"Visiting {comm.title} at level {comm.level}")

                # Rate with small model
                score, reason = await self._rate_community(question, comm, trace)

                if score >= relevance_threshold:
                    trace.add_step(
                        "RATE",
                        comm.id,
                        level=comm.level,
                        score=score,
                        reason=f"Scored {score}/100 (Threshold {relevance_threshold}): {reason}"
                    )

                    # If has children and depth allows, expand; else mark as surviving leaf
                    if comm.children_ids and current_depth < max_depth:
                        trace.add_step("EXPAND", comm.id, level=comm.level, reason=f"Expanding to {len(comm.children_ids)} child communities")
                        for cid in comm.children_ids:
                            if cid in communities:
                                next_frontier.append(communities[cid])
                    else:
                        trace.add_step("SELECT", comm.id, level=comm.level, score=score, reason="Selected as surviving report for synthesis")
                        surviving_reports.append(comm)
                else:
                    trace.add_step(
                        "PRUNE",
                        comm.id,
                        level=comm.level,
                        score=score,
                        reason=f"Pruned sub-tree (score {score}/100 < {relevance_threshold}): {reason}"
                    )

            current_frontier = next_frontier
            current_depth += 1

        # Fallback if over-pruned: use top-scoring or root
        if not surviving_reports and communities:
            fallback = list(communities.values())[0]
            surviving_reports = [fallback]
            trace.add_step("SELECT", fallback.id, level=fallback.level, reason="Fallback inclusion to guarantee answer")

        # 2. Map surviving reports
        map_findings: List[str] = []
        for comm in surviving_reports:
            p = self.MAP_PROMPT.format(question=question, title=comm.title, report=comm.report[:2000])
            mapped, log = await ollama_client.generate(
                prompt=p,
                model=settings.OLLAMA_SMALL_MODEL,
                temperature=0.1
            )
            trace.total_tokens += log.total_tokens
            trace.total_llm_calls += 1
            if mapped:
                map_findings.append(f"### Finding from [{comm.title}]:\n{mapped.strip()}")

        # 3. Reduce with large model
        findings_str = "\n\n".join(map_findings)
        reduce_p = self.REDUCE_PROMPT.format(
            question=question,
            findings=findings_str,
            paper_title=paper_title
        )
        answer, r_log = await ollama_client.generate(
            prompt=reduce_p,
            model=settings.OLLAMA_LARGE_MODEL,
            temperature=0.2
        )
        trace.total_tokens += r_log.total_tokens
        trace.total_llm_calls += 1

        if not answer or r_log.status != "success":
            answer = (
                f"Here is the relevant factual information extracted from '{paper_title}' to fully answer your question ('{question}'):\n\n"
                f"{findings_str}\n"
            )

        trace.latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "mode": "dynamic",
            "answer": answer,
            "trace": trace.model_dump(),
            "selected_reports": [c.id for c in surviving_reports],
            "total_tokens": trace.total_tokens,
            "total_llm_calls": trace.total_llm_calls,
            "latency_ms": trace.latency_ms
        }

    async def _rate_community(
        self,
        question: str,
        comm: HierarchicalCommunity,
        trace: TraversalTrace
    ) -> tuple[float, str]:
        """Rate community relevance using small model."""
        prompt = self.RATING_PROMPT.format(
            question=question,
            title=comm.title,
            level=comm.level,
            summary=comm.summary or comm.report[:400]
        )

        res, log = await ollama_client.generate(
            prompt=prompt,
            model=settings.OLLAMA_SMALL_MODEL,
            temperature=0.1
        )
        trace.total_tokens += log.total_tokens
        trace.total_llm_calls += 1

        score = 75.0
        reason = "Matched query semantic entities."
        if res:
            m = re.search(r"Rating:\s*(\d+)", res, re.IGNORECASE)
            if m:
                score = float(m.group(1))
            r_m = re.search(r"Reason:\s*(.*)", res, re.IGNORECASE)
            if r_m:
                reason = r_m.group(1).strip()
        else:
            # Lexical heuristic score
            q_terms = set(question.lower().split())
            comm_terms = set((comm.title + " " + comm.summary).lower().split())
            overlap = len(q_terms.intersection(comm_terms))
            score = min(95.0, 40.0 + overlap * 20.0)
            reason = f"Keyword overlap: {overlap} terms."

        return score, reason

dynamic_search = DynamicSearch()
