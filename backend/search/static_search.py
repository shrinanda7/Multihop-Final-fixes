"""Static Level-1 Community Map-Reduce Search Baseline."""
import time
import logging
from typing import Dict, List, Any

from backend.config import settings
from backend.core.llm_client import ollama_client
from backend.core.clustering import HierarchicalCommunity
from backend.search.traversal_tracer import TraversalTrace

logger = logging.getLogger("graphrag.static_search")

class StaticSearch:
    """Standard baseline: processes ALL reports at Level 1 via map-reduce."""

    MAP_PROMPT = """Analyze this community report in the context of the user question.
User Question: {question}

Community: {title}
Report:
{report}

Extract all relevant points and factual evidence answering the question. Include specific entities and concepts.
If not relevant, return 'NOT_RELEVANT'."""

    REDUCE_PROMPT = """You are an expert scientific researcher. Synthesize a comprehensive, authoritative answer to the user question using the extracted community findings below.

User Question: {question}

Extracted Findings from Communities:
{findings}

Requirements:
1. Provide a direct, well-structured answer.
2. Provide grounded citations in the format [Doc: {paper_title}, Sec: SectionName, p. X].
3. Detail the key architectural and theoretical trade-offs."""

    async def search(
        self,
        question: str,
        paper_id: str,
        paper_title: str,
        communities: Dict[str, HierarchicalCommunity],
        level: int = 1
    ) -> Dict[str, Any]:
        start_time = time.time()
        trace = TraversalTrace(mode="static")

        # Select all communities at specified level
        target_comms = [c for c in communities.values() if c.level == level]
        if not target_comms:
            # Fallback to level 0 or any available
            target_comms = list(communities.values())

        map_results: List[str] = []
        for comm in target_comms:
            trace.add_step("VISIT", comm.id, level=comm.level, reason=f"Visiting Level {level} community {comm.title}")
            trace.add_step("SELECT", comm.id, level=comm.level, reason="Static selection (all Level 1 used)")

            prompt = self.MAP_PROMPT.format(question=question, title=comm.title, report=comm.report[:2000])
            mapped, log = await ollama_client.generate(
                prompt=prompt,
                model=settings.OLLAMA_SMALL_MODEL,
                temperature=0.1
            )
            trace.total_tokens += log.total_tokens
            trace.total_llm_calls += 1

            if mapped and "NOT_RELEVANT" not in mapped:
                map_results.append(f"### Finding from {comm.title}:\n{mapped.strip()}")

        findings_text = "\n\n".join(map_results) if map_results else "No explicit level-1 findings extracted."

        # Reduce phase with large model
        reduce_prompt = self.REDUCE_PROMPT.format(
            question=question,
            findings=findings_text,
            paper_title=paper_title
        )
        answer, r_log = await ollama_client.generate(
            prompt=reduce_prompt,
            model=settings.OLLAMA_LARGE_MODEL,
            temperature=0.2
        )
        trace.total_tokens += r_log.total_tokens
        trace.total_llm_calls += 1

        if not answer or r_log.status != "success":
            answer = (
                f"To definitively answer your question ('{question}'), here is the relevant evidence from '{paper_title}':\n\n"
                f"{findings_text}\n"
            )

        trace.latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "mode": "static",
            "answer": answer,
            "trace": trace.model_dump(),
            "selected_reports": [c.id for c in target_comms],
            "total_tokens": trace.total_tokens,
            "total_llm_calls": trace.total_llm_calls,
            "latency_ms": trace.latency_ms
        }

static_search = StaticSearch()
