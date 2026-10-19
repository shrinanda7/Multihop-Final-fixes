"""Bottom-up Hierarchical Community Report Generator with LLM Synthesis, Embeddings, and Caching."""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.config import settings
from backend.core.clustering import HierarchicalCommunity
from backend.core.llm_client import ollama_client, validate_embedding

logger = logging.getLogger("graphrag.reports")

class CommunityReportGenerator:
    """Generates bottom-up hierarchical reports for communities with caching and embeddings."""

    REPORT_PROMPT = """You are a lead scientific literature researcher. Write a comprehensive community report for this cluster of concepts from a research paper.

Community Title: {title}
Level: {level}
Member Entities:
{entities_text}

Supporting Child Summaries / Evidence:
{evidence_text}

Provide a structured report with:
1. EXECUTIVE SUMMARY: High-level overview of this cluster's role in the paper.
2. KEY SCIENTIFIC FINDINGS: Detailed points explaining how these entities interact, mechanisms used, or results achieved.
3. RELEVANCE & IMPACT: Significance within the paper's overarching methodology.

Format clearly with bold headers."""

    def __init__(self):
        self.cache_dir = settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, str] = {}

    async def generate_reports_for_hierarchy(
        self,
        communities: Dict[str, HierarchicalCommunity],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> Dict[str, HierarchicalCommunity]:
        """Generate reports bottom-up: Level 2 -> Level 1 -> Level 0."""
        node_map = {n["id"]: n for n in nodes}

        # Group by level descending (leaves first)
        by_level: Dict[int, List[HierarchicalCommunity]] = {}
        for comm in communities.values():
            by_level.setdefault(comm.level, []).append(comm)

        sorted_levels = sorted(by_level.keys(), reverse=True)

        for lvl in sorted_levels:
            for comm in by_level[lvl]:
                cache_file = self.cache_dir / f"{comm.id}_report.json"

                # Check cache
                if use_cache and cache_file.exists():
                    try:
                        with open(cache_file, "r") as f:
                            data = json.load(f)
                            comm.report = data.get("report", "")
                            comm.summary = data.get("summary", "")
                            raw_embed = data.get("embedding", [])
                            # Validate cached embedding: must be non-empty and well-formed
                            if raw_embed:
                                try:
                                    validate_embedding(raw_embed)
                                    comm.embedding = raw_embed
                                except Exception:
                                    comm.embedding = []
                            else:
                                comm.embedding = []
                            if comm.report:
                                continue
                    except Exception:
                        pass

                # Build prompt context
                members = [node_map[nid] for nid in comm.member_node_ids if nid in node_map]
                entities_str = "\n".join(
                    f"- {m['name']} ({m.get('type', 'Concept')}): {m.get('description', '')[:200]}"
                    for m in members[:15]
                ) or "No explicitly named members."

                # Gather child summaries if higher level
                evidence_list = []
                for cid in comm.children_ids:
                    child_comm = communities.get(cid)
                    if child_comm and child_comm.summary:
                        evidence_list.append(f"Sub-community [{child_comm.title}]: {child_comm.summary}")

                evidence_str = "\n".join(evidence_list) if evidence_list else "Primary entity extraction."

                prompt = self.REPORT_PROMPT.format(
                    title=comm.title,
                    level=comm.level,
                    entities_text=entities_str,
                    evidence_text=evidence_str
                )

                # Generate report with large model
                report_text, call_log = await ollama_client.generate(
                    prompt=prompt,
                    model=settings.OLLAMA_LARGE_MODEL,
                    temperature=0.2
                )

                if not report_text or call_log.status != "success":
                    # Evidence-only fallback: do not fabricate methodology, impact, or causal claims.
                    names = [str(m.get("name", m.get("id", "unnamed entity"))) for m in members]
                    descriptions = [
                        f"- {m.get('name', m.get('id', 'Unnamed'))}: {str(m.get('description', '')).strip()}"
                        for m in members if str(m.get("description", "")).strip()
                    ]
                    member_ids = {m.get("id") for m in members}
                    observed_relations = [
                        f"- {node_map.get(e.get('source'), {}).get('name', e.get('source'))} --[{e.get('type', 'RELATED_TO')}]--> {node_map.get(e.get('target'), {}).get('name', e.get('target'))}"
                        for e in edges if e.get("source") in member_ids and e.get("target") in member_ids
                    ]
                    report_text = (
                        f"### Evidence Summary\n"
                        f"This community contains the extracted concepts: {', '.join(names[:10]) or 'no named concepts'}. "
                        f"An LLM report was unavailable, so this summary is limited to stored extraction data.\n\n"
                        f"### Extracted Descriptions\n" + ("\n".join(descriptions[:20]) or "No source descriptions were stored.") + "\n\n"
                        f"### Extracted Relationships\n" + ("\n".join(observed_relations[:20]) or "No internal relationships were stored for this community.")
                    )

                comm.report = report_text
                # First paragraph as concise summary
                paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip() and not p.startswith("#")]
                comm.summary = paragraphs[0] if paragraphs else report_text[:300]

                # Generate embedding for vector retrieval
                try:
                    embed_vec, _ = await ollama_client.get_embedding(comm.summary, raise_on_error=False)
                    comm.embedding = embed_vec or []
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for community {comm.id}: {e}")
                    comm.embedding = []

                # Save cache (only persist embedding if valid and non-empty)
                try:
                    cache_payload = {
                        "id": comm.id,
                        "title": comm.title,
                        "report": comm.report,
                        "summary": comm.summary,
                    }
                    if comm.embedding and len(comm.embedding) > 0:
                        cache_payload["embedding"] = comm.embedding
                    with open(cache_file, "w") as f:
                        json.dump(cache_payload, f)
                except Exception as e:
                    logger.warning(f"Failed to cache report: {e}")

        return communities

report_generator = CommunityReportGenerator()
