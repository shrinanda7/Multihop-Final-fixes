"""Comparative Evaluation and Ablation Suite with LLM-as-a-Judge and CSV Exporter."""
import csv
import io
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.core.llm_client import ollama_client
from backend.evaluation.question_generator import EVAL_QUESTIONS, EvalQuestion
from backend.search.query_router import query_router

logger = logging.getLogger("graphrag.benchmark")

CONFIGURATIONS = [
    {"name": "Static (Level 1)", "mode": "static", "tarjan": True, "bridge": True, "budget": True},
    {"name": "Dynamic (Reference)", "mode": "dynamic", "tarjan": True, "bridge": True, "budget": True},
    {"name": "Ours (Full)", "mode": "ours", "tarjan": True, "bridge": True, "budget": True},
    {"name": "Ablation: No Tarjan", "mode": "ours", "tarjan": False, "bridge": True, "budget": True},
    {"name": "Ablation: No Bridge-Aware", "mode": "ours", "tarjan": True, "bridge": False, "budget": True},
    {"name": "Ablation: No Budget-Aware", "mode": "ours", "tarjan": True, "bridge": True, "budget": False},
]

class BenchmarkRunner:
    """Executes comparative evaluation across 6 configurations and exports metrics."""

    JUDGE_PROMPT = """You are an expert scientific judge evaluating an AI generated answer against a ground truth rubric.

User Question: {question}
Ground Truth Hint: {hint}

Generated Answer to Evaluate:
{answer}

Evaluate on a strict 1 to 5 rubric based on:
1. Factuality & Soundness (Is the information strictly true and grounded?)
2. Comprehensiveness (Does it address core mechanisms and trade-offs?)
3. Precision & Citations (Are specific numbers/sections cited accurately?)

Return ONLY a JSON object:
{{"score": 4.5, "justification": "Brief 1-sentence explanation"}}"""

    def __init__(self):
        self.results_dir = settings.DATA_DIR / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.latest_results: List[Dict[str, Any]] = []

    async def run_benchmark(self, num_questions: int = 6, randomize_order: bool = True) -> Dict[str, Any]:
        """Run benchmark on a subset or all 50 questions across 6 system configurations."""
        questions_to_eval = EVAL_QUESTIONS[:num_questions]
        all_eval_records: List[Dict[str, Any]] = []

        config_list = list(CONFIGURATIONS)
        if randomize_order:
            random.shuffle(config_list)

        for q in questions_to_eval:
            for cfg in config_list:
                # Set ablation switches
                orig_tarjan = settings.ENABLE_TARJAN_COMPRESSION
                orig_bridge = settings.ENABLE_BRIDGE_AWARE_PRUNING
                orig_budget = settings.ENABLE_BUDGET_AWARE_TRAVERSAL

                settings.ENABLE_TARJAN_COMPRESSION = cfg["tarjan"]
                settings.ENABLE_BRIDGE_AWARE_PRUNING = cfg["bridge"]
                settings.ENABLE_BUDGET_AWARE_TRAVERSAL = cfg["budget"]

                t0 = time.time()
                try:
                    res = await query_router.execute_query(
                        question=q.question,
                        paper_id=q.paper_id,
                        paper_title=q.paper_title,
                        mode=cfg["mode"]
                    )
                    latency = round((time.time() - t0) * 1000.0, 2)

                    # Do not synthesize research metrics from configuration flags.
                    # Until the query response includes the actual selected subgraph and
                    # original/compressed graph sizes, these values are explicitly unknown.
                    trace = res.get("trace", {})
                    bridge_kept = len(trace.get("bridge_kept_nodes", []))
                    connectivity = None

                    comp_graph = res.get("compressed_graph", {})
                    comp_ratio = comp_graph.get("compression_ratio")

                    # An unavailable judge result must remain missing, not become a fake score.
                    score, just = await self._judge_answer(q.question, q.ground_truth_hint, res.get("answer", ""))

                    record = {
                        "question_id": q.id,
                        "category": q.category,
                        "paper_id": q.paper_id,
                        "configuration": cfg["name"],
                        "mode": cfg["mode"],
                        "score": score,
                        "justification": just,
                        "total_tokens": res.get("total_tokens", 0),
                        "total_llm_calls": res.get("total_llm_calls", 0),
                        "latency_ms": latency,
                        "compression_ratio": comp_ratio,
                        "graph_preservation": connectivity,
                        "bridge_kept_count": bridge_kept,
                        "status": "ok",
                    }
                    all_eval_records.append(record)

                except Exception as e:
                    logger.exception("Benchmark error for %s on %s", cfg["name"], q.id)
                    # Record failed attempts explicitly so a configuration with
                    # failures cannot appear better merely because failed samples
                    # vanished from the denominator.
                    all_eval_records.append({
                        "question_id": q.id,
                        "category": q.category,
                        "paper_id": q.paper_id,
                        "configuration": cfg["name"],
                        "mode": cfg["mode"],
                        "score": None,
                        "justification": f"Query failed: {type(e).__name__}",
                        "total_tokens": 0,
                        "total_llm_calls": 0,
                        "latency_ms": round((time.time() - t0) * 1000.0, 2),
                        "compression_ratio": None,
                        "graph_preservation": None,
                        "bridge_kept_count": 0,
                        "status": "error",
                    })

                finally:
                    # Restore original switches
                    settings.ENABLE_TARJAN_COMPRESSION = orig_tarjan
                    settings.ENABLE_BRIDGE_AWARE_PRUNING = orig_bridge
                    settings.ENABLE_BUDGET_AWARE_TRAVERSAL = orig_budget

        self.latest_results = all_eval_records

        # Save to file
        with open(self.results_dir / "latest_benchmark.json", "w") as f:
            json.dump(all_eval_records, f, indent=2)

        # Compute aggregate summary per configuration
        summary = self._compute_summary(all_eval_records)
        return {"summary": summary, "records": all_eval_records}

    async def _judge_answer(self, question: str, hint: str, answer: str) -> tuple[Optional[float], str]:
        """Blind randomized evaluation of answer quality."""
        prompt = self.JUDGE_PROMPT.format(question=question, hint=hint, answer=answer[:1500])
        res, log = await ollama_client.generate(
            prompt=prompt,
            model=settings.OLLAMA_LARGE_MODEL,
            temperature=0.0,
            format_json=True
        )

        if not res:
            return None, "Judge unavailable; score not computed."
        try:
            data = json.loads(res)
            score = float(data["score"])
            if not 1.0 <= score <= 5.0:
                raise ValueError("Judge score must be between 1 and 5")
            justification = str(data.get("justification", "No justification supplied."))
            return score, justification
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None, "Judge returned an invalid response; score not computed."

    def _compute_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average metrics per configuration."""
        by_cfg: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            by_cfg.setdefault(r["configuration"], []).append(r)

        summary_table = []
        for cfg_name, cfg_records in by_cfg.items():
            n = len(cfg_records)
            if n == 0:
                continue
            valid_scores = [r["score"] for r in cfg_records if isinstance(r.get("score"), (int, float))]
            avg_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None
            successful_records = [r for r in cfg_records if r.get("status", "ok") == "ok"]
            successful_n = len(successful_records)
            avg_tokens = round(sum(r["total_tokens"] for r in successful_records) / successful_n, 1) if successful_n else None
            avg_calls = round(sum(r["total_llm_calls"] for r in successful_records) / successful_n, 1) if successful_n else None
            avg_latency = round(sum(r["latency_ms"] for r in cfg_records) / n, 1)
            valid_comp = [r["compression_ratio"] for r in cfg_records if isinstance(r.get("compression_ratio"), (int, float))]
            valid_pres = [r["graph_preservation"] for r in cfg_records if isinstance(r.get("graph_preservation"), (int, float))]
            avg_comp = round(sum(valid_comp) / len(valid_comp), 2) if valid_comp else None
            avg_pres = round(sum(valid_pres) / len(valid_pres), 2) if valid_pres else None

            summary_table.append({
                "configuration": cfg_name,
                "avg_score": avg_score,
                "avg_tokens": avg_tokens,
                "avg_llm_calls": avg_calls,
                "avg_latency_ms": avg_latency,
                "compression_ratio": avg_comp,
                "graph_preservation": avg_pres,
                "attempted_samples": n,
                "successful_samples": successful_n,
                "failed_samples": n - successful_n,
                "valid_judge_scores": len(valid_scores),
            })

        return {"configurations": summary_table}

    def export_csv(self) -> str:
        """Export latest benchmark records to CSV format."""
        if not self.latest_results:
            # Generate mock starter if not run yet
            return "question_id,category,paper_id,configuration,score,total_tokens,total_llm_calls,latency_ms,compression_ratio,graph_preservation\n"

        output = io.StringIO()
        fieldnames = [
            "question_id", "category", "paper_id", "configuration", "score",
            "total_tokens", "total_llm_calls", "latency_ms", "compression_ratio", "graph_preservation"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in self.latest_results:
            writer.writerow(r)
        return output.getvalue()

benchmark_runner = BenchmarkRunner()
