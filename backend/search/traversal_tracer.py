"""Structured Traversal Tracer for Cytoscape Graph Animation and Telemetry."""
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TraversalStep(BaseModel):
    step_index: int
    action: str  # "VISIT", "RATE", "EXPAND", "PRUNE", "BRIDGE_PRESERVE", "SELECT"
    node_or_comm_id: str
    target_id: Optional[str] = None
    level: int = 1
    score: float = 0.0
    reason: str = ""
    timestamp_ms: float = Field(default_factory=lambda: round(time.time() * 1000.0, 2))

class TraversalTrace(BaseModel):
    mode: str  # "static", "dynamic", "ours", "local"
    steps: List[TraversalStep] = Field(default_factory=list)
    visited_nodes: List[str] = Field(default_factory=list)
    expanded_nodes: List[str] = Field(default_factory=list)
    pruned_nodes: List[str] = Field(default_factory=list)
    bridge_kept_nodes: List[str] = Field(default_factory=list)
    selected_reports: List[str] = Field(default_factory=list)
    ratings: Dict[str, float] = Field(default_factory=dict)
    total_tokens: int = 0
    total_llm_calls: int = 0
    latency_ms: float = 0.0

    def add_step(
        self,
        action: str,
        node_or_comm_id: str,
        level: int = 1,
        score: float = 0.0,
        reason: str = "",
        target_id: Optional[str] = None
    ):
        step_idx = len(self.steps)
        self.steps.append(TraversalStep(
            step_index=step_idx,
            action=action,
            node_or_comm_id=node_or_comm_id,
            target_id=target_id,
            level=level,
            score=score,
            reason=reason
        ))
        if action == "VISIT" and node_or_comm_id not in self.visited_nodes:
            self.visited_nodes.append(node_or_comm_id)
        elif action == "EXPAND" and node_or_comm_id not in self.expanded_nodes:
            self.expanded_nodes.append(node_or_comm_id)
        elif action == "PRUNE" and node_or_comm_id not in self.pruned_nodes:
            self.pruned_nodes.append(node_or_comm_id)
        elif action == "BRIDGE_PRESERVE" and node_or_comm_id not in self.bridge_kept_nodes:
            self.bridge_kept_nodes.append(node_or_comm_id)
        elif action == "SELECT" and node_or_comm_id not in self.selected_reports:
            self.selected_reports.append(node_or_comm_id)

        if score > 0.0:
            self.ratings[node_or_comm_id] = score
