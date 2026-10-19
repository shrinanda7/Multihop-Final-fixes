"""Query-relevant traversal over the Tarjan quotient graph."""

import re
from typing import Any, Dict, List, Set

import networkx as nx


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what",
    "how", "why", "does", "do", "of", "in", "on", "for",
    "and", "or", "to", "with", "by", "this", "that",
    "it", "its", "from", "paper", "explain", "describe",
}


def _tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def select_quotient_subgraph(
    question: str,
    compressed_graph: Dict[str, Any],
    max_seeds: int = 4,
    max_selected_nodes: int = 12,
    max_path_nodes: int = 8,
    bridge_aware: bool = True,
    budget_aware: bool = True,
) -> Dict[str, Any]:
    """Select relevant quotient nodes and preserve directed paths between seeds.

    This function performs deterministic lexical seed selection. It does not
    generate answers, retrieve chunk contents, or claim semantic relevance.
    """

    nodes = compressed_graph.get("nodes", [])
    edges = compressed_graph.get("edges", [])

    # Ablations must change the retrieval procedure, not merely their labels.
    # The unbudgeted ablation keeps all selected path nodes; synthesis still
    # enforces the model's hard context limit downstream.
    if not budget_aware:
        max_selected_nodes = max(len(nodes), 1)
        max_path_nodes = max(len(nodes), 1)

    if not nodes:
        return {
            "nodes": [],
            "edges": [],
            "seed_node_ids": [],
            "selected_node_ids": [],
            "selected_original_node_ids": [],
            "selected_chunk_ids": [],
            "connector_component_ids": [],
            "scores": {},
        }

    node_map = {str(node["id"]): node for node in nodes if "id" in node}
    graph = nx.DiGraph()
    graph.add_nodes_from(node_map)

    valid_edges = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))

        if source in node_map and target in node_map:
            graph.add_edge(source, target)
            valid_edges.append(edge)

    # Map original entity IDs to their quotient-node IDs.
    original_to_component: Dict[str, str] = {}

    for component_id, node in node_map.items():
        member_ids = node.get("member_node_ids") or [component_id]
        for member_id in member_ids:
            original_to_component[str(member_id)] = component_id

    articulation_points = {
        str(node_id)
        for node_id in compressed_graph.get("articulation_points", [])
    }

    connector_components = {
        original_to_component[node_id]
        for node_id in articulation_points
        if node_id in original_to_component
    }

    query_tokens = _tokens(question)
    scores: Dict[str, int] = {}

    for component_id, node in node_map.items():
        text_parts = [
            str(node.get("name", "")),
            str(node.get("description", "")),
            str(node.get("type", "")),
        ]

        # Include member descriptions when the component represents an SCC.
        for member in node.get("member_nodes", []):
            if isinstance(member, dict):
                text_parts.extend([
                    str(member.get("name", "")),
                    str(member.get("description", "")),
                ])

        node_tokens = _tokens(" ".join(text_parts))
        scores[component_id] = len(query_tokens & node_tokens)

    ranked_seeds = sorted(
        (node_id for node_id in node_map if scores[node_id] > 0),
        key=lambda node_id: (
            -scores[node_id],
            node_id,
        ),
    )[:max_seeds]

    if not ranked_seeds:
        return {
            "nodes": [],
            "edges": [],
            "seed_node_ids": [],
            "selected_node_ids": [],
            "selected_original_node_ids": [],
            "selected_chunk_ids": [],
            "connector_component_ids": [],
            "scores": scores,
        }

    selected: Set[str] = set(ranked_seeds)

    # Preserve directed connecting paths between relevant seeds.
    # Try both endpoint orders, but never reverse an edge in the path.
    for i, source in enumerate(ranked_seeds):
        for target in ranked_seeds[i + 1:]:
            candidates = []

            for start, end in ((source, target), (target, source)):
                try:
                    path = nx.shortest_path(graph, start, end)
                    if len(path) <= max_path_nodes:
                        candidates.append(path)
                except nx.NetworkXNoPath:
                    continue

            if candidates:
                path = min(candidates, key=lambda p: (len(p), p))
                selected.update(path)

    # If no path joins two seeds, retain directly adjacent connector
    # components rather than silently discarding their structural role.
    if bridge_aware and len(selected) == len(ranked_seeds):
        for seed in ranked_seeds:
            for neighbor in graph.successors(seed):
                if neighbor in connector_components:
                    selected.add(neighbor)
            for neighbor in graph.predecessors(seed):
                if neighbor in connector_components:
                    selected.add(neighbor)

    # Bound context size deterministically. Keep seed nodes first.
    ordered = list(ranked_seeds)
    ordered.extend(sorted(selected - set(ordered)))
    selected_ids = ordered if not budget_aware else ordered[:max_selected_nodes]
    selected_set = set(selected_ids)

    selected_edges = [
        edge for edge in valid_edges
        if str(edge["source"]) in selected_set
        and str(edge["target"]) in selected_set
    ]

    original_ids: Set[str] = set()
    chunk_ids: Set[str] = set()

    for component_id in selected_ids:
        node = node_map[component_id]
        members = node.get("member_node_ids") or [component_id]

        original_ids.update(str(member_id) for member_id in members)
        chunk_ids.update(
            str(chunk_id)
            for chunk_id in node.get("source_chunk_ids", [])
        )

    for edge in selected_edges:
        chunk_ids.update(
            str(chunk_id)
            for chunk_id in edge.get("source_chunk_ids", [])
        )

    return {
        "nodes": [node_map[node_id] for node_id in selected_ids],
        "edges": selected_edges,
        "seed_node_ids": ranked_seeds,
        "selected_node_ids": selected_ids,
        "selected_original_node_ids": sorted(original_ids),
        "selected_chunk_ids": sorted(chunk_ids),
        "connector_component_ids": sorted(
            selected_set & connector_components
        ),
        "scores": scores,
    }