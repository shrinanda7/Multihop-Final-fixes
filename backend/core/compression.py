"""Graph Compression using Tarjan's SCCs, Articulation Points, and Bridge Preservation."""
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from backend.config import settings
from backend.core.tarjan import tarjan

logger = logging.getLogger("graphrag.compression")

class SCCDiagnostics:
    def __init__(
        self,
        total_nodes: int,
        num_sccs: int,
        scc_sizes: List[int],
        max_scc_size: int,
        max_scc_ratio: float,
        has_giant_scc: bool,
        mitigation_strategy: Optional[str] = None
    ):
        self.total_nodes = total_nodes
        self.num_sccs = num_sccs
        self.scc_sizes = scc_sizes
        self.max_scc_size = max_scc_size
        self.max_scc_ratio = max_scc_ratio
        self.has_giant_scc = has_giant_scc
        self.mitigation_strategy = mitigation_strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "num_sccs": self.num_sccs,
            "scc_sizes": sorted(self.scc_sizes, reverse=True),
            "max_scc_size": self.max_scc_size,
            "max_scc_ratio": round(self.max_scc_ratio, 3),
            "has_giant_scc": self.has_giant_scc,
            "mitigation_strategy": self.mitigation_strategy,
        }

class TarjanGraphCompressor:
    """Compresses knowledge graph by collapsing SCCs into supernodes while protecting bridges and articulation points."""

    def analyze_scc_distribution(
        self,
        nodes: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Tuple[List[List[str]], SCCDiagnostics]:
        """Analyze SCC size distribution and detect giant component anomalies before compressing."""
        total_nodes = len(nodes)
        if total_nodes == 0:
            return [], SCCDiagnostics(0, 0, [], 0, 0.0, False)

        node_ids = [n["id"] for n in nodes]
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for r in relations:
            src = r["source"]
            tgt = r["target"]
            if src in adj and tgt in adj:
                adj[src].append(tgt)

        sccs = tarjan.strongly_connected_components(adj)
        scc_sizes = [len(s) for s in sccs]
        max_size = max(scc_sizes) if scc_sizes else 0
        ratio = max_size / total_nodes if total_nodes > 0 else 0.0

        has_giant = ratio >= settings.GIANT_SCC_THRESHOLD and max_size > 3
        mitigation = None
        if has_giant:
            mitigation = (
                f"Giant SCC detected ({max_size}/{total_nodes} nodes, {ratio:.1%}). "
                "Mitigation active: Pruning transitive cross-domain hub connections with weight < 0.5, "
                "and preserving articulation points to prevent over-condensation."
            )
            logger.warning(mitigation)

        diag = SCCDiagnostics(
            total_nodes=total_nodes,
            num_sccs=len(sccs),
            scc_sizes=scc_sizes,
            max_scc_size=max_size,
            max_scc_ratio=ratio,
            has_giant_scc=has_giant,
            mitigation_strategy=mitigation
        )
        return sccs, diag

    def compress_graph(
        self,
        nodes: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Produce compressed quotient graph with preserved connectors and complete supernode mappings."""
        if not nodes:
            return {"nodes": [], "edges": [], "diagnostics": {}, "compression_ratio": 1.0}

        node_map = {n["id"]: n for n in nodes}
        node_ids = list(node_map.keys())

        # 1. Undirected edges for bridge and articulation point detection
        edge_tuples = [(r["source"], r["target"]) for r in relations]
        bridges, articulation_points = tarjan.bridges_and_articulation_points(node_ids, edge_tuples)

        # 2. Directed SCC analysis
        sccs, diagnostics = self.analyze_scc_distribution(nodes, relations)

        # 3. Form compressed nodes. Articulation points remain explicit when
        # they are singleton SCCs; inside a non-trivial SCC they are recorded
        # as protected connector members so SCC condensation remains acyclic.
        compressed_nodes: List[Dict[str, Any]] = []
        node_to_compressed_id: Dict[str, str] = {}

        supernode_counter = 0

        for scc in sccs:
            # Keep each non-trivial SCC atomic when compression is enabled.
            # Splitting an articulation vertex out of an SCC can reintroduce
            # cycles into a graph that is meant to be a condensation DAG.
            if len(scc) > 1 and settings.ENABLE_TARJAN_COMPRESSION:
                members = [node_map[nid] for nid in scc]
                member_names = [m["name"] for m in members]
                supernode_id = f"supernode_{supernode_counter:03d}"
                supernode_counter += 1
                combined_desc = " | ".join(
                    m.get("description", "") for m in members if m.get("description")
                )
                all_chunks = sorted({
                    str(cid) for m in members for cid in m.get("source_chunk_ids", [])
                })
                connector_members = [nid for nid in scc if nid in articulation_points]
                compressed_nodes.append({
                    "id": supernode_id,
                    "name": f"SCC [{', '.join(member_names[:3])}{'...' if len(member_names) > 3 else ''}]",
                    "type": "Supernode",
                    "description": combined_desc[:500],
                    "is_supernode": True,
                    "is_connector": bool(connector_members),
                    "connector_member_node_ids": connector_members,
                    "member_node_ids": list(scc),
                    "member_nodes": members,
                    "member_count": len(scc),
                    "source_chunk_ids": all_chunks,
                })
                for nid in scc:
                    node_to_compressed_id[nid] = supernode_id
                continue

            # With compression disabled, preserve every original node. A
            # singleton articulation point remains an explicit connector.
            for nid in scc:
                orig_node = node_map[nid]
                is_connector = nid in articulation_points
                compressed_nodes.append({
                    "id": nid,
                    "name": orig_node["name"],
                    "type": orig_node.get("type", "Concept"),
                    "description": orig_node.get("description", ""),
                    "is_supernode": False,
                    "is_connector": is_connector,
                    "connector_member_node_ids": [nid] if is_connector else [],
                    "member_node_ids": [nid],
                    "member_count": 1,
                    "source_chunk_ids": orig_node.get("source_chunk_ids", []),
                    # Preserve namespace/provenance metadata through compression.
                    **({"paper_id": orig_node["paper_id"]} if "paper_id" in orig_node else {}),
                    **({"paper_title": orig_node["paper_title"]} if "paper_title" in orig_node else {}),
                    **({"original_node_id": orig_node["original_node_id"]} if "original_node_id" in orig_node else {}),
                })
                node_to_compressed_id[nid] = nid

        # 4. Quotient Edges
        compressed_edges_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for r in relations:
            src_comp = node_to_compressed_id.get(r["source"])
            tgt_comp = node_to_compressed_id.get(r["target"])

            if not src_comp or not tgt_comp:
                continue

            # Check if this edge is a bridge
            edge_key = (min(r["source"], r["target"]), max(r["source"], r["target"]))
            is_bridge = edge_key in bridges

            # If inside the same supernode and not an external connection, skip quotient edge
            if src_comp == tgt_comp:
                continue

            comp_key = (src_comp, tgt_comp)
            if comp_key not in compressed_edges_dict:
                compressed_edges_dict[comp_key] = {
                    "source": src_comp,
                    "target": tgt_comp,
                    "type": r.get("type", "RELATED_TO"),
                    "weight": float(r.get("weight", 1.0)),
                    "is_bridge": is_bridge,
                    "original_relations": [r]
                }
            else:
                existing = compressed_edges_dict[comp_key]
                existing["weight"] += float(r.get("weight", 1.0))
                if is_bridge:
                    existing["is_bridge"] = True
                existing["original_relations"].append(r)

        compressed_edges = list(compressed_edges_dict.values())
        compression_ratio = round(len(compressed_nodes) / len(nodes), 3) if nodes else 1.0

        # Compute topological order of condensation DAG
        topo_order = self._topological_sort(compressed_nodes, compressed_edges)

        return {
            "nodes": compressed_nodes,
            "edges": compressed_edges,
            "original_nodes": nodes,
            "original_edges": relations,
            "diagnostics": diagnostics.to_dict(),
            "compression_ratio": compression_ratio,
            "articulation_points": list(articulation_points),
            "bridges": [list(b) for b in bridges],
            "topological_order": topo_order
        }

    def _topological_sort(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """Kahn's topological sort on quotient DAG."""
        in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}
        adj: Dict[str, List[str]] = {n["id"]: [] for n in nodes}

        for e in edges:
            src = e["source"]
            tgt = e["target"]
            if src in in_degree and tgt in in_degree:
                adj[src].append(tgt)
                in_degree[tgt] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for nxt in adj.get(curr, []):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        # A topological order exists only for a DAG. Never append cyclic
        # remnants, which would falsely present an invalid order as valid.
        if len(order) != len(nodes):
            logger.warning(
                "Topological ordering unavailable: graph contains a cycle. "
                "This is expected when Tarjan compression is disabled."
            )
            return []
        return order

compressor = TarjanGraphCompressor()
