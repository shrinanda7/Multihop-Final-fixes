"""Hierarchical Leiden/Louvain Community Clustering for Knowledge Graphs."""
import logging
from typing import Dict, List, Set, Any, Optional
import networkx as nx
try:
    import community as community_louvain
except ImportError:  # NetworkX has a Louvain implementation in supported versions.
    community_louvain = None

logger = logging.getLogger("graphrag.clustering")

class HierarchicalCommunity:
    def __init__(
        self,
        community_id: str,
        level: int,
        title: str,
        member_node_ids: List[str],
        parent_id: Optional[str] = None,
        children_ids: Optional[List[str]] = None
    ):
        self.id = community_id
        self.level = level
        self.title = title
        self.member_node_ids = member_node_ids
        self.parent_id = parent_id
        self.children_ids = children_ids or []
        self.report: str = ""
        self.summary: str = ""
        self.embedding: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "member_node_ids": self.member_node_ids,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "report": self.report,
            "summary": self.summary
        }

class HierarchicalClusterer:
    """Partitions graph into multi-level hierarchical community tree."""

    def build_hierarchy(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        paper_id: str
    ) -> Dict[str, HierarchicalCommunity]:
        """Detect hierarchical communities using multi-resolution Louvain/Leiden modularity."""
        if not nodes:
            return {}

        # Construct undirected graph for community detection
        G = nx.Graph()
        node_map = {n["id"]: n for n in nodes}
        for n in nodes:
            G.add_node(n["id"])
        for e in edges:
            u, v = e["source"], e["target"]
            if u in node_map and v in node_map and u != v:
                w = float(e.get("weight", 1.0))
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)

        # Handle disconnected components / isolated nodes gracefully
        communities: Dict[str, HierarchicalCommunity] = {}

        # 1. Level 0: Root community embracing all components
        all_node_ids = list(node_map.keys())
        root_id = f"{paper_id}_comm_L0_000"
        root_comm = HierarchicalCommunity(
            community_id=root_id,
            level=0,
            title=f"Global Research Context ({len(all_node_ids)} nodes)",
            member_node_ids=all_node_ids,
            parent_id=None
        )
        communities[root_id] = root_comm

        # 2. Partition Level 1: Moderate resolution
        if len(G.nodes()) > 2 and len(G.edges()) > 0:
            try:
                # Level 1 partition (coarse)
                part_l1 = community_louvain.best_partition(G, weight="weight", resolution=0.8, random_state=42)
                l1_groups: Dict[int, List[str]] = {}
                for nid, cid in part_l1.items():
                    l1_groups.setdefault(cid, []).append(nid)

                # Level 2 partition (fine-grained)
                part_l2 = community_louvain.best_partition(G, weight="weight", resolution=1.5, random_state=42)
                l2_groups: Dict[int, List[str]] = {}
                for nid, cid in part_l2.items():
                    l2_groups.setdefault(cid, []).append(nid)

            except Exception as e:
                logger.warning(f"Community partitioning fallback due to: {e}")
                l1_groups = {0: all_node_ids}
                l2_groups = {0: all_node_ids}
        else:
            l1_groups = {0: all_node_ids}
            l2_groups = {0: all_node_ids}

        # Build Level 1 community objects
        l1_comm_ids = []
        for c_idx, (cid, members) in enumerate(l1_groups.items()):
            l1_id = f"{paper_id}_comm_L1_{c_idx:03d}"
            names = [node_map[m]["name"] for m in members if m in node_map]
            title = f"Domain: {', '.join(names[:2])}{'...' if len(names) > 2 else ''}"

            comm_obj = HierarchicalCommunity(
                community_id=l1_id,
                level=1,
                title=title,
                member_node_ids=members,
                parent_id=root_id
            )
            communities[l1_id] = comm_obj
            l1_comm_ids.append(l1_id)

        root_comm.children_ids = l1_comm_ids

        # Build Level 2 (Sub-communities) under Level 1
        for l1_id in l1_comm_ids:
            l1_comm = communities[l1_id]
            l1_member_set = set(l1_comm.member_node_ids)

            # Find matching L2 subgroups
            matching_l2_groups = []
            for l2_cid, l2_members in l2_groups.items():
                intersection = [m for m in l2_members if m in l1_member_set]
                if intersection:
                    matching_l2_groups.append(intersection)

            l2_comm_ids = []
            if len(matching_l2_groups) > 1:
                for sub_idx, sub_members in enumerate(matching_l2_groups):
                    l2_id = f"{l1_id}_sub_{sub_idx:02d}"
                    sub_names = [node_map[m]["name"] for m in sub_members if m in node_map]
                    sub_title = f"Topic: {', '.join(sub_names[:2])}"
                    sub_comm = HierarchicalCommunity(
                        community_id=l2_id,
                        level=2,
                        title=sub_title,
                        member_node_ids=sub_members,
                        parent_id=l1_id
                    )
                    communities[l2_id] = sub_comm
                    l2_comm_ids.append(l2_id)
            else:
                # Single subgroup
                l2_id = f"{l1_id}_sub_00"
                sub_comm = HierarchicalCommunity(
                    community_id=l2_id,
                    level=2,
                    title=f"{l1_comm.title} Details",
                    member_node_ids=l1_comm.member_node_ids,
                    parent_id=l1_id
                )
                communities[l2_id] = sub_comm
                l2_comm_ids.append(l2_id)

            l1_comm.children_ids = l2_comm_ids

        return communities

clusterer = HierarchicalClusterer()
