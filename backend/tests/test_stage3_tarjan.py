"""Stage 3 Verification Tests: Tarjan SCC, Bridges, and Articulation Points verified against NetworkX."""
import pytest
import networkx as nx
import random
from backend.core.tarjan import tarjan
from backend.core.compression import compressor
from backend.config import settings

def test_tarjan_scc_vs_networkx_cycles():
    """Verify SCC equivalence with networkx on cyclic topologies."""
    # Graph with two distinct SCC cycles and singleton nodes:
    # Cycle 1: a -> b -> c -> a
    # Cycle 2: d -> e -> d
    # Bridge edge: c -> d
    # Outgoing to singleton: e -> f
    adj = {
        "a": ["b"],
        "b": ["c"],
        "c": ["a", "d"],
        "d": ["e"],
        "e": ["d", "f"],
        "f": []
    }

    # Custom iterative Tarjan
    custom_sccs = [set(scc) for scc in tarjan.strongly_connected_components(adj)]

    # NetworkX reference
    G = nx.DiGraph()
    for u, neighbors in adj.items():
        G.add_node(u)
        for v in neighbors:
            G.add_edge(u, v)

    nx_sccs = [set(scc) for scc in nx.strongly_connected_components(G)]

    # Verify identical partition
    assert len(custom_sccs) == len(nx_sccs)
    for scc in custom_sccs:
        assert scc in nx_sccs

def test_tarjan_scc_vs_networkx_random_graphs():
    """Verify iterative Tarjan on multiple randomized directed graphs against NetworkX."""
    random.seed(42)
    for num_nodes in [10, 25, 50]:
        for edge_prob in [0.05, 0.15, 0.3]:
            G = nx.fast_gnp_random_graph(num_nodes, edge_prob, directed=True, seed=42)
            nodes = [str(n) for n in G.nodes()]
            adj = {str(n): [] for n in G.nodes()}
            for u, v in G.edges():
                adj[str(u)].append(str(v))

            custom_sccs = [set(scc) for scc in tarjan.strongly_connected_components(adj)]
            nx_sccs = [set(str(n) for n in scc) for scc in nx.strongly_connected_components(G)]

            assert len(custom_sccs) == len(nx_sccs)
            for scc in custom_sccs:
                assert scc in nx_sccs

def test_bridges_and_articulation_points_vs_networkx():
    """Verify bridges and articulation points match NetworkX exactly."""
    # Classic bridge & articulation point graph:
    # Triangle (1, 2, 3) connected via bridge (3, 4) to barbell (4, 5, 6)
    # Node 3 and Node 4 are articulation points, edge (3, 4) is a bridge
    nodes = ["1", "2", "3", "4", "5", "6"]
    edges = [
        ("1", "2"), ("2", "3"), ("3", "1"),  # cycle 1
        ("3", "4"),                          # bridge!
        ("4", "5"), ("5", "6"), ("6", "4")   # cycle 2
    ]

    custom_bridges, custom_ap = tarjan.bridges_and_articulation_points(nodes, edges)

    # NetworkX reference on undirected graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nx_bridges = {tuple(sorted((str(u), str(v)))) for u, v in nx.bridges(G)}
    nx_ap = {str(n) for n in nx.articulation_points(G)}

    assert custom_bridges == nx_bridges
    assert custom_ap == nx_ap
    assert ("3", "4") in custom_bridges
    assert "3" in custom_ap and "4" in custom_ap

def test_graph_compression_preserves_connectors_and_mapping():
    """Verify that compression collapses SCCs, preserves articulation points, and stores member mappings."""
    nodes = [
        {"id": "a1", "name": "Node A1"},
        {"id": "a2", "name": "Node A2"},
        {"id": "b", "name": "Bridge Point B"},
        {"id": "c", "name": "Connector C"},
        {"id": "d", "name": "Bridge Point D"},
        {"id": "e1", "name": "Node E1"},
        {"id": "e2", "name": "Node E2"}
    ]
    # Triangle a1 <-> a2 <-> b <-> a1 connected to c, c connected to d, d connected to e1 <-> e2 <-> d <-> e1
    relations = [
        {"source": "a1", "target": "a2", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "a2", "target": "a1", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "a2", "target": "b", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "b", "target": "a2", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "b", "target": "a1", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "a1", "target": "b", "type": "CONNECTED_TO", "weight": 1.0},

        {"source": "b", "target": "c", "type": "BRIDGES_TO", "weight": 1.0},
        {"source": "c", "target": "d", "type": "BRIDGES_TO", "weight": 1.0},

        {"source": "d", "target": "e1", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "e1", "target": "d", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "e1", "target": "e2", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "e2", "target": "e1", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "e2", "target": "d", "type": "CONNECTED_TO", "weight": 1.0},
        {"source": "d", "target": "e2", "type": "CONNECTED_TO", "weight": 1.0},
    ]

    compressed = compressor.compress_graph(nodes, relations)

    assert "nodes" in compressed
    assert "edges" in compressed
    assert "diagnostics" in compressed
    assert compressed["compression_ratio"] < 1.0

    comp_nodes = compressed["nodes"]
    # Check that articulation point 'c' is preserved as a connector
    c_nodes = [n for n in comp_nodes if n["id"] == "c"]
    assert len(c_nodes) == 1
    assert c_nodes[0]["is_connector"] is True
    assert c_nodes[0]["is_supernode"] is False

    # Check supernodes have member mappings
    supernodes = [n for n in comp_nodes if n["is_supernode"]]
    assert len(supernodes) >= 1
    for sn in supernodes:
        assert len(sn["member_node_ids"]) > 1
        assert "member_nodes" in sn

def test_giant_scc_alerting():
    """Verify that giant SCCs exceeding threshold trigger diagnostics with mitigation recommendations."""
    # Graph with 10 nodes where 8 form one giant SCC cycle
    nodes = [{"id": f"n{i}", "name": f"Node {i}"} for i in range(10)]
    relations = []
    # 8-node cycle
    for i in range(8):
        relations.append({"source": f"n{i}", "target": f"n{(i+1)%8}"})
    # 2 trailing nodes
    relations.append({"source": "n0", "target": "n8"})
    relations.append({"source": "n8", "target": "n9"})

    sccs, diag = compressor.analyze_scc_distribution(nodes, relations)
    assert diag.has_giant_scc is True
    assert diag.max_scc_ratio >= 0.30
    assert diag.mitigation_strategy is not None

def test_compression_disabled_multi_node_scc_preservation(monkeypatch):
    """When ENABLE_TARJAN_COMPRESSION=False, multi-node SCC nodes and valid edges must be preserved without supernode merging."""
    monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", False)

    nodes = [
        {"id": "A", "name": "Node A", "type": "Concept", "description": "Desc A", "source_chunk_ids": ["c1"]},
        {"id": "B", "name": "Node B", "type": "Method", "description": "Desc B", "source_chunk_ids": ["c2"]},
        {"id": "C", "name": "Node C", "type": "Metric", "description": "Desc C", "source_chunk_ids": ["c3"]},
    ]
    relations = [
        {"source": "A", "target": "B", "type": "CALLS", "weight": 2.0},
        {"source": "B", "target": "C", "type": "CALLS", "weight": 1.5},
        {"source": "C", "target": "A", "type": "CALLS", "weight": 1.0},
    ]

    compressed = compressor.compress_graph(nodes, relations)

    assert compressed["compression_ratio"] == 1.0
    output_nodes = compressed["nodes"]
    output_edges = compressed["edges"]

    # All 3 nodes must exist as standalone nodes (no supernode)
    assert len(output_nodes) == 3
    node_ids = {n["id"] for n in output_nodes}
    assert node_ids == {"A", "B", "C"}
    for n in output_nodes:
        assert n["is_supernode"] is False
        assert n["member_count"] == 1
        assert n["member_node_ids"] == [n["id"]]
        assert n["name"] in ["Node A", "Node B", "Node C"]
        assert n["description"].startswith("Desc ")

    # All 3 cycle edges must be preserved
    assert len(output_edges) == 3
    edge_pairs = {(e["source"], e["target"]) for e in output_edges}
    assert edge_pairs == {("A", "B"), ("B", "C"), ("C", "A")}

def test_compression_singleton_node(monkeypatch):
    """Verify singleton node handling with compression enabled and disabled."""
    nodes = [{"id": "single", "name": "Isolated Concept", "type": "Concept", "description": "Alone", "source_chunk_ids": ["c0"]}]
    relations = []

    for flag in [True, False]:
        monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", flag)
        res = compressor.compress_graph(nodes, relations)
        assert len(res["nodes"]) == 1
        assert res["nodes"][0]["id"] == "single"
        assert res["nodes"][0]["is_supernode"] is False
        assert res["nodes"][0]["is_connector"] is False
        assert res["nodes"][0]["member_node_ids"] == ["single"]
        assert len(res["edges"]) == 0
        assert res["compression_ratio"] == 1.0

def test_compression_disconnected_graph_with_cycles(monkeypatch):
    """Verify disconnected components with multiple cycles preserve all nodes and edges when compression disabled."""
    monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", False)

    nodes = [
        {"id": "c1_a", "name": "Cycle 1 A"},
        {"id": "c1_b", "name": "Cycle 1 B"},
        {"id": "c2_a", "name": "Cycle 2 A"},
        {"id": "c2_b", "name": "Cycle 2 B"},
        {"id": "iso", "name": "Isolated Node"},
    ]
    relations = [
        {"source": "c1_a", "target": "c1_b", "type": "LINK"},
        {"source": "c1_b", "target": "c1_a", "type": "LINK"},
        {"source": "c2_a", "target": "c2_b", "type": "LINK"},
        {"source": "c2_b", "target": "c2_a", "type": "LINK"},
    ]

    res = compressor.compress_graph(nodes, relations)
    assert len(res["nodes"]) == 5
    assert {n["id"] for n in res["nodes"]} == {"c1_a", "c1_b", "c2_a", "c2_b", "iso"}
    assert len(res["edges"]) == 4
    edge_pairs = {(e["source"], e["target"]) for e in res["edges"]}
    assert edge_pairs == {("c1_a", "c1_b"), ("c1_b", "c1_a"), ("c2_a", "c2_b"), ("c2_b", "c2_a")}

def test_compression_self_loop_handling(monkeypatch):
    """Self-loops must not create redundant quotient edges and must not drop the node."""
    nodes = [
        {"id": "self_node", "name": "Self Looping Entity", "description": "Loops back"},
        {"id": "target_node", "name": "Target Entity", "description": "Connected"}
    ]
    relations = [
        {"source": "self_node", "target": "self_node", "type": "SELF_LOOP", "weight": 1.0},
        {"source": "self_node", "target": "target_node", "type": "CONNECTS_TO", "weight": 2.0},
    ]

    for flag in [True, False]:
        monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", flag)
        res = compressor.compress_graph(nodes, relations)
        assert len(res["nodes"]) == 2
        assert {n["id"] for n in res["nodes"]} == {"self_node", "target_node"}
        # Self-loop edge inside same node (src_comp == tgt_comp) is skipped in quotient graph
        assert len(res["edges"]) == 1
        assert res["edges"][0]["source"] == "self_node"
        assert res["edges"][0]["target"] == "target_node"
        assert res["edges"][0]["weight"] == 2.0

def test_compression_enabled_still_collapses_scc(monkeypatch):
    """When ENABLE_TARJAN_COMPRESSION=True, multi-node SCC is collapsed into a supernode while connectors are preserved."""
    monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", True)

    nodes = [
        {"id": "A", "name": "Node A", "description": "Desc A"},
        {"id": "B", "name": "Node B", "description": "Desc B"},
        {"id": "C", "name": "Node C", "description": "Desc C"},
    ]
    relations = [
        {"source": "A", "target": "B", "type": "TO"},
        {"source": "B", "target": "C", "type": "TO"},
        {"source": "C", "target": "A", "type": "TO"},
    ]

    res = compressor.compress_graph(nodes, relations)
    assert len(res["nodes"]) == 1
    supernode = res["nodes"][0]
    assert supernode["is_supernode"] is True
    assert supernode["member_count"] == 3
    assert set(supernode["member_node_ids"]) == {"A", "B", "C"}
    assert res["compression_ratio"] < 1.0
    # Internal cycle edges collapsed into supernode
    assert len(res["edges"]) == 0

def test_compression_preserves_reachability_across_scc(monkeypatch):
    """Compression should preserve reachability between original nodes."""

    monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", True)

    nodes = [
        {"id": "A", "name": "Evidence A"},
        {"id": "B", "name": "Evidence B"},
        {"id": "C", "name": "Evidence C"},
        {"id": "D", "name": "Evidence D"},
    ]

    relations = [
        {"source": "A", "target": "B", "type": "LINK"},
        {"source": "B", "target": "C", "type": "LINK"},
        {"source": "C", "target": "A", "type": "LINK"},
        {"source": "C", "target": "D", "type": "LINK"},
    ]

    compressed = compressor.compress_graph(nodes, relations)

    # Map each original node to its compressed representation.
    node_to_component = {}
    for node in compressed["nodes"]:
        for member_id in node["member_node_ids"]:
            node_to_component[member_id] = node["id"]

    compressed_graph = nx.DiGraph()
    compressed_graph.add_nodes_from(node_to_component.values())
    compressed_graph.add_edges_from(
        (edge["source"], edge["target"])
        for edge in compressed["edges"]
    )

    original_graph = nx.DiGraph()
    original_graph.add_nodes_from(node["id"] for node in nodes)
    original_graph.add_edges_from(
        (rel["source"], rel["target"]) for rel in relations
    )

    # Every reachable pair of distinct original nodes must remain
    # reachable in the compressed graph.
    for source in original_graph.nodes:
        for target in original_graph.nodes:
            if source == target:
                continue

            if nx.has_path(original_graph, source, target):
                assert nx.has_path(
                    compressed_graph,
                    node_to_component[source],
                    node_to_component[target],
                ), f"Lost path: {source} -> {target}"


def test_compression_keeps_articulation_member_inside_scc_to_preserve_dag(monkeypatch):
    """An articulation vertex inside an SCC must not be split from its SCC quotient."""
    monkeypatch.setattr(settings, "ENABLE_TARJAN_COMPRESSION", True)
    nodes = [{"id": name, "name": f"Node {name}"} for name in ("A", "B", "C", "D")]
    relations = [
        {"source": "A", "target": "B"}, {"source": "B", "target": "A"},
        {"source": "B", "target": "C"}, {"source": "C", "target": "B"},
        {"source": "B", "target": "D"}, {"source": "D", "target": "B"},
    ]

    compressed = compressor.compress_graph(nodes, relations)

    assert len(compressed["nodes"]) == 1
    supernode = compressed["nodes"][0]
    assert set(supernode["member_node_ids"]) == {"A", "B", "C", "D"}
    assert supernode["is_connector"] is True
    assert supernode["connector_member_node_ids"] == ["B"]
    assert compressed["edges"] == []
    assert len(compressed["topological_order"]) == 1
