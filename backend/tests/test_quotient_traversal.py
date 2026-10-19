from backend.search.quotient_traversal import select_quotient_subgraph


def test_selects_directed_path_and_preserves_connector():
    graph = {
        "nodes": [
            {
                "id": "MODEL",
                "name": "Alpha Model",
                "member_node_ids": ["A"],
                "source_chunk_ids": ["chunk-a"],
            },
            {
                "id": "METHOD",
                "name": "Bridge Method",
                "member_node_ids": ["B"],
                "source_chunk_ids": ["chunk-b"],
            },
            {
                "id": "RESULT",
                "name": "Gamma Result",
                "member_node_ids": ["C"],
                "source_chunk_ids": ["chunk-c"],
            },
            {
                "id": "UNRELATED",
                "name": "Unrelated Astronomy",
                "member_node_ids": ["D"],
            },
        ],
        "edges": [
            {
                "source": "MODEL",
                "target": "METHOD",
                "type": "USES",
                "source_chunk_ids": ["chunk-ab"],
            },
            {
                "source": "METHOD",
                "target": "RESULT",
                "type": "PRODUCES",
                "source_chunk_ids": ["chunk-bc"],
            },
        ],
        "articulation_points": ["B"],
    }

    result = select_quotient_subgraph(
        "Alpha Model Gamma Result",
        graph,
    )

    selected = set(result["selected_node_ids"])

    assert {"MODEL", "METHOD", "RESULT"} <= selected
    assert "UNRELATED" not in selected
    assert result["connector_component_ids"] == ["METHOD"]

    assert [
        (edge["source"], edge["target"])
        for edge in result["edges"]
    ] == [
        ("MODEL", "METHOD"),
        ("METHOD", "RESULT"),
    ]

    assert set(result["selected_chunk_ids"]) >= {
        "chunk-a",
        "chunk-b",
        "chunk-c",
        "chunk-ab",
        "chunk-bc",
    }


def test_does_not_invent_a_path_against_edge_direction():
    graph = {
        "nodes": [
            {"id": "A", "name": "Alpha Model"},
            {"id": "B", "name": "Gamma Result"},
        ],
        "edges": [
            {"source": "B", "target": "A", "type": "REVERSE"},
        ],
        "articulation_points": [],
    }

    result = select_quotient_subgraph("Alpha Model Gamma Result", graph)

    # Both seeds are retained, but the original edge direction is unchanged.
    assert set(result["seed_node_ids"]) == {"A", "B"}
    assert result["edges"][0]["source"] == "B"
    assert result["edges"][0]["target"] == "A"


def test_empty_graph_returns_empty_selection():
    result = select_quotient_subgraph("How does the model work?", {})

    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["selected_chunk_ids"] == []

def test_bridge_ablation_disables_connector_only_fallback():
    graph = {
        "nodes": [
            {"id": "A", "name": "Alpha Model", "member_node_ids": ["A"]},
            {"id": "B", "name": "Bridge Connector", "member_node_ids": ["B"]},
            {"id": "C", "name": "Gamma Result", "member_node_ids": ["C"]},
        ],
        "edges": [{"source": "A", "target": "B"}],
        "articulation_points": ["B"],
    }

    enabled = select_quotient_subgraph("Alpha Model Gamma Result", graph, bridge_aware=True)
    disabled = select_quotient_subgraph("Alpha Model Gamma Result", graph, bridge_aware=False)

    assert "B" in enabled["selected_node_ids"]
    assert "B" not in disabled["selected_node_ids"]


def test_budget_ablation_disables_selected_node_cap():
    graph = {
        "nodes": [
            {"id": "A", "name": "Alpha"},
            {"id": "B", "name": "Beta"},
            {"id": "C", "name": "Gamma"},
            {"id": "D", "name": "Delta"},
        ],
        "edges": [],
        "articulation_points": [],
    }

    budgeted = select_quotient_subgraph(
        "Alpha Beta Gamma Delta", graph,
        max_selected_nodes=2, budget_aware=True,
    )
    unbudgeted = select_quotient_subgraph(
        "Alpha Beta Gamma Delta", graph,
        max_selected_nodes=2, budget_aware=False,
    )

    assert len(budgeted["selected_node_ids"]) == 2
    assert len(unbudgeted["selected_node_ids"]) == 4
