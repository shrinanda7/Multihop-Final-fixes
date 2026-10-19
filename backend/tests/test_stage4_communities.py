"""Stage 4 Verification Tests: Hierarchical Communities, Bottom-up Reports, and Caching."""
import pytest
from backend.core.clustering import clusterer
from backend.pipeline.report_generator import report_generator
from backend.core.llm_client import validate_embedding

@pytest.mark.asyncio
async def test_hierarchical_clustering_and_reports():
    """Verify multi-level hierarchy construction and bottom-up report generation."""
    nodes = [
        {"id": "gcn", "name": "GCN", "type": "Architecture", "description": "Graph Convolutional Network"},
        {"id": "spectral", "name": "Spectral Convolutions", "type": "Method", "description": "Chebyshev polynomial approximation"},
        {"id": "cora", "name": "Cora Dataset", "type": "Dataset", "description": "Citation network benchmark"},
        {"id": "citeseer", "name": "Citeseer", "type": "Dataset", "description": "Citation network benchmark"},
        {"id": "semi_sup", "name": "Semi-Supervised Learning", "type": "Task", "description": "Node classification with few labels"},
    ]
    edges = [
        {"source": "gcn", "target": "spectral", "weight": 1.0},
        {"source": "gcn", "target": "cora", "weight": 0.8},
        {"source": "cora", "target": "citeseer", "weight": 0.9},
        {"source": "gcn", "target": "semi_sup", "weight": 1.0},
    ]

    communities = clusterer.build_hierarchy(nodes, edges, "test_paper_stage4")
    assert len(communities) >= 2, "Should create at least root and level 1 communities"

    # Verify Root (Level 0)
    root_nodes = [c for c in communities.values() if c.level == 0]
    assert len(root_nodes) == 1
    root = root_nodes[0]
    assert len(root.member_node_ids) == len(nodes)
    assert len(root.children_ids) > 0

    # Verify Level 1
    l1_nodes = [c for c in communities.values() if c.level == 1]
    assert len(l1_nodes) >= 1
    for l1 in l1_nodes:
        assert l1.parent_id == root.id

    # Test report generation
    updated_comms = await report_generator.generate_reports_for_hierarchy(
        communities, nodes, edges, use_cache=False
    )
    for c in updated_comms.values():
        assert c.report != ""
        assert c.summary != ""
        # Offline mode may legitimately have no embedding. Never accept a pseudo-vector.
        assert isinstance(c.embedding, list)
        if c.embedding:
            validate_embedding(c.embedding)
