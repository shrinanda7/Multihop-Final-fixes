import { GraphEdge, GraphNode, GraphTopology } from "./types";

/** Drop duplicate / dangling graph pieces so Cytoscape cannot throw. */
export function sanitizeGraphTopology(
  graph: GraphTopology | null | undefined
): GraphTopology | null {
  if (!graph || !Array.isArray(graph.nodes)) return graph ?? null;

  const nodes: GraphNode[] = [];
  const seen = new Set<string>();

  for (const node of graph.nodes) {
    if (!node?.id) continue;
    const id = String(node.id);
    if (seen.has(id)) continue;
    seen.add(id);
    nodes.push({ ...node, id });
  }

  const edges: GraphEdge[] = [];
  const edgeKeys = new Set<string>();

  for (const edge of graph.edges || []) {
    if (!edge?.source || !edge?.target) continue;
    if (!seen.has(edge.source) || !seen.has(edge.target)) continue;
    const key = `${edge.source}->${edge.target}:${edge.type || ""}`;
    if (edgeKeys.has(key)) continue;
    edgeKeys.add(key);
    edges.push(edge);
  }

  return {
    ...graph,
    nodes,
    edges,
    node_count: nodes.length,
    edge_count: edges.length,
  };
}
