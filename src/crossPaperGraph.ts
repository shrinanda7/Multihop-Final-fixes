import { GraphEdge, GraphNode, GraphTopology } from "./types";
import { buildQuotientGraph } from "./graphEngine";

const STOP = new Set([
  "the", "and", "for", "with", "from", "that", "this", "what", "how", "why",
  "into", "onto", "using", "used", "based", "via", "over", "under", "between",
  "paper", "model", "method", "approach", "network", "system", "layer",
]);

const GENERIC = new Set([
  "method", "model", "graph", "paper", "approach", "result", "dataset",
  "experiment", "framework", "network", "system", "layer", "module",
  "baseline", "performance", "architecture", "algorithm", "embedding",
]);

const DISTINCTIVE = new Set([
  "attention", "transformer", "bert", "gpt", "rag", "gcn", "gat", "graphsage",
  "sage", "deepwalk", "node2vec", "convolution", "inductive", "transductive",
  "retriever", "generator", "walk", "embedding", "message", "aggregation",
  "self", "multihead", "positional", "encoder", "decoder",
]);

export function originalNodeId(id: string): string {
  const idx = id.lastIndexOf("::");
  return idx >= 0 ? id.slice(idx + 2) : id;
}

export function paperIdFromNode(node: GraphNode): string {
  if (node.paper_id) return node.paper_id;
  const idx = node.id.indexOf("::");
  return idx > 0 ? node.id.slice(0, idx) : "";
}

export function normalizeEntityName(value: string): string {
  return (value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

export function entityTokens(value: string): Set<string> {
  return new Set(
    normalizeEntityName(value)
      .split(/\s+/)
      .filter((tok) => tok.length > 2 && !STOP.has(tok) && !GENERIC.has(tok))
  );
}

export function namespaceNode(paperId: string, node: GraphNode): GraphNode {
  const original = originalNodeId(node.id);
  return {
    ...node,
    id: `${paperId}::${original}`,
    paper_id: paperId,
    member_node_ids: node.member_node_ids?.map(
      (id) => `${paperId}::${originalNodeId(id)}`
    ),
    member_nodes: undefined,
  };
}

function nodesRelated(a: GraphNode, b: GraphNode): { type: string; weight: number } | null {
  const na = normalizeEntityName(a.name);
  const nb = normalizeEntityName(b.name);
  if (na.length < 3 || nb.length < 3) return null;
  if (GENERIC.has(na) || GENERIC.has(nb)) return null;

  if (na === nb) {
    return { type: "SAME_CONCEPT", weight: 1.0 };
  }

  const ta = entityTokens(`${a.name} ${a.type || ""} ${a.description || ""}`);
  const tb = entityTokens(`${b.name} ${b.type || ""} ${b.description || ""}`);
  if (ta.size === 0 || tb.size === 0) return null;

  let overlap = 0;
  const shared: string[] = [];
  for (const tok of ta) {
    if (tb.has(tok)) {
      if (DISTINCTIVE.has(tok)) {
        overlap += 2;
        shared.push(tok);
      }
    }
  }
  if (overlap === 0 || shared.length === 0) return null;

  const denom = Math.min(ta.size, tb.size);
  if (overlap / denom >= 0.5) {
    return { type: "SHARED_PARADIGM", weight: Math.min(0.95, 0.5 + overlap * 0.1) };
  }
  return null;
}

export function buildCrossPaperBridges(nodes: GraphNode[]): GraphEdge[] {
  const byPaper = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const pid = paperIdFromNode(node);
    if (!pid) continue;
    const list = byPaper.get(pid) || [];
    list.push(node);
    byPaper.set(pid, list);
  }

  const paperIds = Array.from(byPaper.keys());
  const bridges: GraphEdge[] = [];
  const seen = new Set<string>();

  const push = (edge: GraphEdge) => {
    const key = `${edge.source}->${edge.target}:${edge.type}`;
    if (seen.has(key) || edge.source === edge.target) return;
    seen.add(key);
    bridges.push(edge);
  };

  // Only create bridges where genuine shared concepts exist across papers
  for (let i = 0; i < paperIds.length; i++) {
    for (let j = i + 1; j < paperIds.length; j++) {
      const leftNodes = byPaper.get(paperIds[i]) || [];
      const rightNodes = byPaper.get(paperIds[j]) || [];

      for (const left of leftNodes) {
        for (const right of rightNodes) {
          const related = nodesRelated(left, right);
          if (!related) continue;

          push({
            source: left.id,
            target: right.id,
            type: related.type,
            is_bridge: true,
            weight: related.weight,
            paper_ids: [paperIds[i], paperIds[j]],
          });
          push({
            source: right.id,
            target: left.id,
            type: related.type,
            is_bridge: true,
            weight: related.weight,
            paper_ids: [paperIds[j], paperIds[i]],
          });
        }
      }
    }
  }

  return bridges;
}

export function mergeNamespacedPaperGraphs(
  papers: { paperId: string; graph: GraphTopology }[]
): { original_graph: GraphTopology; compressed_graph: GraphTopology } {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeIds = new Set<string>();
  const edgeKeys = new Set<string>();

  for (const { paperId, graph } of papers) {
    for (const node of graph.nodes || []) {
      if (!node?.id) continue;
      const namespaced = namespaceNode(paperId, node);
      if (nodeIds.has(namespaced.id)) continue;
      nodeIds.add(namespaced.id);
      nodes.push(namespaced);
    }
    for (const edge of graph.edges || []) {
      if (!edge?.source || !edge?.target) continue;
      const source = `${paperId}::${originalNodeId(edge.source)}`;
      const target = `${paperId}::${originalNodeId(edge.target)}`;
      const key = `${source}->${target}:${edge.type || ""}`;
      if (!nodeIds.has(source) || !nodeIds.has(target) || edgeKeys.has(key)) continue;
      edgeKeys.add(key);
      edges.push({ ...edge, source, target });
    }
  }

  for (const bridge of buildCrossPaperBridges(nodes)) {
    const key = `${bridge.source}->${bridge.target}:${bridge.type}`;
    if (edgeKeys.has(key)) continue;
    edgeKeys.add(key);
    edges.push(bridge);
  }

  const compressed = buildQuotientGraph(nodes, edges, "SCC");
  return {
    original_graph: {
      nodes,
      edges,
      node_count: nodes.length,
      edge_count: edges.length,
    },
    compressed_graph: compressed,
  };
}
