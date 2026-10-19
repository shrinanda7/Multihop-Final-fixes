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

const KNOWN_BRIDGES: { source: string; target: string; type: string }[] = [
  { source: "transformer", target: "bert", type: "PRETRAINED_ARCHITECTURE" },
  { source: "transformer", target: "rag", type: "GENERATOR_BACKBONE" },
  { source: "gcn", target: "graphsage", type: "SPECTRAL_TO_INDUCTIVE" },
  { source: "gcn", target: "gat", type: "ANISOTROPIC_ATTENTION_EVOLUTION" },
  { source: "graphsage", target: "gat", type: "NEIGHBORHOOD_AGGREGATION" },
  { source: "deepwalk", target: "node2vec", type: "ALGORITHMIC_EVOLUTION" },
  { source: "bert", target: "cot", type: "PRETRAINED_REASONING_BASE" },
  { source: "rag", target: "cot", type: "GROUNDED_MULTI_STEP_REASONING" },
  { source: "mha", target: "gat", type: "ATTENTION_MECHANISM_TRANSFER" },
  { source: "scaled_dot", target: "gat", type: "ATTENTION_SCORING" },
];

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

function namespaceNode(paperId: string, node: GraphNode): GraphNode {
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

const CONCEPT_FAMILIES: string[][] = [
  ["attention", "transformer", "bert", "multihead", "self", "gat", "positional", "encoder", "decoder"],
  ["gcn", "convolution", "spectral", "graphsage", "gat", "message", "aggregation", "inductive", "sage"],
  ["deepwalk", "node2vec", "walk", "embedding", "skipgram"],
  ["rag", "retriever", "retrieval", "generator", "bert", "dense"],
  ["cot", "reasoning", "prompt", "chain"],
];

function familyHits(tokens: Set<string>): Set<number> {
  const hits = new Set<number>();
  CONCEPT_FAMILIES.forEach((family, index) => {
    if (family.some((tok) => tokens.has(tok))) hits.add(index);
  });
  return hits;
}

function nodesRelated(a: GraphNode, b: GraphNode): { type: string; weight: number } | null {
  const na = normalizeEntityName(a.name);
  const nb = normalizeEntityName(b.name);
  if (na.length < 3 || nb.length < 3) return null;
  if (GENERIC.has(na) || GENERIC.has(nb)) return null;

  if (na === nb) {
    return { type: "SAME_ENTITY_NAME", weight: 1 };
  }

  const ta = entityTokens(`${a.name} ${a.type || ""} ${a.description || ""}`);
  const tb = entityTokens(`${b.name} ${b.type || ""} ${b.description || ""}`);
  if (ta.size === 0 || tb.size === 0) return null;

  const familiesA = familyHits(ta);
  const familiesB = familyHits(tb);
  for (const fam of familiesA) {
    if (familiesB.has(fam)) {
      return { type: "SHARED_PARADIGM", weight: 0.88 };
    }
  }

  let overlap = 0;
  const shared: string[] = [];
  for (const tok of ta) {
    if (tb.has(tok)) {
      overlap += DISTINCTIVE.has(tok) ? 2 : 1;
      shared.push(tok);
    }
  }
  if (overlap === 0) return null;

  const denom = Math.min(ta.size, tb.size);
  if (shared.some((tok) => DISTINCTIVE.has(tok)) || overlap / denom >= 0.3) {
    return { type: "RELATED_CONCEPT", weight: Math.min(0.95, 0.45 + overlap * 0.12) };
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

  const byOriginal = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const orig = originalNodeId(node.id);
    const list = byOriginal.get(orig) || [];
    list.push(node);
    byOriginal.set(orig, list);
  }

  for (const known of KNOWN_BRIDGES) {
    const lefts = byOriginal.get(known.source) || [];
    const rights = byOriginal.get(known.target) || [];
    for (const left of lefts) {
      for (const right of rights) {
        if (paperIdFromNode(left) === paperIdFromNode(right)) continue;
        push({
          source: left.id,
          target: right.id,
          type: known.type,
          is_bridge: true,
          weight: 0.95,
        });
      }
    }
  }

  for (let i = 0; i < paperIds.length; i++) {
    for (let j = i + 1; j < paperIds.length; j++) {
      const leftNodes = (byPaper.get(paperIds[i]) || []).slice(0, 48);
      const rightNodes = (byPaper.get(paperIds[j]) || []).slice(0, 48);
      let pairLinks = 0;

      for (const left of leftNodes) {
        for (const right of rightNodes) {
          if (pairLinks >= 14) break;
          const related = nodesRelated(left, right);
          if (!related) continue;
          push({
            source: left.id,
            target: right.id,
            type: related.type,
            is_bridge: true,
            weight: related.weight,
          });
          push({
            source: right.id,
            target: left.id,
            type: related.type,
            is_bridge: true,
            weight: related.weight * 0.9,
          });
          pairLinks += 1;
        }
      }

      const hubs = (nodesForPaper: GraphNode[]) => {
        const preferred = nodesForPaper.filter((n) =>
          ["Architecture", "Method", "Theory"].includes(n.type)
        );
        return (preferred.length ? preferred : nodesForPaper).slice(0, 3);
      };
      if (pairLinks === 0) {
        const leftHubs = hubs(leftNodes);
        const rightHubs = hubs(rightNodes);
        for (let h = 0; h < Math.min(leftHubs.length, rightHubs.length); h++) {
          push({
            source: leftHubs[h].id,
            target: rightHubs[h].id,
            type: "CROSS_PAPER_BRIDGE",
            is_bridge: true,
            weight: 0.7,
          });
          pairLinks += 1;
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
