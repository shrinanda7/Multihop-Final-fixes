import { GraphTopology, GraphNode, GraphEdge, SCCDiagnostics } from "./types";

/**
 * Executes Tarjan's Strongly Connected Components (SCC) algorithm on a directed graph.
 * Time Complexity: O(|V| + |E|)
 */
export function runTarjanSCC(nodes: GraphNode[], edges: GraphEdge[]): {
  sccs: string[][];
  nodeToScc: Map<string, number>;
} {
  let index = 0;
  const indices = new Map<string, number>();
  const lowlinks = new Map<string, number>();
  const onStack = new Map<string, boolean>();
  const stack: string[] = [];
  const sccs: string[][] = [];
  const nodeToScc = new Map<string, number>();

  // Build adjacency list for directed edges
  const adj = new Map<string, string[]>();
  for (const n of nodes) {
    adj.set(n.id, []);
  }
  for (const e of edges) {
    if (adj.has(e.source)) {
      adj.get(e.source)!.push(e.target);
    }
  }

  function strongConnect(u: string) {
    indices.set(u, index);
    lowlinks.set(u, index);
    index++;
    stack.push(u);
    onStack.set(u, true);

    const neighbors = adj.get(u) || [];
    for (const v of neighbors) {
      if (!indices.has(v)) {
        strongConnect(v);
        lowlinks.set(u, Math.min(lowlinks.get(u)!, lowlinks.get(v)!));
      } else if (onStack.get(v)) {
        lowlinks.set(u, Math.min(lowlinks.get(u)!, indices.get(v)!));
      }
    }

    if (lowlinks.get(u) === indices.get(u)) {
      const scc: string[] = [];
      let w = "";
      while (w !== u) {
        w = stack.pop()!;
        onStack.set(w, false);
        scc.push(w);
      }
      const sccIndex = sccs.length;
      sccs.push(scc);
      for (const member of scc) {
        nodeToScc.set(member, sccIndex);
      }
    }
  }

  for (const n of nodes) {
    if (!indices.has(n.id)) {
      strongConnect(n.id);
    }
  }

  return { sccs, nodeToScc };
}

/**
 * Finds all bridge edges and articulation points using Tarjan / Hopcroft DFS.
 * Considers graph connectivity to protect critical cut-edges.
 */
export function findBridgesAndArticulationPoints(
  nodes: GraphNode[],
  edges: GraphEdge[]
): {
  bridges: [string, string][];
  articulationPoints: string[];
} {
  let timer = 0;
  const tin = new Map<string, number>();
  const low = new Map<string, number>();
  const visited = new Set<string>();
  const bridges: [string, string][] = [];
  const articulationPoints = new Set<string>();

  // Build undirected adjacency list for cut-edge connectivity
  const adj = new Map<string, string[]>();
  for (const n of nodes) {
    adj.set(n.id, []);
  }
  for (const e of edges) {
    if (adj.has(e.source) && adj.has(e.target)) {
      adj.get(e.source)!.push(e.target);
      adj.get(e.target)!.push(e.source);
    }
  }

  function dfs(u: string, p = "") {
    visited.add(u);
    tin.set(u, timer);
    low.set(u, timer);
    timer++;
    let children = 0;

    for (const to of adj.get(u) || []) {
      if (to === p) continue;
      if (visited.has(to)) {
        low.set(u, Math.min(low.get(u)!, tin.get(to)!));
      } else {
        dfs(to, u);
        low.set(u, Math.min(low.get(u)!, low.get(to)!));
        if (low.get(to)! > tin.get(u)!) {
          bridges.push([u, to]);
        }
        if (low.get(to)! >= tin.get(u)! && p !== "") {
          articulationPoints.add(u);
        }
        children++;
      }
    }
    if (p === "" && children > 1) {
      articulationPoints.add(u);
    }
  }

  for (const n of nodes) {
    if (!visited.has(n.id)) {
      dfs(n.id);
    }
  }

  return {
    bridges,
    articulationPoints: Array.from(articulationPoints)
  };
}

/**
 * Condenses a knowledge graph into its Tarjan quotient graph.
 * Any SCC with > 1 node is contracted into a single supernode.
 * Internal cyclic edges are collapsed; boundary edges and bridges are preserved.
 */
export function buildQuotientGraph(
  originalNodes: GraphNode[],
  originalEdges: GraphEdge[],
  supernodeNamePrefix = "SCC"
): GraphTopology {
  const { sccs, nodeToScc } = runTarjanSCC(originalNodes, originalEdges);
  const { bridges, articulationPoints } = findBridgesAndArticulationPoints(originalNodes, originalEdges);

  const nodeMap = new Map<string, GraphNode>();
  for (const n of originalNodes) {
    nodeMap.set(n.id, n);
  }

  const bridgeSet = new Set(bridges.map(([u, v]) => `${u}---${v}`));
  const apSet = new Set(articulationPoints);

  const compressedNodes: GraphNode[] = [];
  const sccIdMap = new Map<number, string>(); // sccIndex -> supernodeId or single nodeId

  // Form nodes
  for (let i = 0; i < sccs.length; i++) {
    const members = sccs[i];
    if (members.length > 1) {
      const memberObjs = members.map((id) => nodeMap.get(id)!).filter(Boolean);
      const superId = `scc_cluster_${i}_${members[0].slice(0, 8)}`;
      sccIdMap.set(i, superId);

      const superName = `${supernodeNamePrefix} [${memberObjs[0]?.name?.split(" ")[0] || "Core"} Cyclic Loop]`;
      compressedNodes.push({
        id: superId,
        name: superName,
        type: "Supernode",
        description: `Contracted quotient supernode containing ${members.length} cyclically interdependent components: ${memberObjs.map((m) => m.name).join(", ")}.`,
        is_supernode: true,
        member_count: members.length,
        member_node_ids: members,
        member_nodes: memberObjs
      });
    } else {
      const singleId = members[0];
      sccIdMap.set(i, singleId);
      const nodeObj = nodeMap.get(singleId);
      if (nodeObj) {
        compressedNodes.push({
          ...nodeObj,
          is_connector: apSet.has(singleId) || nodeObj.is_connector
        });
      }
    }
  }

  // Form edges in quotient graph
  const edgeKeySet = new Set<string>();
  const compressedEdges: GraphEdge[] = [];

  for (const e of originalEdges) {
    const sccU = nodeToScc.get(e.source);
    const sccV = nodeToScc.get(e.target);

    if (sccU === undefined || sccV === undefined) continue;

    // If within same SCC, internal cycle is collapsed into the supernode
    if (sccU === sccV) continue;

    const sourceCompId = sccIdMap.get(sccU)!;
    const targetCompId = sccIdMap.get(sccV)!;

    if (sourceCompId === targetCompId) continue;

    const edgeKey = `${sourceCompId}->${targetCompId}`;
    if (!edgeKeySet.has(edgeKey)) {
      edgeKeySet.add(edgeKey);
      const isBridge =
        bridgeSet.has(`${e.source}---${e.target}`) ||
        bridgeSet.has(`${e.target}---${e.source}`) ||
        e.is_bridge;

      compressedEdges.push({
        source: sourceCompId,
        target: targetCompId,
        type: e.type,
        weight: e.weight || 1.0,
        is_bridge: isBridge
      });
    }
  }

  const sccSizes = sccs.map((s) => s.length);
  const maxSccSize = Math.max(...sccSizes, 1);
  const totalNodes = originalNodes.length;
  const maxSccRatio = totalNodes > 0 ? maxSccSize / totalNodes : 0;
  const compressionRatio = totalNodes > 0 ? compressedNodes.length / totalNodes : 1.0;

  const diagnostics: SCCDiagnostics = {
    total_nodes: totalNodes,
    num_sccs: sccs.length,
    scc_sizes: sccSizes,
    max_scc_size: maxSccSize,
    max_scc_ratio: parseFloat(maxSccRatio.toFixed(3)),
    has_giant_scc: maxSccRatio >= 0.35
  };

  return {
    nodes: compressedNodes,
    edges: compressedEdges,
    node_count: compressedNodes.length,
    edge_count: compressedEdges.length,
    articulation_points: Array.from(apSet),
    bridges,
    compression_ratio: parseFloat(compressionRatio.toFixed(3)),
    diagnostics
  };
}
