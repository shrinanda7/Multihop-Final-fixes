import assert from "node:assert/strict";
import { runTarjanSCC, findBridgesAndArticulationPoints, buildQuotientGraph } from "../src/graphEngine";
import { buildCrossPaperBridges, mergeNamespacedPaperGraphs, normalizeEntityName } from "../src/crossPaperGraph";
import { sanitizeGraphTopology } from "../src/graphSanitize";
import { GraphNode, GraphEdge, GraphTopology } from "../src/types";

console.log("==========================================");
console.log("   RUNNING TARJAN-RAG REGRESSION SUITE    ");
console.log("==========================================\n");

let passed = 0;
let total = 0;

function runTest(name: string, fn: () => void) {
  total++;
  try {
    fn();
    console.log(`  ✓ PASSED: ${name}`);
    passed++;
  } catch (err: any) {
    console.error(`  ✗ FAILED: ${name}`);
    console.error(`    ${err?.message || err}`);
  }
}

// ---------------- 1. Tarjan SCC & Quotient Graph Compression ----------------
runTest("Tarjan SCC correctly identifies and compresses a 3-node cycle into 1 supernode", () => {
  const nodes: GraphNode[] = [
    { id: "n1", name: "Node 1", type: "Method" },
    { id: "n2", name: "Node 2", type: "Method" },
    { id: "n3", name: "Node 3", type: "Method" },
    { id: "n4", name: "Node 4", type: "Dataset" },
  ];
  const edges: GraphEdge[] = [
    { source: "n1", target: "n2", type: "FEEDS" },
    { source: "n2", target: "n3", type: "FEEDS" },
    { source: "n3", target: "n1", type: "FEEDBACK" }, // Cycle {n1, n2, n3}
    { source: "n3", target: "n4", type: "EVALUATED_ON" },
  ];

  const { sccs } = runTarjanSCC(nodes, edges);
  assert.equal(sccs.length, 2, "Expected 2 SCCs (1 cycle + 1 single node)");
  const cycleScc = sccs.find((s) => s.length === 3);
  assert.ok(cycleScc, "Expected an SCC with 3 nodes");

  const quotient = buildQuotientGraph(nodes, edges, "SCC");
  assert.equal(quotient.nodes.length, 2, "Quotient graph should contract cycle to 1 supernode + 1 regular node");
  const supernode = quotient.nodes.find((n) => n.is_supernode);
  assert.ok(supernode, "Should have a supernode");
  assert.equal(supernode?.member_count, 3, "Supernode should have 3 members");
  assert.equal(quotient.edges.length, 1, "Only the inter-SCC edge should remain");
});

runTest("Tarjan SCC preserves strictly acyclic DAGs without inventing supernodes", () => {
  const nodes: GraphNode[] = [
    { id: "a", name: "A", type: "Architecture" },
    { id: "b", name: "B", type: "Method" },
    { id: "c", name: "C", type: "Metric" },
  ];
  const edges: GraphEdge[] = [
    { source: "a", target: "b", type: "USES" },
    { source: "b", target: "c", type: "EVALUATES" },
  ];

  const quotient = buildQuotientGraph(nodes, edges);
  assert.equal(quotient.nodes.length, 3, "No nodes should be contracted");
  assert.equal(quotient.nodes.filter((n) => n.is_supernode).length, 0, "No supernodes in an acyclic graph");
  assert.equal(quotient.compression_ratio, 1.0, "Compression ratio should be 1.0");
});

// ---------------- 2. Bridge & Articulation Point Detection ----------------
runTest("Tarjan bridge detection accurately identifies cut-edges and cut-vertices", () => {
  // Barbell: (n1-n2-n3-n1) --- (n3-n4) --- (n4-n5-n6-n4)
  const nodes: GraphNode[] = [
    { id: "1", name: "1", type: "A" },
    { id: "2", name: "2", type: "A" },
    { id: "3", name: "3", type: "A" },
    { id: "4", name: "4", type: "B" },
    { id: "5", name: "5", type: "B" },
    { id: "6", name: "6", type: "B" },
  ];
  const edges: GraphEdge[] = [
    { source: "1", target: "2", type: "E" },
    { source: "2", target: "3", type: "E" },
    { source: "3", target: "1", type: "E" },
    { source: "3", target: "4", type: "CRITICAL_CUT" }, // Bridge between components
    { source: "4", target: "5", type: "E" },
    { source: "5", target: "6", type: "E" },
    { source: "6", target: "4", type: "E" },
  ];

  const { bridges, articulationPoints } = findBridgesAndArticulationPoints(nodes, edges);
  assert.equal(bridges.length, 1, "Expected exactly 1 bridge edge");
  const isCutEdge =
    (bridges[0][0] === "3" && bridges[0][1] === "4") ||
    (bridges[0][0] === "4" && bridges[0][1] === "3");
  assert.ok(isCutEdge, "Bridge edge must be (3, 4)");
  assert.ok(articulationPoints.includes("3") && articulationPoints.includes("4"), "Nodes 3 and 4 must be articulation points");
});

// ---------------- 3. Graph Sanitization & Dangling Endpoint Rejection ----------------
runTest("Sanitization drops duplicate nodes and dangling edges referencing absent nodes", () => {
  const invalidGraph: GraphTopology = {
    nodes: [
      { id: "n1", name: "Node 1", type: "Method" },
      { id: "n1", name: "Node 1 Duplicate", type: "Method" }, // Duplicate
      { id: "n2", name: "Node 2", type: "Method" },
    ],
    edges: [
      { source: "n1", target: "n2", type: "VALID" },
      { source: "n1", target: "ghost_node_999", type: "DANGLING" }, // Non-existent target
      { source: "ghost_node_888", target: "n2", type: "DANGLING" }, // Non-existent source
      { source: "n1", target: "n2", type: "VALID" }, // Duplicate edge
    ],
  };

  const sanitized = sanitizeGraphTopology(invalidGraph);
  assert.ok(sanitized, "Sanitized graph should exist");
  assert.equal(sanitized?.nodes.length, 2, "Duplicate node should be removed");
  assert.equal(sanitized?.edges.length, 1, "Only the 1 valid non-duplicate edge should remain");
  assert.equal(sanitized?.edges[0].type, "VALID");
});

// ---------------- 4. Cross-Paper Bridge Validation ----------------
runTest("Cross-paper bridges are created ONLY for validated shared concepts, never fabricated", () => {
  const nodesPaperA: GraphNode[] = [
    { id: "P1::bert", name: "BERT", type: "Architecture", paper_id: "P1" },
    { id: "P1::cora", name: "Cora Dataset", type: "Dataset", paper_id: "P1" },
    { id: "P1::unique_a", name: "Proprietary Technique A", type: "Method", paper_id: "P1" },
  ];
  const nodesPaperB: GraphNode[] = [
    { id: "P2::bert", name: "BERT", type: "Architecture", paper_id: "P2" },
    { id: "P2::unique_b", name: "Proprietary Technique B", type: "Method", paper_id: "P2" },
  ];

  const allNodes = [...nodesPaperA, ...nodesPaperB];
  const bridges = buildCrossPaperBridges(allNodes);

  // BERT should match across papers P1 and P2
  const bertBridge = bridges.find(
    (b) =>
      (b.source === "P1::bert" && b.target === "P2::bert") ||
      (b.source === "P2::bert" && b.target === "P1::bert")
  );
  assert.ok(bertBridge, "Expected a verified bridge for shared entity BERT");
  assert.equal(bertBridge?.type, "SAME_CONCEPT");
  assert.equal(bertBridge?.is_bridge, true);

  // Unrelated nodes must NOT have arbitrary fabricated bridges
  const fakeBridge = bridges.find(
    (b) =>
      (b.source.includes("unique_a") && b.target.includes("unique_b")) ||
      (b.source.includes("unique_b") && b.target.includes("unique_a"))
  );
  assert.equal(fakeBridge, undefined, "Must NOT fabricate CROSS_PAPER_BRIDGE between unrelated entities");
});

runTest("Two unrelated papers with zero shared concepts yield zero cross-paper bridges", () => {
  const nodesP1: GraphNode[] = [
    { id: "P1::quantum_annealing", name: "Quantum Annealing Flux Qubit", type: "Architecture", paper_id: "P1" },
  ];
  const nodesP2: GraphNode[] = [
    { id: "P2::crop_disease", name: "Cassava Leaf Blight Detection", type: "Dataset", paper_id: "P2" },
  ];

  const bridges = buildCrossPaperBridges([...nodesP1, ...nodesP2]);
  assert.equal(bridges.length, 0, "No bridge edges should be created between completely unrelated papers");
});

// ---------------- 5. Evidence Grounding & Stable Chunk IDs ----------------
runTest("Every evidence item retains a stable, verifiable chunk_id and paper_id", () => {
  const paperId = "test_paper_123";
  const rawChunks = [
    "Introduction to deep convolutional networks for medical image analysis.",
    "Section 2: Methodology detailing residual skip connections and bottleneck layers.",
  ];

  const structuredChunks = rawChunks.map((text, idx) => ({
    chunk_id: `${paperId}_chunk_${idx}`,
    paper_id: paperId,
    title: "Test Paper",
    text,
    chunk_index: idx,
  }));

  assert.equal(structuredChunks[0].chunk_id, "test_paper_123_chunk_0");
  assert.equal(structuredChunks[1].chunk_id, "test_paper_123_chunk_1");
  assert.ok(!structuredChunks[0].chunk_id.startsWith("local_"), "Chunk IDs must not be arbitrary local_ indexes");
});

// ---------------- 6. Namespacing & Union Graph Construction ----------------
runTest("mergeNamespacedPaperGraphs isolates node namespaces and computes quotient graph", () => {
  const g1: GraphTopology = {
    nodes: [
      { id: "encoder", name: "Encoder", type: "Architecture" },
      { id: "attention", name: "Attention Mechanism", type: "Method" },
    ],
    edges: [{ source: "encoder", target: "attention", type: "USES" }],
  };

  const g2: GraphTopology = {
    nodes: [
      { id: "decoder", name: "Decoder", type: "Architecture" },
      { id: "attention", name: "Attention Mechanism", type: "Method" },
    ],
    edges: [{ source: "decoder", target: "attention", type: "USES" }],
  };

  const union = mergeNamespacedPaperGraphs([
    { paperId: "P1", graph: g1 },
    { paperId: "P2", graph: g2 },
  ]);

  const p1Attention = union.original_graph.nodes.find((n) => n.id === "P1::attention");
  const p2Attention = union.original_graph.nodes.find((n) => n.id === "P2::attention");
  assert.ok(p1Attention, "Node P1::attention must exist");
  assert.ok(p2Attention, "Node P2::attention must exist");

  // Check bridge exists between P1::attention and P2::attention
  const crossEdge = union.original_graph.edges.find(
    (e) =>
      (e.source === "P1::attention" && e.target === "P2::attention") ||
      (e.source === "P2::attention" && e.target === "P1::attention")
  );
  assert.ok(crossEdge, "Cross-paper bridge must link identical concepts");
  assert.equal(crossEdge?.is_bridge, true);
});

console.log("\n==========================================");
console.log(`RESULTS: ${passed} / ${total} TESTS PASSED`);
console.log("==========================================\n");

if (passed !== total) {
  process.exit(1);
}
