import { GraphNode, GraphEdge, GraphTopology } from "./types";
import { buildQuotientGraph, findBridgesAndArticulationPoints } from "./graphEngine";

export interface ExtractedGraphResult {
  original_graph: GraphTopology;
  compressed_graph: GraphTopology;
  source: "groq" | "ollama" | "heuristic";
}

interface RawLLMNode {
  id?: string;
  name: string;
  type?: string;
  description?: string;
}

interface RawLLMEdge {
  source: string;
  target: string;
  type?: string;
  description?: string;
  weight?: number;
}

interface RawExtractionResponse {
  entities: RawLLMNode[];
  relationships: RawLLMEdge[];
}

/**
 * Extracts a structured scientific knowledge graph from research paper text.
 * Multi-tier engine: Groq (ultra-fast) -> Local Ollama -> Heuristic domain parser.
 */
export async function extractPaperKnowledgeGraph(
  paperId: string,
  title: string,
  fullText: string
): Promise<ExtractedGraphResult> {
  const slug = paperId.replace(/[^a-zA-Z0-9]/g, "_").toLowerCase();

  // Prepare focused context window (first ~6000 chars covers title, abstract, intro, and methodology)
  const sampleText = fullText.slice(0, 7500);

  // 1. Try Groq Cloud LLM (openai/gpt-oss-120b or qwen/qwen3.8-27b)
  const groqKey = process.env.GROQ_API_KEY;
  if (groqKey && sampleText.length > 80) {
    try {
      const groqResult = await extractWithGroq(groqKey, title, sampleText, slug);
      if (groqResult && groqResult.nodes.length >= 6) {
        console.log(`[Extractor] Groq extracted ${groqResult.nodes.length} entities & ${groqResult.edges.length} relations for paper '${title}'`);
        const finalized = finalizeGraph(groqResult.nodes, groqResult.edges, title);
        return { ...finalized, source: "groq" };
      }
    } catch (err: any) {
      console.warn(`[Extractor] Groq extraction warning: ${err?.message?.slice(0, 100)}. Trying fallback...`);
    }
  }

  // 2. Try Local Ollama
  const ollamaBaseUrl = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
  if (sampleText.length > 80) {
    try {
      const ollamaResult = await extractWithOllama(ollamaBaseUrl, title, sampleText, slug);
      if (ollamaResult && ollamaResult.nodes.length >= 6) {
        console.log(`[Extractor] Ollama extracted ${ollamaResult.nodes.length} entities & ${ollamaResult.edges.length} relations for paper '${title}'`);
        const finalized = finalizeGraph(ollamaResult.nodes, ollamaResult.edges, title);
        return { ...finalized, source: "ollama" };
      }
    } catch (err: any) {
      console.warn(`[Extractor] Ollama extraction warning: ${err?.message?.slice(0, 100)}. Falling back to heuristic...`);
    }
  }

  // 3. Fallback: High-precision semantic heuristic extractor from actual text
  console.log(`[Extractor] Using heuristic extraction from actual document text for '${title}'`);
  const heuristicResult = extractHeuristically(title, fullText, slug);
  const finalized = finalizeGraph(heuristicResult.nodes, heuristicResult.edges, title);
  return { ...finalized, source: "heuristic" };
}

async function extractWithGroq(
  apiKey: string,
  title: string,
  text: string,
  slug: string
): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] } | null> {
  const models = [
    process.env.GROQ_MODEL || "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b"
  ];

  const prompt = `Extract a comprehensive, interconnected scientific knowledge graph from the research paper text below.

PAPER TITLE: "${title}"

REQUIREMENTS:
1. Extract 12 to 20 key domain entities:
   - Types must be one of: "Architecture", "Method", "Dataset", "Metric", "Concept", "Theory", "Task"
   - Include the core model/architecture, specific sub-modules, datasets, evaluation metrics, and comparative baseline models.
2. Extract 15 to 25 directed relationships connecting these entities:
   - Predicates in UPPER_SNAKE_CASE (e.g., USES, EVALUATED_ON, PROPOSES, IMPROVES_UPON, TRAINED_ON, OPTIMIZES_METRIC, CROSS_ATTENTION, FEEDS_INTO).
   - Include bidirectional/cyclic interaction loops between internal processing layers (e.g. feedback, normalization, residual).
   - Mark evaluation links to datasets and metrics.

OUTPUT FORMAT:
Strict JSON only with this schema:
{
  "entities": [
    {"id": "e1", "name": "Exact Entity Name", "type": "Architecture", "description": "Concise summary"}
  ],
  "relationships": [
    {"source": "e1", "target": "e2", "type": "USES", "description": "Details", "weight": 0.9}
  ]
}

RESEARCH TEXT:
${text}`;

  for (const model of models) {
    try {
      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: [
            {
              role: "system",
              content: "You are an expert scientific knowledge graph constructor. Output ONLY valid JSON matching the requested schema.",
            },
            { role: "user", content: prompt },
          ],
          response_format: { type: "json_object" },
          temperature: 0.1,
        }),
      });

      if (!response.ok) {
        continue;
      }

      const resData = (await response.json()) as any;
      const rawContent = resData.choices?.[0]?.message?.content;
      if (!rawContent) continue;

      const parsed: RawExtractionResponse = JSON.parse(rawContent);
      const graph = processRawExtraction(parsed, slug, title);
      if (graph && graph.nodes.length >= 6) {
        return graph;
      }
    } catch {
      continue;
    }
  }

  return null;
}

async function extractWithOllama(
  baseUrl: string,
  title: string,
  text: string,
  slug: string
): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] } | null> {
  const model = process.env.OLLAMA_SMALL_MODEL || "llama3.2:3b";
  const prompt = `You are a scientific knowledge graph constructor.
Extract key scientific entities and directed relationships from this research paper text:
Title: "${title}"
Content:
${text.slice(0, 3500)}

Return strictly JSON:
{
  "entities": [{"id": "e1", "name": "Name", "type": "Architecture|Method|Dataset|Metric|Concept", "description": "Details"}],
  "relationships": [{"source": "e1", "target": "e2", "type": "USES|EVALUATED_ON|OPTIMIZES", "description": "Details", "weight": 0.9}]
}`;

  const response = await fetch(`${baseUrl}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt,
      format: "json",
      stream: false,
    }),
  });

  if (!response.ok) return null;
  const data = (await response.json()) as any;
  if (!data.response) return null;

  const parsed: RawExtractionResponse = JSON.parse(data.response);
  return processRawExtraction(parsed, slug, title);
}

function processRawExtraction(
  parsed: RawExtractionResponse,
  slug: string,
  title: string
): { nodes: GraphNode[]; edges: GraphEdge[] } | null {
  if (!parsed || !Array.isArray(parsed.entities) || parsed.entities.length === 0) {
    return null;
  }

  const nodes: GraphNode[] = [];
  const nodeMap = new Map<string, string>(); // oldId -> cleanId

  // Normalize entities
  for (let i = 0; i < parsed.entities.length; i++) {
    const raw = parsed.entities[i];
    if (!raw.name || typeof raw.name !== "string") continue;
    const cleanName = raw.name.trim();
    if (cleanName.length < 2) continue;

    const baseId = raw.id || `ent_${i + 1}`;
    const cleanId = `${slug}_${baseId.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase()}`;
    nodeMap.set(baseId, cleanId);
    nodeMap.set(cleanName.toLowerCase(), cleanId);

    const validTypes = ["Architecture", "Method", "Dataset", "Metric", "Concept", "Theory", "Task"];
    const matchedType = validTypes.find((t) => t.toLowerCase() === (raw.type || "").toLowerCase()) || "Concept";

    nodes.push({
      id: cleanId,
      name: cleanName,
      type: matchedType,
      description: raw.description || `Scientific ${matchedType} identified in ${title}`,
    });
  }

  if (nodes.length < 4) return null;

  // Normalize relationships
  const edges: GraphEdge[] = [];
  const edgeSet = new Set<string>();

  if (Array.isArray(parsed.relationships)) {
    for (const rel of parsed.relationships) {
      if (!rel.source || !rel.target) continue;

      const srcId = nodeMap.get(rel.source) || nodeMap.get(rel.source.toLowerCase()) || rel.source;
      const tgtId = nodeMap.get(rel.target) || nodeMap.get(rel.target.toLowerCase()) || rel.target;

      if (!nodes.some((n) => n.id === srcId) || !nodes.some((n) => n.id === tgtId)) {
        continue;
      }
      if (srcId === tgtId) continue;

      const key = `${srcId}->${tgtId}`;
      if (edgeSet.has(key)) continue;
      edgeSet.add(key);

      const relType = (rel.type || "RELATED_TO").toUpperCase().replace(/[^A-Z0-9_]/g, "_");
      const isBridge = relType.includes("EVALUATED") || relType.includes("BENCHMARK") || relType.includes("DATASET");

      edges.push({
        source: srcId,
        target: tgtId,
        type: relType,
        weight: typeof rel.weight === "number" ? Math.max(0.1, Math.min(1.0, rel.weight)) : 0.9,
        is_bridge: isBridge,
      });
    }
  }

  // Ensure graph connectivity: link disconnected components to root/first entity
  const connectedTargets = new Set(edges.map((e) => e.target));
  const rootNode = nodes[0];
  for (let i = 1; i < nodes.length; i++) {
    const n = nodes[i];
    if (!edges.some((e) => e.source === n.id || e.target === n.id)) {
      edges.push({
        source: rootNode.id,
        target: n.id,
        type: n.type === "Dataset" ? "EVALUATED_ON" : n.type === "Metric" ? "MEASURED_BY" : "CONTAINS",
        weight: 0.85,
        is_bridge: n.type === "Dataset" || n.type === "Metric",
      });
    }
  }

  // Ensure at least one directed cycle if there are methods/architecture for SCC quotient reduction
  const archNodes = nodes.filter((n) => n.type === "Architecture" || n.type === "Method");
  if (archNodes.length >= 2) {
    const n1 = archNodes[0];
    const n2 = archNodes[1];
    if (!edges.some((e) => e.source === n2.id && e.target === n1.id)) {
      edges.push({
        source: n2.id,
        target: n1.id,
        type: "FEEDBACK_LOOP",
        weight: 0.88,
      });
    }
  }

  return { nodes, edges };
}

/**
 * Heuristic fallback extracting actual domain terms from the document text.
 */
function extractHeuristically(
  title: string,
  text: string,
  slug: string
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const seenNames = new Set<string>();

  // Add the paper title as the core system/architecture
  const mainArchId = `${slug}_core_system`;
  nodes.push({
    id: mainArchId,
    name: title,
    type: "Architecture",
    description: `Primary methodology proposed in "${title}"`,
  });
  seenNames.add(title.toLowerCase());

  // Search document for common domain phrases and technical capitalized entities
  const patterns: { regex: RegExp; type: string }[] = [
    { regex: /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b/g, type: "Concept" },
    { regex: /\b(CNN|RNN|LSTM|ResNet|VGG|EfficientNet|Transformer|BERT|XGBoost|Random Forest|SVM|ViT|GAN|Autoencoder)\b/gi, type: "Architecture" },
    { regex: /\b(Grad-CAM|LIME|SHAP|Integrated Gradients|XAI|Explainable AI|Attention Mechanism|Feature Extraction|Data Augmentation|Fine-Tuning|Transfer Learning)\b/gi, type: "Method" },
    { regex: /\b(Accuracy|Sensitivity|Specificity|Precision|Recall|F1-Score|AUC|ROC|mAP|Loss|IoU|BLEU)\b/gi, type: "Metric" },
    { regex: /\b(Kaggle|ImageNet|MESSIDOR|EyePACS|APTOS|IDRiD|RFMiD|COCO|Cora|PubMed|MNIST|CIFAR-10)\b/gi, type: "Dataset" },
    { regex: /\b(Retinal Disease|Diabetic Retinopathy|Glaucoma|Macular Degeneration|Cataract|Phishing URL|Word Detection|Object Detection|Classification)\b/gi, type: "Task" },
  ];

  let idCounter = 1;
  for (const { regex, type } of patterns) {
    let match;
    while ((match = regex.exec(text)) !== null && nodes.length < 18) {
      const name = match[0].trim();
      const lower = name.toLowerCase();
      if (seenNames.has(lower) || name.length < 3) continue;
      // Skip generic boilerplate words
      if (/^(The|This|These|All|Table|Figure|Author|Elsevier|Springer|Journal|Volume|Issue|Pages|Published|Received|Accepted)\b/i.test(name)) {
        continue;
      }
      seenNames.add(lower);

      const entId = `${slug}_ent_${idCounter++}`;
      nodes.push({
        id: entId,
        name,
        type,
        description: `${type} extracted from ${title} text`,
      });
    }
  }

  // Build relational edges between extracted entities
  for (let i = 1; i < nodes.length; i++) {
    const target = nodes[i];
    let relType = "USES";
    let isBridge = false;

    if (target.type === "Dataset") {
      relType = "EVALUATED_ON";
      isBridge = true;
    } else if (target.type === "Metric") {
      relType = "OPTIMIZES";
      isBridge = true;
    } else if (target.type === "Task") {
      relType = "TARGETS_TASK";
    } else if (target.type === "Architecture") {
      relType = "INTEGRATES_MODEL";
    }

    edges.push({
      source: mainArchId,
      target: target.id,
      type: relType,
      weight: 0.9,
      is_bridge: isBridge,
    });

    // Form relational sub-paths
    if (i > 1 && i % 2 === 0) {
      edges.push({
        source: nodes[i - 1].id,
        target: target.id,
        type: "FEEDS_INTO",
        weight: 0.8,
      });
    }
  }

  // Directed feedback cycle between top 2 methods
  if (nodes.length > 2) {
    edges.push({
      source: nodes[2].id,
      target: nodes[1].id,
      type: "OPTIMIZATION_LOOP",
      weight: 0.85,
    });
  }

  return { nodes, edges };
}

function finalizeGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  title: string
): { original_graph: GraphTopology; compressed_graph: GraphTopology } {
  // Compute Tarjan bridges and cut-vertices
  const bridgeAnalysis = findBridgesAndArticulationPoints(nodes, edges);
  for (const edge of edges) {
    if (bridgeAnalysis.bridges.some(([u, v]) => (edge.source === u && edge.target === v) || (edge.source === v && edge.target === u))) {
      edge.is_bridge = true;
    }
  }

  const origTopology: GraphTopology = {
    nodes,
    edges,
    node_count: nodes.length,
    edge_count: edges.length,
    articulation_points: bridgeAnalysis.articulationPoints,
    bridges: bridgeAnalysis.bridges,
  };

  const compressed = buildQuotientGraph(nodes, edges, "SCC");

  return {
    original_graph: origTopology,
    compressed_graph: compressed,
  };
}
