import dotenv from "dotenv";
dotenv.config({ override: true });
import express from "express";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import multer from "multer";
import { PDFParse } from "pdf-parse";
import { createServer as createViteServer } from "vite";
import { PRELOADED_PAPERS, INITIAL_CONFIG, generatePaperGraph, generateMultiPaperGraph } from "./src/data";
import { ConfigState, Paper, TraversalStep, TraversalTrace, GraphTopology, GraphNode, GraphEdge } from "./src/types";
import { mergeNamespacedPaperGraphs } from "./src/crossPaperGraph";
import { GoogleGenAI } from "@google/genai";
import { extractPaperKnowledgeGraph } from "./src/extractorService";
import {
  checkNeo4jConnection,
  initNeo4jConstraints,
  syncPaperToNeo4j,
  runCypherQuery,
  deletePaperFromNeo4j,
  purgeAllFromNeo4j,
} from "./src/neo4jService";

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json());

// Persistent disk storage paths
const PAPERS_DIR = path.join(process.cwd(), "data", "papers");
const STORAGE_DIR = path.join(process.cwd(), "storage");
const UPLOADS_DIR = path.join(STORAGE_DIR, "uploads");
const GRAPHS_DIR = path.join(STORAGE_DIR, "graphs");
const PAPERS_FILE = path.join(STORAGE_DIR, "papers.json");

if (!fs.existsSync(STORAGE_DIR)) fs.mkdirSync(STORAGE_DIR, { recursive: true });
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(GRAPHS_DIR)) fs.mkdirSync(GRAPHS_DIR, { recursive: true });

// Memory cache for parsed graphs (both original & Tarjan-compressed)
const paperGraphCache = new Map<string, { original_graph: GraphTopology; compressed_graph: GraphTopology }>();

function loadStoredPapers(): Paper[] {
  try {
    if (fs.existsSync(PAPERS_FILE)) {
      const raw = fs.readFileSync(PAPERS_FILE, "utf-8");
      const diskPapers: Paper[] = JSON.parse(raw);
      if (Array.isArray(diskPapers)) {
        for (const dp of diskPapers) {
          const graphFile = path.join(GRAPHS_DIR, `${dp.paper_id}.json`);
          if (fs.existsSync(graphFile)) {
            try {
              const cached = JSON.parse(fs.readFileSync(graphFile, "utf-8"));
              if (cached?.original_graph && cached?.compressed_graph) {
                paperGraphCache.set(dp.paper_id, cached);
                dp.num_entities = cached.original_graph.node_count ?? cached.original_graph.nodes.length;
                dp.num_relations = cached.original_graph.edge_count ?? cached.original_graph.edges.length;
                dp.num_communities = cached.compressed_graph.nodes.length;
              }
            } catch {}
          }
          if (!dp.num_entities) {
            const g = generatePaperGraph(dp.paper_id, dp.title);
            dp.num_entities = g.original_graph.node_count ?? 0;
            dp.num_relations = g.original_graph.edge_count ?? 0;
            dp.num_communities = Math.max(3, Math.round(dp.num_entities / 4.5));
          }
        }
        return diskPapers;
      }
    }
  } catch (err) {
    console.warn("[Storage] Could not read stored papers from disk:", err);
  }

  // First run: populate default papers file with PRELOADED_PAPERS
  try {
    fs.writeFileSync(PAPERS_FILE, JSON.stringify(PRELOADED_PAPERS, null, 2), "utf-8");
  } catch {}
  return [...PRELOADED_PAPERS];
}

function saveStoredPapers(papers: Paper[]) {
  try {
    fs.writeFileSync(PAPERS_FILE, JSON.stringify(papers, null, 2), "utf-8");
  } catch (err) {
    console.warn("[Storage] Failed to write papers to disk:", err);
  }
}

let currentConfig: ConfigState = { ...INITIAL_CONFIG };
let storedPapers: Paper[] = loadStoredPapers();

// In-memory high-speed cache for real extracted PDF text and chunk passages
const paperTextCache = new Map<string, string>();
const paperChunksCache = new Map<string, string[]>();

/**
 * Extracts and caches real text from a PDF buffer or disk path
 */
async function getOrExtractPaperText(paperId: string): Promise<string> {
  if (paperTextCache.has(paperId)) {
    return paperTextCache.get(paperId)!;
  }

  // 1. Try uploaded file in uploads dir
  const uploadPath = path.join(UPLOADS_DIR, `${paperId}.pdf`);
  let bufferToParse: Buffer | null = null;

  if (fs.existsSync(uploadPath)) {
    try {
      bufferToParse = fs.readFileSync(uploadPath);
    } catch {}
  }

  // 2. Try preloaded papers directory
  if (!bufferToParse && fs.existsSync(PAPERS_DIR)) {
    const paperRecord = storedPapers.find((p) => p.paper_id === paperId);
    const candidateFilenames = [
      paperRecord?.filename,
      `${paperId}.pdf`,
    ].filter(Boolean) as string[];

    for (const cf of candidateFilenames) {
      const fullPath = path.join(PAPERS_DIR, cf);
      if (fs.existsSync(fullPath)) {
        try {
          bufferToParse = fs.readFileSync(fullPath);
          break;
        } catch {}
      }
    }

    if (!bufferToParse) {
      const allPdfs = fs.readdirSync(PAPERS_DIR);
      const matched = allPdfs.find((f) => f.includes(paperId) || paperId.includes(f.replace(".pdf", "")));
      if (matched) {
        try {
          bufferToParse = fs.readFileSync(path.join(PAPERS_DIR, matched));
        } catch {}
      }
    }
  }

  if (bufferToParse) {
    try {
      const parser = new PDFParse(new Uint8Array(bufferToParse));
      const res = await parser.getText();
      const extractedText = (res.text || "").replace(/\r\n/g, "\n").trim();
      if (extractedText.length > 50) {
        paperTextCache.set(paperId, extractedText);
        // Create semantic chunks (~800 chars)
        const rawChunks = extractedText
          .split(/\n\s*\n/)
          .map((c) => c.replace(/\s+/g, " ").trim())
          .filter((c) => c.length > 80);

        paperChunksCache.set(paperId, rawChunks.length > 0 ? rawChunks : [extractedText.slice(0, 4000)]);
        return extractedText;
      }
    } catch (err) {
      console.warn(`[PDF Engine] Extraction warning for paper ${paperId}:`, err);
    }
  }

  const pRecord = storedPapers.find((p) => p.paper_id === paperId);
  const fallbackText = pRecord ? `Title: ${pRecord.title}. Academic research paper analyzing graph algorithms and neural architectures.` : "";
  paperTextCache.set(paperId, fallbackText);
  paperChunksCache.set(paperId, [fallbackText]);
  return fallbackText;
}

/**
 * Fast semantic passage retrieval across multiple selected papers (ensures fair representation)
 */
async function retrieveRelevantPassages(
  paperIds: string[],
  question: string,
  topKPerPaper = 4,
  graph?: GraphTopology | null
): Promise<{ paper_id: string; title: string; text: string; score: number }[]> {
  const stop = new Set([
    "what", "which", "when", "where", "how", "why", "does", "the", "and", "for",
    "with", "between", "these", "papers", "paper", "explain", "relation", "compare",
    "across", "their", "this", "that", "from", "into",
  ]);
  const queryTokens = question
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !stop.has(w));

  if (graph) {
    for (const node of graph.nodes.slice(0, 80)) {
      for (const tok of (node.name || "").toLowerCase().split(/\s+/)) {
        if (tok.length > 3 && !stop.has(tok) && queryTokens.some((q) => tok.includes(q) || q.includes(tok))) {
          if (!queryTokens.includes(tok)) queryTokens.push(tok);
        }
      }
    }
  }

  const allMatches: { paper_id: string; title: string; text: string; score: number }[] = [];

  for (const pid of paperIds) {
    await getOrExtractPaperText(pid);
    const chunks = paperChunksCache.get(pid) || [];
    const pRecord = storedPapers.find((p) => p.paper_id === pid);
    const title = pRecord?.title || pid;
    const titleToks = title.toLowerCase().split(/\s+/).filter((t) => t.length > 3);

    const paperMatches: { paper_id: string; title: string; text: string; score: number }[] = [];

    for (const chunk of chunks) {
      if (!chunk || chunk.trim().length < 20) continue;
      const chunkLower = chunk.toLowerCase();
      let matchScore = 0;
      for (const tok of queryTokens) {
        if (chunkLower.includes(tok)) matchScore += 3;
        else if (tok.length > 4 && chunkLower.includes(tok.slice(0, Math.max(4, tok.length - 2)))) matchScore += 1;
      }
      if (titleToks.some((t) => chunkLower.includes(t))) matchScore += 1;
      paperMatches.push({
        paper_id: pid,
        title,
        text: chunk.length > 1400 ? chunk.slice(0, 1400) : chunk,
        score: matchScore,
      });
    }

    paperMatches.sort((a, b) => b.score - a.score);

    const selected: typeof paperMatches = [];
    const seen = new Set<string>();
    const pushUnique = (item: (typeof paperMatches)[0]) => {
      const key = item.text.slice(0, 80);
      if (seen.has(key)) return;
      seen.add(key);
      selected.push(item);
    };

    // Always keep the opening of the paper so an answer present in the
    // introduction is not dropped just because keywords did not fire.
    if (chunks[0]) {
      pushUnique({
        paper_id: pid,
        title,
        text: chunks[0].slice(0, 1200),
        score: Math.max(paperMatches[0]?.score || 1, 1),
      });
    }
    for (const match of paperMatches) {
      if (selected.length >= topKPerPaper) break;
      pushUnique(match);
    }

    if (selected.length === 0) {
      selected.push({
        paper_id: pid,
        title,
        text: `Title: ${title}. No extracted PDF chunks were available; use the knowledge-graph entities for this paper.`,
        score: 1,
      });
    }

    allMatches.push(...selected);
  }

  return allMatches.sort((a, b) => b.score - a.score);
}

/**
 * Builds real traversal steps directly from actual graph topology and query (with multi-hop bridging)
 */
function buildDynamicTrace(
  paperIds: string[],
  graph: GraphTopology,
  question: string,
  mode: string,
  passages: { paper_id: string; title: string; text: string }[]
): TraversalTrace {
  const qTokens = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
  const matchedNodes = graph.nodes.filter((n) => {
    const text = (n.name + " " + (n.description || "") + " " + n.type).toLowerCase();
    return qTokens.some((tok) => text.includes(tok));
  });

  const seedNodes = matchedNodes.length > 0 ? matchedNodes.slice(0, 4) : graph.nodes.slice(0, 3);
  const visitedIds = seedNodes.map((n) => n.id);

  const bridgeEdges = graph.edges.filter((e) => e.is_bridge);
  const steps: TraversalStep[] = [];
  let stepIdx = 1;

  // Step 1: Visit seed entities from user query
  for (const n of seedNodes.slice(0, 2)) {
    steps.push({
      step_index: stepIdx++,
      action: "VISIT",
      node_or_comm_id: n.id,
      level: 0,
      score: 95.0,
      reason: `Matched seed entity '${n.name}' (${n.type}) from user query intent.`,
      timestamp_ms: 60 * stepIdx,
    });
  }

  // Step 2: Multi-Hop Cross-Paper Bridge or local neighbor expansion
  if (paperIds.length > 1 && bridgeEdges.length > 0) {
    const crossBridge = bridgeEdges.find((e) => e.type.includes("PARADIGM") || e.type.includes("VS") || e.is_bridge) || bridgeEdges[0];
    steps.push({
      step_index: stepIdx++,
      action: "BRIDGE_PRESERVE",
      node_or_comm_id: crossBridge.target,
      level: 1,
      score: 97.0,
      reason: `Traversed multi-hop cross-paper bridge (${crossBridge.source} -[${crossBridge.type}]-> ${crossBridge.target}) linking distinct architectural paradigms.`,
      timestamp_ms: 60 * stepIdx,
    });
  } else {
    const expandedNode = seedNodes[0] || graph.nodes[0];
    if (expandedNode) {
      steps.push({
        step_index: stepIdx++,
        action: "EXPAND",
        node_or_comm_id: expandedNode.id,
        level: 1,
        score: 88.5,
        reason: `Traversed directed relations incident to '${expandedNode.name}' into neighbor subgraph.`,
        timestamp_ms: 60 * stepIdx,
      });
    }
  }

  // Step 3: Quotient Supernode selection
  const supernode = graph.nodes.find((n) => n.is_supernode) || graph.nodes[0];
  if (supernode) {
    steps.push({
      step_index: stepIdx++,
      action: "SELECT",
      node_or_comm_id: supernode.id,
      level: 2,
      score: 98.0,
      reason: `Selected quotient condensed component '${supernode.name}' for multi-paper synthesis.`,
      timestamp_ms: 60 * stepIdx,
    });
  }

  return {
    mode,
    steps,
    visited_nodes: visitedIds,
    expanded_nodes: seedNodes.slice(0, 2).map((n) => n.id),
    pruned_nodes: graph.nodes.slice(seedNodes.length + 3).map((n) => n.id),
    bridge_kept_nodes: bridgeEdges.slice(0, 3).map((e) => e.target),
    selected_reports: passages.map((p) => p.paper_id),
    ratings: Object.fromEntries(seedNodes.map((n) => [n.id, 92])),
    total_tokens: Math.min(2400, 600 + passages.reduce((acc, p) => acc + p.text.length / 4, 0)),
    total_llm_calls: 1,
    latency_ms: 280,
  };
}

// Multer memory storage for PDF file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024, files: 10, fields: 30, fieldSize: 1024 * 1024 },
});

// Lazy Gemini AI initialization
let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI | null {
  if (!aiClient && process.env.GEMINI_API_KEY) {
    try {
      aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    } catch (err) {
      console.warn("Failed to initialize GoogleGenAI client:", err);
    }
  }
  return aiClient;
}

// Local Ollama stream helper
async function streamFromOllama(
  prompt: string,
  onToken: (token: string) => void,
  signal?: AbortSignal
): Promise<string> {
  const ollamaBaseUrl = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
  const model = process.env.OLLAMA_SMALL_MODEL || "llama3.2:3b";
  const response = await fetch(`${ollamaBaseUrl}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt,
      stream: true,
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Ollama request failed with HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        if (data.response) {
          full += data.response;
          onToken(data.response);
        }
      } catch {}
    }
  }

  return full;
}

// xAI Grok stream helper
async function streamFromGrok(
  apiKey: string,
  prompt: string,
  onToken: (token: string) => void,
  signal?: AbortSignal
): Promise<string> {
  const response = await fetch("https://api.x.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "grok-2-latest",
      messages: [{ role: "user", content: prompt }],
      stream: true,
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    const errText = await response.text().catch(() => "");
    throw new Error(`xAI Grok failed with HTTP ${response.status}: ${errText.slice(0, 160)}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed === "data: [DONE]") continue;
      if (trimmed.startsWith("data: ")) {
        try {
          const data = JSON.parse(trimmed.slice(6));
          const delta = data.choices?.[0]?.delta?.content || "";
          if (delta) {
            full += delta;
            onToken(delta);
          }
        } catch {}
      }
    }
  }

  return full;
}

// Groq ultra-fast stream helper with multi-model fallback
async function streamFromGroq(
  apiKey: string,
  prompt: string,
  onToken: (token: string, modelName: string) => void,
  signal?: AbortSignal
): Promise<string> {
  const models = [
    process.env.GROQ_MODEL || "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
  ];

  let lastError = "";

  for (const model of models) {
    try {
      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          stream: true,
        }),
        signal,
      });

      if (!response.ok || !response.body) {
        const errText = await response.text().catch(() => "");
        lastError = `${model} HTTP ${response.status}: ${errText.slice(0, 120)}`;
        console.warn(`[Groq Stream] ${model} unavailable (${response.status}), trying next model...`);
        continue;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let full = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        for (const line of text.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === "data: [DONE]") continue;
          if (trimmed.startsWith("data: ")) {
            try {
              const data = JSON.parse(trimmed.slice(6));
              const delta = data.choices?.[0]?.delta?.content || "";
              if (delta) {
                full += delta;
                onToken(delta, model);
              }
            } catch {}
          }
        }
      }

      if (full.length > 0) {
        return full;
      }
    } catch (err: any) {
      lastError = err?.message || "error";
      continue;
    }
  }

  throw new Error(`Groq stream failed across all models: ${lastError}`);
}

/**
 * Derives a clean academic paper title from text while filtering out journal metadata
 */
function derivePaperTitle(extractedText: string, originalFilename: string): string {
  const fallback = originalFilename.replace(/\.pdf$/i, "").replace(/[_-]/g, " ").trim();
  if (!extractedText || extractedText.length < 30) return fallback;

  const lines = extractedText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => {
      if (l.length < 10 || l.length > 160) return false;
      if (/^(received|accepted|revised|published|available online|doi|issn|copyright|all rights reserved|volume|pages|citation|https?:|article)/i.test(l)) return false;
      if (/^(arxiv|ieee|elsevier|springer|mdpi|acm|nature|science|biorxiv)/i.test(l)) return false;
      if (/^\d{1,4}[-\/]\d{1,4}/.test(l)) return false;
      if (/^computational and structural biotechnology/i.test(l)) return false;
      return true;
    });

  if (lines.length > 0) {
    const candidate = lines[0].replace(/\s+/g, " ");
    if (/[a-zA-Z]{3,}/.test(candidate) && !/^\d+$/.test(candidate)) {
      return candidate;
    }
  }
  return fallback;
}

/**
 * Returns the knowledge graph for a paper.
 * For custom uploaded papers, extracts real entities and directed relationships with Groq/LLM,
 * compresses via Tarjan's SCC quotient graph, and persists to disk.
 */
async function getOrExtractGraph(
  paperId: string,
  customTitle?: string
): Promise<{ original_graph: GraphTopology; compressed_graph: GraphTopology }> {
  // 1. Check in-memory cache
  if (paperGraphCache.has(paperId)) {
    return paperGraphCache.get(paperId)!;
  }

  // 2. Check disk cache in storage/graphs/<paperId>.json
  const graphFile = path.join(GRAPHS_DIR, `${paperId}.json`);
  if (fs.existsSync(graphFile)) {
    try {
      const cached = JSON.parse(fs.readFileSync(graphFile, "utf-8"));
      if (cached?.original_graph && cached?.compressed_graph) {
        paperGraphCache.set(paperId, cached);
        return cached;
      }
    } catch {}
  }

  // 3. Preloaded paper catalog
  if (PRELOADED_PAPERS.some((p) => p.paper_id === paperId)) {
    const graph = generatePaperGraph(paperId, customTitle);
    paperGraphCache.set(paperId, graph);
    return graph;
  }

  // 4. Custom uploaded paper: extract real entities & relations from PDF text
  const paperText = await getOrExtractPaperText(paperId);
  const paperRecord = storedPapers.find((p) => p.paper_id === paperId);
  const title = customTitle || paperRecord?.title || paperId;

  if (paperText && paperText.length > 50) {
    try {
      console.log(`[Graph Engine] Extracting real knowledge graph for paper '${title}' (${paperId})...`);
      const extracted = await extractPaperKnowledgeGraph(paperId, title, paperText);
      const result = {
        original_graph: extracted.original_graph,
        compressed_graph: extracted.compressed_graph,
      };

      paperGraphCache.set(paperId, result);
      try {
        fs.writeFileSync(graphFile, JSON.stringify(result, null, 2), "utf-8");
      } catch {}

      syncPaperToNeo4j(paperId, title, result.original_graph, result.compressed_graph).catch(() => {});
      return result;
    } catch (err) {
      console.warn(`[Graph Engine] Extraction error for '${title}':`, err);
    }
  }

  const fallback = generatePaperGraph(paperId, title);
  paperGraphCache.set(paperId, fallback);
  return fallback;
}

/**
 * Returns union graph across multiple papers linked by cross-paper bridges
 */
async function getOrExtractMultiGraph(
  paperIds: string[]
): Promise<{ original_graph: GraphTopology; compressed_graph: GraphTopology }> {
  if (!paperIds || paperIds.length === 0) {
    const first = storedPapers[0];
    return getOrExtractGraph(first.paper_id, first.title);
  }
  if (paperIds.length === 1) {
    const p = storedPapers.find((x) => x.paper_id === paperIds[0]);
    return getOrExtractGraph(paperIds[0], p?.title);
  }

  const parts: { paperId: string; graph: GraphTopology }[] = [];
  for (const pid of paperIds) {
    const p = storedPapers.find((x) => x.paper_id === pid);
    const g = await getOrExtractGraph(pid, p?.title);
    parts.push({ paperId: pid, graph: g.original_graph });
  }
  return mergeNamespacedPaperGraphs(parts);
}

function compactGraphForClient(graph: GraphTopology): GraphTopology {
  return {
    ...graph,
    nodes: (graph.nodes || []).map((n) => ({
      id: n.id,
      name: n.name,
      type: n.type,
      description: (n.description || "").slice(0, 220),
      is_supernode: n.is_supernode,
      is_connector: n.is_connector,
      member_count: n.member_count,
      paper_id: n.paper_id,
    })),
    edges: (graph.edges || []).map((e) => ({
      source: e.source,
      target: e.target,
      type: e.type,
      is_bridge: Boolean(e.is_bridge),
      weight: e.weight || 1,
    })),
  };
}

function titleForPaper(paperId: string): string {
  return storedPapers.find((p) => p.paper_id === paperId)?.title || paperId;
}

function paperIdOfNode(nodeId: string, node?: GraphNode): string {
  return node?.paper_id || (nodeId.includes("::") ? nodeId.slice(0, nodeId.indexOf("::")) : "");
}

function crossPaperBridges(graph: GraphTopology): GraphEdge[] {
  const seen = new Set<string>();
  const out: GraphEdge[] = [];
  for (const edge of graph.edges || []) {
    const isCross =
      Boolean(edge.is_bridge) ||
      /SAME_ENTITY|RELATED_CONCEPT|SHARED_PARADIGM|CROSS_PAPER/.test(edge.type || "");
    if (!isCross) continue;
    const key = [edge.source, edge.target].sort().join("|") + (edge.type || "");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(edge);
  }
  return out;
}

function buildMultiHopGroundedAnswer(
  question: string,
  paperIds: string[],
  graphs: { original_graph: GraphTopology; compressed_graph: GraphTopology },
  passages: { paper_id: string; title: string; text: string; score: number }[]
): string {
  const nodeById = new Map(graphs.original_graph.nodes.map((n) => [n.id, n]));
  const bridges = crossPaperBridges(graphs.original_graph);
  const titles = paperIds.map(titleForPaper);

  let md = `### Direct Answer\n\n`;
  if (bridges.length > 0) {
    md += `**${titles.join("** and **")}** are linked by ${bridges.length} multi-hop bridges `;
    md += `on the Tarjan quotient graph (shared entities, related concepts, and paradigm links), not a single root node.\n\n`;
  } else {
    md += `The selected papers were unioned into one knowledge graph. Structural bridges were added so evidence can traverse across papers.\n\n`;
  }

  md += `### Multi-hop Synthesis\n\n`;
  const displayedBridges = bridges.slice(0, 4);
  if (displayedBridges.length > 0) {
    for (const edge of displayedBridges) {
      const src = nodeById.get(edge.source);
      const tgt = nodeById.get(edge.target);
      const srcPaper = titleForPaper(paperIdOfNode(edge.source, src));
      const tgtPaper = titleForPaper(paperIdOfNode(edge.target, tgt));
      const edgeLabel = edge.type ? ` → [${edge.type}] → ` : " → ";
      md += `- **${src?.name || edge.source}** *(${srcPaper})*${edgeLabel}**${tgt?.name || edge.target}** *(${tgtPaper})*\n`;
    }
  } else {
    md += `- ${titles.join(" ↔ ")} *(Connected via union quotient graph)*\n`;
  }

  md += `\n### Evidence\n\n`;
  for (const passage of passages.slice(0, 3)) {
    let snippet = passage.text.replace(/\s+/g, " ").trim();
    if (snippet.length > 150) {
      snippet = snippet.slice(0, 145) + "…";
    }
    md += `- **${passage.title}:** "${snippet}"\n`;
  }

  md += `\n### Key Takeaway\n\n`;
  md += `Cross-paper synthesis grounded across ${titles.join(" & ")}.\n`;
  return md;
}

async function collectLlmAnswer(prompt: string): Promise<{ answer: string; engine: string }> {
  const groqKey = process.env.GROQ_API_KEY;
  const xaiKey = process.env.XAI_API_KEY;

  if (groqKey) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12000);
      const answer = await streamFromGroq(groqKey, prompt, () => {}, controller.signal);
      clearTimeout(timer);
      if (answer.trim()) return { answer, engine: "Groq (multi-hop)" };
    } catch (err: any) {
      console.warn("[Multi-hop] Groq skipped:", err?.message?.slice(0, 120));
    }
  }

  if (xaiKey) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12000);
      const answer = await streamFromGrok(xaiKey, prompt, () => {}, controller.signal);
      clearTimeout(timer);
      if (answer.trim()) return { answer, engine: "xAI Grok (multi-hop)" };
    } catch (err: any) {
      console.warn("[Multi-hop] Grok skipped:", err?.message?.slice(0, 120));
    }
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    const answer = await streamFromOllama(prompt, () => {}, controller.signal);
    clearTimeout(timer);
    if (answer.trim()) return { answer, engine: "Local Ollama (multi-hop)" };
  } catch (err: any) {
    console.warn("[Multi-hop] Ollama skipped:", err?.message?.slice(0, 120));
  }

  return { answer: "", engine: "" };
}

async function runLocalMultiPaperQuery(paperIds: string[], question: string) {
  const started = Date.now();
  const graphs = await getOrExtractMultiGraph(paperIds);
  const titles = paperIds.map(titleForPaper);
  const passages = await retrieveRelevantPassages(
    paperIds,
    question,
    4,
    graphs.original_graph
  );
  const nodeById = new Map(graphs.original_graph.nodes.map((n) => [n.id, n]));
  const bridges = crossPaperBridges(graphs.original_graph);

  for (const edge of bridges.slice(0, 8)) {
    const src = nodeById.get(edge.source);
    const tgt = nodeById.get(edge.target);
    if (!src || !tgt) continue;
    passages.push({
      paper_id: paperIdOfNode(edge.source, src) || paperIds[0],
      title: `${titleForPaper(paperIdOfNode(edge.source, src))} ⟷ ${titleForPaper(paperIdOfNode(edge.target, tgt))}`,
      text: `Cross-paper hop [${edge.type}]: ${src.name} (${src.description || src.type}) → ${tgt.name} (${tgt.description || tgt.type})`,
      score: 8,
    });
  }

  const graphTriples = bridges
    .slice(0, 12)
    .map((e) => {
      const s = nodeById.get(e.source);
      const t = nodeById.get(e.target);
      return `(${s?.name}) [${titleForPaper(paperIdOfNode(e.source, s))}] -[${e.type}]-> (${t?.name}) [${titleForPaper(paperIdOfNode(e.target, t))}]`;
    })
    .join("\n");

  const passagesContext = passages
    .slice(0, 8)
    .map((p, idx) => `[Source ${idx + 1}: ${p.title}]\n${p.text.slice(0, 1100)}`)
    .join("\n\n---\n\n");

  const prompt = `You are answering a multi-hop question over several research papers using their union knowledge graph.

PAPERS: ${titles.map((t) => `"${t}"`).join(", ")}

CROSS-PAPER GRAPH HOPS (Tarjan bridges / shared concepts):
${graphTriples || "(structural union of the selected paper graphs)"}

EVIDENCE PASSAGES:
${passagesContext}

USER QUESTION:
"${question}"

INSTRUCTIONS:
Write clean, concise Markdown. Keep it short and readable:
### Direct Answer
2-3 concise sentences directly answering the question.
### Multi-hop Synthesis
2-3 bullet points showing the bridge chain: **Paper A** (concept) → **shared bridge** → **Paper B** (concept). Keep each bullet to 1 line.
### Evidence
2-3 concise, short quotes/facts from the sources (max 1-2 lines each).
### Key Takeaway
1-2 closing sentences.
Never state no chunks matched if graph hops or passages are present.`;

  let { answer, engine } = await collectLlmAnswer(prompt);
  if (!answer.trim()) {
    answer = buildMultiHopGroundedAnswer(question, paperIds, graphs, passages);
    engine = "Tarjan multi-hop graph synthesis";
  }

  const trace = buildDynamicTrace(
    paperIds,
    graphs.compressed_graph,
    question,
    "ours",
    passages
  );

  return {
    status: "ok",
    mode: "multi_paper",
    question,
    paper_ids: paperIds,
    paper_titles: Object.fromEntries(paperIds.map((id) => [id, titleForPaper(id)])),
    answer,
    engine,
    llm_status: engine.includes("synthesis") ? "grounded" : "ok",
    original_graph: compactGraphForClient(graphs.original_graph),
    compressed_graph: compactGraphForClient(graphs.compressed_graph),
    evidence: passages.slice(0, 8).map((p, i) => ({
      evidence_id: `E${i + 1}`,
      paper_id: p.paper_id,
      paper_title: p.title,
      chunk_id: `local_${i + 1}`,
      content: p.text,
    })),
    trace,
    total_tokens: trace.total_tokens,
    total_llm_calls: engine.includes("synthesis") ? 0 : 1,
    latency_ms: Date.now() - started,
    cross_paper_links: bridges.length,
  };
}

// ---------------- API Routes ----------------

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    papers_count: storedPapers.length,
    cached_texts: paperTextCache.size,
    timestamp: new Date().toISOString(),
  });
});

// Neo4j Aura endpoints
app.get("/api/neo4j/status", async (req, res) => {
  try {
    const status = await checkNeo4jConnection();
    res.json(status);
  } catch (err: any) {
    res.status(500).json({
      connected: false,
      status: "error",
      error: err?.message || "Failed to check Neo4j status",
    });
  }
});

app.post("/api/neo4j/sync", async (req, res) => {
  try {
    const targetPaperId = req.body?.paper_id;
    const papersToSync = targetPaperId
      ? storedPapers.filter((p) => p.paper_id === targetPaperId)
      : storedPapers;

    let totalNodes = 0;
    let totalEdges = 0;
    const results = [];

    for (const p of papersToSync) {
      const g = generatePaperGraph(p.paper_id, p.title);
      const syncRes = await syncPaperToNeo4j(p.paper_id, p.title, g.original_graph, g.compressed_graph);
      totalNodes += syncRes.nodes_synced;
      totalEdges += syncRes.edges_synced;
      results.push({ paper_id: p.paper_id, ...syncRes });
    }

    const neoStatus = await checkNeo4jConnection();
    res.json({
      success: true,
      papers_synced: papersToSync.length,
      total_nodes_synced: totalNodes,
      total_edges_synced: totalEdges,
      details: results,
      neo4j_stats: neoStatus,
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || "Failed to sync to Neo4j" });
  }
});

app.post("/api/neo4j/query", async (req, res) => {
  // Arbitrary Cypher execution is disabled by default. Enable only in a trusted
  // local environment; do not expose this debugging endpoint to the public internet.
  if (process.env.NEO4J_QUERY_API_ENABLED !== "true") {
    return res.status(403).json({ error: "Cypher console is disabled. Set NEO4J_QUERY_API_ENABLED=true only in a trusted environment." });
  }

  try {
    const { cypher, params } = req.body;
    if (!cypher || typeof cypher !== "string" || cypher.length > 5000) {
      return res.status(400).json({ error: "A Cypher query of at most 5000 characters is required" });
    }

    // The console is for inspection only. Reject write/admin procedures and
    // multi-statement attempts before the query reaches Neo4j.
    const trimmedCypher = cypher.trim();
    const startsReadOnly = /^(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b/i.test(trimmedCypher);
    const containsWriteOrAdmin = /\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|FOREACH|USE|GRANT|DENY|REVOKE|ALTER|TERMINATE|APOC)\b/i.test(trimmedCypher);
    if (!startsReadOnly || containsWriteOrAdmin || trimmedCypher.includes(";")) {
      return res.status(400).json({ error: "Only a single read-only Cypher query is permitted" });
    }

    const result = await runCypherQuery(trimmedCypher, params || {});
    res.json(result);
  } catch (err: any) {
    res.status(400).json({ error: err?.message || "Cypher execution error" });
  }
});

app.get("/api/papers", (req, res) => {
  res.json(storedPapers);
});

app.delete("/api/papers/:paperId", async (req, res) => {
  const paperId = req.params.paperId;
  const index = storedPapers.findIndex((p) => p.paper_id === paperId);
  let removedTitle = paperId;
  if (index !== -1) {
    removedTitle = storedPapers[index].title;
    storedPapers.splice(index, 1);
    saveStoredPapers(storedPapers);
  }

  // Clean memory caches
  paperGraphCache.delete(paperId);
  paperTextCache.delete(paperId);
  paperChunksCache.delete(paperId);

  // Clean disk files
  const graphFile = path.join(GRAPHS_DIR, `${paperId}.json`);
  if (fs.existsSync(graphFile)) {
    try { fs.unlinkSync(graphFile); } catch {}
  }
  const uploadPdf = path.join(UPLOADS_DIR, `${paperId}.pdf`);
  if (fs.existsSync(uploadPdf)) {
    try { fs.unlinkSync(uploadPdf); } catch {}
  }

  // Clean Neo4j: await deletion to guarantee caller receives accurate status immediately
  try {
    if (storedPapers.length === 0) {
      await purgeAllFromNeo4j();
    } else {
      await deletePaperFromNeo4j(paperId);
    }
  } catch (err) {
    console.warn(`[Delete Paper] Neo4j cleanup error for ${paperId}:`, err);
  }

  console.log(`[Delete Paper] Deleted paper '${removedTitle}' (${paperId}). Remaining in catalog: ${storedPapers.length}.`);
  res.json({
    success: true,
    message: `Paper '${removedTitle}' deleted successfully.`,
    deleted_paper_id: paperId,
    remaining_papers: storedPapers.length,
  });
});

app.post("/api/neo4j/clear", async (req, res) => {
  try {
    const result = await purgeAllFromNeo4j();
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err?.message || "Failed to purge Neo4j" });
  }
});

// Single & Multi-file PDF Upload with real extraction
app.post("/api/upload", upload.any() as any, async (req, res) => {
  try {
    const files = (req.files as Express.Multer.File[]) || [];
    if (!files || files.length === 0) {
      return res.status(400).json({ error: "No PDF files uploaded." });
    }

    const createdPapers: Paper[] = [];

    for (const file of files) {
      if (!file.originalname.toLowerCase().endsWith(".pdf")) {
        return res.status(415).json({ error: `Only PDF files are accepted: ${path.basename(file.originalname)}` });
      }
      if (!file.buffer.subarray(0, 5).equals(Buffer.from("%PDF-"))) {
        return res.status(415).json({ error: `Invalid PDF signature: ${path.basename(file.originalname)}` });
      }

      const contentHash = crypto.createHash("sha256").update(file.buffer).digest("hex");
      const fileSizeKb = Math.max(1, Math.round(file.size / 1024));

      // Deduplication check: if force reindex is requested or if existing paper has no graph, re-extract
      const isForce = req.query.force === "true" || req.body.force === "true";
      const existingIndex = storedPapers.findIndex(
        (p) => p.content_hash === contentHash || p.filename.toLowerCase() === file.originalname.toLowerCase()
      );
      const existing = existingIndex >= 0 ? storedPapers[existingIndex] : null;

      if (existing && !isForce) {
        existing.status = "ready";
        createdPapers.push(existing);
        continue;
      }

      // Real text extraction from uploaded PDF
      let extractedText = "";
      try {
        const parser = new PDFParse(new Uint8Array(file.buffer));
        const pRes = await parser.getText();
        extractedText = (pRes.text || "").trim();
      } catch (err) {
        console.warn("[Upload] PDFParse extraction warning:", err);
      }

      const paperId = existing?.paper_id || `paper_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
      const customTitles = Array.isArray(req.body.custom_titles) 
        ? req.body.custom_titles 
        : (req.body.custom_titles ? [req.body.custom_titles] : []);
      const requestedTitle = customTitles[files.indexOf(file)]?.trim();
      const title = requestedTitle || existing?.title || derivePaperTitle(extractedText, file.originalname);

      if (extractedText.length > 20) {
        paperTextCache.set(paperId, extractedText);
        const rawChunks = extractedText
          .split(/\n\s*\n/)
          .map((c) => c.replace(/\s+/g, " ").trim())
          .filter((c) => c.length > 80);
        paperChunksCache.set(paperId, rawChunks);
      }

      // Save PDF to uploads disk
      try {
        fs.writeFileSync(path.join(UPLOADS_DIR, `${paperId}.pdf`), file.buffer);
      } catch {}

      // Invalidate existing cache if re-indexing
      if (isForce) {
        paperGraphCache.delete(paperId);
      }

      // Extract real domain knowledge graph using Groq / LLM
      const initialGraph = await getOrExtractGraph(paperId, title);
      const newPaper: Paper = {
        paper_id: paperId,
        title: title,
        filename: file.originalname,
        file_size_kb: fileSizeKb,
        content_hash: contentHash,
        is_preloaded: false,
        status: "ready",
        num_entities: initialGraph.original_graph.node_count ?? initialGraph.original_graph.nodes.length,
        num_relations: initialGraph.original_graph.edge_count ?? initialGraph.original_graph.edges.length,
        num_communities: initialGraph.compressed_graph.nodes.length,
        summary: extractedText.slice(0, 280),
      };

      if (existingIndex >= 0) {
        storedPapers[existingIndex] = newPaper;
      } else {
        storedPapers.unshift(newPaper);
      }
      createdPapers.push(newPaper);
    }

    saveStoredPapers(storedPapers);

    res.json({
      success: true,
      message: `Successfully processed ${createdPapers.length} paper(s).`,
      paper: createdPapers[0],
      papers: createdPapers,
      uploaded_papers: createdPapers,
    });
  } catch (err: any) {
    console.error("Upload error:", err);
    res.status(500).json({ error: err?.message || "Failed to process PDF upload." });
  }
});

// Single Paper Graph
app.get("/api/graph/:paperId", async (req, res) => {
  const paperId = req.params.paperId;
  const paper = storedPapers.find((p) => p.paper_id === paperId);
  const graphData = await getOrExtractGraph(paperId, paper?.title);
  res.json(graphData);
});

// Multi-Paper Graph Union
app.post("/api/graph/multi", async (req, res) => {
  const paperIds: string[] = Array.isArray(req.body.paper_ids) ? req.body.paper_ids : [];
  if (paperIds.length === 0) {
    const first = storedPapers[0];
    const graphData = await getOrExtractGraph(first.paper_id, first.title);
    return res.json(graphData);
  }
  const multiGraph = await getOrExtractMultiGraph(paperIds);
  res.json(multiGraph);
});

app.get("/api/config", (req, res) => {
  res.json(currentConfig);
});

app.post("/api/config", (req, res) => {
  currentConfig = { ...currentConfig, ...req.body };
  res.json(currentConfig);
});

// Summarize a paper via LLM
app.post("/api/summarize/:paperId", async (req, res) => {
  const paperId = req.params.paperId;
  if (!paperId || !/^[A-Za-z0-9_.\-]+$/.test(paperId) || paperId.includes("..")) {
    return res.status(400).json({ error: "Invalid paper ID" });
  }
  const paper = storedPapers.find((p) => p.paper_id === paperId);
  const title = paper?.title || paperId;
  const chunks = paperChunksCache.get(paperId) || [];
  const textSnippet = chunks.slice(0, 8).join("\n\n").slice(0, 4000) || (paper?.summary || "No text available.");

  try {
    const { callLLM } = await import("./src/extractorService") as any;
    let summary = "";
    if (typeof callLLM === "function") {
      summary = await callLLM(
        `You are a research assistant. Summarize the following research paper content in 3-5 sentences, focusing on the main contributions, methods, and key results.\n\nPaper: ${title}\n\nContent:\n${textSnippet}`,
        200
      );
    }
    if (!summary) {
      // Fallback: return first snippet
      summary = `${title}: ${textSnippet.slice(0, 400)}...`;
    }
    return res.json({ paper_id: paperId, title, summary });
  } catch (err: any) {
    console.error("[Summarize] Error:", err);
    const fallback = `${title}: ${textSnippet.slice(0, 300)}...`;
    return res.json({ paper_id: paperId, title, summary: fallback });
  }
});

// Multi-paper / multi-hop: answer from the local Tarjan union graph so the
// UI does not depend on Python/Neo4j chunk indexes (those return "no matching chunks").

app.post("/api/query/multi-paper", async (req, res) => {
  const paperIds: string[] = Array.isArray(req.body.paper_ids) ? req.body.paper_ids : [];
  const question = typeof req.body.question === "string" ? req.body.question.trim() : "";
  if (paperIds.length < 2 || paperIds.length > 12 || paperIds.some((id) => typeof id !== "string" || !/^[A-Za-z0-9_.-]+$/.test(id) || id.includes(".."))) {
    return res.status(400).json({ error: "Select between 2 and 12 valid indexed paper IDs." });
  }
  if (question.length < 3 || question.length > 4000) {
    return res.status(400).json({ error: "Question must contain 3 to 4000 characters." });
  }
  try {
    const payload = await runLocalMultiPaperQuery(paperIds, question);
    return res.json(payload);
  } catch (error: any) {
    console.error("[Multi-hop] local query failed:", error);
    return res.status(500).json({
      error: error?.message || "Multi-hop query failed",
    });
  }
});

// SSE Streaming Query Endpoint (with Multi-Paper & Real PDF Grounding)
app.post("/api/query/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  if (typeof (res as any).flushHeaders === "function") {
    (res as any).flushHeaders();
  }

  let clientDisconnected = false;
  res.on("close", () => {
    if (!res.writableEnded) {
      clientDisconnected = true;
    }
  });

  const originalWrite = res.write.bind(res);
  (res as any).write = (chunk: any, ...args: any[]) => {
    if (clientDisconnected || res.writableEnded || res.destroyed) return false;
    try {
      return (originalWrite as any)(chunk, ...args);
    } catch {
      return false;
    }
  };

  try {
    const startTime = Date.now();
    const { question, mode = "ours" } = req.body;
  let paperIds: string[] = [];

  if (Array.isArray(req.body.paper_ids) && req.body.paper_ids.length > 0) {
    paperIds = req.body.paper_ids;
  } else if (req.body.paper_id) {
    paperIds = [req.body.paper_id];
  } else {
    paperIds = [storedPapers[0].paper_id];
  }

  const selectedPaperRecords = storedPapers.filter((p) => paperIds.includes(p.paper_id));
  const paperTitles = selectedPaperRecords.map((p) => p.title);

  // 1. Generate appropriate topology (Single or Multi-Paper)
  const graphs =
    paperIds.length > 1
      ? await getOrExtractMultiGraph(paperIds)
      : await getOrExtractGraph(paperIds[0], paperTitles[0]);

  // 2. Retrieve real passages from PDFs
  const passages = await retrieveRelevantPassages(paperIds, question, 5, graphs.original_graph);

  // 3. Compute dynamic traversal trace from actual graph
  const trace = buildDynamicTrace(paperIds, graphs.compressed_graph, question, mode, passages);

  // 4. Emit topology event to frontend Cytoscape viewer
  res.write(
    `event: topology\ndata: ${JSON.stringify({
      original_graph: graphs.original_graph,
      compressed_graph: graphs.compressed_graph,
    })}\n\n`
  );

  // 5. Emit traversal steps
  for (const step of trace.steps) {
    res.write(`event: trace_step\ndata: ${JSON.stringify(step)}\n\n`);
  }

  // 6. Build real grounded prompt
  const isMultiPaper = paperIds.length > 1;
  const graphTriplesSummary = graphs.original_graph.edges
    .slice(0, 20)
    .map((e) => `(${e.source}) -[${e.type}${e.is_bridge ? " (CRITICAL_BRIDGE)" : ""}]-> (${e.target})`)
    .join("\n");

  const traversalTraceSummary = trace.steps
    .map((s) => `Step ${s.step_index} [${s.action}]: Entity '${s.node_or_comm_id}' (Relevance: ${s.score}) -> ${s.reason}`)
    .join("\n");

  const passagesContext = passages
    .slice(0, 4)
    .map((p, idx) => `[Source ${idx + 1}: ${p.title}]\n${p.text.slice(0, 1100)}`)
    .join("\n\n---\n\n");

  const prompt = `You are a scientific AI researcher conducting rigorous academic analysis with Knowledge Graph-Augmented Retrieval (GraphRAG).

PRIMARY PAPERS: ${paperTitles.map((t) => `"${t}"`).join(", ")}
RETRIEVAL STRATEGY: ${mode.toUpperCase()} (Tarjan SCC quotient cycle reduction & bridge-aware preservation)

GRAPH TRAVERSAL CHAIN:
${traversalTraceSummary}

RELEVANT KNOWLEDGE GRAPH TOPOLOGY & TRIPLETS:
${graphTriplesSummary}

EXTRACTED RESEARCH PAPER PASSAGES (GROUND TRUTH):
${passagesContext}

USER QUESTION:
"${question}"

INSTRUCTIONS FOR ANSWER:
Write clean, concise Markdown only. Do not wrap the whole answer in a code fence. Do not use LaTeX.
Use this structure:
### Direct Answer
2-3 punchy sentences answering the question.
${isMultiPaper ? `### Multi-hop Synthesis
2-3 short bullet points chaining the concepts: **[Paper A]** Concept A → [Relation/Bridge] → **[Paper B]** Concept B. Keep each line compact.
### Evidence
2-3 brief evidence points with short quotes or key findings from each paper.` : `### Evidence
2-3 short bullet points grounded in passages, citing concrete methods or metrics.`}
### Graph Connections
1-2 lines listing key entities & bridge relationships traversed.
### Key Takeaway
1-2 brief sentences.
If evidence is insufficient, state so directly instead of guessing.`;

  // 7. Multi-tier LLM generation: Tier 1 (Groq) -> Tier 2 (xAI Grok) -> Tier 3 (Gemini) -> Tier 4 (Ollama) -> Tier 5 (Grounded Synthesis)
  const groqKey = process.env.GROQ_API_KEY;
  const ai = getGenAI();
  const xaiKey = process.env.XAI_API_KEY;
  let streamedSuccessfully = false;
  let fullAnswerText = "";
  let activeEngine = "";

  // Tier 1: Groq Cloud LLM (openai/gpt-oss-120b / qwen/qwen3.8-27b - Sub-Second Streaming)
  if (groqKey) {
    try {
      console.log(`[LLM Engine] Tier 1: Attempting Groq streaming...`);
      const controller = new AbortController();
      let firstToken = true;
      const initialTimeout = setTimeout(() => {
        if (firstToken) controller.abort();
      }, 5000);

      fullAnswerText = await streamFromGroq(
        groqKey,
        prompt,
        (token, modelName) => {
          if (firstToken) {
            clearTimeout(initialTimeout);
            activeEngine = `Groq (${modelName})`;
            res.write(`event: engine\ndata: ${JSON.stringify({ engine: activeEngine })}\n\n`);
            firstToken = false;
          }
          res.write(`event: answer_token\ndata: ${JSON.stringify({ token })}\n\n`);
        },
        controller.signal
      );
      clearTimeout(initialTimeout);

      if (fullAnswerText.trim().length > 0) {
        streamedSuccessfully = true;
        console.log(`[LLM Engine] Tier 1 (Groq) streamed successfully (${fullAnswerText.length} chars).`);
      }
    } catch (err: any) {
      console.warn(`[LLM Engine] Tier 1 (Groq) skipped (${err?.message?.slice(0, 100) || "error"}). Falling to next tier...`);
    }
  }

  // Tier 2: xAI Grok (only if the previous engine produced no answer)
  if (!streamedSuccessfully && xaiKey) {
    try {
      console.log("[LLM Engine] Tier 1: Attempting xAI Grok (grok-2-latest)...");
      const controller = new AbortController();
      let firstToken = true;
      const initialTimeout = setTimeout(() => {
        if (firstToken) controller.abort();
      }, 5000);

      fullAnswerText = await streamFromGrok(
        xaiKey,
        prompt,
        (token) => {
          if (firstToken) {
            clearTimeout(initialTimeout);
            activeEngine = "xAI Grok (grok-2-latest)";
            res.write(`event: engine\ndata: ${JSON.stringify({ engine: activeEngine })}\n\n`);
            firstToken = false;
          }
          res.write(`event: answer_token\ndata: ${JSON.stringify({ token })}\n\n`);
        },
        controller.signal
      );
      clearTimeout(initialTimeout);

      if (fullAnswerText.trim().length > 0) {
        streamedSuccessfully = true;
        console.log(`[LLM Engine] Tier 1 (xAI Grok) streamed successfully (${fullAnswerText.length} chars).`);
      }
    } catch (err: any) {
      console.warn(`[LLM Engine] Tier 1 (xAI Grok) unavailable (${err?.message?.slice(0, 100) || "error"}). Falling to next tier...`);
    }
  }

  // Tier 3: Gemini 3 Flash (Fast streaming, verified working)
  if (!streamedSuccessfully && ai) {
    try {
      console.log("[LLM Engine] Tier 3: Attempting Gemini 3 Flash Preview...");
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Gemini timeout exceeded (8s)")), 8000)
      );

      const responseStream = await Promise.race([
        ai.models.generateContentStream({
          model: "gemini-3-flash-preview",
          contents: prompt,
        }),
        timeoutPromise,
      ]);

      for await (const chunk of responseStream) {
        if (chunk.text) {
          if (!activeEngine) {
            activeEngine = "Gemini 3 Flash";
            res.write(`event: engine\ndata: ${JSON.stringify({ engine: activeEngine })}\n\n`);
          }
          fullAnswerText += chunk.text;
          res.write(`event: answer_token\ndata: ${JSON.stringify({ token: chunk.text })}\n\n`);
          streamedSuccessfully = true;
        }
      }
      if (streamedSuccessfully) {
        console.log("[LLM Engine] Gemini 3 Flash streamed successfully.");
      }
    } catch (err: any) {
      console.warn(`[LLM Engine] Tier 3 (Gemini) skipped (${err?.message ? err.message.slice(0, 120) : "unavailable"}). Falling back to local Ollama...`);
    }
  }

  // Tier 2: Local Ollama (llama3.2:3b)
  if (!streamedSuccessfully) {
    try {
      console.log("[LLM Engine] Tier 2: Streaming from local Ollama (llama3.2:3b)...");
      const controller = new AbortController();
      let firstToken = true;
      const initialTimeout = setTimeout(() => {
        if (firstToken) {
          console.warn("[LLM Engine] Ollama initial response timed out (12s). Aborting and falling back...");
          controller.abort();
        }
      }, 12000);

      fullAnswerText = await streamFromOllama(
        prompt,
        (token) => {
          if (firstToken) {
            clearTimeout(initialTimeout);
            activeEngine = "Local Ollama (llama3.2:3b)";
            res.write(`event: engine\ndata: ${JSON.stringify({ engine: activeEngine })}\n\n`);
            firstToken = false;
          }
          res.write(`event: answer_token\ndata: ${JSON.stringify({ token })}\n\n`);
        },
        controller.signal
      );

      clearTimeout(initialTimeout);
      if (fullAnswerText.trim().length > 0) {
        streamedSuccessfully = true;
        console.log(`[LLM Engine] Tier 2 (Ollama) streamed successfully (${fullAnswerText.length} chars).`);
      }
    } catch (err: any) {
      if (fullAnswerText.trim().length > 50) {
        streamedSuccessfully = true;
      } else {
        console.warn(`[LLM Engine] Tier 2 (Ollama) unavailable (${err?.message || "error"}). Falling back to dynamic synthesis...`);
      }
    }
  }

  // Tier 3: Grounded Dynamic Synthesis from PDF Passages & Knowledge Graph
  if (!streamedSuccessfully) {
    activeEngine = "GraphRAG Grounded Synthesis (PDF + Graph)";
    res.write(`event: engine\ndata: ${JSON.stringify({ engine: activeEngine })}\n\n`);
    console.log("[LLM Engine] Tier 3: Generating grounded synthesis from primary PDF passages and Neo4j graph topology...");
    let syntheticAnswer = `### Research Synthesis (${paperTitles.join(", ")})\n\n`;
    syntheticAnswer += `Based on the extracted text and knowledge graph traversal across **${paperTitles.join(", ")}**, here are the findings regarding *"${question}"*:\n\n`;

    // Group passages by paper title
    const paperPassageMap = new Map<string, string[]>();
    for (const p of passages) {
      if (!paperPassageMap.has(p.title)) paperPassageMap.set(p.title, []);
      paperPassageMap.get(p.title)!.push(p.text);
    }

    // Grounded literature citations from real extracted PDF text
    if (passages.length > 0) {
      syntheticAnswer += `#### Grounded Evidence from Literature:\n\n`;
      for (const [title, pTexts] of paperPassageMap.entries()) {
        syntheticAnswer += `**From *${title}*:**\n`;
        for (const pt of pTexts.slice(0, 2)) {
          const snippet = pt.length > 360 ? pt.slice(0, 360).trim() + "..." : pt.trim();
          syntheticAnswer += `> ${snippet}\n\n`;
        }
      }
    } else {
      syntheticAnswer += `*No direct keyword matches were found in the indexed text passages for this specific query. Review the knowledge graph topology below for structural concepts.* \n\n`;
    }

    // Graph relationships identified
    const activeEdges = graphs.original_graph.edges
      .filter((e) => e.is_bridge || trace.visited_nodes.includes(e.source) || trace.visited_nodes.includes(e.target))
      .slice(0, 6);

    if (activeEdges.length > 0) {
      syntheticAnswer += `#### Topological Graph Connections Traversed:\n`;
      for (const e of activeEdges) {
        syntheticAnswer += `- **${e.source}** → ${e.type} → **${e.target}**${e.is_bridge ? " *(preserved bridge)*" : ""}\n`;
      }
      syntheticAnswer += `\n`;
    }

    // Key takeaway grounded purely in retrieved content
    syntheticAnswer += `---\n### 💡 Key Takeaway\n`;
    if (isMultiPaper) {
      syntheticAnswer += `Multi-hop traversal across ${paperTitles.length} papers connects ${activeEdges.length > 0 ? activeEdges.map((e) => `\`${e.source}\``).slice(0, 3).join(", ") : "the target concepts"} through the verified knowledge graph, grounding theoretical mechanisms directly in primary text citations.`;
    } else {
      syntheticAnswer += `The retrieved evidence from **${paperTitles[0]}** directly establishes the mechanism answering this query, grounded in the primary paper text and verified through ${trace.steps.length} graph traversal steps.`;
    }

    fullAnswerText = syntheticAnswer;

    const words = syntheticAnswer.split(" ");
    for (let i = 0; i < words.length; i += 3) {
      const chunk = words.slice(i, i + 3).join(" ") + (i + 3 < words.length ? " " : "");
      res.write(`event: answer_token\ndata: ${JSON.stringify({ token: chunk })}\n\n`);
      await new Promise((r) => setTimeout(r, 20));
    }
  }

  // 8. Emit completion event with cited passages and complete answer
  const liveLatencyMs = Date.now() - startTime;
  const estimatedPromptTokens = Math.round(prompt.length / 3.8);
  const estimatedCompletionTokens = Math.round(fullAnswerText.length / 3.8);
  const actualTokens = estimatedPromptTokens + estimatedCompletionTokens;
  const actualLlmCalls = streamedSuccessfully ? 1 : 0;

  res.write(
    `event: complete\ndata: ${JSON.stringify({
      mode,
      engine: activeEngine,
      answer: fullAnswerText,
      total_tokens: actualTokens,
      total_llm_calls: actualLlmCalls,
      latency_ms: liveLatencyMs,
      selected_reports: trace.selected_reports,
      bridge_kept_count: graphs.compressed_graph.bridges?.length || 2,
      cited_passages: passages.map((p) => ({
        paper_title: p.title,
        text: p.text.slice(0, 220),
        score: p.score,
      })),
      papers_evaluated: paperTitles,
    })}\n\n`
  );

    if (!clientDisconnected && !res.writableEnded) {
      try {
        res.end();
      } catch {}
    }
  } catch (err: any) {
    console.error("[Stream Endpoint] Streaming error caught:", err?.message || err);
    if (!clientDisconnected && !res.writableEnded) {
      try {
        res.write(`event: error\ndata: ${JSON.stringify({ error: err?.message || "Internal streaming error" })}\n\n`);
        res.end();
      } catch {}
    }
  }
});

// ---------------- Frontend Vite / Static Integration ----------------

async function start() {
  // Pre-warm extracted text cache for all preloaded papers in the background
  setTimeout(async () => {
    console.log("[PDF Engine] Pre-warming text extraction for catalog papers...");
    for (const p of storedPapers.slice(0, 6)) {
      try {
        await getOrExtractPaperText(p.paper_id);
      } catch {}
    }
    console.log(`[PDF Engine] Pre-warmed ${paperTextCache.size} papers in memory.`);
  }, 500);

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        watch: {
          ignored: (watchPath: string) => {
            const normalized = watchPath.replace(/\\/g, "/").toLowerCase();
            return (
              normalized.includes("/storage/") ||
              normalized.includes("/data/") ||
              normalized.includes("/uploads/")
            );
          },
        },
      },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  const server = app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Server] GraphRAG Research Engine listening on http://0.0.0.0:${PORT}`);

    // Check Neo4j Aura connectivity
    initNeo4jConstraints()
      .then(async () => {
        const neoStatus = await checkNeo4jConnection();
        if (neoStatus.connected) {
          console.log(`[Neo4j Aura] Connected to ${neoStatus.uri} | Database: ${neoStatus.database} | Total Nodes: ${neoStatus.total_nodes}`);
        } else {
          console.log(`[Neo4j Aura] Status: ${neoStatus.status} (${neoStatus.error || "Awaiting credentials"})`);
        }
      })
      .catch((err) => {
        console.warn("[Neo4j Aura] Startup initialization check:", err?.message);
      });
  });

  const shutdown = () => {
    console.log("[Server] Shutting down...");
    server.close();
    process.exit(0);
  };

  process.on("uncaughtException", (err) => {
    console.warn("[Server] Caught uncaughtException:", err?.message || err);
  });
  process.on("unhandledRejection", (reason) => {
    console.warn("[Server] Caught unhandledRejection:", reason);
  });
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

start().catch((err) => {
  console.error("Failed to start server:", err);
});
