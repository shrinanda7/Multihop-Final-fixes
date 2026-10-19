import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { PaperSidebar } from "./components/PaperSidebar";
import { CytoscapeGraphViewer } from "./components/CytoscapeGraphViewer";
import { ChatPanel } from "./components/ChatPanel";
import { Neo4jAuraModal } from "./components/Neo4jAuraModal";
import { NotesDrawer } from "./components/NotesDrawer";
import { SettingsModal } from "./components/SettingsModal";
import { PaperSummaryModal } from "./components/PaperSummaryModal";
import {
Paper,
GraphTopology,
QueryResult,
TraversalTrace,
TraversalStep,
Neo4jStatus,
} from "./types";
import {
PRELOADED_PAPERS,
generatePaperGraph,
generateMultiPaperGraph,
} from "./data";
import { sanitizeGraphTopology } from "./graphSanitize";
import { formatGeneratedAnswer } from "./answerFormat";

export default function App() {
const [theme, setTheme] = useState<"dark" | "light">(() => {
try {
const saved = localStorage.getItem("graphrag_theme");
if (saved === "light" || saved === "dark") return saved;
} catch { }
return "dark";
});

const toggleTheme = () => {
setTheme((prev) => {
const next = prev === "dark" ? "light" : "dark";
try {
localStorage.setItem("graphrag_theme", next);
} catch { }
return next;
});
};

const [papers, setPapers] = useState<Paper[]>([]);
const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);

// Graph state
const [originalGraph, setOriginalGraph] = useState<GraphTopology | null>(null);
const [compressedGraph, setCompressedGraph] = useState<GraphTopology | null>(null);
const [trace, setTrace] = useState<TraversalTrace | null>(null);

// Query and answer state
const [isQuerying, setIsQuerying] = useState(false);
const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
const [streamedAnswer, setStreamedAnswer] = useState("");
const [activeEngine, setActiveEngine] = useState(
"Groq (openai/gpt-oss-120b)"
);

// Neo4j modal state
const [neo4jStatus, setNeo4jStatus] = useState<Neo4jStatus | null>(null);
const [isNeo4jOpen, setIsNeo4jOpen] = useState(false);

// Notebook inspired features state
const [isNotesOpen, setIsNotesOpen] = useState(false);
const [isSettingsOpen, setIsSettingsOpen] = useState(false);
const [isSummarizeOpen, setIsSummarizeOpen] = useState(false);
const [summarizePaper, setSummarizePaper] = useState<Paper | null>(null);
const [notesCount, setNotesCount] = useState<number>(() => {
  try {
    const saved = localStorage.getItem("graphrag_notes");
    return saved ? JSON.parse(saved).length : 0;
  } catch {
    return 0;
  }
});

const refreshNotesCount = () => {
  try {
    const saved = localStorage.getItem("graphrag_notes");
    setNotesCount(saved ? JSON.parse(saved).length : 0);
  } catch {}
};

useEffect(() => {
void loadPapers();
void loadNeo4jStatus();
}, []);

useEffect(() => {
void loadGraphForSelection(selectedPaperIds);
}, [selectedPaperIds]);

const loadPapers = async () => {
try {
const res = await fetch("/api/papers");

if (res.ok) {
const data = await res.json();

if (Array.isArray(data)) {
setPapers(data);
if (data.length > 0) {
setSelectedPaperIds((current) =>
current.length === 0 || !data.some((p) => current.includes(p.paper_id))
  ? [data[0].paper_id]
  : current
);
} else {
setSelectedPaperIds([]);
setOriginalGraph(null);
setCompressedGraph(null);
}
}
}
} catch (err) {
console.warn(
"Backend papers endpoint offline, using preloaded catalog:",
err
);
}
};

const handlePapersUploaded = async (newPaperId?: string) => {
try {
const res = await fetch("/api/papers");
if (res.ok) {
const data = await res.json();
if (Array.isArray(data) && data.length > 0) {
setPapers(data);
if (newPaperId) {
setSelectedPaperIds([newPaperId]);
}
}
}
} catch (err) {
console.warn("Failed to refresh papers after upload:", err);
}
};

const loadNeo4jStatus = async () => {
try {
const res = await fetch("/api/neo4j/status");

if (res.ok) {
setNeo4jStatus(await res.json());
}
} catch (err) {
console.warn("Neo4j status check failed:", err);
}
};

const loadGraphForSelection = async (ids: string[]) => {
if (!ids || ids.length === 0) {
setOriginalGraph(null);
setCompressedGraph(null);
return;
}

if (ids.length === 1) {
try {
const res = await fetch(
`/api/graph/${encodeURIComponent(ids[0])}`
);

if (res.ok) {
const data = await res.json();

if (data.original_graph) {
setOriginalGraph(data.original_graph);
}
if (data.compressed_graph) {
setCompressedGraph(data.compressed_graph);
}
return;
}
} catch (err) {
console.warn("Unable to load graph from backend:", err);
}

const paperExists = papers.some((p) => p.paper_id === ids[0]);
if (paperExists) {
  const fallback = generatePaperGraph(ids[0]);
  setOriginalGraph(fallback.original_graph);
  setCompressedGraph(fallback.compressed_graph);
} else {
  setOriginalGraph(null);
  setCompressedGraph(null);
}
return;
}

try {
const res = await fetch("/api/graph/multi", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ paper_ids: ids }),
});

if (res.ok) {
const data = await res.json();

if (data.original_graph) {
setOriginalGraph(data.original_graph);
}
if (data.compressed_graph) {
setCompressedGraph(data.compressed_graph);
}
return;
}
} catch (err) {
console.warn("Unable to load multi-paper graph:", err);
}

const anyExist = ids.some((id) => papers.some((p) => p.paper_id === id));
if (anyExist) {
  const fallback = generateMultiPaperGraph(ids);
  setOriginalGraph(fallback.original_graph);
  setCompressedGraph(fallback.compressed_graph);
} else {
  setOriginalGraph(null);
  setCompressedGraph(null);
}
};

// Paper selection handlers
const handleTogglePaper = (paperId: string) => {
setSelectedPaperIds((prev) => {
if (prev.includes(paperId)) {
if (prev.length === 1) return prev;
return prev.filter((id) => id !== paperId);
}
return [...prev, paperId];
});
};

const handleSelectOnlyPaper = (paperId: string) => {
setSelectedPaperIds([paperId]);
};

const handleSelectAll = () => {
setSelectedPaperIds(papers.map((paper) => paper.paper_id));
};

const handleClearSelection = () => {
setSelectedPaperIds(
papers.length > 0 ? [papers[0].paper_id] : []
);
};

const handleDeletePaper = async (paperId: string) => {
const remaining = papers.filter((p) => p.paper_id !== paperId);
setPapers(remaining);

if (remaining.length === 0) {
  setSelectedPaperIds([]);
  setOriginalGraph(null);
  setCompressedGraph(null);
} else {
  setSelectedPaperIds((prev) => {
    const updated = prev.filter((id) => id !== paperId);
    return updated.length > 0 ? updated : [remaining[0].paper_id];
  });
}

// Purge paper from backend (memory caches, disk files, and Neo4j graph database)
try {
  await fetch(`/api/papers/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
} catch (err) {
  console.warn("Failed to delete paper from primary backend:", err);
}

try {
  await fetch(`http://127.0.0.1:8001/api/papers/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  }).catch(() => {});
} catch {}

// If all papers deleted, ensure Neo4j is completely purged
if (remaining.length === 0) {
  try {
    await fetch("/api/neo4j/clear", { method: "POST" });
  } catch {}
  try {
    await fetch("http://127.0.0.1:8001/api/neo4j/clear", { method: "POST" });
  } catch {}
}

// Update Neo4j connection status and metrics
await loadNeo4jStatus();
};

const handleExecuteQuery = async (
question: string,
mode: string
) => {
if (selectedPaperIds.length === 0 || isQuerying) return;

setIsQuerying(true);
setStreamedAnswer("");
setQueryResult(null);

const paperIdsForQuery = [...selectedPaperIds];

const freshTrace: TraversalTrace = {
mode,
steps: [],
visited_nodes: [],
expanded_nodes: [],
pruned_nodes: [],
bridge_kept_nodes: [],
selected_reports: [],
ratings: {},
total_tokens: 0,
total_llm_calls: 0,
latency_ms: 0,
};

setTrace(freshTrace);

try {
// Multi-paper Ours: local Tarjan union graph + multi-hop answer.
if (paperIdsForQuery.length > 1 && mode === "ours") {
try {
const response = await fetch("/api/query/multi-paper", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({
paper_ids: paperIdsForQuery,
question,
}),
});

const payload = await response.json().catch(() => ({}));
const rawAnswer =
typeof payload.answer === "string" ? payload.answer.trim() : "";
const noEvidence =
/no source chunks matched|no matching chunks|no_indexed_graph|no_evidence/i.test(
`${rawAnswer} ${payload.status || ""} ${payload.error || ""}`
);

if (!response.ok || !rawAnswer || noEvidence) {
console.warn(
"Multi-paper endpoint unusable, falling back to stream:",
payload?.error || payload?.status || response.status
);
} else {
const original = sanitizeGraphTopology(payload.original_graph);
const compressed = sanitizeGraphTopology(payload.compressed_graph);
if (original) setOriginalGraph(original);
if (compressed) setCompressedGraph(compressed);

const mappedSteps: TraversalStep[] = (
payload.trace?.steps || []
).map((step: any, index: number) => ({
step_index: index + 1,
action: [
"VISIT",
"RATE",
"EXPAND",
"PRUNE",
"BRIDGE_PRESERVE",
"SELECT",
].includes(step.action)
? step.action
: "SELECT",
node_or_comm_id: String(
step.node_id || step.chunk_id || "evidence"
),
level: 0,
score:
typeof step.score === "number" ? step.score : 0,
reason: String(
step.reason ||
"Selected by multi-paper quotient-graph retrieval"
),
timestamp_ms: Date.now(),
}));

const multiTrace: TraversalTrace = {
...freshTrace,
mode: "multi_paper",
steps: mappedSteps,
visited_nodes: mappedSteps.map(
(step) => step.node_or_comm_id
),
total_tokens: payload.total_tokens || 0,
total_llm_calls: payload.total_llm_calls || 0,
latency_ms: payload.latency_ms || 0,
};

const answer = formatGeneratedAnswer(rawAnswer);

setTrace(multiTrace);
setStreamedAnswer(answer);

setQueryResult({
mode: "multi_paper",
engine: payload.engine || `Tarjan multi-hop (${payload.llm_status || payload.status || "ok"})`,
answer,
trace: multiTrace,
original_graph: original || undefined,
compressed_graph: compressed || undefined,
total_tokens: payload.total_tokens || 0,
total_llm_calls: payload.total_llm_calls || 0,
latency_ms: payload.latency_ms || 0,
bridge_kept_count:
payload.compressed_graph?.articulation_points?.length ||
0,
cited_passages: (payload.evidence || []).map(
(item: any, index: number) => ({
paper_title:
item.paper_title || item.paper_id || "Unknown paper",
text: `[${item.evidence_id || `E${index + 1}`}] ${item.content || ""
} (Chunk ${item.chunk_id || "unknown"}; page ${item.page ?? "unknown"
})`,
score: 1,
})
),
papers_evaluated:
payload.paper_ids || paperIdsForQuery,
});

return;
}
} catch (multiErr) {
console.warn("Multi-paper query failed, falling back to stream:", multiErr);
}
}

const response = await fetch("/api/query/stream", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({
paper_ids: paperIdsForQuery,
question,
mode,
}),
});

if (!response.ok || !response.body) {
const errorText = await response.text().catch(() => "");
throw new Error(
errorText || `Query stream failed (${response.status})`
);
}

const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");

let buffer = "";
let answerBuffer = "";
let latestEngine = activeEngine;
let completed = false;

const processSseEvent = (rawEvent: string) => {
if (!rawEvent.trim()) return;

let eventType = "message";
const dataLines: string[] = [];

for (const line of rawEvent.split("\n")) {
if (line.startsWith("event:")) {
eventType =
line.slice("event:".length).trim() || "message";
} else if (line.startsWith("data:")) {
dataLines.push(
line.slice("data:".length).replace(/^ /, "")
);
}
}

const eventData = dataLines.join("\n");
if (!eventData) return;

let parsed: any;

try {
parsed = JSON.parse(eventData);
} catch (err) {
console.error("Invalid SSE JSON payload:", err, eventData);
return;
}

if (eventType === "engine") {
if (
typeof parsed.engine === "string" &&
parsed.engine.trim()
) {
latestEngine = parsed.engine;
setActiveEngine(parsed.engine);
}
return;
}

if (eventType === "topology") {
if (
parsed.original_graph &&
(!originalGraph || originalGraph.nodes.length === 0)
) {
setOriginalGraph(parsed.original_graph);
}

if (
parsed.compressed_graph &&
(!compressedGraph || compressedGraph.nodes.length === 0)
) {
setCompressedGraph(parsed.compressed_graph);
}
return;
}

if (eventType === "trace_step") {
freshTrace.steps.push(parsed as TraversalStep);

if (
parsed.node_or_comm_id &&
!freshTrace.visited_nodes.includes(
parsed.node_or_comm_id
)
) {
freshTrace.visited_nodes.push(
parsed.node_or_comm_id
);
}

setTrace({
...freshTrace,
steps: [...freshTrace.steps],
visited_nodes: [...freshTrace.visited_nodes],
});
return;
}

if (eventType === "answer_token") {
const token =
typeof parsed.token === "string" ? parsed.token : "";

if (token) {
answerBuffer += token;
setStreamedAnswer(answerBuffer);
}
return;
}

if (eventType === "error") {
throw new Error(
String(
parsed.message ||
parsed.error ||
"The answer stream failed."
)
);
}

if (eventType === "complete") {
const finalAnswer =
typeof parsed.answer === "string" &&
parsed.answer.length > 0
? parsed.answer
: answerBuffer;

answerBuffer = finalAnswer;
setStreamedAnswer(finalAnswer);

const finalTrace: TraversalTrace = {
...freshTrace,
steps: [...freshTrace.steps],
visited_nodes: [...freshTrace.visited_nodes],
};

setTrace(finalTrace);
setQueryResult({
mode: parsed.mode || mode,
engine: parsed.engine || latestEngine,
answer: finalAnswer,
trace: finalTrace,
total_tokens:
typeof parsed.total_tokens === "number"
? parsed.total_tokens
: 0,
total_llm_calls:
typeof parsed.total_llm_calls === "number"
? parsed.total_llm_calls
: 0,
latency_ms:
typeof parsed.latency_ms === "number"
? parsed.latency_ms
: 0,
bridge_kept_count:
typeof parsed.bridge_kept_count === "number"
? parsed.bridge_kept_count
: 0,
cited_passages: Array.isArray(parsed.cited_passages)
? parsed.cited_passages
: [],
papers_evaluated: Array.isArray(
parsed.papers_evaluated
)
? parsed.papers_evaluated
: [],
});

completed = true;
}
};

while (true) {
const { done, value } = await reader.read();
if (done) break;

buffer += decoder.decode(value, { stream: true });
buffer = buffer.replace(/\r\n/g, "\n");

const events = buffer.split("\n\n");
buffer = events.pop() || "";

for (const event of events) {
processSseEvent(event);
}
}

// Flush decoder state and process a final event that lacks the
// usual terminating blank line.
buffer += decoder.decode();
buffer = buffer.replace(/\r\n/g, "\n");

if (buffer.trim()) {
processSseEvent(buffer);
}

// Keep a completed streamed answer visible even if the server
// closes the stream without emitting a separate completion event.
if (!completed && answerBuffer.trim()) {
const finalTrace: TraversalTrace = {
...freshTrace,
steps: [...freshTrace.steps],
visited_nodes: [...freshTrace.visited_nodes],
};

setTrace(finalTrace);
setQueryResult({
mode,
engine: latestEngine,
answer: answerBuffer,
trace: finalTrace,
total_tokens: freshTrace.total_tokens || 0,
total_llm_calls: freshTrace.total_llm_calls || 0,
latency_ms: freshTrace.latency_ms || 0,
bridge_kept_count:
freshTrace.bridge_kept_nodes.length,
cited_passages: [],
papers_evaluated: paperIdsForQuery,
});
}
} catch (err: any) {
console.error("Query execution failed:", err);

const errorMessage =
err instanceof Error ? err.message : String(err);

setStreamedAnswer((previous) =>
previous ||
`Unable to complete query: ${errorMessage}`
);
} finally {
setIsQuerying(false);
}
};

const selectedPapers = papers.filter((paper) =>
selectedPaperIds.includes(paper.paper_id)
);
const selectedPaperTitles = selectedPapers.map(
(paper) => paper.title
);

return (
<div
className={`flex flex-col h-screen w-screen font-sans antialiased overflow-hidden transition-colors duration-200 ${theme === "dark"
? "bg-slate-950 text-slate-100"
: "bg-slate-50 text-slate-900"
}`}
>
<Header
  theme={theme}
  onToggleTheme={toggleTheme}
  selectedPapersCount={selectedPaperIds.length}
  totalPapersCount={papers.length}
  neo4jStatus={neo4jStatus}
  onOpenNeo4j={() => setIsNeo4jOpen(true)}
  onOpenNotes={() => {
    refreshNotesCount();
    setIsNotesOpen(true);
  }}
  notesCount={notesCount}
  onOpenSettings={() => setIsSettingsOpen(true)}
/>

<div className="flex-1 flex overflow-hidden">
  <PaperSidebar
    papers={papers}
    selectedPaperIds={selectedPaperIds}
    onTogglePaper={handleTogglePaper}
    onSelectOnlyPaper={handleSelectOnlyPaper}
    onSelectAll={handleSelectAll}
    onClearSelection={handleClearSelection}
    onDeletePaper={handleDeletePaper}
    onPapersUploaded={handlePapersUploaded}
    onSummarizePaper={(paper) => {
      setSummarizePaper(paper);
      setIsSummarizeOpen(true);
    }}
    theme={theme}
  />

  <CytoscapeGraphViewer
    originalGraph={originalGraph}
    compressedGraph={compressedGraph}
    trace={trace}
    selectedPaperTitles={selectedPaperTitles}
    theme={theme}
  />

  <ChatPanel
    selectedPapers={selectedPapers}
    onExecuteQuery={handleExecuteQuery}
    isQuerying={isQuerying}
    queryResult={queryResult}
    streamedAnswer={streamedAnswer}
    trace={trace}
    activeEngine={activeEngine}
    theme={theme}
  />
</div>

<Neo4jAuraModal
  isOpen={isNeo4jOpen}
  onClose={() => setIsNeo4jOpen(false)}
  neo4jStatus={neo4jStatus}
  onRefreshStatus={loadNeo4jStatus}
/>

<NotesDrawer
  isOpen={isNotesOpen}
  onClose={() => {
    setIsNotesOpen(false);
    refreshNotesCount();
  }}
  theme={theme}
/>

<SettingsModal
  isOpen={isSettingsOpen}
  onClose={() => setIsSettingsOpen(false)}
  theme={theme}
/>

<PaperSummaryModal
  paper={summarizePaper}
  isOpen={isSummarizeOpen}
  onClose={() => {
    setIsSummarizeOpen(false);
    setSummarizePaper(null);
    refreshNotesCount();
  }}
  theme={theme}
  onAddToNotes={() => refreshNotesCount()}
/>
</div>
);
}