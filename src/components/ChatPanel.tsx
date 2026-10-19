import React, { useState, useRef, useEffect } from "react";
import {
  CornerDownLeft, Copy, Check, Sparkles, BookOpen, Clock, Cpu,
  Layers, GitCommit, ChevronDown, ChevronRight, MessageSquare,
  History, Plus, Trash2, X
} from "lucide-react";
import Markdown, { Components } from "react-markdown";
import { QueryResult, Paper, TraversalTrace } from "../types";

export interface ChatSessionItem {
  id: string;
  title: string;
  question: string;
  answer: string;
  mode: string;
  engine?: string;
  queryResult?: QueryResult | null;
  updatedAt: number;
}

interface ChatPanelProps {
  selectedPapers: Paper[];
  onExecuteQuery: (question: string, mode: string) => Promise<void>;
  isQuerying: boolean;
  queryResult: QueryResult | null;
  streamedAnswer: string;
  trace?: TraversalTrace | null;
  activeEngine?: string;
  theme: "dark" | "light";
}

const SINGLE_PAPER_PRESETS: Record<string, string[]> = {
  "attention": [
    "How does Multi-Head Attention replace recurrence in sequence modeling?",
    "Explain why Scaled Dot-Product Attention divides by sqrt(d_k).",
    "What were the BLEU score results on WMT 2014 English-to-German?",
  ],
  "gcn": [
    "What is the renormalization trick and why is it necessary for deep GCN layers?",
    "What classification accuracy did GCN achieve on Cora and Citeseer?",
  ],
  "gat": [
    "How do attention coefficients alpha_ij allow anisotropic message passing?",
    "What Micro-averaged F1 score was achieved on the inductive PPI dataset?",
  ],
  "graphsage": [
    "How does GraphSAGE transition from transductive learning to inductive aggregation?",
    "Explain the difference between the Mean, LSTM, and Pooling aggregators.",
  ],
  "rag": [
    "Compare RAG-Sequence and RAG-Token formulations in terms of marginalization.",
    "What are the primary components of RAG's dense retriever and parametric generator?",
  ],
  "bert": [
    "Explain the Masked Language Model (MLM) and Next Sentence Prediction (NSP) objectives.",
    "How does BERT's bidirectional representation differ from GPT's left-to-right approach?",
  ],
  "cot": [
    "What is Chain-of-Thought prompting, and how does it elicit multi-step reasoning?",
    "What accuracy did Chain-of-Thought achieve on the GSM8K math reasoning benchmark?",
  ],
};

const MULTI_PAPER_PRESETS = [
  "Compare the attention mechanism in Transformer vs Graph Attention Networks (GAT).",
  "How does inductive learning in GraphSAGE compare with transductive learning in GCN and DeepWalk?",
  "Synthesize how BERT and RAG leverage transformer foundations for representation vs retrieval.",
  "What are the common graph neural network paradigms across the selected papers?",
];

// Styled Markdown renderer with crisp typography and clean hierarchy
const StyledMarkdown: React.FC<{ content: string; isDark: boolean }> = ({ content, isDark }) => {
  const c = isDark
    ? {
        text: "#cbd5e1",
        heading: "#f8fafc",
        headingSec: "#e2e8f0",
        code: "#a5b4fc",
        codeBg: "#0f172a",
        border: "#334155",
        link: "#818cf8",
        blockquote: "#94a3b8",
        blockquoteBg: "#1e1b4b33",
        strong: "#ffffff",
        accent: "#6366f1",
      }
    : {
        text: "#1e293b",
        heading: "#0f172a",
        headingSec: "#1e293b",
        code: "#4338ca",
        codeBg: "#f1f5f9",
        border: "#cbd5e1",
        link: "#4f46e5",
        blockquote: "#475569",
        blockquoteBg: "#f8fafc",
        strong: "#0f172a",
        accent: "#4f46e5",
      };

  const components: Components = {
    h1: ({ children }) => (
      <h1
        style={{
          color: c.heading,
          fontSize: "1.1rem",
          fontWeight: 700,
          marginBottom: "0.6rem",
          marginTop: "1rem",
          lineHeight: 1.35,
          borderBottom: `1px solid ${c.border}`,
          paddingBottom: "0.35rem",
          letterSpacing: "-0.01em",
        }}
      >
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2
        style={{
          color: c.heading,
          fontSize: "0.98rem",
          fontWeight: 700,
          marginBottom: "0.5rem",
          marginTop: "0.9rem",
          lineHeight: 1.35,
          letterSpacing: "-0.01em",
        }}
      >
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3
        style={{
          color: c.headingSec,
          fontSize: "0.9rem",
          fontWeight: 600,
          marginBottom: "0.45rem",
          marginTop: "0.8rem",
          lineHeight: 1.35,
        }}
      >
        {children}
      </h3>
    ),
    h4: ({ children }) => (
      <h4
        style={{
          color: c.headingSec,
          fontSize: "0.82rem",
          fontWeight: 600,
          marginBottom: "0.35rem",
          marginTop: "0.6rem",
        }}
      >
        {children}
      </h4>
    ),
    p: ({ children }) => (
      <p
        style={{
          color: c.text,
          fontSize: "0.82rem",
          lineHeight: 1.72,
          marginBottom: "0.75rem",
        }}
      >
        {children}
      </p>
    ),
    strong: ({ children }) => (
      <strong style={{ color: c.strong, fontWeight: 700 }}>{children}</strong>
    ),
    em: ({ children }) => (
      <em style={{ color: c.blockquote, fontStyle: "italic" }}>{children}</em>
    ),
    ul: ({ children }) => (
      <ul
        style={{
          color: c.text,
          fontSize: "0.82rem",
          lineHeight: 1.7,
          paddingLeft: "1.25rem",
          marginBottom: "0.75rem",
          listStyleType: "disc",
        }}
      >
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol
        style={{
          color: c.text,
          fontSize: "0.82rem",
          lineHeight: 1.7,
          paddingLeft: "1.25rem",
          marginBottom: "0.75rem",
          listStyleType: "decimal",
        }}
      >
        {children}
      </ol>
    ),
    li: ({ children }) => (
      <li style={{ marginBottom: "0.25rem" }}>{children}</li>
    ),
    code: ({ children, className }) => {
      const isBlock = className?.includes("language-");
      return isBlock ? (
        <code
          style={{
            display: "block",
            backgroundColor: c.codeBg,
            color: c.code,
            fontSize: "0.75rem",
            padding: "0.65rem 0.85rem",
            borderRadius: "8px",
            overflowX: "auto",
            whiteSpace: "pre",
            marginBottom: "0.75rem",
            border: `1px solid ${c.border}`,
            fontFamily: "ui-monospace, monospace",
            lineHeight: 1.5,
          }}
        >
          {children}
        </code>
      ) : (
        <code
          style={{
            backgroundColor: c.codeBg,
            color: c.code,
            fontSize: "0.75rem",
            padding: "0.1rem 0.35rem",
            borderRadius: "4px",
            border: `1px solid ${c.border}`,
            fontFamily: "ui-monospace, monospace",
          }}
        >
          {children}
        </code>
      );
    },
    pre: ({ children }) => (
      <pre style={{ background: "none", margin: 0, padding: 0 }}>{children}</pre>
    ),
    blockquote: ({ children }) => (
      <blockquote
        style={{
          borderLeft: `3px solid ${c.accent}`,
          backgroundColor: c.blockquoteBg,
          padding: "0.5rem 0.75rem",
          borderRadius: "0 6px 6px 0",
          marginLeft: 0,
          marginBottom: "0.75rem",
          color: c.blockquote,
          fontStyle: "normal",
          fontSize: "0.8rem",
          lineHeight: 1.6,
        }}
      >
        {children}
      </blockquote>
    ),
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        style={{ color: c.link, textDecoration: "underline", fontSize: "0.82rem" }}
      >
        {children}
      </a>
    ),
    hr: () => (
      <hr
        style={{
          borderColor: c.border,
          marginTop: "0.75rem",
          marginBottom: "0.75rem",
          opacity: 0.5,
        }}
      />
    ),
    table: ({ children }) => (
      <table
        style={{
          width: "100%",
          fontSize: "0.78rem",
          borderCollapse: "collapse",
          marginBottom: "0.75rem",
        }}
      >
        {children}
      </table>
    ),
    th: ({ children }) => (
      <th
        style={{
          color: c.heading,
          fontWeight: 600,
          padding: "0.35rem 0.6rem",
          borderBottom: `1px solid ${c.border}`,
          textAlign: "left",
        }}
      >
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td
        style={{
          color: c.text,
          padding: "0.3rem 0.6rem",
          borderBottom: `1px solid ${c.border}`,
        }}
      >
        {children}
      </td>
    ),
  };

  return (
    <div className="select-text cursor-text" style={{ userSelect: "text", WebkitUserSelect: "text" }}>
      <Markdown components={components}>{content}</Markdown>
    </div>
  );
};

export const ChatPanel: React.FC<ChatPanelProps> = ({
  selectedPapers,
  onExecuteQuery,
  isQuerying,
  queryResult,
  streamedAnswer,
  trace,
  activeEngine,
  theme,
}) => {
  const isDark = theme === "dark";
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [mode, setMode] = useState<string>("ours");
  const [copied, setCopied] = useState(false);
  const [showPassages, setShowPassages] = useState(true);
  const [showTrace, setShowTrace] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Sessions state stored in localStorage
  const [sessions, setSessions] = useState<ChatSessionItem[]>(() => {
    try {
      const saved = localStorage.getItem("graphrag_sessions");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Auto-save query to sessions
  useEffect(() => {
    if (queryResult && lastQuestion) {
      const newOrUpdatedSession: ChatSessionItem = {
        id: activeSessionId || `session-${Date.now()}`,
        title: lastQuestion.length > 40 ? lastQuestion.slice(0, 40) + "…" : lastQuestion,
        question: lastQuestion,
        answer: queryResult.answer,
        mode: queryResult.mode || mode,
        engine: queryResult.engine || activeEngine,
        queryResult,
        updatedAt: Date.now(),
      };

      setSessions((prev) => {
        const filtered = prev.filter((s) => s.id !== newOrUpdatedSession.id);
        const next = [newOrUpdatedSession, ...filtered];
        try {
          localStorage.setItem("graphrag_sessions", JSON.stringify(next));
        } catch {}
        return next;
      });

      if (!activeSessionId) {
        setActiveSessionId(newOrUpdatedSession.id);
      }
    }
  }, [queryResult]);

  useEffect(() => {
    if (streamedAnswer || isQuerying) {
      if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      }
    }
  }, [streamedAnswer, isQuerying]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isQuerying) return;
    const q = question.trim();
    setLastQuestion(q);
    setQuestion("");
    onExecuteQuery(q, mode);
  };

  const handleSelectPreset = (q: string) => {
    setLastQuestion(q);
    setQuestion("");
    onExecuteQuery(q, mode);
  };

  const handleNewChat = () => {
    setActiveSessionId(null);
    setLastQuestion("");
    setShowHistory(false);
  };

  const handleSelectSession = (s: ChatSessionItem) => {
    setActiveSessionId(s.id);
    setLastQuestion(s.question);
    setMode(s.mode);
    setShowHistory(false);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      try {
        localStorage.setItem("graphrag_sessions", JSON.stringify(next));
      } catch {}
      return next;
    });
    if (activeSessionId === id) {
      setActiveSessionId(null);
      setLastQuestion("");
    }
  };

  const handleCopy = async () => {
    const textToCopy = streamedAnswer || queryResult?.answer || "";
    if (!textToCopy) return;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(textToCopy);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = textToCopy;
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        textArea.style.top = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand("copy");
        textArea.remove();
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn("Failed to copy answer to clipboard:", err);
    }
  };

  // Determine dynamic presets
  let presets: string[] = [];
  if (selectedPapers.length > 1) {
    presets = MULTI_PAPER_PRESETS;
  } else if (selectedPapers.length === 1) {
    const titleLower = selectedPapers[0].title.toLowerCase();
    const matchedKey = Object.keys(SINGLE_PAPER_PRESETS).find((k) => titleLower.includes(k));
    presets = matchedKey ? SINGLE_PAPER_PRESETS[matchedKey] : [
      "What are the main theoretical contributions of this paper?",
      "What datasets and empirical evaluation benchmarks were used?",
      "Explain the core mathematical formulation and equations.",
    ];
  } else {
    presets = [
      "Select one or more research papers from the left sidebar to begin graph-grounded analysis.",
    ];
  }

  const activeAnswer = streamedAnswer || queryResult?.answer;

  return (
    <aside
      className={`w-96 h-full border-l flex flex-col transition-colors duration-200 relative ${
        isDark
          ? "bg-slate-950 border-slate-800/80 text-slate-200"
          : "bg-slate-50 border-slate-200 text-slate-800"
      }`}
    >
      {/* Chat Header */}
      <div
        className={`p-3.5 border-b flex flex-col gap-2.5 ${
          isDark ? "border-slate-800/80 bg-slate-900/40" : "border-slate-200 bg-white"
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                GraphRAG Assistant
              </h2>
              {/* History button */}
              <button
                type="button"
                onClick={() => setShowHistory(!showHistory)}
                className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border transition-colors ${
                  showHistory
                    ? "bg-indigo-600 text-white border-indigo-500"
                    : isDark
                    ? "bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300"
                    : "bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700"
                }`}
                title="View Chat Sessions History"
              >
                <History className="w-3 h-3 text-indigo-400" />
                <span>Sessions ({sessions.length})</span>
              </button>
            </div>
            <div className="text-xs text-slate-500 mt-0.5 truncate max-w-[210px]">
              {selectedPapers.length === 0
                ? "No papers loaded"
                : selectedPapers.length === 1
                ? `Grounded: ${selectedPapers[0].title}`
                : `Synthesizing ${selectedPapers.length} papers`}
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {/* New Chat */}
            <button
              onClick={handleNewChat}
              className={`p-1.5 rounded-md border text-xs transition-colors flex items-center gap-1 ${
                isDark
                  ? "bg-slate-900 hover:bg-slate-800 border-slate-700 text-slate-300"
                  : "bg-white hover:bg-slate-100 border-slate-300 text-slate-700 shadow-2xs"
              }`}
              title="Start a new chat session"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>

            {/* Clean Research Search Mode Selector */}
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={isQuerying}
              className={`text-[11px] font-medium py-1 px-1.5 rounded-lg border outline-none transition-colors ${
                isDark
                  ? "bg-slate-900 border-slate-700/80 text-indigo-300 focus:border-indigo-500"
                  : "bg-white border-slate-300 text-indigo-700 focus:border-indigo-600 shadow-2xs"
              }`}
              title="Select retrieval & search method for research comparison"
            >
              <option value="ours">Ours (Quotient + Bridges)</option>
              <option value="static">Static GraphRAG</option>
              <option value="dynamic">Dynamic GraphRAG</option>
              <option value="local">Local Search</option>
            </select>

            {/* Copy button */}
            {activeAnswer && (
              <button
                onClick={handleCopy}
                className={`p-1.5 rounded-md border text-xs transition-colors flex items-center gap-1 ${
                  isDark
                    ? "bg-slate-900 hover:bg-slate-800 border-slate-700 text-slate-300"
                    : "bg-white hover:bg-slate-100 border-slate-300 text-slate-700 shadow-2xs"
                }`}
                title="Copy answer to clipboard"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            )}
          </div>
        </div>

        {/* Sessions Dropdown Panel if toggled */}
        {showHistory && (
          <div
            className={`p-2 rounded-lg border shadow-xl flex flex-col gap-1 max-h-48 overflow-y-auto animate-in fade-in duration-100 ${
              isDark ? "bg-slate-900 border-slate-700 text-white" : "bg-white border-slate-300 text-slate-900"
            }`}
          >
            <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400 pb-1 border-b border-slate-800 px-1">
              <span>Saved Chat Sessions</span>
              <button onClick={() => setShowHistory(false)} className="hover:text-slate-200">
                <X className="w-3 h-3" />
              </button>
            </div>
            {sessions.length === 0 ? (
              <div className="text-center py-4 text-xs text-slate-500">
                No past sessions yet. Ask a question to start.
              </div>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => handleSelectSession(s)}
                  className={`px-2 py-1.5 rounded-md text-xs cursor-pointer flex items-center justify-between gap-2 transition-colors ${
                    activeSessionId === s.id
                      ? "bg-indigo-600/20 text-indigo-300 font-medium border border-indigo-500/30"
                      : isDark
                      ? "hover:bg-slate-800 text-slate-300"
                      : "hover:bg-slate-100 text-slate-700"
                  }`}
                >
                  <div className="truncate flex-1">
                    <div className="truncate font-medium">{s.title}</div>
                    <div className="text-[10px] text-slate-500 flex items-center gap-1 mt-0.5">
                      <span>{new Date(s.updatedAt).toLocaleDateString()}</span>
                      <span>·</span>
                      <span className="uppercase text-[9px] font-mono">{s.mode}</span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    className="text-slate-500 hover:text-rose-400 p-1 shrink-0"
                    title="Delete session"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Answer & Chat Stream Area */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {activeAnswer ? (
          <div className="space-y-4">
            {/* Active Engine Indicator Badge */}
            <div className="flex items-center justify-between gap-2">
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {queryResult?.engine || activeEngine || "Local Ollama (llama3.2:3b)"}
              </span>
              {isQuerying && (
                <span className="text-[10px] text-indigo-400 animate-pulse font-medium">
                  Streaming tokens...
                </span>
              )}
            </div>

            {/* User Question Bubble */}
            {lastQuestion && (
              <div
                className={`p-3 rounded-xl border text-xs font-medium leading-relaxed select-text cursor-text ${
                  isDark
                    ? "bg-indigo-950/40 border-indigo-800/60 text-indigo-200"
                    : "bg-indigo-50 border-indigo-200 text-indigo-900"
                }`}
              >
                <span className="text-[10px] uppercase font-bold text-indigo-400 block mb-0.5 tracking-wider select-none">
                  Your Query
                </span>
                "{lastQuestion}"
              </div>
            )}

            {/* Metric strip */}
            {queryResult && (
              <div
                className={`p-2.5 rounded-lg border text-[11px] flex items-center justify-between gap-2 ${
                  isDark
                    ? "bg-slate-900/60 border-slate-800 text-slate-400"
                    : "bg-white border-slate-200 text-slate-600 shadow-2xs"
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3 h-3 text-cyan-500" />
                  <span>{queryResult.latency_ms ?? 0}ms</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Cpu className="w-3 h-3 text-indigo-500" />
                  <span>{queryResult.total_tokens ?? 0} tokens</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Layers className="w-3 h-3 text-emerald-500" />
                  <span>{queryResult.bridge_kept_count ?? 0} bridges kept</span>
                </div>
              </div>
            )}

            {/* Answer Card with Direct Copy Action & Fully Selectable Content */}
            <div
              className={`rounded-xl border p-3.5 shadow-sm transition-all select-text cursor-auto ${
                isDark
                  ? "bg-slate-900/40 border-slate-800/80"
                  : "bg-white border-slate-200/90 shadow-2xs"
              }`}
            >
              <div className="flex items-center justify-between pb-2.5 mb-3 border-b border-slate-800/40 select-none">
                <div className="flex items-center gap-1.5">
                  <div className="w-5 h-5 rounded-md bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                    <Sparkles className="w-3 h-3" />
                  </div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Answer Output
                  </span>
                </div>
                <button
                  type="button"
                  onClick={handleCopy}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-all shadow-2xs ${
                    copied
                      ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-400"
                      : isDark
                      ? "bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300 hover:text-white"
                      : "bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700 hover:text-slate-900"
                  }`}
                  title="Copy complete answer to clipboard"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-[11px] font-semibold text-emerald-400">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-slate-400" />
                      <span className="text-[11px]">Copy Complete Output</span>
                    </>
                  )}
                </button>
              </div>

              {/* Markdown Answer */}
              <div className="leading-relaxed select-text cursor-text" style={{ userSelect: "text" }}>
                <StyledMarkdown content={activeAnswer} isDark={isDark} />
              </div>
            </div>

            {/* Graph Traversal Discovery Path Accordion */}
            {trace && trace.steps && trace.steps.length > 0 && (
              <div className="pt-3 border-t border-slate-800/60 space-y-2">
                <button
                  type="button"
                  onClick={() => setShowTrace(!showTrace)}
                  className="w-full flex items-center justify-between text-[11px] font-semibold text-slate-400 hover:text-indigo-400 transition-colors uppercase tracking-wider"
                >
                  <div className="flex items-center gap-1.5">
                    <GitCommit className="w-3.5 h-3.5 text-amber-500" />
                    <span>Graph Traversal Chain ({trace.steps.length} steps)</span>
                  </div>
                  {showTrace ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                </button>

                {showTrace && (
                  <div className="space-y-1.5">
                    {trace.steps.map((s, idx) => (
                      <div
                        key={idx}
                        className={`p-2 rounded-lg border text-[11px] leading-normal flex items-start gap-2 ${
                          isDark
                            ? "bg-slate-900/50 border-slate-800/80 text-slate-300"
                            : "bg-white border-slate-200 text-slate-700 shadow-2xs"
                        }`}
                      >
                        <span className="font-mono text-amber-500 text-[10px] font-bold shrink-0 mt-0.5">
                          #{s.step_index}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold text-indigo-400">
                              {s.action}
                            </span>
                            <span className="text-slate-500">·</span>
                            <span className="font-mono text-[10px] text-slate-400 truncate">
                              '{s.node_or_comm_id}'
                            </span>
                          </div>
                          <p className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                            {s.reason}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Cited Real Passages Accordion */}
            {queryResult?.cited_passages && queryResult.cited_passages.length > 0 && (
              <div className="pt-3 border-t border-slate-800/60 space-y-2">
                <button
                  type="button"
                  onClick={() => setShowPassages(!showPassages)}
                  className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-400 hover:text-indigo-400 transition-colors uppercase tracking-wider"
                >
                  <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Grounded Document Passages ({queryResult.cited_passages.length})</span>
                </button>

                {showPassages && (
                  <div className="space-y-1.5">
                    {queryResult.cited_passages.map((p, idx) => (
                      <div
                        key={idx}
                        className={`p-2 rounded-lg border text-[11px] leading-snug ${
                          isDark
                            ? "bg-slate-900/60 border-slate-800/80 text-slate-300"
                            : "bg-white border-slate-200 text-slate-700 shadow-2xs"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-0.5">
                          <span className="font-semibold text-indigo-400 text-[10px] truncate">
                            {p.paper_title}
                          </span>
                          <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 shrink-0">
                            Evidence #{idx + 1}
                          </span>
                        </div>
                        <p className="line-clamp-2 text-[10.5px] italic text-slate-400">
                          "{p.text}"
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : isQuerying ? (
          <div className="h-full flex flex-col justify-center items-center text-center px-4 space-y-4">
            {lastQuestion && (
              <div
                className={`w-full p-3 rounded-xl border text-xs font-medium text-left leading-relaxed ${
                  isDark
                    ? "bg-indigo-950/40 border-indigo-800/60 text-indigo-200"
                    : "bg-indigo-50 border-indigo-200 text-indigo-900"
                }`}
              >
                <span className="text-[10px] uppercase font-bold text-indigo-400 block mb-0.5 tracking-wider">
                  Analyzing Question
                </span>
                "{lastQuestion}"
              </div>
            )}
            <div className="relative">
              <div className="w-12 h-12 rounded-2xl bg-indigo-500/20 flex items-center justify-center text-indigo-400 animate-pulse">
                <Sparkles className="w-6 h-6 animate-spin" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-white animate-ping" />
              </div>
            </div>
            <div>
              <h3 className={`text-xs font-semibold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                Analyzing Graph & Traversing Evidence...
              </h3>
              <p className="text-[11px] text-slate-500 mt-1 leading-normal max-w-xs">
                Retrieving topological subgraphs, evaluating cut-edges, and generating grounded scientific synthesis.
              </p>
            </div>
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-indigo-400">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" />
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.2s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.4s]" />
              <span className="ml-1 text-slate-400 font-sans">{activeEngine || "Groq (openai/gpt-oss-120b)"}</span>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col justify-center text-center px-4 space-y-3">
            <div className="w-10 h-10 mx-auto rounded-xl bg-indigo-500/10 flex items-center justify-center text-indigo-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3
                className={`text-xs font-semibold ${
                  isDark ? "text-slate-200" : "text-slate-800"
                }`}
              >
                Ask Anything About the Papers
              </h3>
              <p className="text-[11px] text-slate-500 mt-1 leading-normal">
                Answers are grounded in real PDF passages and topological relationships in Neo4j.
              </p>
            </div>

            {/* Preset Query Chips */}
            <div className="space-y-1.5 pt-2 text-left">
              <span className="text-[10px] uppercase font-semibold text-slate-500 tracking-wider block">
                Suggested Research Queries:
              </span>
              {presets.map((preset, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    handleSelectPreset(preset);
                  }}
                  className={`w-full text-left p-2 rounded-lg border text-[11px] transition-all leading-snug cursor-pointer ${
                    isDark
                      ? "bg-slate-900/60 hover:bg-slate-900 border-slate-800/80 text-slate-300 hover:border-slate-700"
                      : "bg-white hover:bg-slate-100 border-slate-200 text-slate-700 shadow-2xs"
                  }`}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* Input Form */}
      <div
        className={`p-3 border-t ${
          isDark ? "border-slate-800/80 bg-slate-950" : "border-slate-200 bg-white"
        }`}
      >
        <form onSubmit={handleSubmit} className="space-y-2">
          <div className="relative">
            <textarea
              rows={2}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!question.trim() || isQuerying || selectedPapers.length === 0) return;
                  const q = question.trim();
                  setLastQuestion(q);
                  setQuestion("");
                  onExecuteQuery(q, mode);
                }
              }}
              placeholder={
                selectedPapers.length === 0
                  ? "Select a paper first..."
                  : selectedPapers.length === 1
                  ? `Ask about ${selectedPapers[0].title}...`
                  : `Ask question synthesizing ${selectedPapers.length} papers...`
              }
              disabled={isQuerying || selectedPapers.length === 0}
              className={`w-full p-2.5 pr-10 text-xs rounded-xl border outline-none resize-none transition-all ${
                isDark
                  ? "bg-slate-900 border-slate-800 text-white placeholder-slate-500 focus:border-indigo-500"
                  : "bg-slate-100/70 border-slate-200 text-slate-900 placeholder-slate-400 focus:border-indigo-600 focus:bg-white"
              }`}
            />
            <button
              type="button"
              onClick={() => {
                if (!question.trim() || isQuerying || selectedPapers.length === 0) return;
                const q = question.trim();
                setLastQuestion(q);
                setQuestion("");
                onExecuteQuery(q, mode);
              }}
              disabled={isQuerying || !question.trim() || selectedPapers.length === 0}
              className="absolute right-2.5 bottom-3.5 p-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white transition-all shadow-xs cursor-pointer"
            >
              <CornerDownLeft className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>Enter to send · Shift+Enter for newline</span>
            <span className="font-mono">{isQuerying ? "Streaming tokens..." : "Ready"}</span>
          </div>
        </form>
      </div>
    </aside>
  );
};
