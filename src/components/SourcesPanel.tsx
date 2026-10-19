import React, { useState, useRef } from "react";
import {
  FolderOpen, Folder, Plus, Upload, Trash2, Search,
  FileText, ChevronRight, ChevronDown, Loader2,
  CheckSquare, Square, BookOpen, X, AlertCircle, Check
} from "lucide-react";
import { Paper } from "../types";

export interface SourceFolder {
  id: string;
  name: string;
  paperIds: string[];
}

interface SourcesPanelProps {
  papers: Paper[];
  selectedPaperIds: string[];
  onTogglePaper: (id: string) => void;
  onSelectOnlyPaper: (id: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onDeletePaper: (id: string) => void;
  onPapersUploaded: () => void;
  onSummarizePaper?: (paperId: string) => Promise<string>;
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({
  papers,
  selectedPaperIds,
  onTogglePaper,
  onSelectOnlyPaper,
  onSelectAll,
  onClearSelection,
  onDeletePaper,
  onPapersUploaded,
  onSummarizePaper,
}) => {
  const [search, setSearch] = useState("");
  const [folders, setFolders] = useState<SourceFolder[]>([
    { id: "all", name: "All Sources", paperIds: [] },
  ]);
  const [activeFolder, setActiveFolder] = useState("all");
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["all"]));
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [summaryPaperId, setSummaryPaperId] = useState<string | null>(null);
  const [summaryText, setSummaryText] = useState<string>("");
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [uploadTarget, setUploadTarget] = useState<string>("all");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const getPapersForFolder = (folderId: string) => {
    if (folderId === "all") return papers;
    const folder = folders.find(f => f.id === folderId);
    if (!folder) return [];
    return papers.filter(p => folder.paperIds.includes(p.paper_id));
  };

  const filteredPapers = getPapersForFolder(activeFolder).filter(p => {
    const q = search.toLowerCase();
    return p.title.toLowerCase().includes(q) || (p.filename && p.filename.toLowerCase().includes(q));
  });

  const handleFileUpload = async (files: FileList | null, folderId = activeFolder) => {
    if (!files || files.length === 0) return;
    const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) { alert("Please upload PDF files."); return; }

    const titles: string[] = [];
    for (const file of pdfs) {
      const def = file.name.replace(/\.pdf$/i, "").replace(/[_-]/g, " ").trim();
      const t = window.prompt(`Title for "${file.name}":`, def);
      titles.push((t && t.trim()) ? t.trim() : def);
    }

    setIsUploading(true);
    setUploadMsg(`Indexing ${pdfs.length} PDF(s)…`);
    try {
      const fd = new FormData();
      pdfs.forEach((f, i) => { fd.append("files", f); fd.append("custom_titles", titles[i]); });
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || "Upload failed"); }
      const data = await res.json();
      setUploadMsg(`Indexed ${pdfs.length} paper(s)!`);
      if (folderId !== "all") {
        const paperIds = (data.papers || []).map((p: any) => p.paper_id).filter(Boolean);
        setFolders(prev => prev.map(f =>
          f.id === folderId ? { ...f, paperIds: [...new Set([...f.paperIds, ...paperIds])] } : f
        ));
      }
      setTimeout(() => { setIsUploading(false); setUploadMsg(null); onPapersUploaded(); }, 700);
    } catch (err: any) {
      alert(err.message || "Upload failed"); setIsUploading(false); setUploadMsg(null);
    }
  };

  const handleCreateFolder = () => {
    const name = newFolderName.trim();
    if (!name) return;
    const id = `folder-${Date.now()}`;
    setFolders(prev => [...prev, { id, name, paperIds: [] }]);
    setNewFolderName("");
    setShowNewFolder(false);
    setActiveFolder(id);
  };

  const handleDeleteFolder = (folderId: string) => {
    if (folderId === "all") return;
    setFolders(prev => prev.filter(f => f.id !== folderId));
    if (activeFolder === folderId) setActiveFolder("all");
  };

  const toggleFolder = (id: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleSummarize = async (paperId: string) => {
    if (!onSummarizePaper) return;
    setSummaryPaperId(paperId); setSummaryText(""); setIsSummarizing(true);
    try {
      const s = await onSummarizePaper(paperId);
      setSummaryText(s);
    } catch { setSummaryText("Failed to generate summary."); }
    finally { setIsSummarizing(false); }
  };

  const allSelected = papers.length > 0 && selectedPaperIds.length === papers.length;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Panel header */}
      <div className="panel-header">
        <div className="flex-between" style={{ marginBottom: 10 }}>
          <span className="panel-title">Sources</span>
          <div style={{ display: "flex", gap: 4 }}>
            <button className="btn-icon" title="Select all" onClick={allSelected ? onClearSelection : onSelectAll}>
              {allSelected ? <CheckSquare size={14} /> : <Square size={14} />}
            </button>
            <button className="btn-icon" title="New folder" onClick={() => setShowNewFolder(v => !v)}>
              <Plus size={14} />
            </button>
          </div>
        </div>

        {showNewFolder && (
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            <input
              className="input input-sm"
              placeholder="Folder name…"
              value={newFolderName}
              onChange={e => setNewFolderName(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") handleCreateFolder(); if (e.key === "Escape") { setShowNewFolder(false); setNewFolderName(""); } }}
              autoFocus
            />
            <button className="btn btn-primary btn-sm" onClick={handleCreateFolder}>Add</button>
          </div>
        )}

        {/* Search */}
        <div style={{ position: "relative" }}>
          <Search size={12} style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", color: "var(--text-tertiary)", pointerEvents: "none" }} />
          <input
            className="input input-sm"
            style={{ paddingLeft: 28 }}
            placeholder="Search sources…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Folder nav */}
      <div style={{ padding: "8px 8px 0", flexShrink: 0 }}>
        {folders.map(folder => (
          <div key={folder.id} style={{ marginBottom: 2 }}>
            <div
              className={`folder-item ${activeFolder === folder.id ? "active" : ""}`}
              onClick={() => setActiveFolder(folder.id)}
            >
              <button
                className="btn-icon"
                style={{ padding: 0, width: 16, height: 16 }}
                onClick={e => { e.stopPropagation(); toggleFolder(folder.id); }}
              >
                {expandedFolders.has(folder.id) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              </button>
              {activeFolder === folder.id ? <FolderOpen size={13} style={{ color: "var(--accent)" }} /> : <Folder size={13} />}
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{folder.name}</span>
              <span style={{ fontSize: 10, color: "var(--text-tertiary)", flexShrink: 0 }}>
                {folder.id === "all" ? papers.length : folder.paperIds.length}
              </span>
              {folder.id !== "all" && (
                <button
                  className="btn-icon"
                  style={{ padding: 2, marginLeft: 2, opacity: 0.6 }}
                  title="Delete folder"
                  onClick={e => { e.stopPropagation(); handleDeleteFolder(folder.id); }}
                >
                  <X size={11} />
                </button>
              )}
            </div>
          </div>
        ))}
        <div className="divider" />
      </div>

      {/* Paper list */}
      <div className="panel-body" style={{ padding: "0 8px 8px" }}>
        {filteredPapers.length === 0 ? (
          <div className="empty-state">
            <FileText />
            <p>No sources{search ? ` matching "${search}"` : " in this folder"}.<br />Upload a PDF below.</p>
          </div>
        ) : (
          filteredPapers.map(paper => {
            const isSelected = selectedPaperIds.includes(paper.paper_id);
            return (
              <div
                key={paper.paper_id}
                className={`source-card ${isSelected ? "active" : ""}`}
                style={{ marginBottom: 4, position: "relative" }}
                onClick={() => onTogglePaper(paper.paper_id)}
              >
                <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                  <button
                    className="btn-icon"
                    style={{ padding: 0, marginTop: 2, flexShrink: 0 }}
                    onClick={e => { e.stopPropagation(); onTogglePaper(paper.paper_id); }}
                  >
                    {isSelected
                      ? <CheckSquare size={13} style={{ color: "var(--accent)" }} />
                      : <Square size={13} />}
                  </button>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="source-card-title">{paper.title}</div>
                    <div className="source-card-meta">
                      <span>{paper.num_entities || 0} ent.</span>
                      <span>·</span>
                      <span>{paper.num_relations || 0} rel.</span>
                      {paper.file_size_kb ? <><span>·</span><span>{Math.round(paper.file_size_kb)}KB</span></> : null}
                      <span>·</span>
                      <span className={`chip ${paper.status === "ready" ? "chip-green" : paper.status === "processing" ? "chip-amber" : "chip-red"}`} style={{ padding: "1px 5px", fontSize: 9 }}>
                        {paper.status}
                      </span>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 2, flexShrink: 0 }}>
                    <button
                      className="btn-icon"
                      style={{ padding: 3, fontSize: 10, fontFamily: "var(--font-sans)" }}
                      title="Focus only this paper"
                      onClick={e => { e.stopPropagation(); onSelectOnlyPaper(paper.paper_id); }}
                    >
                      <span style={{ fontSize: 9, fontWeight: 600 }}>Only</span>
                    </button>
                    {onSummarizePaper && (
                      <button
                        className="btn-icon"
                        style={{ padding: 3 }}
                        title="Summarize paper"
                        onClick={e => { e.stopPropagation(); handleSummarize(paper.paper_id); }}
                      >
                        <BookOpen size={12} />
                      </button>
                    )}
                    <button
                      className="btn-icon"
                      style={{ padding: 3, color: "var(--danger)" }}
                      title="Delete"
                      onClick={e => { e.stopPropagation(); if (window.confirm(`Remove "${paper.title}"?`)) onDeletePaper(paper.paper_id); }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Upload dropzone */}
      <div style={{ padding: "0 8px 8px", flexShrink: 0 }}>
        <div
          className={`dropzone ${dragActive ? "dragging" : ""}`}
          onDragOver={e => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={e => { e.preventDefault(); setDragActive(false); handleFileUpload(e.dataTransfer.files); }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" multiple accept=".pdf" className="hidden" style={{ display: "none" }}
            onChange={e => { handleFileUpload(e.target.files); if (fileInputRef.current) fileInputRef.current.value = ""; }} />
          {isUploading ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
              <div className="spinner" />
              <span style={{ fontSize: 12 }}>{uploadMsg}</span>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <Upload size={18} />
              <span style={{ fontSize: 12, fontWeight: 500 }}>Upload PDFs to {activeFolder === "all" ? "All Sources" : folders.find(f => f.id === activeFolder)?.name}</span>
              <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>Click or drag & drop</span>
            </div>
          )}
        </div>
      </div>

      {/* Summary Modal */}
      {summaryPaperId && (
        <div className="modal-overlay" onClick={() => setSummaryPaperId(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 560 }}>
            <div className="modal-header">
              <span className="modal-title">
                Paper Summary — {papers.find(p => p.paper_id === summaryPaperId)?.title}
              </span>
              <button className="btn-icon" onClick={() => setSummaryPaperId(null)}><X size={15} /></button>
            </div>
            <div className="modal-body">
              {isSummarizing ? (
                <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "20px 0" }}>
                  <div className="spinner" />
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Generating summary…</span>
                </div>
              ) : (
                <p style={{ fontSize: 13, lineHeight: 1.7, color: "var(--text-secondary)", whiteSpace: "pre-wrap" }}>
                  {summaryText || "No summary available."}
                </p>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={() => setSummaryPaperId(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
