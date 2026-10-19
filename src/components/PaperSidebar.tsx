import React, { useState, useRef, useEffect } from "react";
import {
  Upload, Search, CheckSquare, Square, FileText, Check, Plus, Loader2,
  Trash2, Folder, FolderPlus, Sparkles, Tag, ChevronDown
} from "lucide-react";
import { Paper } from "../types";

interface PaperSidebarProps {
  papers: Paper[];
  selectedPaperIds: string[];
  onTogglePaper: (paperId: string) => void;
  onSelectOnlyPaper: (paperId: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onDeletePaper: (paperId: string) => void;
  onPapersUploaded: (newPaperId?: string) => void;
  onSummarizePaper?: (paper: Paper) => void;
  theme: "dark" | "light";
}

const DEFAULT_FOLDERS = [
  "All Sources",
  "Transformers & Attention",
  "Graph Neural Networks",
  "RAG & Multi-Hop",
  "Custom Uploads",
];

export const PaperSidebar: React.FC<PaperSidebarProps> = ({
  papers,
  selectedPaperIds,
  onTogglePaper,
  onSelectOnlyPaper,
  onSelectAll,
  onClearSelection,
  onDeletePaper,
  onPapersUploaded,
  onSummarizePaper,
  theme,
}) => {
  const isDark = theme === "dark";
  const [searchQuery, setSearchQuery] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Folder states
  const [folders, setFolders] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem("graphrag_folders");
      return saved ? JSON.parse(saved) : DEFAULT_FOLDERS;
    } catch {
      return DEFAULT_FOLDERS;
    }
  });

  const [activeFolder, setActiveFolder] = useState<string>("All Sources");
  const [isCreatingFolder, setIsCreatingFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");

  // Paper-to-folder mapping
  const [paperFolderMap, setPaperFolderMap] = useState<Record<string, string>>(() => {
    try {
      const saved = localStorage.getItem("graphrag_paper_folders");
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("graphrag_folders", JSON.stringify(folders));
    } catch {}
  }, [folders]);

  useEffect(() => {
    try {
      localStorage.setItem("graphrag_paper_folders", JSON.stringify(paperFolderMap));
    } catch {}
  }, [paperFolderMap]);

  // Assign default categories to built-in papers if not set
  useEffect(() => {
    setPaperFolderMap((prev) => {
      const updated = { ...prev };
      let changed = false;
      papers.forEach((p) => {
        if (!updated[p.paper_id]) {
          const t = p.title.toLowerCase();
          if (t.includes("attention") || t.includes("bert")) {
            updated[p.paper_id] = "Transformers & Attention";
            changed = true;
          } else if (t.includes("gcn") || t.includes("gat") || t.includes("graphsage")) {
            updated[p.paper_id] = "Graph Neural Networks";
            changed = true;
          } else if (t.includes("rag") || t.includes("thought")) {
            updated[p.paper_id] = "RAG & Multi-Hop";
            changed = true;
          } else {
            updated[p.paper_id] = "Custom Uploads";
            changed = true;
          }
        }
      });
      return changed ? updated : prev;
    });
  }, [papers]);

  const handleCreateFolder = () => {
    const trimmed = newFolderName.trim();
    if (!trimmed) return;
    if (!folders.includes(trimmed)) {
      setFolders((prev) => [...prev, trimmed]);
      setActiveFolder(trimmed);
    }
    setNewFolderName("");
    setIsCreatingFolder(false);
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const pdfFiles = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfFiles.length === 0) {
      alert("Please upload valid PDF files.");
      return;
    }

    const titles: string[] = [];
    for (const file of pdfFiles) {
      const defaultTitle = file.name.replace(/\.pdf$/i, "").replace(/[_-]/g, " ").trim();
      const enteredTitle = window.prompt(
        `Enter a title for "${file.name}" (or leave blank to use the filename):`,
        defaultTitle
      );
      titles.push(enteredTitle && enteredTitle.trim() ? enteredTitle.trim() : defaultTitle);
    }

    setIsUploading(true);
    setUploadMessage(`Extracting & indexing ${pdfFiles.length} PDF(s)...`);

    try {
      const formData = new FormData();
      pdfFiles.forEach((file, index) => {
        formData.append("files", file);
        formData.append("custom_titles", titles[index]);
      });

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Failed to process upload");
      }

      const data = await res.json();
      const uploadedList = data.uploaded_papers || data.papers || (data.paper ? [data.paper] : []);

      // Tag uploaded papers with the current folder
      if (uploadedList.length > 0) {
        setPaperFolderMap((prev) => {
          const next = { ...prev };
          const target = activeFolder === "All Sources" ? "Custom Uploads" : activeFolder;
          uploadedList.forEach((up: any) => {
            if (up.paper_id) next[up.paper_id] = target;
          });
          return next;
        });
      }

      setUploadMessage(`Indexed ${pdfFiles.length} paper(s) successfully!`);
      setTimeout(() => {
        setIsUploading(false);
        setUploadMessage(null);
        const newestId = uploadedList[0]?.paper_id;
        onPapersUploaded(newestId);
      }, 700);
    } catch (err: any) {
      alert(err.message || "Upload failed");
      setIsUploading(false);
      setUploadMessage(null);
    }
  };

  // Filter papers by search query and active folder
  const filteredPapers = papers.filter((p) => {
    const q = searchQuery.toLowerCase();
    const matchesSearch =
      p.title.toLowerCase().includes(q) ||
      (p.filename && p.filename.toLowerCase().includes(q));

    if (!matchesSearch) return false;
    if (activeFolder === "All Sources") return true;

    const paperFolder = paperFolderMap[p.paper_id] || "Custom Uploads";
    return paperFolder === activeFolder;
  });

  const allSelected = papers.length > 0 && selectedPaperIds.length === papers.length;

  return (
    <aside
      className={`w-80 h-full border-r flex flex-col transition-colors duration-200 select-none ${
        isDark
          ? "bg-slate-950 border-slate-800/80 text-slate-200"
          : "bg-slate-50 border-slate-200 text-slate-800"
      }`}
    >
      {/* Sidebar Header */}
      <div
        className={`p-4 border-b flex flex-col gap-3 ${
          isDark ? "border-slate-800/80 bg-slate-900/40" : "border-slate-200 bg-white"
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Research Corpus
            </h2>
            <div className="text-xs text-slate-500 mt-0.5">
              <span>{selectedPaperIds.length} of {papers.length} active</span>
              <span className="mx-1.5">·</span>
              <span>Multi-select enabled</span>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={allSelected ? onClearSelection : onSelectAll}
              className={`px-2.5 py-1 text-xs font-medium rounded-md border transition-colors ${
                isDark
                  ? "bg-slate-900 hover:bg-slate-800 border-slate-700 text-slate-300"
                  : "bg-white hover:bg-slate-100 border-slate-300 text-slate-700 shadow-2xs"
              }`}
            >
              {allSelected ? "Clear" : "Select All"}
            </button>
          </div>
        </div>

        {/* Search Input */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search papers by title or topic..."
            className={`w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border outline-none transition-all ${
              isDark
                ? "bg-slate-900/80 border-slate-800 text-white placeholder-slate-500 focus:border-indigo-500"
                : "bg-slate-100/70 border-slate-200 text-slate-900 placeholder-slate-400 focus:border-indigo-600 focus:bg-white"
            }`}
          />
        </div>

        {/* Dropzone / Upload Action */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            handleFileUpload(e.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`border border-dashed rounded-lg p-2.5 text-center cursor-pointer transition-all ${
            dragActive
              ? "border-indigo-500 bg-indigo-500/10"
              : isDark
              ? "border-slate-800 hover:border-slate-700 bg-slate-900/30 hover:bg-slate-900/50"
              : "border-slate-300 hover:border-slate-400 bg-white hover:bg-slate-50 shadow-2xs"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf"
            className="hidden"
            onChange={(e) => handleFileUpload(e.target.files)}
          />
          {isUploading ? (
            <div className="flex items-center justify-center gap-2 text-xs text-indigo-400 py-1">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>{uploadMessage || "Indexing PDF text & graph..."}</span>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-2 text-xs font-medium">
              <Upload className="w-3.5 h-3.5 text-indigo-400" />
              <span className={isDark ? "text-slate-300" : "text-slate-700"}>
                Upload PDF to{" "}
                <strong className="text-indigo-400 font-semibold">
                  {activeFolder === "All Sources" ? "Corpus" : activeFolder}
                </strong>
              </span>
            </div>
          )}
        </div>

        {/* Folders Bar (Notebook style) */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1">
              <Folder className="w-3 h-3 text-amber-500" /> Folders / Categories
            </span>
            <button
              onClick={() => setIsCreatingFolder(!isCreatingFolder)}
              className="text-[10px] text-indigo-400 hover:text-indigo-300 flex items-center gap-0.5 font-medium"
              title="Create new folder heading"
            >
              <Plus className="w-3 h-3" /> New
            </button>
          </div>

          {isCreatingFolder && (
            <div className="flex items-center gap-1.5 mb-2">
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateFolder()}
                placeholder="Folder name..."
                className={`flex-1 px-2 py-1 text-xs rounded border outline-none ${
                  isDark
                    ? "bg-slate-800 border-slate-700 text-white"
                    : "bg-white border-slate-300 text-slate-900"
                }`}
                autoFocus
              />
              <button
                onClick={handleCreateFolder}
                className="px-2 py-1 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded font-medium"
              >
                Add
              </button>
            </div>
          )}

          <div className="flex items-center gap-1 overflow-x-auto pb-1 no-scrollbar">
            {folders.map((f) => {
              const count =
                f === "All Sources"
                  ? papers.length
                  : papers.filter((p) => (paperFolderMap[p.paper_id] || "Custom Uploads") === f).length;
              const isActive = activeFolder === f;

              return (
                <button
                  key={f}
                  onClick={() => setActiveFolder(f)}
                  className={`px-2 py-0.5 text-[11px] rounded-full whitespace-nowrap transition-colors flex items-center gap-1 shrink-0 ${
                    isActive
                      ? "bg-indigo-600 text-white font-medium"
                      : isDark
                      ? "bg-slate-800/80 hover:bg-slate-800 text-slate-400 hover:text-slate-200"
                      : "bg-slate-200/80 hover:bg-slate-300 text-slate-700"
                  }`}
                >
                  <span>{f}</span>
                  <span className={`text-[9px] px-1 rounded-full ${isActive ? "bg-indigo-700 text-white" : "text-slate-500"}`}>
                    {count}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Paper List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {filteredPapers.map((paper) => {
          const isSelected = selectedPaperIds.includes(paper.paper_id);
          const folder = paperFolderMap[paper.paper_id] || "Custom Uploads";

          return (
            <div
              key={paper.paper_id}
              className={`p-2.5 rounded-lg border transition-all cursor-pointer group ${
                isSelected
                  ? isDark
                    ? "bg-slate-900/90 border-indigo-500/60 shadow-xs"
                    : "bg-white border-indigo-500/60 shadow-xs"
                  : isDark
                  ? "bg-transparent border-transparent hover:bg-slate-900/50 hover:border-slate-800"
                  : "bg-transparent border-transparent hover:bg-white hover:border-slate-200"
              }`}
              onClick={() => onTogglePaper(paper.paper_id)}
            >
              <div className="flex items-start gap-2.5">
                {/* Checkbox */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onTogglePaper(paper.paper_id);
                  }}
                  className="mt-0.5 text-slate-400 hover:text-indigo-400 transition-colors shrink-0"
                >
                  {isSelected ? (
                    <CheckSquare className="w-4 h-4 text-indigo-500" />
                  ) : (
                    <Square className="w-4 h-4 text-slate-500" />
                  )}
                </button>

                <div className="flex-1 min-w-0">
                  <h3
                    className={`text-xs font-medium leading-snug line-clamp-2 ${
                      isSelected
                        ? isDark
                          ? "text-white"
                          : "text-slate-950 font-semibold"
                        : isDark
                        ? "text-slate-300 group-hover:text-white"
                        : "text-slate-700 group-hover:text-slate-900"
                    }`}
                  >
                    {paper.title}
                  </h3>

                  {/* Clean metadata */}
                  <div className="mt-1 flex items-center gap-1.5 text-[11px] text-slate-500 flex-wrap">
                    <span>{paper.num_entities || 16} entities</span>
                    <span aria-hidden="true">·</span>
                    <span>{paper.num_relations || 22} relations</span>
                    {activeFolder === "All Sources" && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="text-[10px] text-slate-400 font-mono bg-slate-800/40 px-1 rounded">
                          {folder}
                        </span>
                      </>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {/* Summarize paper button */}
                  {onSummarizePaper && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSummarizePaper(paper);
                      }}
                      className={`opacity-0 group-hover:opacity-100 p-1 text-[10px] font-medium rounded transition-opacity ${
                        isDark
                          ? "bg-slate-800 hover:bg-amber-950/80 text-slate-400 hover:text-amber-300"
                          : "bg-slate-100 hover:bg-amber-50 text-slate-500 hover:text-amber-600"
                      }`}
                      title="Summarize Paper"
                    >
                      <Sparkles className="w-3 h-3 text-amber-400" />
                    </button>
                  )}

                  {/* "Focus Only" single selection trigger */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectOnlyPaper(paper.paper_id);
                    }}
                    className={`opacity-0 group-hover:opacity-100 px-1.5 py-0.5 text-[10px] font-medium rounded transition-opacity ${
                      isDark
                        ? "bg-slate-800 hover:bg-slate-700 text-slate-300"
                        : "bg-slate-100 hover:bg-slate-200 text-slate-700"
                    }`}
                    title="Focus solely on this paper"
                  >
                    Only
                  </button>

                  {/* Delete source button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeletePaper(paper.paper_id);
                    }}
                    className={`opacity-0 group-hover:opacity-100 p-1 text-[10px] font-medium rounded transition-opacity ${
                      isDark
                        ? "bg-slate-800/80 hover:bg-rose-950/80 text-slate-400 hover:text-rose-400 border border-transparent hover:border-rose-800/50"
                        : "bg-slate-100 hover:bg-rose-50 text-slate-500 hover:text-rose-600 border border-transparent hover:border-rose-200"
                    }`}
                    title="Remove paper from sources"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>
          );
        })}

        {filteredPapers.length === 0 && (
          <div className="p-6 text-center text-xs text-slate-500">
            No research papers in folder "{activeFolder}" matching "{searchQuery}"
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div
        className={`p-3 border-t text-[11px] text-slate-500 flex items-center justify-between ${
          isDark ? "border-slate-800/80 bg-slate-950" : "border-slate-200 bg-white"
        }`}
      >
        <span>Neo4j Synchronized</span>
        <span className="font-mono">{papers.length} Papers Total</span>
      </div>
    </aside>
  );
};
