import React from "react";
import { Network, Database, Sun, Moon, StickyNote, Settings } from "lucide-react";
import { Neo4jStatus } from "../types";

interface HeaderProps {
  theme: "dark" | "light";
  onToggleTheme: () => void;
  selectedPapersCount: number;
  totalPapersCount: number;
  neo4jStatus: Neo4jStatus | null;
  onOpenNeo4j: () => void;
  onOpenNotes?: () => void;
  notesCount?: number;
  onOpenSettings?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  theme,
  onToggleTheme,
  selectedPapersCount,
  totalPapersCount,
  neo4jStatus,
  onOpenNeo4j,
  onOpenNotes,
  notesCount = 0,
  onOpenSettings,
}) => {
  const isDark = theme === "dark";

  return (
    <header
      className={`px-5 py-3 border-b flex items-center justify-between gap-4 transition-colors duration-200 select-none ${
        isDark
          ? "bg-slate-950 border-slate-800/80 text-white"
          : "bg-white border-slate-200 text-slate-900 shadow-xs"
      }`}
    >
      {/* Brand & Context */}
      <div className="flex items-center gap-3">
        <div
          className={`w-9 h-9 rounded-lg flex items-center justify-center transition-colors ${
            isDark
              ? "bg-indigo-950/70 border border-indigo-700/50 text-indigo-400"
              : "bg-indigo-50 border border-indigo-200 text-indigo-600"
          }`}
        >
          <Network className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-sm font-semibold tracking-tight">GraphRAG Paper Studio</h1>
            <span
              className={`text-xs ${
                isDark ? "text-slate-500" : "text-slate-400"
              }`}
            >
              ·
            </span>
            <span
              className={`text-xs tracking-tight ${
                isDark ? "text-slate-400" : "text-slate-500"
              }`}
            >
              Tarjan SCC & Bridge-Preserving Graph Retrieval
            </span>
          </div>
          <div
            className={`flex items-center gap-2 text-[11px] ${
              isDark ? "text-slate-400" : "text-slate-500"
            }`}
          >
            <span>
              {selectedPapersCount === 0
                ? "No papers selected"
                : selectedPapersCount === 1
                ? "1 paper active"
                : `${selectedPapersCount} papers active (${totalPapersCount} total)`}
            </span>
            <span aria-hidden="true">·</span>
            <span>Multi-document synthesis</span>
          </div>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2">
        {/* Notes Button */}
        {onOpenNotes && (
          <button
            onClick={onOpenNotes}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
              isDark
                ? "bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-200 hover:border-slate-700"
                : "bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-700"
            }`}
            title="Open Research Notes Drawer"
          >
            <StickyNote className="w-3.5 h-3.5 text-amber-400" />
            <span>Notes</span>
            {notesCount > 0 && (
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-amber-500/20 text-amber-300 font-mono">
                {notesCount}
              </span>
            )}
          </button>
        )}

        {/* Model & API Key Settings */}
        {onOpenSettings && (
          <button
            onClick={onOpenSettings}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
              isDark
                ? "bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-200 hover:border-slate-700"
                : "bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-700"
            }`}
            title="Configure LLM Models & API Keys"
          >
            <Settings className="w-3.5 h-3.5 text-indigo-400" />
            <span>Settings</span>
          </button>
        )}

        {/* Neo4j Aura Cloud Button */}
        <button
          onClick={onOpenNeo4j}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
            isDark
              ? "bg-slate-900 hover:bg-slate-850 border-slate-800 text-slate-200 hover:border-slate-700"
              : "bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-700"
          }`}
          title="Open Neo4j Aura Cloud Explorer & Cypher Console"
        >
          <Database className="w-3.5 h-3.5 text-cyan-500" />
          <span className="flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                neo4jStatus?.connected ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span className="font-mono text-[11px]">
              {neo4jStatus?.connected ? "Aura Connected" : "Neo4j Sync"}
            </span>
          </span>
        </button>

        {/* Theme Switcher */}
        <button
          onClick={onToggleTheme}
          className={`p-2 rounded-lg border transition-all ${
            isDark
              ? "bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-300 hover:text-white"
              : "bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-600 hover:text-slate-900"
          }`}
          title={`Switch to ${isDark ? "Light" : "Dark"} mode`}
          aria-label="Toggle theme"
        >
          {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-700" />}
        </button>
      </div>
    </header>
  );
};
