import React, { useState, useEffect } from "react";
import { StickyNote, Plus, Trash2, X, Edit3, Save, Clock, ChevronRight } from "lucide-react";

export interface Note {
  id: string;
  title: string;
  content: string;
  createdAt: number;
  updatedAt: number;
}

interface NotesDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  theme: "dark" | "light";
}

export const NotesDrawer: React.FC<NotesDrawerProps> = ({
  isOpen,
  onClose,
  theme,
}) => {
  const isDark = theme === "dark";
  const [notes, setNotes] = useState<Note[]>(() => {
    try {
      const saved = localStorage.getItem("graphrag_notes");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [search, setSearch] = useState("");

  useEffect(() => {
    try {
      localStorage.setItem("graphrag_notes", JSON.stringify(notes));
    } catch (e) {
      console.warn("Failed to persist notes:", e);
    }
  }, [notes]);

  if (!isOpen) return null;

  const createNote = () => {
    const newNote: Note = {
      id: `note-${Date.now()}`,
      title: "New Note",
      content: "",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setNotes((prev) => [newNote, ...prev]);
    setActiveNoteId(newNote.id);
    setTitle(newNote.title);
    setContent(newNote.content);
  };

  const selectNote = (note: Note) => {
    setActiveNoteId(note.id);
    setTitle(note.title);
    setContent(note.content);
  };

  const saveCurrentNote = () => {
    if (!activeNoteId) return;
    setNotes((prev) =>
      prev.map((n) =>
        n.id === activeNoteId
          ? {
              ...n,
              title: title.trim() || "Untitled Note",
              content,
              updatedAt: Date.now(),
            }
          : n
      )
    );
  };

  const deleteNote = (id: string) => {
    setNotes((prev) => prev.filter((n) => n.id !== id));
    if (activeNoteId === id) {
      setActiveNoteId(null);
      setTitle("");
      setContent("");
    }
  };

  const filteredNotes = notes.filter((n) =>
    n.title.toLowerCase().includes(search.toLowerCase()) ||
    n.content.toLowerCase().includes(search.toLowerCase())
  );

  const activeNote = notes.find((n) => n.id === activeNoteId);

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/50 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className={`w-full max-w-md h-full shadow-2xl flex flex-col border-l transition-all ${
          isDark
            ? "bg-slate-900 border-slate-800 text-white"
            : "bg-white border-slate-200 text-slate-900"
        }`}
      >
        {/* Drawer Header */}
        <div
          className={`px-4 py-3.5 border-b flex items-center justify-between ${
            isDark ? "border-slate-800 bg-slate-950/60" : "border-slate-200 bg-slate-50"
          }`}
        >
          <div className="flex items-center gap-2">
            <StickyNote className="w-4 h-4 text-amber-400" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Research Notes ({notes.length})
            </h2>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={createNote}
              className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white shadow-xs transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New Note</span>
            </button>
            <button
              onClick={onClose}
              className={`p-1.5 rounded-lg border transition-colors ${
                isDark
                  ? "border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-white"
                  : "border-slate-200 hover:bg-slate-100 text-slate-600"
              }`}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content area: split into list & editor or switched */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {activeNote ? (
            /* Note Editor */
            <div className="flex-1 flex flex-col p-4 overflow-hidden">
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800/60">
                <button
                  onClick={() => {
                    saveCurrentNote();
                    setActiveNoteId(null);
                  }}
                  className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
                >
                  <ChevronRight className="w-3.5 h-3.5 rotate-180" />
                  <span>All Notes</span>
                </button>
                <div className="flex items-center gap-2">
                  <button
                    onClick={saveCurrentNote}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-md bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/30 transition-colors"
                  >
                    <Save className="w-3 h-3" />
                    <span>Save</span>
                  </button>
                  <button
                    onClick={() => deleteNote(activeNote.id)}
                    className="p-1 text-rose-400 hover:text-rose-300 transition-colors"
                    title="Delete Note"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                onBlur={saveCurrentNote}
                placeholder="Note title..."
                className={`w-full text-sm font-semibold mb-2 p-2 rounded-lg border outline-none ${
                  isDark
                    ? "bg-slate-800/80 border-slate-700 text-white placeholder-slate-500"
                    : "bg-slate-50 border-slate-200 text-slate-900 placeholder-slate-400"
                }`}
              />

              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                onBlur={saveCurrentNote}
                placeholder="Type your research thoughts, paper insights, or draft summaries here..."
                className={`flex-1 w-full p-3 text-xs rounded-lg border outline-none resize-none leading-relaxed font-sans ${
                  isDark
                    ? "bg-slate-800/50 border-slate-700 text-slate-200 placeholder-slate-500 focus:border-indigo-500"
                    : "bg-slate-50 border-slate-200 text-slate-800 placeholder-slate-400 focus:border-indigo-600"
                }`}
              />

              <div className="mt-2 text-[10px] text-slate-500 text-right">
                Auto-saved · {new Date(activeNote.updatedAt).toLocaleTimeString()}
              </div>
            </div>
          ) : (
            /* Notes List */
            <div className="flex-1 flex flex-col p-3 overflow-hidden">
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search notes..."
                className={`w-full px-3 py-1.5 text-xs rounded-lg border outline-none mb-3 ${
                  isDark
                    ? "bg-slate-800 border-slate-700 text-white placeholder-slate-500"
                    : "bg-slate-50 border-slate-200 text-slate-900 placeholder-slate-400"
                }`}
              />

              <div className="flex-1 overflow-y-auto space-y-2">
                {filteredNotes.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 text-xs">
                    <StickyNote className="w-8 h-8 mx-auto mb-2 opacity-40" />
                    <p>No notes found.</p>
                    <button
                      onClick={createNote}
                      className="mt-2 text-indigo-400 hover:text-indigo-300 font-medium"
                    >
                      + Create first note
                    </button>
                  </div>
                ) : (
                  filteredNotes.map((note) => (
                    <div
                      key={note.id}
                      onClick={() => selectNote(note)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        isDark
                          ? "bg-slate-800/40 border-slate-800 hover:bg-slate-800/80 hover:border-slate-700"
                          : "bg-slate-50 border-slate-200 hover:bg-slate-100"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="text-xs font-semibold truncate flex-1">
                          {note.title || "Untitled Note"}
                        </h4>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteNote(note.id);
                          }}
                          className="text-slate-500 hover:text-rose-400 transition-colors"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-2 mt-1">
                        {note.content || "Empty note"}
                      </p>
                      <div className="flex items-center gap-1 text-[10px] text-slate-500 mt-2">
                        <Clock className="w-2.5 h-2.5" />
                        <span>{new Date(note.updatedAt).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
