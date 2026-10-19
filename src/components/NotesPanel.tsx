import React, { useState } from "react";
import { Plus, FileText, Trash2, X, Edit3, Save, StickyNote } from "lucide-react";

export interface Note {
  id: string;
  title: string;
  content: string;
  createdAt: number;
  updatedAt: number;
}

interface NotesPanelProps {
  /* no external state needed — self-contained with localStorage */
}

export const NotesPanel: React.FC<NotesPanelProps> = () => {
  const [notes, setNotes] = useState<Note[]>(() => {
    try {
      const saved = localStorage.getItem("graphrag_notes");
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editContent, setEditContent] = useState("");
  const [isDirty, setIsDirty] = useState(false);

  const saveToStorage = (updated: Note[]) => {
    try { localStorage.setItem("graphrag_notes", JSON.stringify(updated)); } catch {}
  };

  const createNote = () => {
    const note: Note = {
      id: `note-${Date.now()}`,
      title: "Untitled Note",
      content: "",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    const updated = [note, ...notes];
    setNotes(updated);
    saveToStorage(updated);
    openNote(note);
  };

  const openNote = (note: Note) => {
    setActiveNoteId(note.id);
    setEditTitle(note.title);
    setEditContent(note.content);
    setIsDirty(false);
  };

  const saveNote = () => {
    if (!activeNoteId) return;
    const updated = notes.map(n =>
      n.id === activeNoteId
        ? { ...n, title: editTitle || "Untitled", content: editContent, updatedAt: Date.now() }
        : n
    );
    setNotes(updated);
    saveToStorage(updated);
    setIsDirty(false);
  };

  const deleteNote = (id: string) => {
    const updated = notes.filter(n => n.id !== id);
    setNotes(updated);
    saveToStorage(updated);
    if (activeNoteId === id) { setActiveNoteId(null); setEditTitle(""); setEditContent(""); setIsDirty(false); }
  };

  const fmtDate = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const activeNote = notes.find(n => n.id === activeNoteId);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="panel-header">
        <div className="flex-between">
          <span className="panel-title">Notes</span>
          <button className="btn btn-primary btn-sm" onClick={createNote}>
            <Plus size={12} /> New Note
          </button>
        </div>
      </div>

      {activeNote ? (
        /* Editor view */
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border-subtle)", display: "flex", gap: 8, alignItems: "center" }}>
            <button className="btn-icon" title="Back to list" onClick={() => { if (isDirty && !window.confirm("Discard changes?")) return; setActiveNoteId(null); }}>
              <X size={14} />
            </button>
            <input
              className="input input-sm"
              style={{ flex: 1, fontWeight: 600 }}
              value={editTitle}
              onChange={e => { setEditTitle(e.target.value); setIsDirty(true); }}
              placeholder="Note title"
            />
            {isDirty && (
              <button className="btn btn-primary btn-sm" onClick={saveNote}>
                <Save size={12} /> Save
              </button>
            )}
          </div>
          <textarea
            className="textarea"
            style={{ flex: 1, border: "none", borderRadius: 0, resize: "none", minHeight: 0, padding: "12px 14px", fontSize: 13, lineHeight: 1.75 }}
            placeholder="Write your notes here… supports markdown-ish formatting."
            value={editContent}
            onChange={e => { setEditContent(e.target.value); setIsDirty(true); }}
            onKeyDown={e => { if ((e.metaKey || e.ctrlKey) && e.key === "s") { e.preventDefault(); saveNote(); } }}
          />
          <div style={{ padding: "8px 12px", fontSize: 11, color: "var(--text-tertiary)", borderTop: "1px solid var(--border-subtle)" }}>
            {isDirty ? "Unsaved changes — Ctrl+S to save" : `Last saved ${fmtDate(activeNote.updatedAt)}`}
          </div>
        </div>
      ) : (
        /* List view */
        <div className="panel-body">
          {notes.length === 0 ? (
            <div className="empty-state">
              <StickyNote />
              <p>No notes yet.<br />Create one to start writing.</p>
            </div>
          ) : (
            notes.map(note => (
              <div
                key={note.id}
                className="note-card"
                style={{ marginBottom: 6, position: "relative" }}
                onClick={() => openNote(note)}
              >
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-primary)", marginBottom: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {note.title}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-tertiary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {note.content ? note.content.slice(0, 60) + (note.content.length > 60 ? "…" : "") : "Empty note"}
                    </div>
                    <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 4 }}>{fmtDate(note.updatedAt)}</div>
                  </div>
                  <button
                    className="btn-icon"
                    style={{ padding: 3, color: "var(--danger)", flexShrink: 0 }}
                    onClick={e => { e.stopPropagation(); if (window.confirm("Delete this note?")) deleteNote(note.id); }}
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};
