import React, { useState } from "react";
import { MessageSquare, Plus, Trash2, Clock, X } from "lucide-react";
import { TraversalTrace, QueryResult } from "../types";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  mode?: string;
  queryResult?: QueryResult;
  trace?: TraversalTrace;
}

export interface ChatSession {
  id: string;
  title: string;
  paperIds: string[];
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

interface SessionsPanelProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
}

export const SessionsPanel: React.FC<SessionsPanelProps> = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}) => {
  const fmtTime = (ts: number) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60_000) return "just now";
    if (diff < 3600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86400_000) return `${Math.floor(diff / 3600_000)}h ago`;
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="panel-header">
        <div className="flex-between">
          <span className="panel-title">Sessions</span>
          <button className="btn btn-primary btn-sm" onClick={onNewSession}>
            <Plus size={12} /> New Chat
          </button>
        </div>
      </div>

      <div className="panel-body">
        {sessions.length === 0 ? (
          <div className="empty-state">
            <MessageSquare />
            <p>No sessions yet.<br />Start a new chat.</p>
          </div>
        ) : (
          sessions
            .slice()
            .sort((a, b) => b.updatedAt - a.updatedAt)
            .map(session => (
              <div
                key={session.id}
                className={`session-item ${activeSessionId === session.id ? "active" : ""}`}
                style={{ marginBottom: 4 }}
                onClick={() => onSelectSession(session.id)}
              >
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="session-item-title">{session.title}</div>
                    <div className="session-item-meta" style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <Clock size={10} />
                      <span>{fmtTime(session.updatedAt)}</span>
                      <span>·</span>
                      <span>{session.messages.length} msg{session.messages.length !== 1 ? "s" : ""}</span>
                    </div>
                  </div>
                  <button
                    className="btn-icon"
                    style={{ padding: 3, flexShrink: 0, color: "var(--danger)" }}
                    title="Delete session"
                    onClick={e => {
                      e.stopPropagation();
                      if (window.confirm("Delete this session?")) onDeleteSession(session.id);
                    }}
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            ))
        )}
      </div>
    </div>
  );
};
