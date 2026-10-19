import React, { useState, useEffect } from "react";
import { X, Database, RefreshCw, Terminal, CheckCircle2, AlertTriangle, Layers, ArrowRight, Trash2 } from "lucide-react";
import { Neo4jStatus } from "../types";

interface Neo4jAuraModalProps {
  isOpen: boolean;
  onClose: () => void;
  neo4jStatus: Neo4jStatus | null;
  onRefreshStatus: () => Promise<void>;
}

export const Neo4jAuraModal: React.FC<Neo4jAuraModalProps> = ({
  isOpen,
  onClose,
  neo4jStatus,
  onRefreshStatus,
}) => {
  const [cypherQuery, setCypherQuery] = useState<string>(
    "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.paper_id AS paper_id LIMIT 10"
  );
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<{
    columns: string[];
    records: any[];
    execution_ms: number;
    summary: string;
  } | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  const [syncing, setSyncing] = useState(false);
  const [purging, setPurging] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      onRefreshStatus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handlePurgeAll = async () => {
    if (!window.confirm("Are you sure you want to purge ALL nodes and relationships from Neo4j Aura?")) return;
    setPurging(true);
    setSyncMessage(null);
    try {
      const res = await fetch("/api/neo4j/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      if (res.ok) {
        setSyncMessage(`Purged all ${data.nodes_deleted || 0} nodes from Neo4j Aura database.`);
        await onRefreshStatus();
      } else {
        setSyncMessage(`Purge failed: ${data.error || "Unknown error"}`);
      }
    } catch (err: any) {
      setSyncMessage(`Purge error: ${err.message}`);
    } finally {
      setPurging(false);
    }
  };

  const handleSyncAll = async () => {
    setSyncing(true);
    setSyncMessage(null);
    try {
      const res = await fetch("/api/neo4j/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (res.ok) {
        setSyncMessage(`Synced ${data.papers_synced} papers, ${data.total_nodes_synced} entities, and ${data.total_edges_synced} relations into Neo4j Aura!`);
        await onRefreshStatus();
      } else {
        setSyncMessage(`Sync failed: ${data.error || "Unknown error"}`);
      }
    } catch (err: any) {
      setSyncMessage(`Sync failed: ${err.message}`);
    } finally {
      setSyncing(false);
    }
  };

  const handleRunCypher = async (queryToRun?: string) => {
    const q = queryToRun || cypherQuery;
    setQueryLoading(true);
    setQueryError(null);
    setQueryResult(null);
    try {
      const res = await fetch("/api/neo4j/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cypher: q }),
      });
      const data = await res.json();
      if (res.ok) {
        setQueryResult(data);
      } else {
        setQueryError(data.error || "Execution error");
      }
    } catch (err: any) {
      setQueryError(err.message || "Failed to execute query");
    } finally {
      setQueryLoading(false);
    }
  };

  const sampleQueries = [
    {
      label: "Entities Sample",
      cypher: "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.paper_id AS paper_id LIMIT 10",
    },
    {
      label: "Relationships",
      cypher: "MATCH (s:Entity)-[r:RELATION]->(t:Entity) RETURN s.name AS Source, type(r) AS Relation, t.name AS Target, r.is_bridge AS Bridge LIMIT 10",
    },
    {
      label: "Paper Catalog",
      cypher: "MATCH (p:Paper) RETURN p.title AS Title, p.paper_id AS ID, p.original_nodes AS Nodes, p.original_edges AS Edges LIMIT 10",
    },
    {
      label: "Supernodes",
      cypher: "MATCH (s:Supernode) RETURN s.name AS Supernode, s.member_count AS MemberCount, s.paper_id AS PaperID LIMIT 10",
    },
  ];

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden text-slate-200">
        {/* Modal Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-white">Neo4j Aura Graph Database</h2>
                {neo4jStatus?.connected ? (
                  <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Connected
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    {neo4jStatus?.status || "Connecting..."}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400">
                Instance: {neo4jStatus?.instance_name || "Free instance"} &bull; ID: {neo4jStatus?.instance_id || "fec31fe4"} &bull; Database: {neo4jStatus?.database || "fec31fe4"}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {/* Status Metrics Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-800/40 border border-slate-700/60 p-3.5 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Total Nodes</span>
              <span className="text-xl font-bold font-mono text-cyan-400">{neo4jStatus?.total_nodes ?? "..."}</span>
            </div>
            <div className="bg-slate-800/40 border border-slate-700/60 p-3.5 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Relationships</span>
              <span className="text-xl font-bold font-mono text-indigo-400">{neo4jStatus?.total_relationships ?? "..."}</span>
            </div>
            <div className="bg-slate-800/40 border border-slate-700/60 p-3.5 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Entity Nodes</span>
              <span className="text-xl font-bold font-mono text-emerald-400">{neo4jStatus?.entity_count ?? "..."}</span>
            </div>
            <div className="bg-slate-800/40 border border-slate-700/60 p-3.5 rounded-xl">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">Indexed Papers</span>
              <span className="text-xl font-bold font-mono text-amber-400">{neo4jStatus?.paper_count ?? "..."}</span>
            </div>
          </div>

          {/* Connection URI & Server Info */}
          <div className="bg-slate-800/30 border border-slate-700/50 p-4 rounded-xl flex flex-wrap items-center justify-between gap-3 text-[11px]">
            <div className="space-y-1">
              <div className="text-slate-400">
                <strong className="text-slate-200">URI:</strong> <code className="text-cyan-300 font-mono">{neo4jStatus?.uri}</code>
              </div>
              <div className="text-slate-400">
                <strong className="text-slate-200">Labels in DB:</strong> {neo4jStatus?.labels?.join(", ") || "Entity, Paper, Supernode"}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleSyncAll}
                disabled={syncing || purging}
                className="flex items-center gap-1.5 px-3 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg font-medium transition-colors shadow-sm"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${syncing ? "animate-spin" : ""}`} />
                {syncing ? "Syncing..." : "Sync All Graphs"}
              </button>
              <button
                onClick={handlePurgeAll}
                disabled={purging || syncing}
                className="flex items-center gap-1.5 px-3 py-2 bg-rose-600/90 hover:bg-rose-500 disabled:opacity-50 text-white rounded-lg font-medium transition-colors shadow-sm"
                title="Delete all nodes and relationships from Neo4j database"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {purging ? "Purging..." : "Purge Database"}
              </button>
              <button
                onClick={onRefreshStatus}
                className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors border border-slate-700"
                title="Refresh Neo4j connection status"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {syncMessage && (
            <div className="p-3 rounded-xl bg-indigo-950/60 border border-indigo-500/30 text-indigo-300 text-xs flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-indigo-400 shrink-0" />
              <span>{syncMessage}</span>
            </div>
          )}

          {/* Cypher Explorer Console */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                <h3 className="font-semibold text-white uppercase tracking-wider text-[11px]">
                  Interactive Cypher Query Runner
                </h3>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                {sampleQueries.map((sq, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setCypherQuery(sq.cypher);
                      handleRunCypher(sq.cypher);
                    }}
                    className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] border border-slate-700 transition-colors"
                  >
                    {sq.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <textarea
                value={cypherQuery}
                onChange={(e) => setCypherQuery(e.target.value)}
                rows={3}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-xs text-cyan-300 focus:outline-none focus:border-cyan-500 transition-colors resize-none"
                placeholder="Enter Cypher query (e.g., MATCH (n) RETURN n LIMIT 25)..."
              />
              <div className="flex justify-between items-center">
                <span className="text-[11px] text-slate-500 font-mono">
                  Database: {neo4jStatus?.database || "fec31fe4"}
                </span>
                <button
                  onClick={() => handleRunCypher()}
                  disabled={queryLoading}
                  className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg font-medium transition-colors shadow-sm flex items-center gap-1.5"
                >
                  <ArrowRight className="w-3.5 h-3.5" />
                  {queryLoading ? "Executing..." : "Execute Cypher"}
                </button>
              </div>
            </div>

            {queryError && (
              <div className="p-3 bg-red-950/40 border border-red-500/30 rounded-xl text-red-300 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{queryError}</span>
              </div>
            )}

            {queryResult && (
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 space-y-2">
                <div className="flex items-center justify-between text-[11px] text-slate-400 border-b border-slate-800/80 pb-2">
                  <span className="text-cyan-400 font-mono">{queryResult.summary}</span>
                  <span>{queryResult.records.length} rows</span>
                </div>
                {queryResult.records.length > 0 ? (
                  <div className="overflow-x-auto max-h-56">
                    <table className="w-full text-left font-mono text-[11px]">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400">
                          {queryResult.columns.map((c, i) => (
                            <th key={i} className="py-1.5 px-2 font-semibold">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {queryResult.records.map((r, rowIdx) => (
                          <tr key={rowIdx} className="hover:bg-slate-900/60">
                            {queryResult.columns.map((col, colIdx) => (
                              <td key={colIdx} className="py-1.5 px-2 text-slate-300 max-w-xs truncate">
                                {typeof r[col] === "object"
                                  ? JSON.stringify(r[col])
                                  : String(r[col] ?? "")}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="py-4 text-center text-slate-500 text-xs">No records returned.</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex justify-between items-center text-xs">
          <span className="text-slate-500">
            TLS Encrypted &bull; Neo4j Aura Bolt Protocol
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
