import React, { useState } from "react";
import { Key, Eye, EyeOff, Database, Cpu, Sun, Moon, Check, RefreshCw } from "lucide-react";
import { Neo4jStatus } from "../types";

interface SettingsPanelProps {
  theme: "dark" | "light";
  onToggleTheme: () => void;
  neo4jStatus: Neo4jStatus | null;
  onOpenNeo4j: () => void;
  onRefreshNeo4j: () => void;
}

const LLM_PROVIDERS = [
  {
    id: "groq", label: "Groq", envKey: "VITE_GROQ_API_KEY",
    models: ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
  },
  {
    id: "openai", label: "OpenAI", envKey: "VITE_OPENAI_API_KEY",
    models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
  },
  {
    id: "anthropic", label: "Anthropic", envKey: "VITE_ANTHROPIC_API_KEY",
    models: ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
  },
  {
    id: "ollama", label: "Ollama (local)", envKey: "",
    models: ["llama3.1", "llama3.2", "mistral", "gemma2", "qwen2.5"],
  },
];

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  theme,
  onToggleTheme,
  neo4jStatus,
  onOpenNeo4j,
  onRefreshNeo4j,
}) => {
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [apiKeys, setApiKeys] = useState<Record<string, string>>(() => {
    try {
      const saved = localStorage.getItem("graphrag_api_keys");
      return saved ? JSON.parse(saved) : {};
    } catch { return {}; }
  });
  const [selectedProvider, setSelectedProvider] = useState(() =>
    localStorage.getItem("graphrag_llm_provider") || "groq"
  );
  const [selectedModel, setSelectedModel] = useState(() =>
    localStorage.getItem("graphrag_llm_model") || LLM_PROVIDERS[0].models[0]
  );
  const [saved, setSaved] = useState(false);

  const provider = LLM_PROVIDERS.find(p => p.id === selectedProvider) || LLM_PROVIDERS[0];

  const handleSave = () => {
    try {
      localStorage.setItem("graphrag_api_keys", JSON.stringify(apiKeys));
      localStorage.setItem("graphrag_llm_provider", selectedProvider);
      localStorage.setItem("graphrag_llm_model", selectedModel);
    } catch {}
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleProviderChange = (id: string) => {
    setSelectedProvider(id);
    const p = LLM_PROVIDERS.find(x => x.id === id);
    if (p) setSelectedModel(p.models[0]);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflowY: "auto" }}>
      <div className="panel-header">
        <span className="panel-title">Settings</span>
      </div>

      <div style={{ padding: "12px 12px 0" }}>
        {/* Appearance */}
        <div className="settings-section" style={{ padding: "0 0 16px" }}>
          <span className="settings-label">Appearance</span>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Theme</span>
            <button
              className="btn btn-ghost btn-sm"
              onClick={onToggleTheme}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              {theme === "dark" ? <Sun size={13} style={{ color: "#f59e0b" }} /> : <Moon size={13} />}
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </button>
          </div>
        </div>

        <div className="divider" />

        {/* LLM Provider */}
        <div className="settings-section" style={{ padding: "16px 0" }}>
          <span className="settings-label">LLM Provider</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              <label style={{ fontSize: 11, color: "var(--text-tertiary)", display: "block", marginBottom: 5 }}>Provider</label>
              <select
                className="select"
                style={{ width: "100%" }}
                value={selectedProvider}
                onChange={e => handleProviderChange(e.target.value)}
              >
                {LLM_PROVIDERS.map(p => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: 11, color: "var(--text-tertiary)", display: "block", marginBottom: 5 }}>Model</label>
              <select
                className="select"
                style={{ width: "100%" }}
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
              >
                {provider.models.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            {provider.envKey && (
              <div>
                <label style={{ fontSize: 11, color: "var(--text-tertiary)", display: "block", marginBottom: 5 }}>
                  API Key {provider.id !== "ollama" && <span style={{ color: "var(--danger)" }}>*</span>}
                </label>
                <div style={{ position: "relative" }}>
                  <input
                    className="input input-sm"
                    type={showKeys[provider.id] ? "text" : "password"}
                    placeholder={`${provider.label} API key…`}
                    value={apiKeys[provider.id] || ""}
                    onChange={e => setApiKeys(prev => ({ ...prev, [provider.id]: e.target.value }))}
                    style={{ paddingRight: 34 }}
                  />
                  <button
                    className="btn-icon"
                    style={{ position: "absolute", right: 4, top: "50%", transform: "translateY(-50%)", padding: 4 }}
                    onClick={() => setShowKeys(prev => ({ ...prev, [provider.id]: !prev[provider.id] }))}
                  >
                    {showKeys[provider.id] ? <EyeOff size={12} /> : <Eye size={12} />}
                  </button>
                </div>
                <p style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 4 }}>
                  Keys are stored locally in your browser only.
                </p>
              </div>
            )}

            <button
              className={`btn ${saved ? "btn-primary" : "btn-ghost"}`}
              style={{ width: "100%", justifyContent: "center" }}
              onClick={handleSave}
            >
              {saved ? <><Check size={13} /> Saved!</> : "Save Settings"}
            </button>
          </div>
        </div>

        <div className="divider" />

        {/* Neo4j */}
        <div className="settings-section" style={{ padding: "16px 0" }}>
          <span className="settings-label">Neo4j Cloud</span>
          <div style={{
            padding: "10px 12px", borderRadius: 8,
            background: "var(--bg-raised)", border: "1px solid var(--border-default)",
            marginBottom: 8
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div className={`status-dot ${neo4jStatus?.connected ? "online pulse" : "offline"}`} />
              <span style={{ fontSize: 12, fontWeight: 500, color: "var(--text-primary)" }}>
                {neo4jStatus?.connected ? "Connected" : "Disconnected"}
              </span>
              <button className="btn-icon" style={{ marginLeft: "auto", padding: 4 }} onClick={onRefreshNeo4j}>
                <RefreshCw size={12} />
              </button>
            </div>
            {neo4jStatus?.connected && (
              <>
                <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 2 }}>
                  {neo4jStatus.instance_name || neo4jStatus.uri}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
                  {neo4jStatus.total_nodes?.toLocaleString()} nodes · {neo4jStatus.total_relationships?.toLocaleString()} rels
                </div>
              </>
            )}
            {neo4jStatus?.error && (
              <div style={{ fontSize: 11, color: "var(--danger)", marginTop: 4 }}>
                {neo4jStatus.error}
              </div>
            )}
          </div>
          <button className="btn btn-ghost" style={{ width: "100%", justifyContent: "center" }} onClick={onOpenNeo4j}>
            <Database size={13} /> Configure Neo4j
          </button>
        </div>

        <div className="divider" />

        {/* About */}
        <div className="settings-section" style={{ padding: "16px 0" }}>
          <span className="settings-label">About</span>
          <div style={{ fontSize: 12, color: "var(--text-tertiary)", lineHeight: 1.6 }}>
            <p style={{ marginBottom: 6 }}><strong style={{ color: "var(--text-secondary)" }}>GraphRAG Paper Studio</strong></p>
            <p>Tarjan SCC · Bridge-preserving compression · Multi-document synthesis</p>
          </div>
        </div>
      </div>
    </div>
  );
};
