import React, { useState } from "react";
import { Key, Eye, EyeOff, Cpu, X, Check, Save } from "lucide-react";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: "dark" | "light";
}

export const LLM_PROVIDERS = [
  {
    id: "groq",
    label: "Groq Cloud (Ultra-Fast)",
    envKey: "VITE_GROQ_API_KEY",
    models: ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
  },
  {
    id: "gemini",
    label: "Google Gemini",
    envKey: "VITE_GEMINI_API_KEY",
    models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
  },
  {
    id: "openai",
    label: "OpenAI",
    envKey: "VITE_OPENAI_API_KEY",
    models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
  },
  {
    id: "anthropic",
    label: "Anthropic Claude",
    envKey: "VITE_ANTHROPIC_API_KEY",
    models: ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
  },
  {
    id: "ollama",
    label: "Local Ollama",
    envKey: "",
    models: ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "gemma2:9b", "nomic-embed-text"],
  },
];

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  theme,
}) => {
  const isDark = theme === "dark";
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [apiKeys, setApiKeys] = useState<Record<string, string>>(() => {
    try {
      const saved = localStorage.getItem("graphrag_api_keys");
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });
  const [selectedProvider, setSelectedProvider] = useState(() =>
    localStorage.getItem("graphrag_llm_provider") || "groq"
  );
  const [selectedModel, setSelectedModel] = useState(() =>
    localStorage.getItem("graphrag_llm_model") || "openai/gpt-oss-120b"
  );
  const [ollamaUrl, setOllamaUrl] = useState(() =>
    localStorage.getItem("graphrag_ollama_url") || "http://localhost:11434"
  );
  const [saved, setSaved] = useState(false);

  if (!isOpen) return null;

  const currentProvider =
    LLM_PROVIDERS.find((p) => p.id === selectedProvider) || LLM_PROVIDERS[0];

  const handleProviderChange = (id: string) => {
    setSelectedProvider(id);
    const p = LLM_PROVIDERS.find((x) => x.id === id);
    if (p && p.models.length > 0) {
      setSelectedModel(p.models[0]);
    }
  };

  const handleSave = () => {
    try {
      localStorage.setItem("graphrag_api_keys", JSON.stringify(apiKeys));
      localStorage.setItem("graphrag_llm_provider", selectedProvider);
      localStorage.setItem("graphrag_llm_model", selectedModel);
      localStorage.setItem("graphrag_ollama_url", ollamaUrl);
    } catch (e) {
      console.warn("Failed to save settings to localStorage:", e);
    }
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      onClose();
    }, 800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150">
      <div
        className={`w-full max-w-lg rounded-xl border shadow-2xl flex flex-col overflow-hidden transition-all ${
          isDark
            ? "bg-slate-900 border-slate-700/80 text-white"
            : "bg-white border-slate-200 text-slate-900"
        }`}
      >
        {/* Header */}
        <div
          className={`px-5 py-3.5 border-b flex items-center justify-between ${
            isDark ? "border-slate-800 bg-slate-950/50" : "border-slate-100 bg-slate-50"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold tracking-tight">Model & API Key Settings</h3>
              <p className="text-[11px] text-slate-400">Configure LLM providers and custom API keys</p>
            </div>
          </div>
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

        {/* Body */}
        <div className="p-5 space-y-4 max-h-[70vh] overflow-y-auto">
          {/* Provider selection */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              LLM Provider
            </label>
            <div className="grid grid-cols-2 gap-2">
              {LLM_PROVIDERS.map((p) => {
                const isSelected = selectedProvider === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleProviderChange(p.id)}
                    className={`px-3 py-2 text-left rounded-lg border text-xs font-medium transition-all ${
                      isSelected
                        ? "bg-indigo-600/20 border-indigo-500 text-indigo-300 shadow-xs"
                        : isDark
                        ? "bg-slate-800/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800"
                        : "bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100"
                    }`}
                  >
                    <div className="font-semibold">{p.label}</div>
                    <div className="text-[10px] text-slate-400 truncate mt-0.5">
                      {p.models.length} models available
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model selection */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              Selected Model ({currentProvider.label})
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className={`w-full px-3 py-2 text-xs rounded-lg border outline-none font-mono transition-colors ${
                isDark
                  ? "bg-slate-800 border-slate-700 text-slate-200 focus:border-indigo-500"
                  : "bg-slate-50 border-slate-200 text-slate-800 focus:border-indigo-600"
              }`}
            >
              {currentProvider.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* API Key configuration for the selected provider */}
          {currentProvider.envKey && (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Key className="w-3.5 h-3.5 text-indigo-400" />
                  {currentProvider.label} API Key
                </label>
                <span className="text-[10px] text-slate-500">
                  Stored securely in browser localStorage
                </span>
              </div>
              <div className="relative">
                <input
                  type={showKeys[currentProvider.id] ? "text" : "password"}
                  value={apiKeys[currentProvider.id] || ""}
                  onChange={(e) =>
                    setApiKeys((prev) => ({
                      ...prev,
                      [currentProvider.id]: e.target.value,
                    }))
                  }
                  placeholder={`Enter your ${currentProvider.label} API Key...`}
                  className={`w-full pl-3 pr-10 py-2 text-xs rounded-lg border outline-none font-mono transition-colors ${
                    isDark
                      ? "bg-slate-800 border-slate-700 text-slate-200 focus:border-indigo-500 placeholder-slate-500"
                      : "bg-slate-50 border-slate-200 text-slate-800 focus:border-indigo-600 placeholder-slate-400"
                  }`}
                />
                <button
                  type="button"
                  onClick={() =>
                    setShowKeys((prev) => ({
                      ...prev,
                      [currentProvider.id]: !prev[currentProvider.id],
                    }))
                  }
                  className="absolute right-2.5 top-2.5 text-slate-400 hover:text-slate-200"
                >
                  {showKeys[currentProvider.id] ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Ollama URL config if ollama selected */}
          {selectedProvider === "ollama" && (
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                Ollama Endpoint URL
              </label>
              <input
                type="text"
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
                placeholder="http://localhost:11434"
                className={`w-full px-3 py-2 text-xs rounded-lg border outline-none font-mono transition-colors ${
                  isDark
                    ? "bg-slate-800 border-slate-700 text-slate-200 focus:border-indigo-500"
                    : "bg-slate-50 border-slate-200 text-slate-800 focus:border-indigo-600"
                }`}
              />
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          className={`px-5 py-3 border-t flex items-center justify-between ${
            isDark ? "border-slate-800 bg-slate-950/50" : "border-slate-100 bg-slate-50"
          }`}
        >
          <div className="text-[11px] text-slate-400">
            {saved ? (
              <span className="text-emerald-400 flex items-center gap-1 font-medium">
                <Check className="w-3.5 h-3.5" /> Settings saved successfully
              </span>
            ) : (
              "Settings are saved locally."
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
                isDark
                  ? "border-slate-800 hover:bg-slate-800 text-slate-300"
                  : "border-slate-200 hover:bg-slate-100 text-slate-700"
              }`}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors shadow-xs"
            >
              <Save className="w-3.5 h-3.5" />
              Save Preferences
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
