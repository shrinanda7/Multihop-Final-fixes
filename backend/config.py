"""Central configuration module for Knowledge-Graph RAG."""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file with absolute path
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

class Settings(BaseModel):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PAPERS_DIR: Path = DATA_DIR / "papers"
    CACHE_DIR: Path = DATA_DIR / "cache"

    # Groq Cloud LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_SMALL_MODEL: str = os.getenv("OLLAMA_SMALL_MODEL", "llama3.2:3b")
    OLLAMA_LARGE_MODEL: str = os.getenv("OLLAMA_LARGE_MODEL", "llama3.2:3b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "180.0"))

    # Neo4j Graph Database
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Ablation & Algorithmic Toggles
    ENABLE_TARJAN_COMPRESSION: bool = os.getenv("ENABLE_TARJAN_COMPRESSION", "true").lower() in ("true", "1", "yes")
    ENABLE_BRIDGE_AWARE_PRUNING: bool = os.getenv("ENABLE_BRIDGE_AWARE_PRUNING", "true").lower() in ("true", "1", "yes")
    ENABLE_BUDGET_AWARE_TRAVERSAL: bool = os.getenv("ENABLE_BUDGET_AWARE_TRAVERSAL", "true").lower() in ("true", "1", "yes")

    # Algorithmic Parameters
    GIANT_SCC_THRESHOLD: float = float(os.getenv("GIANT_SCC_THRESHOLD", "0.30"))
    MAX_REPORT_BUDGET: int = int(os.getenv("MAX_REPORT_BUDGET", "10"))
    MAX_TOKEN_BUDGET: int = int(os.getenv("MAX_TOKEN_BUDGET", "4000"))
    TOTAL_LLM_BUDGET: int = int(os.getenv("TOTAL_LLM_BUDGET", "12"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "60.0"))
    MAX_COMMUNITY_DEPTH: int = int(os.getenv("MAX_COMMUNITY_DEPTH", "2"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Server Port
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8001"))

settings = Settings()

# Ensure directories exist
settings.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
