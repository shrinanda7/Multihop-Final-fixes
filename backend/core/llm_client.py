"""Ollama LLM and Embeddings Client with token tracking, latency logging, and error handling."""
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx
from pydantic import BaseModel, Field

from backend.config import settings

logger = logging.getLogger("graphrag.llm")

class LLMCallLog(BaseModel):
    call_type: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None

class EmbeddingError(RuntimeError):
    """Raised when text embedding generation fails, connects to an unavailable service, or yields invalid vectors."""
    pass

MODEL_EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "text-embedding-3-small": 1536,
}

def get_expected_embedding_dim(model_name: str) -> Optional[int]:
    """Return expected vector dimension for well-known models if available."""
    clean_name = model_name.split(":")[0].strip().lower()
    return MODEL_EMBEDDING_DIMENSIONS.get(clean_name)

def validate_embedding(
    vec: Any,
    expected_dim: Optional[int] = None,
    allow_empty: bool = False
) -> None:
    """Validate that an embedding is a nonempty, finite list of numeric floats matching expected dimensions."""
    import math
    if not isinstance(vec, list):
        raise EmbeddingError(f"Embedding must be a list of floats, got {type(vec).__name__}")
    if not vec:
        if allow_empty:
            return
        raise EmbeddingError("Embedding vector is empty")
    if expected_dim is not None and len(vec) != expected_dim:
        raise EmbeddingError(f"Embedding dimension mismatch: expected {expected_dim}, got {len(vec)}")
    for i, val in enumerate(vec):
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise EmbeddingError(f"Embedding contains non-finite or non-numeric value at index {i}: {val!r}")

class OllamaClient:
    """Client for interacting with local Ollama service for completions and embeddings."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.small_model = settings.OLLAMA_SMALL_MODEL
        self.large_model = settings.OLLAMA_LARGE_MODEL
        self.embed_model = settings.OLLAMA_EMBED_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        self.call_logs: List[LLMCallLog] = []
        self._mock_mode: bool = False
        self._mock_dim: int = get_expected_embedding_dim(self.embed_model) or 768

    def set_mock_mode(self, enabled: bool = True, dim: Optional[int] = None):
        """Configure intentional mock embedding mode for unit testing without live daemons."""
        self._mock_mode = enabled
        if dim is not None:
            self._mock_dim = dim
        logger.info(f"OllamaClient mock embedding mode set to: {enabled} (dim={self._mock_dim})")

    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama is accessible and list available models."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return {
                        "status": "connected",
                        "base_url": self.base_url,
                        "available_models": models,
                        "has_small_model": any(self.small_model in m for m in models),
                        "has_large_model": any(self.large_model in m for m in models),
                        "has_embed_model": any(self.embed_model in m for m in models),
                    }
                return {
                    "status": "error",
                    "base_url": self.base_url,
                    "error": f"HTTP {res.status_code}: {res.text}"
                }
        except Exception as e:
            return {
                "status": "unreachable",
                "base_url": self.base_url,
                "error": str(e),
                "suggestion": f"Ensure Ollama is running (`ollama serve`) and models ({self.small_model}, {self.large_model}, {self.embed_model}) are pulled."
            }

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.2,
        format_json: bool = False,
    ) -> Tuple[str, LLMCallLog]:
        """Generate response via Ollama /api/generate with latency and token telemetry."""
        target_model = model or self.large_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        start_time = time.time()
        call_log = LLMCallLog(call_type="generate", model=target_model)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                elapsed_ms = (time.time() - start_time) * 1000.0
                call_log.latency_ms = round(elapsed_ms, 2)

                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "")
                    prompt_tokens = data.get("prompt_eval_count", len(prompt.split()) * 4 // 3)
                    completion_tokens = data.get("eval_count", len(response_text.split()) * 4 // 3)
                    call_log.prompt_tokens = prompt_tokens
                    call_log.completion_tokens = completion_tokens
                    call_log.total_tokens = prompt_tokens + completion_tokens
                    call_log.status = "success"
                    self.call_logs.append(call_log)
                    return response_text, call_log
                else:
                    call_log.status = "failed"
                    call_log.error_message = f"HTTP {res.status_code}: {res.text}"
                    self.call_logs.append(call_log)
                    return "", call_log
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            call_log.latency_ms = round(elapsed_ms, 2)
            call_log.status = "error"
            call_log.error_message = str(e)
            self.call_logs.append(call_log)
            logger.warning(f"Ollama call to {target_model} failed: {e}")
            return "", call_log

    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        expected_dim: Optional[int] = None,
        raise_on_error: bool = True
    ) -> Tuple[List[float], LLMCallLog]:
        """Generate text embedding vector with telemetry and strict validation.

        Args:
            text: Input text string to embed.
            model: Optional model override. Defaults to self.embed_model.
            expected_dim: Optional explicit expected vector length.
            raise_on_error: If True (default), raises EmbeddingError on HTTP, connection, or validation failures.
                            If False, returns ([], call_log) with call_log.status='failed' or 'error'.
        """
        import math
        import hashlib

        target_model = model or self.embed_model
        exp_dim = expected_dim if expected_dim is not None else get_expected_embedding_dim(target_model)
        start_time = time.time()
        call_log = LLMCallLog(call_type="embedding", model=target_model)

        # 1. Intentional Mock Mode (strictly for unit tests explicitly requesting mocks)
        if self._mock_mode:
            elapsed_ms = (time.time() - start_time) * 1000.0
            call_log.latency_ms = round(elapsed_ms, 2)
            call_log.prompt_tokens = len(text.split()) * 4 // 3
            call_log.total_tokens = call_log.prompt_tokens
            call_log.status = "mock"
            dim = exp_dim or self._mock_dim
            vec = [0.0] * dim
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h >> 6) & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            norm_vec = [round(x / norm, 6) for x in vec]
            validate_embedding(norm_vec, expected_dim=dim)
            self.call_logs.append(call_log)
            return norm_vec, call_log

        payload = {"model": target_model, "prompt": text}

        # 2. Live HTTP Generation
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/embeddings", json=payload)
                elapsed_ms = (time.time() - start_time) * 1000.0
                call_log.latency_ms = round(elapsed_ms, 2)

                if res.status_code == 200:
                    try:
                        data = res.json()
                    except Exception as json_err:
                        err_msg = f"Malformed JSON from Ollama embeddings endpoint: {json_err}"
                        call_log.status = "failed"
                        call_log.error_message = err_msg
                        self.call_logs.append(call_log)
                        if raise_on_error:
                            raise EmbeddingError(err_msg) from json_err
                        return [], call_log

                    if not isinstance(data, dict) or "embedding" not in data:
                        err_msg = "Malformed response from Ollama embeddings API: missing 'embedding' field"
                        call_log.status = "failed"
                        call_log.error_message = err_msg
                        self.call_logs.append(call_log)
                        if raise_on_error:
                            raise EmbeddingError(err_msg)
                        return [], call_log

                    embedding = data.get("embedding")
                    try:
                        validate_embedding(embedding, expected_dim=exp_dim)
                    except EmbeddingError as val_err:
                        call_log.status = "failed"
                        call_log.error_message = str(val_err)
                        self.call_logs.append(call_log)
                        if raise_on_error:
                            raise
                        return [], call_log

                    call_log.prompt_tokens = len(text.split()) * 4 // 3
                    call_log.total_tokens = call_log.prompt_tokens
                    call_log.status = "success"
                    self.call_logs.append(call_log)
                    return embedding, call_log
                else:
                    err_msg = f"HTTP {res.status_code}: {res.text}"
                    call_log.status = "failed"
                    call_log.error_message = err_msg
                    self.call_logs.append(call_log)
                    if raise_on_error:
                        raise EmbeddingError(f"Ollama embedding request failed with {err_msg}")
                    return [], call_log
        except EmbeddingError:
            raise
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            call_log.latency_ms = round(elapsed_ms, 2)
            call_log.status = "error"
            call_log.error_message = str(e)
            self.call_logs.append(call_log)
            # NEVER return silent pseudo-embeddings in experiments or production!
            if raise_on_error:
                raise EmbeddingError(f"Failed to generate embedding from Ollama service: {e}") from e
            return [], call_log

# Singleton instance
ollama_client = OllamaClient()
