"""Entity and Directed Relationship Extractor with strict JSON schemas, validation, and retry logic."""
import json
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from backend.core.llm_client import ollama_client
from backend.config import settings

logger = logging.getLogger("graphrag.extractor")

class RawEntity(BaseModel):
    name: str = Field(description="Name of the entity, title-cased or canonical form")
    type: str = Field(description="Category, e.g., Architecture, Model, Dataset, Metric, Concept, Task, Method")
    description: str = Field(description="Comprehensive summary of what this entity is and does in the paper")

class RawRelation(BaseModel):
    source: str = Field(description="Canonical name of the source entity")
    target: str = Field(description="Canonical name of the target entity")
    type: str = Field(description="Directed predicate in UPPER_SNAKE_CASE, e.g. USES, EVALUATED_ON, EXTENDS, OUTPERFORMS")
    description: str = Field(description="Explanation of the relationship between source and target")
    weight: float = Field(default=1.0, description="Strength of the relationship between 0.0 and 1.0")

class ExtractionResult(BaseModel):
    entities: List[RawEntity] = Field(default_factory=list)
    relationships: List[RawRelation] = Field(default_factory=list)

class EntityRelationExtractor:
    """Extracts typed entities and directed relations with schema validation and retries."""

    SYSTEM_PROMPT = """You are an expert scientific knowledge graph constructor.
Extract key scientific entities and directed relationships from the research paper text.
Return ONLY a valid JSON object matching this schema:
{
  "entities": [
    {"name": "Entity Name", "type": "Architecture|Method|Task|Metric|Dataset|Concept", "description": "Details"}
  ],
  "relationships": [
    {"source": "Source Name", "target": "Target Name", "type": "RELATION_TYPE", "description": "Details", "weight": 0.9}
  ]
}
Rules:
1. Relationships MUST be directed (source -> target).
2. Entity names must be concise and consistent.
3. Output STRICT JSON only without Markdown formatting or explanations.
"""

    def __init__(self):
        self.client = ollama_client

    async def _extract_with_groq(self, prompt: str) -> Optional[str]:
        import os
        import httpx
        groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not groq_key:
            return None
        models = [
            settings.GROQ_MODEL or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b"
        ]
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                for model in models:
                    try:
                        res = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {groq_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": self.SYSTEM_PROMPT},
                                    {"role": "user", "content": prompt}
                                ],
                                "response_format": {"type": "json_object"},
                                "temperature": 0.1,
                                "max_tokens": 1500
                            }
                        )
                        if res.status_code == 200:
                            data = res.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if content and len(content.strip()) > 10:
                                return content
                        else:
                            logger.warning(f"[Groq Extract] Model {model} returned HTTP {res.status_code}: {res.text[:150]}")
                    except Exception as err:
                        logger.warning(f"[Groq Extract] {model} attempt error: {err}")
                        continue
        except Exception as e:
            logger.warning(f"[Groq Extract] Client error: {e}")
        return None

    async def extract_from_chunk(self, chunk: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """Extract entities and directed relationships from a single chunk with retry and validation."""
        chunk_id = chunk["chunk_id"]
        section = chunk.get("section", "")
        content = chunk.get("content", "")

        prompt = f"Section: {section}\n\nContent:\n{content[:2500]}\n\nExtract scientific entities and directed relationships:"

        # 1. Tier 1: Try Groq Cloud LLM (ultra-fast, sub-second)
        groq_text = await self._extract_with_groq(prompt)
        if groq_text:
            parsed = self._clean_and_parse_json(groq_text)
            if parsed:
                try:
                    validated = ExtractionResult(**parsed)
                    if validated.entities:
                        return self._format_result(validated, chunk_id)
                except ValidationError:
                    pass

        # 2. Tier 2: Try Local Ollama
        for attempt in range(max_retries + 1):
            raw_text, call_log = await self.client.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                model=settings.OLLAMA_LARGE_MODEL,
                temperature=0.1,
                format_json=True
            )

            if call_log.status == "success" and raw_text:
                parsed = self._clean_and_parse_json(raw_text)
                if parsed:
                    try:
                        validated = ExtractionResult(**parsed)
                        return self._format_result(validated, chunk_id)
                    except ValidationError as ve:
                        logger.warning(f"Schema validation error (attempt {attempt + 1}): {ve}")
            else:
                break

        # If LLM unavailable or output unparseable, employ semantic rule-based fallback
        logger.info(f"Using rule-based extraction fallback for chunk {chunk_id}")
        return self._rule_based_fallback(chunk)

    def _clean_and_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Clean markdown code fences and parse JSON robustly."""
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Attempt to locate first { and last }
            match = re.search(r"(\{.*\})", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def _format_result(self, result: ExtractionResult, chunk_id: str) -> Dict[str, Any]:
        """Format validated Pydantic model into pipeline dictionary."""
        entities = []
        for e in result.entities:
            name = e.name.strip()
            if len(name) < 2:
                continue
            ent_id = re.sub(r"[^a-zA-Z0-9_]+", "_", name.lower()).strip("_")
            entities.append({
                "id": ent_id,
                "name": name,
                "type": e.type or "Concept",
                "description": e.description or "",
                "source_chunk_ids": [chunk_id]
            })

        relations = []
        for r in result.relationships:
            src_name = r.source.strip()
            tgt_name = r.target.strip()
            if not src_name or not tgt_name or src_name.lower() == tgt_name.lower():
                continue
            src_id = re.sub(r"[^a-zA-Z0-9_]+", "_", src_name.lower()).strip("_")
            tgt_id = re.sub(r"[^a-zA-Z0-9_]+", "_", tgt_name.lower()).strip("_")
            rel_type = re.sub(r"[^a-zA-Z0-9_]+", "_", r.type.upper()).strip("_") or "RELATED_TO"
            relations.append({
                "source": src_id,
                "target": tgt_id,
                "type": rel_type,
                "description": r.description or "",
                "weight": max(0.1, min(1.0, float(r.weight or 1.0))),
                "source_chunk_ids": [chunk_id]
            })

        return {
            "entities": entities,
            "relations": relations,
            "origin": "llm",
            "status": "success"
        }

    def _rule_based_fallback(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """High-precision heuristic fallback extracting core AI research entities when LLM is unavailable.

        Never invents directed relationships or ASSOCIATED_WITH co-occurrence edges.
        """
        content = chunk.get("content", "")
        chunk_id = chunk["chunk_id"]

        # Known scientific entities and patterns
        patterns = [
            (r"\b(Transformer|BERT|GPT|GCN|GAT|GraphSAGE|DeepWalk|node2vec|RAG|LLM|Encoder|Decoder)\b", "Architecture"),
            (r"\b(Self-Attention|Multi-Head Attention|Feed-Forward|Positional Encoding|Skip-Connection|Message Passing|Residual)\b", "Method"),
            (r"\b(BLEU|ROUGE|Accuracy|Perplexity|MRR|Hits@10|F1-Score|Loss)\b", "Metric"),
            (r"\b(WMT 2014|Cora|Citeseer|Pubmed|SQuAD|ImageNet|GLUE)\b", "Dataset"),
            (r"\b(Machine Translation|Question Answering|Node Classification|Link Prediction|Summarization)\b", "Task")
        ]

        found_entities: Dict[str, Dict[str, Any]] = {}
        for pat, ent_type in patterns:
            for m in re.finditer(pat, content, re.IGNORECASE):
                name = m.group(0).strip()
                ent_id = re.sub(r"[^a-zA-Z0-9_]+", "_", name.lower()).strip("_")
                if ent_id not in found_entities:
                    found_entities[ent_id] = {
                        "id": ent_id,
                        "name": name,
                        "type": ent_type,
                        "description": f"Scientific {ent_type} mentioned in section '{chunk.get('section', '')}'",
                        "source_chunk_ids": [chunk_id]
                    }

        # Do NOT invent directed edges or heuristic ASSOCIATED_WITH co-occurrences without genuine semantic support
        return {
            "entities": list(found_entities.values()),
            "relations": [],
            "origin": "fallback",
            "status": "fallback"
        }

extractor = EntityRelationExtractor()
