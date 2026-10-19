"""Canonical Entity Resolution and Graph Deduplication."""
import re
from typing import List, Dict, Any, Tuple

class EntityMerger:
    """Merges duplicate entities across chunks and reconciles synonyms and multi-mentions."""

    # Common acronym and alias mappings in AI / Graph papers
    CANONICAL_MAP = {
        "transformer": "Transformer",
        "transformers": "Transformer",
        "gcn": "Graph Convolutional Network",
        "gcns": "Graph Convolutional Network",
        "graph convolutional networks": "Graph Convolutional Network",
        "gat": "Graph Attention Network",
        "gats": "Graph Attention Network",
        "graph attention networks": "Graph Attention Network",
        "rag": "Retrieval-Augmented Generation",
        "graphsage": "GraphSAGE",
        "deepwalk": "DeepWalk",
        "node2vec": "node2vec",
        "bert": "BERT",
        "self attention": "Self-Attention",
        "multi head attention": "Multi-Head Attention",
        "multi head self attention": "Multi-Head Attention",
    }

    def merge_graph(
        self,
        raw_entities: List[Dict[str, Any]],
        raw_relations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Deduplicate entities and combine mentions, then rewrite and deduplicate relationships."""
        merged_entities: Dict[str, Dict[str, Any]] = {}
        alias_to_canonical_id: Dict[str, str] = {}

        # 1. Merge entities
        for ent in raw_entities:
            raw_name = ent.get("name", "").strip()
            norm_name = re.sub(r"[^a-zA-Z0-9]+", " ", raw_name.lower()).strip()

            # Canonical name lookup
            canonical_name = self.CANONICAL_MAP.get(norm_name, raw_name)
            canonical_id = re.sub(r"[^a-zA-Z0-9_]+", "_", canonical_name.lower()).strip("_")
            alias_to_canonical_id[ent.get("id", "")] = canonical_id
            alias_to_canonical_id[raw_name.lower()] = canonical_id

            if canonical_id not in merged_entities:
                merged_entities[canonical_id] = {
                    "id": canonical_id,
                    "name": canonical_name,
                    "type": ent.get("type", "Concept"),
                    "description": ent.get("description", ""),
                    "source_chunk_ids": list(set(ent.get("source_chunk_ids", [])))
                }
            else:
                existing = merged_entities[canonical_id]
                existing["source_chunk_ids"] = list(set(existing["source_chunk_ids"] + ent.get("source_chunk_ids", [])))
                # Append description if complementary
                new_desc = ent.get("description", "")
                if new_desc and new_desc not in existing["description"]:
                    existing["description"] = f"{existing['description']}; {new_desc}".strip("; ")

        # 2. Re-map and deduplicate relationships
        merged_relations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for rel in raw_relations:
            src = alias_to_canonical_id.get(rel.get("source", ""), rel.get("source", ""))
            tgt = alias_to_canonical_id.get(rel.get("target", ""), rel.get("target", ""))
            rel_type = rel.get("type", "RELATED_TO").upper()

            # Discard self-loops and unresolved endpoints
            if not src or not tgt or src == tgt:
                continue
            if src not in merged_entities or tgt not in merged_entities:
                continue

            rel_key = (src, tgt, rel_type)
            if rel_key not in merged_relations:
                merged_relations[rel_key] = {
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "description": rel.get("description", ""),
                    "weight": float(rel.get("weight", 1.0)),
                    "source_chunk_ids": list(set(rel.get("source_chunk_ids", [])))
                }
            else:
                existing = merged_relations[rel_key]
                existing["weight"] = min(2.0, existing["weight"] + 0.2) # reinforce co-occurring edges
                existing["source_chunk_ids"] = list(set(existing["source_chunk_ids"] + rel.get("source_chunk_ids", [])))

        return list(merged_entities.values()), list(merged_relations.values())

entity_merger = EntityMerger()
