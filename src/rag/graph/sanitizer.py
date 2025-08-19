from __future__ import annotations

from typing import Any, List

from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.schema import BaseNode, TransformComponent


def _sanitize_label(raw: Any, fallback: str) -> str:
    if not isinstance(raw, str):
        return fallback
    label = raw.strip().upper()
    if not label:
        return fallback
    # Keep only A-Z, 0-9, and underscore
    safe = []
    for ch in label:
        if ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch == "_":
            safe.append(ch)
        else:
            safe.append("_")
    label = "".join(safe).strip("_")
    if not label:
        return fallback
    return label[:64]


def _sanitize_properties(props: Any) -> dict:
    if not isinstance(props, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for k, v in props.items():
        key = str(k)[:64]
        try:
            if isinstance(v, (str, int, float, bool)) or v is None:
                sanitized[key] = v if not isinstance(v, str) else v[:1000]
            elif isinstance(v, (list, tuple)):
                sanitized[key] = [str(x)[:256] for x in v[:50]]
            elif isinstance(v, dict):
                sanitized[key] = {str(kk)[:64]: str(vv)[:256] for kk, vv in list(v.items())[:20]}
            else:
                sanitized[key] = str(v)[:256]
        except Exception:
            sanitized[key] = None
    return sanitized


class SanitizeKGExtractor(TransformComponent):
    """Post-processor extractor to sanitize KG entities/relations.

    Ensures:
    - Entity labels are valid; fallback to ENTITY
    - Relation labels are valid; fallback to RELATED_TO
    - Properties are JSON-serializable and size-limited
    - Drops relations missing source/target ids
    """

    def __call__(self, llama_nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        for node in llama_nodes:
            entities = node.metadata.get(KG_NODES_KEY, []) or []
            relations = node.metadata.get(KG_RELATIONS_KEY, []) or []

            sanitized_entities: List[EntityNode] = []
            for e in entities:
                label = _sanitize_label(getattr(e, "label", None), fallback="ENTITY")
                name = getattr(e, "name", None)
                if not isinstance(name, str) or not name.strip():
                    name = "UNKNOWN_ENTITY"
                props = _sanitize_properties(getattr(e, "properties", {}) or {})
                # Preserve original IDs to keep relation source/target mapping intact
                eid = getattr(e, "id", None)
                if eid is not None:
                    sanitized_entities.append(
                        EntityNode(id=eid, name=name.strip()[:256], label=label, properties=props)
                    )
                else:
                    sanitized_entities.append(
                        EntityNode(name=name.strip()[:256], label=label, properties=props)
                    )

            sanitized_relations: List[Relation] = []
            for r in relations:
                src = getattr(r, "source_id", None)
                tgt = getattr(r, "target_id", None)
                if not src or not tgt:
                    continue
                label = _sanitize_label(getattr(r, "label", None), fallback="RELATED_TO")
                props = _sanitize_properties(getattr(r, "properties", {}) or {})
                # Preserve any relation identifier if present (optional)
                rid = getattr(r, "id", None)
                if rid is not None:
                    sanitized_relations.append(
                        Relation(id=rid, label=label, source_id=src, target_id=tgt, properties=props)
                    )
                else:
                    sanitized_relations.append(
                        Relation(label=label, source_id=src, target_id=tgt, properties=props)
                    )

            node.metadata[KG_NODES_KEY] = sanitized_entities
            node.metadata[KG_RELATIONS_KEY] = sanitized_relations

        return llama_nodes


