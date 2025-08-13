"""
HybridRAGEngine scaffolding.

Will compose GraphRAG and VectorRAG retrievals at query time and synthesize a
single answer. No ingestion coupling.
"""

from __future__ import annotations

from typing import Optional, List

from .base import RAGEngine
from .vector import VectorRAGEngine
from .graph import GraphRAGEngine
from ..generator import generate_answer


class HybridRAGEngine(RAGEngine):
    def __init__(self):
        self.vector = VectorRAGEngine()
        self.graph = GraphRAGEngine()

    def retrieve(self, room_id: str, query: str):
        # Not used directly; answer() will orchestrate
        return []

    def answer(self, room_id: str, query: str, pdf_summary: Optional[str] = None) -> str:
        # Retrieve from both engines
        graph_evidence = self.graph.retrieve(room_id, query)
        vector_evidence = self.vector.retrieve(room_id, query)

        # Simple blending: take top-N from graph and fill with vector
        combined_hits = []
        combined_hits.extend([ev.raw for ev in graph_evidence[:8]])
        # Avoid duplicates by id
        seen = {getattr(h, 'id', None) for h in combined_hits}
        for ev in vector_evidence:
            if getattr(ev.raw, 'id', None) not in seen:
                combined_hits.append(ev.raw)
            if len(combined_hits) >= 12:
                break

        if pdf_summary:
            pdf_hit = type('Hit', (), {'payload': {'content': pdf_summary[:1000], 'page': 'uploaded_pdf'}})()
            combined_hits = [pdf_hit] + combined_hits[:11]

        if not combined_hits:
            return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"

        return generate_answer(query, combined_hits)


