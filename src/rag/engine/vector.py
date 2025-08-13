from __future__ import annotations

from typing import List, Optional

from .base import RAGEngine, Evidence
from ..retriever import retrieve as vector_retrieve
from ...models.reranker import rerank
from ..generator import generate_answer


class VectorRAGEngine(RAGEngine):
    """Adapter over existing vector RAG pipeline to fit the engine interface.

    This preserves current behavior: use global Qdrant collection retrieval
    and the existing generator with optional reranking.
    """

    def retrieve(self, room_id: str, query: str) -> List[Evidence]:
        # Prefer room-scoped chat uploads in Qdrant, then fall back to global
        hits = vector_retrieve(query=query, top_k=5, score_threshold=0.3, room_id=room_id, prefer_chat_scope=True)
        if hits and len(hits) > 1:
            hits = rerank(query, hits)
        return [Evidence(h) for h in hits]

    def answer(self, room_id: str, query: str, pdf_summary: Optional[str] = None) -> str:
        # When a PDF summary is present, compose a synthetic context by injecting it as a hit
        if pdf_summary:
            hits = self.retrieve(room_id, query)
            pdf_hit = type('Hit', (), {
                'payload': {'content': pdf_summary[:1000], 'page': 'uploaded_pdf'}
            })()
            combined = [pdf_hit] + [ev.raw for ev in hits[:2]]
            return generate_answer(query, combined)

        hits = self.retrieve(room_id, query)
        return generate_answer(query, [ev.raw for ev in hits])


