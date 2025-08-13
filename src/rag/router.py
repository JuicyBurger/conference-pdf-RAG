"""
RAG Router: delegates query answering to the appropriate engine based on room metadata.

Initial scaffolding: routes everything to VectorRAGEngine (legacy behavior).
Graph/Hybrid engines will be plugged in later.
"""

from __future__ import annotations

from typing import Optional

from src.API.services.chat_service import chat_service


class RAGRouter:
    def __init__(self):
        # Lazy imports to avoid heavy deps at import time
        from .engine.vector import VectorRAGEngine

        self.vector_engine = VectorRAGEngine()
        from .engine.graph import GraphRAGEngine
        from .engine.hybrid import HybridRAGEngine
        self.graph_engine = GraphRAGEngine()
        self.hybrid_engine = HybridRAGEngine()

    def _get_room_mode(self, room_id: str) -> str:
        try:
            room = chat_service.rooms.get(room_id)
            if not room:
                # Attempt to fetch if not cached
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    data = loop.run_until_complete(chat_service.get_room(room_id))
                    room = data or {}
                finally:
                    loop.close()
            mode = (room or {}).get("rag_mode")
            return mode or "vector"
        except Exception:
            return "vector"

    def answer(self, room_id: str, query: str, pdf_summary: Optional[str] = None):
        mode = self._get_room_mode(room_id)
        if mode == "graph":
            return self.graph_engine.answer(room_id, query, pdf_summary=pdf_summary)
        if mode == "hybrid":
            return self.hybrid_engine.answer(room_id, query, pdf_summary=pdf_summary)
        return self.vector_engine.answer(room_id, query, pdf_summary=pdf_summary)


# Global router instance
router = RAGRouter()


