from __future__ import annotations

from typing import List, Optional, Protocol, Any


class Evidence:
    """Minimal evidence container for downstream generation.

    For now we reuse the existing 'hit' objects structure used by generator,
    so this is a placeholder for a richer typed DTO in future.
    """

    def __init__(self, raw_hit: Any):
        self.raw = raw_hit


class RAGEngine(Protocol):
    def retrieve(self, room_id: str, query: str) -> List[Evidence]:
        ...

    def answer(self, room_id: str, query: str, pdf_summary: Optional[str] = None) -> str:
        ...


