# src/RAG/retriever.py
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY
from src.embedder import embed

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def retrieve(query: str, top_k: int = 5):
    q_vec = embed(query)
    hits = client.search(
        collection_name="docs",
        query_vector=q_vec,
        limit=top_k
    )
    # each hit has .payload['text'], .payload['page'], etc.
    return hits
