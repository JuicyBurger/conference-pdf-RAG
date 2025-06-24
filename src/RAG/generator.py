# src/RAG/generator.py
from ollama import Client
from config import OLLAMA_HOST, OLLAMA_MODEL
from src.LLM import LLM  # your LLM(...) function

client = Client(host=OLLAMA_HOST)

def generate_answer(query: str, hits):
    # build a single context string
    context = "\n\n".join(
        f"[p{h.payload['page']}] {h.payload['text']}"
        for h in hits
    )
    system_prompt = (
        "You are an investor-briefing assistant. "
        "Use the context to answer factually and cite page numbers."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return LLM(client, OLLAMA_MODEL, system_prompt, user_prompt)
