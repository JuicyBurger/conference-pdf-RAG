# src/RAG/reranker.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model     = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, hits):
    # Handle empty hits list
    if not hits:
        return []
    
    # Filter out hits without proper payload structure
    valid_hits = []
    for h in hits:
        if hasattr(h, 'payload') and h.payload and 'text' in h.payload:
            valid_hits.append(h)
        else:
            print(f"⚠️ Skipping hit without 'text' in payload: {getattr(h, 'id', 'unknown')}")
    
    if not valid_hits:
        print("⚠️ No valid hits with 'text' payload found for reranking")
        return []
    
    pairs = [f"{query} [SEP] {h.payload['text']}" for h in valid_hits]
    enc   = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        scores = model(**enc).logits.squeeze(-1).tolist()
    # zip, sort by score desc
    ranked = sorted(zip(valid_hits, scores), key=lambda x: x[1], reverse=True)
    return [h for h, _ in ranked]