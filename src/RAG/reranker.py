# src/RAG/reranker.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model     = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, hits):
    pairs = [f"{query} [SEP] {h['payload']['text']}" for h in hits]
    enc   = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        scores = model(**enc).logits.squeeze(-1).tolist()
    # zip, sort by score desc
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [h for h, _ in ranked]
