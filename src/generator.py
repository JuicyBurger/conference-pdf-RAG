# src/generator.py

import json

from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from .config import OLLAMA_HOST, OLLAMA_MODEL, QDRANT_URL, QDRANT_API_KEY
from .retriever import retrieve
from .reranker import rerank
from .LLM import LLM

# Initialize clients
ollama = Client(host=OLLAMA_HOST)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def generate_answer(query: str, hits, lang: str = "zh") -> str:
    """
    Generate an answer in either Traditional Chinese ("zh") or English ("en").
    - query: the user's question
    - hits: list of ScoredPoint from Qdrant
    - lang: "zh" or "en" (default "zh")
    """
    if lang == "zh":
        context = "\n\n".join(
            f"[第{h.payload['page']}頁] {h.payload['text']}" for h in hits
        )
        system_prompt = (
            "你是投資人簡報助理。"
            "以下是文件內容，請根據上下文，用繁體中文回答並標註出處頁碼。"
        )
        user_prompt = f"內容：\n{context}\n\n請問：{query}\n回答："
    else:
        context = "\n\n".join(
            f"[p{h.payload['page']}] {h.payload['text']}" for h in hits
        )
        system_prompt = (
            "You are an investor-briefing assistant. "
            "Use the context to answer factually and cite page numbers."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    return LLM(ollama, OLLAMA_MODEL, system_prompt, user_prompt)


def fetch_doc_chunks(doc_id: str, limit: int = 1000):
    """
    Scroll Qdrant for all chunks where payload.doc_id == doc_id.
    Attach full payloads so you can see page/text.
    """
    # Build the filter condition
    scroll_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    # Note: use scroll_filter, and ask for payload
    records, _ = qdrant.scroll(
        collection_name="docs",
        scroll_filter=scroll_filter,
        limit=limit,
        with_payload=True
    )
    return records


def generate_qa_pairs_for_doc(
    doc_id: str,
    num_pairs: int = 5,
    lang: str = "zh"
) -> list:
    """
    For a single document (by doc_id), fetch its chunks, and prompt the LLM
    to produce num_pairs investor-focused Q&A pairs in a JSON array:
    each with id, question, answer, and source.
    """
    points = fetch_doc_chunks(doc_id)

    if lang == "zh":
        context = "\n\n".join(
            f"[第{p.payload['page']}頁] {p.payload['text']}" for p in points
        )
        system_prompt = """你是投資人簡報助理。  
        以下是公司簡報的內容，請以熱情、吸引投資的語氣，根據上下文產出 {num_pairs} 組問答（Question & Answer）。  
        每組必須是一個 JSON 物件，包含以下欄位：  
        - id：唯一識別碼（UUID 或整數皆可），  
        - question：投資人會問的問題，  
        - answer：簡潔、具說服力的回答，  
        - source：出處頁碼，格式「第X頁」。  

        **請僅回傳一個純 JSON 陣列**，不要包含任何額外文字或 Markdown。  
        格式範例：  
        ```json
        [
        {
            "id": "a1b2c3d4",
            "question": "2023 年的稅後淨利是多少？",
            "answer": "2023 年公司稅後淨利為新台幣 10.6 億元，展現穩健成長，值得關注。",
            "source": "第18頁"
        },
        …
        ]
        
        「只回傳 JSON，並在最後一個右中括號後立刻停止。」
        """
        user_prompt = f"內容：\n{context}\n\n請開始產出 Q&A："
    else:
        context = "\n\n".join(
            f"[p{p.payload['page']}] {p.payload['text']}" for p in points
        )
        system_prompt = """
        You are an investor-briefing assistant.  
        Below is the presentation content. In a promotional, pitch-to-invest tone, generate {num_pairs} investor-style Q&A pairs.  
        Return a single **JSON array** (no extra text or markdown), where each item has the keys:  
        - id: unique identifier (UUID or integer),  
        - question: a question an investor might ask,  
        - answer: a concise, compelling answer,  
        - source: the page number in format “pX”.  

        Example output:  
        json
        [
        {
            "id": "1",
            "question": "What was the net profit after tax in 2023?",
            "answer": "In 2023, the company achieved an after-tax net profit of NT$10.6 billion, demonstrating solid financial performance.",
            "source": "p18"
        },
        …
        ]
        
        Only return the JSON array and stop immediately after the closing bracket.
        """
        user_prompt = f"Context:\n{context}\n\Generate Q&A pairs from the context. Response in JSON object with `id`, `question`, `answer`, `source`"

    raw = LLM(ollama, OLLAMA_MODEL, system_prompt, user_prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        snippet = raw.strip().split("\n")[-1]
        data = json.loads(snippet)

    # for item in data:
    #     item.setdefault("id", str(uuid.uuid4()))

    return data
