# src/generator.py

import ast
import re
import json
import uuid
import demjson3
from fuzzy_json import loads

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
    lang: str = "zh",
    timeout: float = 30.0,
    context_top_k: int = 20
) -> list[dict]:
    """
    1) Fetch all chunks for this doc_id
    2) Retrieve + rerank the top `context_top_k` most relevant chunks, using
       a *fixed* text query like "Generate investor questions"
       (or you could pick a better descriptor).
    3) Build context from those top_k
    4) Prompt LLM with timeout
    """
    
    print(f"Number of pairs that will be generated: {num_pairs}")
    
    # 1) Fetch every chunk for this PDF
    records = fetch_doc_chunks(doc_id)
    if not records:
        return []

    # 2) From those records, retrieve the top-k for the *question generation task*
    text_query = "Generate investor-style questions from this content"
    # embed & search that query against the collection; limit= context_top_k
    hits = retrieve(text_query, top_k=context_top_k)
    hits = rerank(text_query, hits)
    
    
    if lang == "zh":
        context = "\n\n".join(
            f"[第{h.payload['page']}頁] {h.payload['text']}"
            for h in hits
        )
        system_prompt = f"""
        你是投資人簡報助理。
        以下提供公司簡報內容，請以熱情、吸引投資的語氣，根據上下文恰好產出 {num_pairs} 組不重複的問答，
        並以第一人稱（我們／本公司）作答，答案不超過兩句，結尾帶投資亮點。

        嚴格要求：
        1. 僅回傳純 JSON 陣列，停在最後一個 ] 後立即停止，不可包含任何額外文字、XML、HTML、Markdown 或程式碼區塊。
        2. 每個物件都要有：
        - question
        - answer
        - source：單頁用 "第X頁"，多頁用 ["第X頁","第Y頁"]
        3. 請勿使用 XML 或任何其他非 JSON 語法。
        4. **生成後，請重新檢查並確保輸出為完全正確且格式良好的 JSON 結構**，再返回結果。

        範例（僅供格式參考，請勿複製）：
        ```json
        [
        {{
            "question": "2023 年的稅後淨利是多少？",
            "answer": "2023 年公司稅後淨利為新台幣 10.6 億元，展現穩健成長，值得長期投資。",
            "source": ["第18頁"]
        }}
        ]
        """
        user_prompt = f"內容：\n{context}\n\n請直接輸出上述格式的 JSON 陣列。"
    else:
        context = "\n\n".join(
            f"[p{h.payload['page']}] {h.payload['text']}"
            for h in hits
        )
        system_prompt = f"""
        You are an investor-briefing assistant.
        Below is a tightly-focused excerpt of the presentation. 
        In a promotional, pitch-to-invest tone, generate exactly {num_pairs} unique investor-style Q&A pairs:
        all answers in first-person (we/our company), max two sentences ending with an investment highlight.

        Strict requirements:
        1. Return **only** a pure JSON array; stop right after the closing bracket.
        2. Each object must have:
        - question
        - answer
        - source: "pX" or ["pX","pY"]
        3. **Do not** use XML, HTML tags, Markdown, or code fences—JSON only.
        4. **Escape rules**:  
        - Any double-quote inside a question or answer must be escaped as `\"`.  
        - Any literal newline must be escaped as `\n`.
        5. **After generation, revalidate and ensure the JSON structure is perfectly well-formed** before returning.

        Example:
        ```json
        [
        {{
            "question": "What was the company's \"governance\" achievement?",
            "answer": "We have been included in the Taiwan Index Company's \"Corporate Governance 100 Index\", demonstrating leadership.\nThis makes us a strong investment.",
            "source": ["p4"]
        }}
        ]
        """
        user_prompt = f"Context:\n{context}\n\nPlease output the JSON array now."

    # 3) Call LLM with timeout
    raw = LLM(ollama, OLLAMA_MODEL, system_prompt, user_prompt, options={"temperature":0.4}, timeout=timeout, raw=True)
    
    print("LLM raw QA output:", raw["response"])
    
    # 4) Parse any broken JSON
    qa_list = loads(raw["response"])
    
    # enforce exact count
    if len(qa_list) != num_pairs:
        print(f"⚠️ Expected {num_pairs} items but got {len(qa_list)}")

    # 6) Ensure UUIDs
    for item in qa_list:
        if "id" not in item:
            item["id"] = str(uuid.uuid4())

    return qa_list

def sanitize_json_via_llm(raw: str, client: Client, model: str, timeout: float = 15) -> str:
    """
    Ask the LLM to repair and re‐emit a valid JSON array.
    """
    system = (
        "You are a JSON validator and reformatter. "
        "The user will give you a string that is _almost_ a valid JSON array. "
        "Your job is to output **only** a well-formed JSON array—nothing else."
    )
    user = f"Here is the raw JSON to fix:\n\n{raw}\n\nPlease output a corrected JSON array."
    
    # Call your LLM wrapper in raw mode (no Pydantic schema)
    fixed = LLM(
        client=client,
        model=model,
        system_prompt=system,
        user_prompt=user,
        options={"temperature": 0},  # deterministic cleanup
        timeout=timeout,
        raw=True
    )
    return fixed.strip()