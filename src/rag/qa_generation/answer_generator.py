"""
Answer generation functionality.

This module provides functions for generating answers from retrieved context.
"""

import logging
import os
from typing import List, Any, Dict

from src.models.LLM import LLM
from src.models.client_factory import get_llm_client, get_default_model
from ..utils import handle_errors, GenerationError, setup_logger

# Configure logging
logger = setup_logger(__name__)

# Initialize clients using the factory
llm_client = get_llm_client()
default_model = get_default_model()

@handle_errors(error_class=GenerationError, fallback_return="抱歉，我在處理您的請求時遇到了問題。請再試一次。")
def generate_answer(query: str, hits: List[Any]) -> str:
    """
    Generate an answer in Traditional Chinese.
    
    Args:
        query: The user's question
        hits: List of ScoredPoint from Qdrant
        
    Returns:
        Generated answer
    """
    # Handle empty or weak hits: ask for clarification instead of fabricating
    if not hits:
        return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"
    
    # Filter valid hits with proper payload structure
    valid_hits = []
    top_score_observed = 0.0
    for h in hits:
        try:
            # Check for both 'text' and 'content' fields (different indexing methods)
            text_field = None
            if hasattr(h, 'payload') and h.payload:
                if 'text' in h.payload:
                    text_field = 'text'
                elif 'content' in h.payload:
                    text_field = 'content'
                
                if text_field and 'page' in h.payload:
                    # Additional validation: ensure text is not empty
                    if h.payload[text_field] and h.payload[text_field].strip():
                        # Create a normalized hit with 'text' field for consistency
                        normalized_hit = type('Hit', (), {
                            'id': getattr(h, 'id', 'unknown'),
                            'score': getattr(h, 'score', 0.0),
                            'payload': {
                                'text': h.payload[text_field],
                                'page': h.payload['page'],
                                'doc_id': h.payload.get('doc_id', 'unknown'),
                                'type': h.payload.get('type', 'unknown')
                            }
                        })()
                        valid_hits.append(normalized_hit)
                        try:
                            top_score_observed = max(top_score_observed, float(getattr(h, 'score', 0.0)))
                        except Exception:
                            pass
                    else:
                        logger.warning(f"⚠️ Skipping hit with empty {text_field}: {getattr(h, 'id', 'unknown')}")
                else:
                    logger.warning(f"⚠️ Skipping hit without proper payload structure: {getattr(h, 'id', 'unknown')}")
                    if hasattr(h, 'payload'):
                        logger.warning(f"   Payload keys: {list(h.payload.keys()) if h.payload else 'None'}")
            else:
                logger.warning(f"⚠️ Skipping hit without payload: {getattr(h, 'id', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ Error processing hit {getattr(h, 'id', 'unknown')}: {e}")
            continue
    
    if not valid_hits:
        return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"

    # Lower threshold to allow more evidence to be processed
    if os.getenv("DISABLE_QUERY_SIGNAL", "0") != "1":
        if top_score_observed < 0.1:  # Lowered significantly to allow more evidence
            return "目前檢索到的內容相關性不足。請提供更明確的關鍵詞、頁碼或文件名稱以便查找。"
    
    # Include doc_id in context to reduce invented labels like "Document 1"
    # Sort hits by score to prioritize the most relevant information
    sorted_hits = sorted(valid_hits, key=lambda h: getattr(h, 'score', 0.0), reverse=True)
    
    # Take more hits for better context (up to 8 instead of all)
    # For table queries, take more hits to get comprehensive table data
    is_table_query = any(keyword in query.lower() for keyword in ['table', '表格', 'show me', 'complete', 'display', 'find', 'contents'])
    if is_table_query:
        context_hits = sorted_hits[:25]  # Much more hits for table queries
    else:
        context_hits = sorted_hits[:8]   # Standard limit for other queries
    
    context_parts = []
    for i, h in enumerate(context_hits, 1):
        doc_id = h.payload.get('doc_id', 'unknown')
        page = h.payload['page']
        text = h.payload['text']
        score = getattr(h, 'score', 0.0)
        
        # Truncate very long texts to keep context manageable
        if len(text) > 800:
            text = text[:800] + "..."
        
        context_parts.append(f"[來源 {i}: {doc_id} 第{page}頁, 相關度: {score:.3f}]\n{text}")
    
    context = "\n\n".join(context_parts)
    system_prompt = (
        """
        你是一位專業的投資人簡報分析師，專門分析公司治理報告、財務報表與年報節錄。只可使用提供的節錄內容作答；若資訊不足，請明確說明不足之處並告知需要的頁碼/關鍵詞。
        原則：忠實、可驗證、可追溯。

        回答規範
        1. 精確數據：所有數字/百分比/金額務必照原文呈現含單位與小數位；不得臆測或補齊缺失值。若需計算（加總/平均），請在文中給出簡短算式並標註來源。
        2. 衝突處理：若節錄出現互相衝突的數字，同時列出並標註各自來源，再給出審慎判讀（不要擇一隱藏）。
        3. 來源標註：每一段關鍵事實後附上來源標註，格式 [doc_id 第X頁]；多個來源以逗號分隔。
        4. 單位與期間：單位以原文為準；期間一律用絕對日期/年度（例如「民國113年」亦可附西元）。
        5. 表格回答（若問題涉及表格）：
            - 以 Markdown 表格重建重點欄列；若表格過長（>50 行），請標示「已截斷」並顯示前 20 行 + 後 5 行，同時提供整體統計（合計/平均/極值）。
            - 解釋表格的結構（欄位/面板/單位）與意義，並指出明顯趨勢或異常。
            - 嚴禁生成原文中不存在的列/欄。
        6. 語氣與可讀性：使用專業但易懂的繁體中文；段落分明、先結論後依據。

        輸出格式（順序固定）
        - 直接回答（一段話）
        - 關鍵數據/表格（必要時使用 Markdown 表格）
        - 解釋與趨勢（含必要的簡短算式）
        - 來源（逐條列出對應的 [doc_id 第X頁]）
        - 侷限與建議（若資料不足或需補件）
        """
    )
    user_prompt = (
        f"以下是可用的文件節錄（含來源標註資料）：\n{context}\n\n"
        f"問題：{query}\n"
        "請依上方節錄內容作答並遵守系統規範，特別注意：\n"
        "- 僅根據節錄內容作答；若不足以回答，請說明不足與需要的頁碼/關鍵詞。\n"
        "- 對所有使用到的關鍵事實逐一標註來源 [doc_id 第X頁]。\n"
        "- 若問題涉及表格，請：以 Markdown 呈現重點列、說明表格結構與單位、指出趨勢/異常，並在表格下方標註來源。\n"
        "- 若節錄包含多個表格或多年度資料，請先說明你的選取依據（例如與問題關鍵詞最相符、最新年度優先）。\n"
        "- 目標篇幅約 200–500 字；若資料極少，允許更短但不得虛構內容。\n"
    )

    return LLM(
        llm_client, 
        default_model, 
        system_prompt, 
        user_prompt, 
        options={
            "temperature": 0.3,  # Slightly lower for more focused responses
            "num_predict": 2048,  # Allow longer responses
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096  # Larger context window
        },
        raw=True
    )
