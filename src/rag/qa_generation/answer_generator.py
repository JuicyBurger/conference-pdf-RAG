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
    context = "\n\n".join(
        f"[{h.payload.get('doc_id', 'unknown')} 第{h.payload['page']}頁] {h.payload['text']}" for h in valid_hits
    )
    system_prompt = (
        "你是投資人簡報助理。"
        "根據提供的節錄內容回答問題。"
        "如果節錄內容與問題相關，請直接回答並標註來源。"
        "如果節錄內容與問題無關，請說明節錄內容的性質，並建議用戶提供更具體的關鍵詞或頁碼。"
        "回答時務必使用繁體中文；若引用，僅標註實際出現於節錄中的 doc_id 與頁碼。"
    )
    user_prompt = (
        f"以下是可用的文件節錄：\n{context}\n\n"
        f"問題：{query}\n"
        "要求：\n"
        "- 如果節錄內容與問題相關，請直接回答並標註 [doc_id 第X頁]；\n"
        "- 如果節錄內容與問題無關，請說明節錄內容的性質（如：財務數據、股東資訊等），並建議用戶提供更具體的關鍵詞或頁碼；\n"
        "- 不要臆測或編造答案。\n"
        "回答："
    )

    return LLM(llm_client, default_model, system_prompt, user_prompt, raw=True)
