"""
Response generation functionality for RAG systems.

This module provides the core answer generation functionality that was previously
in qa_generation/answer_generator.py. It's now properly placed as part of the
main RAG pipeline rather than in a training/QA generation subfolder.
"""

import logging
import os
from typing import List, Any, Dict, Optional

from src.models.LLM import LLM
from src.models.client_factory import get_llm_client, get_default_model
from ..utils import handle_errors, GenerationError, setup_logger

# Configure logging
logger = setup_logger(__name__)


class ResponseGenerator:
    """
    Core response generator for RAG systems.
    
    Handles the generation of answers from retrieved context using LLMs.
    Supports different response modes and context processing strategies.
    """
    
    def __init__(self, 
                 llm_client=None, 
                 model=None,
                 response_language: str = "zh-TW"):
        """
        Initialize the response generator.
        
        Args:
            llm_client: LLM client instance (defaults to factory client)
            model: Model name (defaults to factory model)
            response_language: Language for responses (default: Traditional Chinese)
        """
        self.llm_client = llm_client or get_llm_client()
        self.model = model or get_default_model()
        self.response_language = response_language
        self.logger = logging.getLogger(f"{__name__}.ResponseGenerator")
    
    @handle_errors(error_class=GenerationError, fallback_return="抱歉，我在處理您的請求時遇到了問題。請再試一次。")
    def generate_response(self, 
                         query: str, 
                         hits: List[Any],
                         context_mode: str = "standard") -> str:
        """
        Generate a response from query and retrieved hits.
        
        Args:
            query: The user's question
            hits: List of ScoredPoint from retrieval systems
            context_mode: Context processing mode ("standard", "table", "graph")
            
        Returns:
            Generated response string
        """
        # Handle empty or weak hits
        if not hits:
            return self._get_no_evidence_response()
        
        # Process and validate hits
        valid_hits = self._process_hits(hits)
        if not valid_hits:
            return self._get_no_evidence_response()
        
        # Check relevance threshold
        if not self._check_relevance_threshold(valid_hits):
            return self._get_low_relevance_response()
        
        # Build context based on mode
        context = self._build_context(query, valid_hits, context_mode)
        
        # Generate response using LLM
        return self._call_llm(query, context)
    
    def _get_no_evidence_response(self) -> str:
        """Get response when no evidence is available."""
        return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"
    
    def _get_low_relevance_response(self) -> str:
        """Get response when relevance is too low."""
        return "目前檢索到的內容相關性不足。請提供更明確的關鍵詞、頁碼或文件名稱以便查找。"
    
    def _process_hits(self, hits: List[Any]) -> List[Any]:
        """
        Process and validate hits from retrieval systems.
        
        Args:
            hits: Raw hits from retrieval
            
        Returns:
            List of valid, normalized hits
        """
        valid_hits = []
        
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
                        else:
                            self.logger.warning(f"⚠️ Skipping hit with empty {text_field}: {getattr(h, 'id', 'unknown')}")
                    else:
                        self.logger.warning(f"⚠️ Skipping hit without proper payload structure: {getattr(h, 'id', 'unknown')}")
                        if hasattr(h, 'payload'):
                            self.logger.warning(f"   Payload keys: {list(h.payload.keys()) if h.payload else 'None'}")
                else:
                    self.logger.warning(f"⚠️ Skipping hit without payload: {getattr(h, 'id', 'unknown')}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error processing hit {getattr(h, 'id', 'unknown')}: {e}")
                continue
        
        return valid_hits
    
    def _check_relevance_threshold(self, hits: List[Any]) -> bool:
        """
        Check if hits meet relevance threshold.
        
        Args:
            hits: Valid hits to check
            
        Returns:
            True if hits are relevant enough, False otherwise
        """
        # Skip threshold check if disabled
        if os.getenv("DISABLE_QUERY_SIGNAL", "0") == "1":
            return True
        
        # Find top score
        top_score = 0.0
        for h in hits:
            try:
                score = float(getattr(h, 'score', 0.0))
                top_score = max(top_score, score)
            except Exception:
                pass
        
        # Lowered threshold to allow more evidence
        return top_score >= 0.1
    
    def _build_context(self, 
                      query: str, 
                      hits: List[Any], 
                      mode: str = "standard") -> str:
        """
        Build context string from hits based on mode.
        
        Args:
            query: User query for context optimization
            hits: Valid hits to build context from
            mode: Context building mode
            
        Returns:
            Formatted context string
        """
        # Sort hits by score to prioritize most relevant information
        sorted_hits = sorted(hits, key=lambda h: getattr(h, 'score', 0.0), reverse=True)
        
        # Determine context limit based on query type and mode
        context_limit = self._get_context_limit(query, mode)
        context_hits = sorted_hits[:context_limit]
        
        # Build context parts
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
        
        return "\n\n".join(context_parts)
    
    def _get_context_limit(self, query: str, mode: str) -> int:
        """
        Determine context limit based on query and mode.
        
        Args:
            query: User query
            mode: Context mode
            
        Returns:
            Number of hits to include in context
        """
        if mode == "table":
            return 25  # More hits for table queries
        elif mode == "graph":
            return 15  # Moderate hits for graph queries
        else:
            # Check if query suggests table content
            is_table_query = any(keyword in query.lower() for keyword in [
                'table', '表格', 'show me', 'complete', 'display', 'find', 'contents'
            ])
            return 25 if is_table_query else 8
    
    def _call_llm(self, query: str, context: str) -> str:
        """
        Call LLM to generate response.
        
        Args:
            query: User query
            context: Formatted context
            
        Returns:
            Generated response
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        return LLM(
            self.llm_client, 
            self.model, 
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
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for response generation."""
        return """
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
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with query and context."""
        return (
            f"以下是可用的文件節錄（含來源標註資料）：\n{context}\n\n"
            f"問題：{query}\n"
            "請依上方節錄內容作答並遵守系統規範，特別注意：\n"
            "- 僅根據節錄內容作答；若不足以回答，請說明不足與需要的頁碼/關鍵詞。\n"
            "- 對所有使用到的關鍵事實逐一標註來源 [doc_id 第X頁]。\n"
            "- 若問題涉及表格，請：以 Markdown 呈現重點列、說明表格結構與單位、指出趨勢/異常，並在表格下方標註來源。\n"
            "- 若節錄包含多個表格或多年度資料，請先說明你的選取依據（例如與問題關鍵詞最相符、最新年度優先）。\n"
            "- 目標篇幅約 200–500 字；若資料極少，允許更短但不得虛構內容。\n"
        )


# Global instance for backward compatibility
_default_generator = None

def get_default_generator() -> ResponseGenerator:
    """Get the default response generator instance."""
    global _default_generator
    if _default_generator is None:
        _default_generator = ResponseGenerator()
    return _default_generator


# Convenience function for backward compatibility
def generate_answer(query: str, hits: List[Any]) -> str:
    """
    Generate an answer using the default response generator.
    
    This function maintains backward compatibility with the existing codebase.
    
    Args:
        query: The user's question
        hits: List of ScoredPoint from retrieval systems
        
    Returns:
        Generated answer
    """
    generator = get_default_generator()
    return generator.generate_response(query, hits)
