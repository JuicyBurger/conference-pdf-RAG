"""
Unified question generation core.

This module provides the central question generation logic shared between
QA generation and suggestions generation systems.
"""

import logging
import random
from typing import List, Dict, Any, Optional

from .models import Question, QAPair, QuestionSuggestion, GenerationRequest, GenerationResult
from ..utils import handle_errors, GenerationError, setup_logger
from ...models.LLM import LLM
from ...models.client_factory import get_llm_client, get_default_model

logger = setup_logger(__name__)


class QuestionGenerator:
    """
    Unified question generator for both QA pairs and suggestions.
    
    This class handles the core question generation logic and delegates
    to specialized processors for different output formats.
    """
    
    def __init__(self):
        """Initialize the question generator."""
        self.llm_client = get_llm_client()
        self.default_model = get_default_model()
        logger.info("Initialized unified question generator")
    
    def fetch_document_context(self, doc_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch document chunks for context building.
        
        Args:
            doc_id: Document ID to fetch chunks for
            limit: Maximum number of chunks to fetch
            
        Returns:
            List of document chunk records
        """
        try:
            # Import here to avoid circular dependencies
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            from ...config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
            
            # Use Qdrant client directly to fetch document chunks
            qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # Build filter for specific document
            scroll_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
            
            # Fetch chunks
            records, _ = qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True
            )
            
            logger.debug(f"Fetched {len(records)} chunks for doc_id: {doc_id}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to fetch document context for {doc_id}: {e}")
            return []
    
    def build_document_context(self, records: List[Dict[str, Any]], max_chunks: int = 10) -> str:
        """
        Build document context from chunk records.
        
        Args:
            records: Document chunk records
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted document context string
        """
        if not records:
            return ""
        
        context_blocks = []
        for record in records[:max_chunks]:
            try:
                # Handle both Qdrant record format and dict format
                if hasattr(record, 'payload'):
                    payload = record.payload
                else:
                    payload = record
                
                page = payload.get("page", "未知")
                text = payload.get("text", "")[:200]  # Limit text length for context
                
                if text.strip():
                    context_blocks.append(f"[第{page}頁] {text}")
                    
            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        context = "\n\n".join(context_blocks)
        logger.debug(f"Built context with {len(context_blocks)} blocks, {len(context)} characters")
        return context
    
    def generate_questions_with_llm(self, 
                                   context: str, 
                                   num_questions: int,
                                   request: GenerationRequest) -> List[str]:
        """
        Generate questions using LLM based on document context.
        
        Args:
            context: Document context text
            num_questions: Number of questions to generate
            request: Generation request with parameters
            
        Returns:
            List of generated question strings
        """
        try:
            # Choose prompt based on whether we have chat context
            if request.chat_context:
                system_prompt = self._get_chat_context_system_prompt(num_questions)
                user_prompt = self._get_chat_context_user_prompt(context, request.chat_context, num_questions)
            else:
                system_prompt = self._get_default_system_prompt(num_questions)
                user_prompt = self._get_default_user_prompt(context, num_questions)
            
            # Apply custom prompts if provided
            if request.custom_prompts:
                if "system" in request.custom_prompts:
                    system_prompt = request.custom_prompts["system"]
                if "user" in request.custom_prompts:
                    user_prompt = request.custom_prompts["user"]
            
            # Configure LLM options
            options = {
                "temperature": 0.3,  # Some creativity but not too much
                "max_tokens": 800 if request.use_lightweight else 1200,
                "top_p": 0.9
            }
            
            logger.debug(f"Generating {num_questions} questions with LLM")
            
            # Call LLM
            response = LLM(
                self.llm_client,
                self.default_model,
                system_prompt,
                user_prompt,
                options=options,
                timeout=request.timeout,
                raw=True
            )
            
            if not response or not response.strip():
                logger.warning("LLM returned empty response")
                return []
            
            # Parse questions from response
            questions = self._parse_questions_from_response(response, num_questions)
            logger.info(f"Successfully generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate questions with LLM: {e}")
            return []
    
    def _get_default_system_prompt(self, num_questions: int) -> str:
        """Get default system prompt for question generation."""
        return f"""你是一個專業的財務分析師。基於提供的文檔內容，生成 {num_questions} 個簡潔、深度且有價值的問題。

要求：
1. 問題必須基於文檔中的具體內容
2. 問題要簡潔明瞭，控制在 30 字以內
3. 問題要有深度，能引發深入思考
4. 問題要涵蓋不同面向：財務、營運、風險、治理等
5. 每個問題都要具體且可回答
6. 避免冗長的背景描述，直接提出核心問題
7. 只返回問題列表，不要包含答案或解釋

格式：直接返回問題列表，每行一個問題。

良好範例：
- 營收下滑的主要原因是什麼？
- 公司如何改善獲利能力？
- 主要競爭風險有哪些？
- 現金流量是否健康？"""
    
    def _get_default_user_prompt(self, context: str, num_questions: int) -> str:
        """Get default user prompt for question generation."""
        return f"""基於以下文檔內容，生成 {num_questions} 個簡潔深度問題（每個問題控制在30字以內）：

{context}

請生成 {num_questions} 個簡潔問題："""
    
    def _get_chat_context_system_prompt(self, num_questions: int) -> str:
        """Get system prompt for chat context-aware question generation."""
        return f"""你是一個專業的AI助手。基於提供的文檔內容和對話歷史，生成 {num_questions} 個簡潔、深度且有價值的問題。

要求：
1. 問題要結合文檔內容和對話脈絡
2. 問題要簡潔明瞭，控制在 30 字以內
3. 問題要有深度，能引發深入思考和進一步探討
4. 問題要自然地延續對話主題
5. 每個問題都要具體且可回答
6. 避免重複已經討論過的問題
7. 只返回問題列表，不要包含答案或解釋

格式：直接返回問題列表，每行一個問題。

良好範例：
- 這個趨勢背後的原因是什麼？
- 有哪些替代方案可以考慮？
- 風險評估結果如何？
- 預期的效益有多大？"""
    
    def _get_chat_context_user_prompt(self, context: str, chat_context: str, num_questions: int) -> str:
        """Get user prompt for chat context-aware question generation."""
        return f"""基於以下文檔內容和對話歷史，生成 {num_questions} 個簡潔的後續問題（每個問題控制在30字以內）：

文檔內容：
{context}

對話歷史：
{chat_context}

請基於文檔內容和對話脈絡生成 {num_questions} 個有意義的問題："""
    
    def _parse_questions_from_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse questions from LLM response.
        
        Args:
            response: Raw LLM response
            expected_count: Expected number of questions
            
        Returns:
            List of parsed question strings
        """
        try:
            # Split by lines and clean up
            lines = response.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Remove common prefixes
                for prefix in ['- ', '• ', f'{len(questions)+1}. ', f'{len(questions)+1}、']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                # Skip if too short or looks like metadata
                if len(line) < 5 or line.lower().startswith(('note:', '註:', '說明:')):
                    continue
                
                # Add question mark if missing
                if not line.endswith('？') and not line.endswith('?'):
                    line += '？'
                
                questions.append(line)
                
                # Stop if we have enough questions
                if len(questions) >= expected_count:
                    break
            
            logger.debug(f"Parsed {len(questions)} questions from response")
            return questions[:expected_count]
            
        except Exception as e:
            logger.error(f"Failed to parse questions from response: {e}")
            return []
    
    @handle_errors(error_class=GenerationError, reraise=False)
    def generate_questions(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate questions based on request.
        
        Args:
            request: Generation request parameters
            
        Returns:
            Generation result with questions
        """
        try:
            logger.info(f"Generating {request.num_items} questions for doc_id: {request.doc_id}")
            
            # 1. Fetch document context
            records = self.fetch_document_context(request.doc_id, limit=request.context_top_k)
            if not records:
                return GenerationResult(
                    success=False,
                    items=[],
                    doc_id=request.doc_id,
                    error="No document chunks found"
                )
            
            # 2. Build context
            context = self.build_document_context(records, max_chunks=10)
            if not context:
                return GenerationResult(
                    success=False,
                    items=[],
                    doc_id=request.doc_id,
                    error="No usable context found"
                )
            
            # 3. Generate questions
            question_texts = self.generate_questions_with_llm(context, request.num_items, request)
            if not question_texts:
                return GenerationResult(
                    success=False,
                    items=[],
                    doc_id=request.doc_id,
                    error="Failed to generate questions"
                )
            
            # 4. Create Question objects
            questions = []
            for text in question_texts:
                question = Question(
                    text=text,
                    doc_id=request.doc_id,
                    metadata={
                        "chat_context": bool(request.chat_context),
                        "room_id": request.room_id,
                        "generation_method": "llm"
                    }
                )
                questions.append(question)
            
            return GenerationResult(
                success=True,
                items=questions,
                doc_id=request.doc_id,
                metadata={
                    "context_chunks": len(records),
                    "context_length": len(context),
                    "lightweight": request.use_lightweight
                }
            )
            
        except Exception as e:
            logger.error(f"Question generation failed for {request.doc_id}: {e}")
            return GenerationResult(
                success=False,
                items=[],
                doc_id=request.doc_id,
                error=str(e)
            )
