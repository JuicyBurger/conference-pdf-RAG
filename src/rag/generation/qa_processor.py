"""
QA pair processing for training data generation.

This module handles the conversion of questions into QA pairs with answers
for machine learning training purposes.
"""

import logging
from typing import List, Dict, Any

from .models import Question, QAPair, GenerationRequest, GenerationResult
from .question_generator import QuestionGenerator
from ..utils import handle_errors, GenerationError, setup_logger
from ...models.LLM import LLM
from ...models.client_factory import get_llm_client, get_default_model

logger = setup_logger(__name__)


class QAProcessor:
    """
    Processor for generating QA pairs with answers.
    
    This class takes questions and generates corresponding answers
    using the document context and retrieval system.
    """
    
    def __init__(self):
        """Initialize the QA processor."""
        self.question_generator = QuestionGenerator()
        self.llm_client = get_llm_client()
        self.default_model = get_default_model()
        logger.info("Initialized QA processor")
    
    def generate_answer_for_question(self, 
                                   question: str, 
                                   doc_context: str,
                                   sources: List[Dict[str, Any]],
                                   timeout: float = 60.0) -> str:
        """
        Generate an answer for a specific question using document context.
        
        Args:
            question: The question to answer
            doc_context: Document context for the answer
            sources: Source information for attribution
            timeout: Timeout for LLM generation
            
        Returns:
            Generated answer text
        """
        try:
            system_prompt = """你是一個專業的財務分析師助手。基於提供的文檔內容，為問題提供準確、詳細的回答。

要求：
1. 回答必須基於文檔中的具體內容
2. 回答要準確、客觀，避免推測
3. 如果文檔中沒有相關信息，明確說明
4. 回答要結構清晰，邏輯條理
5. 適當引用具體數據和事實
6. 保持專業的分析角度"""

            user_prompt = f"""基於以下文檔內容，回答問題：

問題：{question}

文檔內容：
{doc_context}

請提供詳細的回答："""

            options = {
                "temperature": 0.1,  # Low temperature for factual answers
                "max_tokens": 800,
                "top_p": 0.9
            }
            
            logger.debug(f"Generating answer for question: {question[:50]}...")
            
            answer = LLM(
                self.llm_client,
                self.default_model,
                system_prompt,
                user_prompt,
                options=options,
                timeout=timeout,
                raw=True
            )
            
            if not answer or not answer.strip():
                logger.warning(f"Empty answer generated for question: {question}")
                return "抱歉，根據提供的文檔內容，無法回答此問題。"
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate answer for question: {e}")
            return "抱歉，生成回答時發生錯誤。"
    
    def enhance_context_with_retrieval(self, 
                                     question: str, 
                                     doc_id: str,
                                     top_k: int = 5) -> tuple[str, List[Dict[str, Any]]]:
        """
        Enhance context using retrieval for better answer generation.
        
        Args:
            question: Question to retrieve context for
            doc_id: Document ID to scope retrieval
            top_k: Number of top chunks to retrieve
            
        Returns:
            Tuple of (enhanced_context, sources)
        """
        try:
            # Import here to avoid circular dependencies
            from ..retrieval.retrieval_service import retrieval_service
            from ...models.reranker import rerank
            
            # Retrieve relevant chunks for this question
            hits = retrieval_service.retrieve(
                query=question,
                top_k=top_k * 2,  # Get more for reranking
                score_threshold=0.3,
                room_id=None,
                prefer_chat_scope=False,
                scope_filter=None,
                extra_filters={"doc_id": doc_id}
            )
            
            if not hits:
                logger.warning(f"No relevant chunks found for question: {question}")
                return "", []
            
            # Rerank for better relevance
            reranked_hits = rerank(question, hits[:top_k])
            
            # Build enhanced context and sources
            context_parts = []
            sources = []
            
            for i, hit in enumerate(reranked_hits):
                try:
                    text = hit.payload.get("text", "")
                    page = hit.payload.get("page", "未知")
                    doc_id_hit = hit.payload.get("doc_id", doc_id)
                    score = getattr(hit, 'score', 0.0)
                    
                    if text.strip():
                        context_parts.append(f"[來源 {i+1}: {doc_id_hit} 第{page}頁]\n{text}")
                        
                        sources.append({
                            "text": text,
                            "page": page,
                            "doc_id": doc_id_hit,
                            "score": score,
                            "source_index": i + 1
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing retrieval hit: {e}")
                    continue
            
            enhanced_context = "\n\n".join(context_parts)
            logger.debug(f"Enhanced context with {len(sources)} sources")
            
            return enhanced_context, sources
            
        except Exception as e:
            logger.error(f"Failed to enhance context with retrieval: {e}")
            return "", []
    
    @handle_errors(error_class=GenerationError, reraise=False)
    def generate_qa_pairs(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate QA pairs for a document.
        
        Args:
            request: Generation request parameters
            
        Returns:
            Generation result with QA pairs
        """
        try:
            logger.info(f"Generating {request.num_items} QA pairs for doc_id: {request.doc_id}")
            
            # 1. First generate questions using the unified generator
            question_result = self.question_generator.generate_questions(request)
            if not question_result.success or not question_result.items:
                return GenerationResult(
                    success=False,
                    items=[],
                    doc_id=request.doc_id,
                    error=f"Failed to generate questions: {question_result.error}"
                )
            
            questions = question_result.items
            logger.info(f"Generated {len(questions)} questions, now generating answers")
            
            # 2. Generate answers for each question
            qa_pairs = []
            for i, question in enumerate(questions):
                try:
                    logger.debug(f"Generating answer {i+1}/{len(questions)}")
                    
                    # Enhance context using retrieval for this specific question
                    enhanced_context, sources = self.enhance_context_with_retrieval(
                        question.text, 
                        request.doc_id,
                        top_k=5
                    )
                    
                    # Generate answer
                    answer = self.generate_answer_for_question(
                        question.text,
                        enhanced_context,
                        sources,
                        timeout=request.timeout / request.num_items  # Distribute timeout
                    )
                    
                    # Create QA pair
                    qa_pair = QAPair(
                        question=question.text,
                        answer=answer,
                        doc_id=request.doc_id,
                        sources=sources,
                        confidence=question.confidence,
                        metadata={
                            "question_metadata": question.metadata,
                            "context_enhanced": bool(enhanced_context),
                            "source_count": len(sources)
                        }
                    )
                    
                    qa_pairs.append(qa_pair)
                    
                except Exception as e:
                    logger.error(f"Failed to generate answer for question {i+1}: {e}")
                    # Continue with other questions
                    continue
            
            if not qa_pairs:
                return GenerationResult(
                    success=False,
                    items=[],
                    doc_id=request.doc_id,
                    error="Failed to generate any QA pairs"
                )
            
            logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
            
            return GenerationResult(
                success=True,
                items=qa_pairs,
                doc_id=request.doc_id,
                metadata={
                    "questions_generated": len(questions),
                    "qa_pairs_completed": len(qa_pairs),
                    "success_rate": len(qa_pairs) / len(questions) if questions else 0
                }
            )
            
        except Exception as e:
            logger.error(f"QA pair generation failed for {request.doc_id}: {e}")
            return GenerationResult(
                success=False,
                items=[],
                doc_id=request.doc_id,
                error=str(e)
            )


# Convenience function for backward compatibility
def generate_qa_pairs_for_doc(doc_id: str, 
                             num_pairs: int = 5, 
                             timeout: float = 120.0,
                             context_top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Legacy function for generating QA pairs.
    
    Args:
        doc_id: Document ID to generate QA pairs for
        num_pairs: Number of QA pairs to generate  
        timeout: Timeout for generation
        context_top_k: Maximum context chunks to use
        
    Returns:
        List of QA pair dictionaries
    """
    request = GenerationRequest(
        doc_id=doc_id,
        num_items=num_pairs,
        timeout=timeout,
        context_top_k=context_top_k,
        use_lightweight=False  # QA generation uses full context
    )
    
    processor = QAProcessor()
    result = processor.generate_qa_pairs(request)
    
    if result.success:
        return [qa_pair.to_dict() for qa_pair in result.items]
    else:
        logger.error(f"QA generation failed: {result.error}")
        return []
