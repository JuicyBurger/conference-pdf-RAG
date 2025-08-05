# src/rag/suggestions/generator.py

import logging
from typing import List

from .catalog import QuestionSuggestion, store_questions, init_suggestion_collection
from ..generator import generate_qa_pairs_for_doc

logger = logging.getLogger(__name__)

def generate_questions_only(
    doc_id: str, 
    num_questions: int = 8,
    timeout: float = 30.0
) -> List[str]:
    """
    Generate only questions (without answers) for faster suggestion generation.
    This is much lighter than full QA generation.
    """
    try:
        logger.info(f"🤖 Generating {num_questions} questions only for doc_id: {doc_id}")
        
        # Use a much simpler prompt for question generation only
        system_prompt = """你是一個專業的財務分析師。基於提供的文檔內容，生成 {num_questions} 個深度且有價值的問題。

要求：
1. 問題必須基於文檔中的具體內容
2. 問題要有深度，能引發深入思考
3. 問題要涵蓋不同面向：財務、營運、風險、治理等
4. 每個問題都要具體且可回答
5. 只返回問題列表，不要包含答案

格式：直接返回問題列表，每行一個問題。"""

        user_prompt = f"""基於以下文檔內容，生成 {num_questions} 個深度問題：

{{context}}

請生成 {num_questions} 個問題："""

        # Get document chunks for context
        from ..generator import fetch_doc_chunks
        records = fetch_doc_chunks(doc_id, limit=50)  # Reduced limit for speed
        
        if not records:
            logger.warning(f"No chunks found for doc_id: {doc_id}")
            return []
        
        # Build simple context (just first 10 chunks)
        context_blocks = []
        for record in records[:10]:  # Limit to first 10 chunks
            # Access Qdrant Record payload attributes
            page = record.payload.get("page", "未知")
            text = record.payload.get("text", "")[:200]  # Limit text length
            context_blocks.append(f"[第{page}頁] {text}")
        
        context = "\n\n".join(context_blocks)
        
        # Call LLM with lighter parameters
        from ...models.LLM import LLM
        from ...models.client_factory import get_llm_client, get_default_model
        
        llm_client = get_llm_client()
        default_model = get_default_model()
        
        response = LLM(
            llm_client, 
            default_model, 
            system_prompt.format(num_questions=num_questions),
            user_prompt.format(context=context),
            options={"temperature": 0.3, "max_length": 1024},  # Much smaller for questions only
            timeout=timeout,
            raw=True
        )
        
        if not response:
            logger.warning("No response from LLM for question generation")
            return []
        
        # Parse questions from response
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('*'):
                # Clean up the question
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    line = line[2:].strip()
                elif line.startswith(('-', '•', '·')):
                    line = line[1:].strip()
                
                if line and len(line) > 10:  # Minimum question length
                    questions.append(line)
        
        # Limit to requested number
        questions = questions[:num_questions]
        
        logger.info(f"✅ Generated {len(questions)} questions for doc_id: {doc_id}")
        return questions
        
    except Exception as e:
        logger.error(f"Failed to generate questions for doc_id {doc_id}: {e}")
        return []

def generate_suggestions_for_doc(
    doc_id: str, 
    num_questions: int = 10,
    auto_init_collection: bool = True,
    use_lightweight: bool = True  # New parameter for lightweight generation
) -> bool:
    """
    Generate question suggestions for a document and store them in the suggestion catalog.
    
    Args:
        doc_id: Document ID to generate suggestions for
        num_questions: Number of questions to generate
        auto_init_collection: Whether to auto-initialize the collection if it doesn't exist
        use_lightweight: Whether to use lightweight question-only generation (faster)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"🤖 Generating suggestions for doc_id: {doc_id}")
        
        # Initialize suggestion collection if needed
        if auto_init_collection:
            try:
                init_suggestion_collection()
            except Exception as e:
                # Collection might already exist, that's fine
                logger.debug(f"Collection init result: {e}")
        
        if use_lightweight:
            # Use lightweight question-only generation
            questions = generate_questions_only(
                doc_id=doc_id,
                num_questions=num_questions,
                timeout=30.0  # Shorter timeout
            )
            
            if not questions:
                logger.warning(f"No questions generated for doc_id: {doc_id}")
                return False
            
            # Convert questions to QuestionSuggestion objects
            suggestions = []
            for question_text in questions:
                # Create simple section metadata
                sections = [{
                    "page": "未知",
                    "chunk_id": doc_id,
                    "source_text": "基於文檔內容生成"
                }]
                
                suggestion = QuestionSuggestion(
                    question_text=question_text,
                    doc_id=doc_id,
                    sections=sections,
                    tags=None,
                    popularity_score=0.0
                )
                
                suggestions.append(suggestion)
                
        else:
            # Use full QA generation (original method)
            qa_pairs = generate_qa_pairs_for_doc(
                doc_id=doc_id,
                num_pairs=num_questions,
                timeout=60.0
            )
            
            if not qa_pairs:
                logger.warning(f"No QA pairs generated for doc_id: {doc_id}")
                return False
            
            # Extract questions and convert to QuestionSuggestion objects
            suggestions = []
            for qa_pair in qa_pairs:
                try:
                    question_text = qa_pair.get("question", "").strip()
                    if not question_text:
                        continue
                    
                    # Extract section metadata from sources
                    sections = []
                    sources = qa_pair.get("sources", [])
                    
                    for source in sources:
                        section_info = {
                            "page": source.get("page"),
                            "chunk_id": source.get("doc_id"),
                            "source_text": source.get("text", "")[:200] + "..." if len(source.get("text", "")) > 200 else source.get("text", "")
                        }
                        sections.append(section_info)
                    
                    # Create suggestion object
                    suggestion = QuestionSuggestion(
                        question_text=question_text,
                        doc_id=doc_id,
                        sections=sections,
                        tags=None,
                        popularity_score=0.0
                    )
                    
                    suggestions.append(suggestion)
                    
                except Exception as e:
                    logger.warning(f"Failed to process QA pair: {e}")
                    continue
        
        if not suggestions:
            logger.warning(f"No valid suggestions extracted for doc_id: {doc_id}")
            return False
        
        # Store suggestions in catalog
        store_questions(suggestions)
        
        logger.info(f"✅ Generated and stored {len(suggestions)} suggestions for doc_id: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate suggestions for doc_id {doc_id}: {e}")
        return False

def batch_generate_suggestions(
    doc_ids: List[str], 
    num_questions_per_doc: int = 8,
    use_lightweight: bool = True  # Default to lightweight
) -> dict:
    """
    Generate suggestions for multiple documents in batch.
    
    Args:
        doc_ids: List of document IDs
        num_questions_per_doc: Number of questions to generate per document
        use_lightweight: Whether to use lightweight generation (faster)
    
    Returns:
        dict: Results summary with success/failure counts
    """
    results = {
        "total": len(doc_ids),
        "successful": 0,
        "failed": 0,
        "failed_docs": []
    }
    
    logger.info(f"🚀 Starting batch suggestion generation for {len(doc_ids)} documents (lightweight: {use_lightweight})")
    
    for i, doc_id in enumerate(doc_ids, 1):
        logger.info(f"Processing document {i}/{len(doc_ids)}: {doc_id}")
        
        success = generate_suggestions_for_doc(
            doc_id=doc_id,
            num_questions=num_questions_per_doc,
            auto_init_collection=(i == 1),
            use_lightweight=use_lightweight
        )
        
        if success:
            results["successful"] += 1
        else:
            results["failed"] += 1
            results["failed_docs"].append(doc_id)
    
    logger.info(f"🎉 Batch generation complete: {results['successful']} successful, {results['failed']} failed")
    return results