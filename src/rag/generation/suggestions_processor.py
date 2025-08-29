"""
Suggestions processing for frontend question display.

This module handles the conversion of questions into suggestions
and their storage in Qdrant for fast frontend retrieval.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from .models import Question, QuestionSuggestion, GenerationRequest, GenerationResult
from .question_generator import QuestionGenerator
from ..utils import handle_errors, GenerationError, DatabaseError, setup_logger

logger = setup_logger(__name__)


class SuggestionsProcessor:
    """
    Processor for generating and storing question suggestions.
    
    This class takes questions and converts them into suggestions
    for frontend display, storing them in Qdrant.
    """
    
    def __init__(self):
        """Initialize the suggestions processor."""
        self.question_generator = QuestionGenerator()
        logger.info("Initialized suggestions processor")
    
    def convert_questions_to_suggestions(self, 
                                       questions: List[Question],
                                       room_id: str = None) -> List[QuestionSuggestion]:
        """
        Convert Question objects to QuestionSuggestion objects.
        
        Args:
            questions: List of Question objects
            room_id: Optional room ID for scoping
            
        Returns:
            List of QuestionSuggestion objects
        """
        suggestions = []
        
        for question in questions:
            suggestion = QuestionSuggestion(
                question=question.text,
                doc_id=question.doc_id,
                page=question.page,
                room_id=room_id,
                popularity_score=0.0,  # Initial score
                created_at=datetime.now(),
                metadata={
                    "question_metadata": question.metadata,
                    "confidence": question.confidence
                }
            )
            suggestions.append(suggestion)
        
        logger.debug(f"Converted {len(questions)} questions to suggestions")
        return suggestions
    
    def store_suggestions_in_qdrant(self, 
                                  suggestions: List[QuestionSuggestion],
                                  collection_name: str = None) -> bool:
        """
        Store suggestions in Qdrant collection.
        
        Args:
            suggestions: List of QuestionSuggestion objects to store
            collection_name: Qdrant collection name (defaults to config value)
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            # Import here to avoid circular dependencies
            # Import catalog functions from unified generation catalog
            from .catalog import store_questions, init_suggestion_collection
            from ...config import QDRANT_QA_DB
            
            if collection_name is None:
                collection_name = QDRANT_QA_DB
            
            # Ensure collection exists
            try:
                init_suggestion_collection(collection_name)
            except Exception as e:
                logger.warning(f"Collection initialization failed: {e}")
                # Continue anyway in case collection already exists
            
            # Store suggestions
            store_questions(suggestions, collection_name)
            
            logger.info(f"Successfully stored {len(suggestions)} suggestions in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store suggestions in Qdrant: {e}")
            return False
    
    @handle_errors(error_class=GenerationError, reraise=False)
    def generate_suggestions(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate question suggestions for a document.
        
        Args:
            request: Generation request parameters
            
        Returns:
            Generation result with suggestions
        """
        try:
            logger.info(f"Generating {request.num_items} suggestions for doc_id: {request.doc_id}")
            
            # 1. Generate questions using the unified generator
            # Use lightweight mode for suggestions (faster)
            lightweight_request = GenerationRequest(
                doc_id=request.doc_id,
                num_items=request.num_items,
                timeout=request.timeout,
                context_top_k=min(request.context_top_k, 20),  # Limit context for speed
                chat_context=request.chat_context,
                room_id=request.room_id,
                use_lightweight=True,  # Always use lightweight for suggestions
                custom_prompts=request.custom_prompts
            )
            
            question_result = self.question_generator.generate_questions(lightweight_request)
            if not question_result.success or not question_result.items:
                return GenerationResult(
                    success=False,
                    items=[],
                    doc_id=request.doc_id,
                    error=f"Failed to generate questions: {question_result.error}"
                )
            
            questions = question_result.items
            logger.info(f"Generated {len(questions)} questions, converting to suggestions")
            
            # 2. Convert questions to suggestions
            suggestions = self.convert_questions_to_suggestions(questions, request.room_id)
            
            # 3. Store in Qdrant if requested (auto_init_collection equivalent)
            storage_success = True
            if hasattr(request, 'store_in_qdrant') and request.store_in_qdrant:
                storage_success = self.store_suggestions_in_qdrant(suggestions)
            
            if not storage_success:
                logger.warning("Suggestion storage failed, but generation succeeded")
            
            logger.info(f"Successfully generated {len(suggestions)} suggestions")
            
            return GenerationResult(
                success=True,
                items=suggestions,
                doc_id=request.doc_id,
                metadata={
                    "questions_generated": len(questions),
                    "suggestions_created": len(suggestions),
                    "stored_in_qdrant": storage_success,
                    "lightweight_mode": True
                }
            )
            
        except Exception as e:
            logger.error(f"Suggestions generation failed for {request.doc_id}: {e}")
            return GenerationResult(
                success=False,
                items=[],
                doc_id=request.doc_id,
                error=str(e)
            )
    
    @handle_errors(error_class=DatabaseError, reraise=False)
    def store_and_generate_suggestions(self, request: GenerationRequest) -> bool:
        """
        Generate suggestions and store them in Qdrant.
        
        Args:
            request: Generation request parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate suggestions
            result = self.generate_suggestions(request)
            
            if not result.success:
                logger.error(f"Failed to generate suggestions: {result.error}")
                return False
            
            # Store in Qdrant
            storage_success = self.store_suggestions_in_qdrant(result.items)
            
            if storage_success:
                logger.info(f"Successfully generated and stored {len(result.items)} suggestions")
                return True
            else:
                logger.error("Failed to store suggestions despite successful generation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate and store suggestions: {e}")
            return False


# Convenience functions for backward compatibility
def generate_suggestions_for_doc(doc_id: str,
                               num_questions: int = 10,
                               auto_init_collection: bool = True,
                               use_lightweight: bool = True,
                               chat_context: str = None) -> bool:
    """
    Legacy function for generating suggestions.
    
    Args:
        doc_id: Document ID to generate suggestions for
        num_questions: Number of questions to generate
        auto_init_collection: Whether to initialize collection (ignored, always true)
        use_lightweight: Whether to use lightweight mode
        chat_context: Optional chat context
        
    Returns:
        True if successful, False otherwise
    """
    request = GenerationRequest(
        doc_id=doc_id,
        num_items=num_questions,
        timeout=30.0 if use_lightweight else 60.0,
        context_top_k=10 if use_lightweight else 20,
        chat_context=chat_context,
        use_lightweight=use_lightweight
    )
    
    # Add flag to store in Qdrant
    request.store_in_qdrant = True
    
    processor = SuggestionsProcessor()
    return processor.store_and_generate_suggestions(request)


def batch_generate_suggestions(doc_ids: List[str],
                             num_questions_per_doc: int = 8,
                             use_lightweight: bool = True) -> Dict[str, Any]:
    """
    Legacy function for batch generating suggestions.
    
    Args:
        doc_ids: List of document IDs
        num_questions_per_doc: Number of questions per document
        use_lightweight: Whether to use lightweight mode
        
    Returns:
        Dictionary with batch results
    """
    processor = SuggestionsProcessor()
    results = {
        "total": len(doc_ids),
        "successful": 0,
        "failed": 0,
        "doc_results": {}
    }
    
    for i, doc_id in enumerate(doc_ids, 1):
        logger.info(f"Processing document {i}/{len(doc_ids)}: {doc_id}")
        
        request = GenerationRequest(
            doc_id=doc_id,
            num_items=num_questions_per_doc,
            timeout=30.0 if use_lightweight else 60.0,
            context_top_k=10 if use_lightweight else 20,
            use_lightweight=use_lightweight
        )
        request.store_in_qdrant = True
        
        success = processor.store_and_generate_suggestions(request)
        
        if success:
            results["successful"] += 1
            results["doc_results"][doc_id] = "success"
        else:
            results["failed"] += 1
            results["doc_results"][doc_id] = "failed"
    
    logger.info(f"Batch generation completed: {results['successful']}/{results['total']} successful")
    return results
