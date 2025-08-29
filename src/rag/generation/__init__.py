"""
Unified generation module for RAG systems.

This module provides:
- Answer generation from retrieved context (ResponseGenerator)
- QA pair generation for training data (QAProcessor)
- Question suggestions for frontend UI (SuggestionsProcessor)
- Unified question generation core (QuestionGenerator)
"""

# Core generation components
from .response_generator import ResponseGenerator, generate_answer
from .question_generator import QuestionGenerator
from .qa_processor import QAProcessor, generate_qa_pairs_for_doc
from .suggestions_processor import SuggestionsProcessor, generate_suggestions_for_doc, batch_generate_suggestions

# Catalog functions for suggestions management
from .catalog import (
    init_suggestion_collection,
    store_questions,
    retrieve_suggestions,
    retrieve_random_suggestions,
    get_all_doc_ids_with_suggestions,
    update_popularity_score,
    store_question_suggestions  # legacy compatibility
)

# Shared models
from .models import Question, QAPair, QuestionSuggestion, GenerationRequest, GenerationResult

__all__ = [
    # Answer generation (existing)
    'ResponseGenerator',
    'generate_answer',
    
    # Unified generation system (new)
    'QuestionGenerator',
    'QAProcessor', 
    'SuggestionsProcessor',
    
    # Legacy compatibility functions
    'generate_qa_pairs_for_doc',
    'generate_suggestions_for_doc',
    'batch_generate_suggestions',
    
    # Catalog functions
    'init_suggestion_collection',
    'store_questions',
    'retrieve_suggestions',
    'retrieve_random_suggestions',
    'get_all_doc_ids_with_suggestions',
    'update_popularity_score',
    'store_question_suggestions',
    
    # Data models
    'Question',
    'QAPair', 
    'QuestionSuggestion',
    'GenerationRequest',
    'GenerationResult'
]
