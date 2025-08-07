# src/rag/suggestions/__init__.py

from .catalog import (
    QuestionSuggestion,
    init_suggestion_collection,
    store_questions,
    retrieve_suggestions,
    retrieve_random_suggestions,
    get_all_doc_ids_with_suggestions,
    update_popularity_score
)
from .generator import (
    generate_suggestions_for_doc,
    batch_generate_suggestions
)

__all__ = [
    "QuestionSuggestion",
    "init_suggestion_collection", 
    "store_questions",
    "retrieve_suggestions",
    "retrieve_random_suggestions",
    "get_all_doc_ids_with_suggestions",
    "update_popularity_score",
    "generate_suggestions_for_doc",
    "batch_generate_suggestions"
]