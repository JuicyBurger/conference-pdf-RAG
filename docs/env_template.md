# Environment Variables Template

This document provides a template for the required environment variables in the `.env` file.
Only essential credentials and connection settings are included in the environment variables.
Hyperparameters and internal settings are defined directly in the `config.py` file.

## Required Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Qdrant settings
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=docs
QDRANT_CHAT_DB=chat_history
QDRANT_ROOMS_DB=chat_rooms
QDRANT_QA_DB=qa_suggestions

# Neo4j settings
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

# LLM provider and model settings
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q8_0
GRAPH_EXTRACT_MODEL=qwen2.5:7b-instruct-q4_K_M
DEFAULT_MODEL=gpt-oss:20b
OLLAMA_MODEL=gpt-oss:20b

# Machine-specific endpoints
M416_3090=
M416_3090ti=
M416_4090=
PRODUCTION_OLLAMA=
```

## Notes

- Fill in the empty values with your specific credentials and connection details
- `PRODUCTION_OLLAMA` is used as the `OLLAMA_HOST` for the LLM client
- Only these variables need to be set in the environment; all other settings are defined in `config.py`
- For development, you can use the `.env` file
- For production deployment, set these environment variables in your deployment environment
