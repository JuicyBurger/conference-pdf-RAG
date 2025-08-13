from dotenv import load_dotenv
import os

load_dotenv()  # reads .env

# Qdrant settings
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_QA_DB = os.getenv("QDRANT_QA_DB", "qa_suggestions")

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "production"
PRODUCTION_HOST = os.getenv("PRODUCTION_HOST", "http://localhost:8000")
OLLAMA_HOST = os.getenv("M416_3090")  # Keep existing env var for backward compatibility

# Model settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b-instruct-fp16")  # Default for Ollama
if LLM_PROVIDER == "production":
    # For vLLM, use the exact model name that vLLM serves (with forward slash and capital B)
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")  # Default for vLLM Production

# Pipeline settings
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 800))

# Backward compatibility
OLLAMA_MODEL = DEFAULT_MODEL  # For existing code that uses OLLAMA_MODEL

# Graph / Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI")
# Support both NEO4J_USERNAME and legacy NEO4J_USER
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# Database name (Neo4j 4.x+ multi-DB). Prefer NEO4J_DATABASE; default 'neo4j'
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or "neo4j"
TRAINING_CORPUS_ID = os.getenv("TRAINING_CORPUS_ID", "default")

# RAG routing defaults
RAG_DEFAULT_MODE = os.getenv("RAG_DEFAULT_MODE", "vector")  # vector | graph | hybrid

# Graph traversal / retrieval knobs
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", 2))
GRAPH_EXPANSION_TOP_K = int(os.getenv("GRAPH_EXPANSION_TOP_K", 12))