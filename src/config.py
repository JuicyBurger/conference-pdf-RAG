from dotenv import load_dotenv
import os

load_dotenv()  # reads .env

# Qdrant settings
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "production"
PRODUCTION_HOST = os.getenv("PRODUCTION_HOST", "http://localhost:8000")
OLLAMA_HOST = os.getenv("M416_3090")  # Keep existing env var for backward compatibility

# Model settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b-instruct-fp16")  # Default for Ollama
if LLM_PROVIDER == "production":
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama-3.2-11b-vision")  # Default for Production

# Pipeline settings
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 800))

# Backward compatibility
OLLAMA_MODEL = DEFAULT_MODEL  # For existing code that uses OLLAMA_MODEL