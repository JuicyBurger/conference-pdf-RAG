from dotenv import load_dotenv
import os

load_dotenv()  # reads .env

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_HOST    = os.getenv("M416_3090")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL")
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP"))

# Pipeline settings
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 800))

# Qdrant collection name
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")