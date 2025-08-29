# Configuration Structure

This document explains the organization of the configuration system in the document-analyzer project.

## Overview

The configuration system is designed to be:

1. **Simple**: All configuration values are defined in a single file (`src/config.py`)
2. **Organized**: Settings are grouped into logical sections with clear comments
3. **Deployment-friendly**: Only essential credentials and connection settings are loaded from environment variables
4. **Maintainable**: Hyperparameters and internal settings are defined directly in the code

## Configuration Philosophy

The configuration follows these principles:

1. **Environment variables** are used only for:
   - Credentials (API keys, passwords)
   - Connection settings (URLs, hostnames)
   - Database names
   - Model names and endpoints

2. **Direct code definitions** are used for:
   - Hyperparameters
   - Processing settings
   - Feature flags
   - Default values
   - Internal constants

This approach makes deployment simpler by minimizing the number of environment variables that need to be set.

## Configuration Sections

The configuration is organized into the following sections:

### Credentials and Connection Settings (from environment variables)

- **Qdrant settings**: Vector database credentials and collection names
- **Neo4j settings**: Graph database credentials and database name
- **LLM provider and model settings**: Model names and endpoints
- **Machine-specific endpoints**: URLs for specific deployment machines

### Hyperparameters and Internal Settings (defined directly in code)

- **Document chunking parameters**: Chunk sizes and overlap settings
- **File paths and directories**: Paths for data storage
- **RAG settings**: Retrieval modes and parameters
- **LLM generation parameters**: Token limits, temperature, etc.
- **Knowledge graph extraction**: Entity and relation types, extraction settings
- **Table extraction and processing**: HTML chunking and file retention settings
- **External API endpoints**: URLs for OCR and other services

## Usage

To use configuration values in your code, simply import them directly from the config module:

```python
from src.config import QDRANT_URL, QDRANT_API_KEY, DEFAULT_MODEL, CHUNK_MAX_CHARS

# Use the values
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = LLMClient(model=DEFAULT_MODEL)
chunker = DocumentChunker(max_chars=CHUNK_MAX_CHARS)
```

## Environment Variables

Only essential credentials and connection settings are loaded from environment variables. See the [Environment Variables Template](env_template.md) for the complete list of required environment variables.

## Best Practices

1. **Keep environment variables minimal**: Only use environment variables for credentials and connection settings
2. **Define hyperparameters directly**: Define hyperparameters and internal settings directly in the code
3. **Group related settings**: Keep related settings together in the same section
4. **Add clear comments**: Document the purpose of each setting
5. **Use descriptive names**: Choose clear and descriptive names for configuration values