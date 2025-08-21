"""
Embedding service for vector representations.

This module provides embedding functionality using Ollama models
and includes LlamaIndex adapter classes for integration with LlamaIndex components.
"""

from __future__ import annotations

from typing import Union, List, Any, AsyncGenerator, Optional, Dict
import logging
import requests
import json
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Ollama configuration
from src.config import OLLAMA_HOST, EMBEDDING_MODEL

class OllamaEmbeddingModel:
    """Ollama-based embedding model using Qwen3-Embedding-4B."""
    
    def __init__(self, base_url: str = OLLAMA_HOST, model_name: str = EMBEDDING_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.embedding_url = f"{self.base_url}/api/embeddings"
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info(f"✅ Connected to Ollama server at {self.base_url}")
                logger.info(f"🔧 Using embedding model: {self.model_name}")
            else:
                logger.warning(f"⚠️ Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama server: {e}")
            raise
        
        # Cache the embedding dimension during initialization
        self._cached_dimension = None
        try:
            self._cached_dimension = self._get_dimension_fallback()
            logger.info(f"🔍 Cached embedding dimension: {self._cached_dimension}")
        except Exception as e:
            logger.warning(f"⚠️ Could not cache embedding dimension: {e}")
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        # Use cached dimension if available
        if self._cached_dimension is not None:
            return self._cached_dimension
        
        # Fallback: try to get dimension dynamically (this should rarely happen)
        try:
            # Test with a simple text to get the actual dimension
            test_embedding = self.encode("test")
            dimension = len(test_embedding)
            logger.info(f"🔍 Dynamically detected embedding dimension: {dimension}")
            # Cache it for future use
            self._cached_dimension = dimension
            return dimension
        except Exception as e:
            logger.warning(f"⚠️ Could not detect embedding dimension dynamically: {e}")
            # Fallback to known dimension for Qwen3-Embedding-4B
            fallback_dimension = 2560
            self._cached_dimension = fallback_dimension
            return fallback_dimension
    
    def _get_dimension_fallback(self) -> int:
        """Get embedding dimension without calling encode (to avoid circular dependency)."""
        # Use cached dimension if available
        if self._cached_dimension is not None:
            return self._cached_dimension
        
        # Known dimension for Qwen3-Embedding-4B
        return 2560
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s) using Ollama.
        
        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional arguments (ignored for Ollama)
            
        Returns:
            Single embedding list or list of embedding lists
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            # Handle empty or whitespace-only texts
            if not text or not text.strip():
                logger.warning(f"⚠️ Empty text detected, using zero vector")
                embeddings.append([0.0] * self._get_dimension_fallback())
                continue
                
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                
                response = requests.post(
                    self.embedding_url,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    logger.error(f"❌ Ollama embedding failed with status {response.status_code}: {response.text}")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * self._get_dimension_fallback())
                    
            except Exception as e:
                logger.error(f"❌ Error generating embedding: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self._get_dimension_fallback())
        
        # Return single embedding for single text, list for multiple texts
        if len(texts) == 1:
            return embeddings[0]
        else:
            return embeddings


# Global model instance
model = OllamaEmbeddingModel()


def embed(text: Union[str, List[str]]):
    """
    Generate embeddings for single text or batch of texts using Ollama.
    
    Args:
        text: Single text string or list of text strings
        
    Returns:
        Single embedding list or list of embedding lists
    """
    return model.encode(text)


# LlamaIndex integration components
try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM, ChatMessage, MessageRole, CompletionResponse, ChatResponse
    from llama_index.core.base.llms.types import LLMMetadata
    from src.models.client_factory import get_llm_client, get_default_model
    from src.config import LLM_PROVIDER, OLLAMA_HOST, DEFAULT_MODEL

    class LlamaIndexJinaEmbedding(BaseEmbedding):
        """Adapter to use our Ollama embedding model as LlamaIndex embedding backend.
        
        This ensures GraphRAG (and other LlamaIndex components) do not attempt to use
        OpenAI embeddings by default.
        """
        
        def __init__(self):
            # Initialize the base class first
            super().__init__()
            # Then set our model
            self._model = model
        
        @classmethod
        def class_name(cls) -> str:
            return "jina_adapter"
        
        def _get_query_embedding(self, query: str, **kwargs) -> List[float]:
            """Get embedding for a query."""
            vec = self._model.encode(query)
            # Ollama returns list of floats directly
            return list(vec)
        
        def _get_text_embedding(self, text: str, **kwargs) -> List[float]:
            """Get embedding for a single text."""
            vec = self._model.encode(text)
            # Ollama returns list of floats directly
            return list(vec)
        
        def _get_text_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
            """Get embeddings for multiple texts."""
            if not texts:
                return []
            
            try:
                # Debug: check if _model exists and has encode method
                if not hasattr(self, '_model'):
                    logger.error("self._model does not exist!")
                    raise AttributeError("self._model does not exist")
                
                if not hasattr(self._model, 'encode'):
                    logger.error(f"self._model has no encode method. Available methods: {dir(self._model)}")
                    raise AttributeError("self._model has no encode method")
                
                # Ollama handles batching internally
                vecs = self._model.encode(texts)
                # Ollama returns list of lists directly
                return [list(vec) for vec in vecs]
                
            except Exception as e:
                logger.error(f"Error in _get_text_embeddings: {e}")
                logger.error(f"self._model type: {type(self._model)}")
                logger.error(f"texts: {texts[:2]}...")  # Show first 2 texts
                raise
        
        def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
            """Get embeddings for multiple texts (public interface)."""
            return self._get_text_embeddings(texts, **kwargs)
        
        async def _aget_query_embedding(self, query: str, **kwargs) -> List[float]:
            """Async version of _get_query_embedding."""
            return self._get_query_embedding(query, **kwargs)
        
        async def _aget_text_embedding(self, text: str, **kwargs) -> List[float]:
            """Async version of _get_text_embedding."""
            return self._get_text_embedding(text, **kwargs)
        
        async def _aget_text_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
            """Async version of _get_text_embeddings."""
            try:
                return self._get_text_embeddings(texts, **kwargs)
            except AttributeError as e:
                # Debug the issue
                logger.error(f"AttributeError in _aget_text_embeddings: {e}")
                logger.error(f"self._model type: {type(self._model)}")
                logger.error(f"self._model attributes: {dir(self._model)}")
                raise
        
        async def aget_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
            """Async version of get_text_embedding_batch."""
            return await self._aget_text_embeddings(texts, **kwargs)


    class LlamaIndexLocalLLM(LLM):
        """Adapter to use our local LLM (Ollama/vLLM) as LlamaIndex LLM backend.
        
        This ensures GraphRAG (and other LlamaIndex components) do not attempt to use
        OpenAI LLM by default.
        """
        
        def __init__(self):
            super().__init__()
            # Use our existing client factory
            self._client = get_llm_client()
            self._model_name = get_default_model()
        
        @property
        def metadata(self) -> LLMMetadata:
            """Return structured LLM metadata expected by LlamaIndex components."""
            return LLMMetadata(
                model_name=self._model_name,
                is_chat_model=True,
                is_function_calling_model=False,
            )
        
        @property
        def is_chat_model(self) -> bool:
            return True
        
        @property
        def is_function_calling_model(self) -> bool:
            return False
        
        def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            """Complete a prompt using our local LLM."""
            try:
                # Use our existing LLM function from src.models.LLM
                from src.models.LLM import LLM as llm_function
                
                response = llm_function(
                    client=self._client,
                    model=self._model_name,
                    system_prompt="You are a helpful assistant.",
                    user_prompt=prompt,
                    options=kwargs,
                    raw=True  # Get raw text response
                )
                return CompletionResponse(text=response)
            except Exception as e:
                # Fallback to a simple response if LLM fails
                logger.error(f"LLM completion error: {e}")
                return CompletionResponse(text=f"Error generating response: {str(e)}")
        
        def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
            """Chat with the model using our local LLM."""
            try:
                # Convert LlamaIndex ChatMessage to our format
                system_prompt = ""
                user_prompt = ""
                
                for msg in messages:
                    if msg.role == MessageRole.SYSTEM:
                        system_prompt = msg.content
                    elif msg.role == MessageRole.USER:
                        user_prompt = msg.content
                    elif msg.role == MessageRole.ASSISTANT:
                        # For now, just append assistant messages to user prompt
                        user_prompt += f"\nAssistant: {msg.content}"
                
                # Use our existing LLM function
                from src.models.LLM import LLM as llm_function
                
                response = llm_function(
                    client=self._client,
                    model=self._model_name,
                    system_prompt=system_prompt or "You are a helpful assistant.",
                    user_prompt=user_prompt,
                    options=kwargs,
                    raw=True
                )
                return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=response))
            except Exception as e:
                # Fallback to a simple response if LLM fails
                logger.error(f"LLM chat error: {e}")
                return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Error generating response: {str(e)}"))
        
        def stream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
            """Stream complete a prompt using our local LLM."""
            # For now, use non-streaming version since our clients don't support streaming
            async def _stream():
                try:
                    response = self.complete(prompt, **kwargs)
                    yield response
                except Exception as e:
                    logger.error(f"LLM stream complete error: {e}")
                    yield CompletionResponse(text=f"Error generating response: {str(e)}")
            return _stream()
        
        def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
            """Stream chat with the model using our local LLM."""
            # For now, use non-streaming version since our clients don't support streaming
            async def _stream():
                try:
                    response = self.chat(messages, **kwargs)
                    yield response
                except Exception as e:
                    logger.error(f"LLM stream chat error: {e}")
                    yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Error generating response: {str(e)}"))
            return _stream()
        
        async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            """Async complete a prompt using our local LLM."""
            # For now, use sync version since our client might not support async
            return self.complete(prompt, **kwargs)
        
        async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
            """Async chat with the model using our local LLM."""
            # For now, use sync version since our client might not support async
            return self.chat(messages, **kwargs)
        
        async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
            """Async stream complete a prompt using our local LLM."""
            # For now, use sync version since our client might not support async
            async for response in self.stream_complete(prompt, **kwargs):
                yield response
        
        async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
            """Async stream chat with the model using our local LLM."""
            # For now, use sync version since our client might not support async
            async for response in self.stream_chat(messages, **kwargs):
                yield response


    def configure_llamaindex_for_local_models():
        """Configure LlamaIndex to use our local models instead of OpenAI."""
        from llama_index.core import Settings
        from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
        
        # Import configuration
        from src.config import LLM_PROVIDER, DEFAULT_MODEL, OLLAMA_HOST, GRAPH_EXTRACT_MODEL
        
        # Prefer LlamaIndex built-in LLM wrappers to satisfy extractor expectations
        try:
            if LLM_PROVIDER.lower() == "ollama":
                from llama_index.llms.ollama import Ollama
                # Default (answering) model for general usage
                default_llm = Ollama(
                    model=DEFAULT_MODEL,
                    base_url=OLLAMA_HOST,
                    request_timeout=360.0,
                    temperature=0.1,
                )
                Settings.llm = default_llm
                # Expose a helper to switch temporarily inside extraction code
                Settings._default_llm = default_llm
                # Prefer strict JSON mode for KG extraction; fall back to 'format: json'
                # Avoid json_mode/format to keep compatibility; enforce JSON in the prompt instead
                Settings._extract_llm = Ollama(
                    model=GRAPH_EXTRACT_MODEL,
                    base_url=OLLAMA_HOST,
                    request_timeout=360.0,
                    temperature=0.0,
                )
            else:
                # Fallback to our adapter
                Settings.llm = LlamaIndexLocalLLM()
                Settings._default_llm = Settings.llm
                Settings._extract_llm = Settings.llm
        except Exception:
            # Final fallback to adapter if built-in wrapper unavailable
            Settings.llm = LlamaIndexLocalLLM()
            Settings._default_llm = Settings.llm
            Settings._extract_llm = Settings.llm
        
        # Set embedding model
        Settings.embed_model = LlamaIndexJinaEmbedding()
        
        # Enable detailed tracing to surface hidden extractor errors
        debug_handler = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([debug_handler])
        
        logger.info("✅ Configured LlamaIndex to use local Jina embeddings and local LLM")

except ImportError:
    # LlamaIndex is not available
    logger.warning("LlamaIndex not available, skipping adapter classes")
    
    # Define stub functions to avoid errors
    def configure_llamaindex_for_local_models():
        """Stub function for when LlamaIndex is not available."""
        logger.warning("LlamaIndex not available, cannot configure models")
        
    class LlamaIndexJinaEmbedding:
        """Stub class for when LlamaIndex is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("LlamaIndex is not available")
            
    class LlamaIndexLocalLLM:
        """Stub class for when LlamaIndex is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("LlamaIndex is not available")