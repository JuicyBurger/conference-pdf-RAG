from __future__ import annotations

from typing import List, Any, AsyncGenerator, Optional, Dict

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage, MessageRole, CompletionResponse, ChatResponse
from llama_index.core.base.llms.types import LLMMetadata
from src.models.embedder import model as st_model
from src.models.client_factory import get_llm_client, get_default_model
from src.config import LLM_PROVIDER, OLLAMA_HOST, DEFAULT_MODEL


class LlamaIndexJinaEmbedding(BaseEmbedding):
    """Adapter to use our local SentenceTransformer model as LlamaIndex embedding backend.

    This ensures GraphRAG (and other LlamaIndex components) do not attempt to use
    OpenAI embeddings by default.
    """

    def __init__(self):
        # Reuse the globally loaded sentence-transformers model from src.models.embedder
        self._model = st_model
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "jina_adapter"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        vec = self._model.encode(query, convert_to_numpy=False)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        vec = self._model.encode(text, convert_to_numpy=False)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
        # vecs can be numpy array or list
        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [list(v) for v in vecs]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of _get_text_embeddings."""
        return self._get_text_embeddings(texts)


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
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Error generating response: {str(e)}"))
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        """Stream complete a prompt using our local LLM."""
        # For now, use non-streaming version since our clients don't support streaming
        try:
            response = self.complete(prompt, **kwargs)
            yield response
        except Exception as e:
            yield CompletionResponse(text=f"Error generating response: {str(e)}")
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        """Stream chat with the model using our local LLM."""
        # For now, use non-streaming version since our clients don't support streaming
        try:
            response = self.chat(messages, **kwargs)
            yield response
        except Exception as e:
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Error generating response: {str(e)}"))
    
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
    # Prefer LlamaIndex built-in LLM wrappers to satisfy extractor expectations
    try:
        if LLM_PROVIDER.lower() == "ollama":
            from llama_index.llms.ollama import Ollama
            # Aggressive timeout settings for faster processing
            Settings.llm = Ollama(
                model=DEFAULT_MODEL, 
                base_url=OLLAMA_HOST, 
                request_timeout=360.0,
                temperature=0.1,
            )
        else:
            # Fallback to our adapter
            Settings.llm = LlamaIndexLocalLLM()
    except Exception:
        # Final fallback to adapter if built-in wrapper unavailable
        Settings.llm = LlamaIndexLocalLLM()
    
    # Set embedding model
    Settings.embed_model = LlamaIndexJinaEmbedding()

    # Enable detailed tracing to surface hidden extractor errors
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([debug_handler])
    
    print("✅ Configured LlamaIndex to use local Jina embeddings and local LLM")


