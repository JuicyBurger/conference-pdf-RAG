#!/usr/bin/env python3
"""
Unified client factory for LLM providers
Supports Ollama (OpenAI interface), Legacy Ollama, and Production servers
"""

import os
from typing import Union
from ollama import Client  # For backward compatibility
from .LLM import OllamaOpenAIClient, ProductionAPIClient
from ..config import OLLAMA_HOST, DEFAULT_MODEL

def get_llm_client() -> Union[OllamaOpenAIClient, Client, ProductionAPIClient]:
    """
    Get the appropriate LLM client based on configuration
    
    Returns:
        Client instance (OllamaOpenAIClient, Legacy Client, or ProductionAPIClient)
    """
    # Use OLLAMA_HOST as the production host
    if OLLAMA_HOST:
        # print(f"🔧 Using Ollama OpenAI LLM client: {OLLAMA_HOST}")
        return OllamaOpenAIClient(host=OLLAMA_HOST)
    else:
        raise ValueError("OLLAMA_HOST not configured. Please set PRODUCTION_OLLAMA environment variable.")

def get_default_model() -> str:
    """
    Get the default model name based on provider
    
    Returns:
        Model name string
    """
    return DEFAULT_MODEL

 