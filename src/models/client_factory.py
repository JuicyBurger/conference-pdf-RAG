#!/usr/bin/env python3
"""
Unified client factory for LLM providers
Supports both Ollama and Production servers
"""

import os
from typing import Union
from ollama import Client
from .LLM import ProductionAPIClient
from ..config import LLM_PROVIDER, PRODUCTION_HOST, OLLAMA_HOST, DEFAULT_MODEL

def get_llm_client() -> Union[Client, ProductionAPIClient]:
    """
    Get the appropriate LLM client based on configuration
    
    Returns:
        Client instance (Ollama or Production)
    """
    provider = LLM_PROVIDER.lower()
    
    if provider == "production":
        # print(f"🔧 Using Production LLM client: {PRODUCTION_HOST}")
        return ProductionAPIClient(host=PRODUCTION_HOST)
    elif provider == "ollama":
        # print(f"🔧 Using Ollama LLM client: {OLLAMA_HOST}")
        return Client(host=OLLAMA_HOST)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'ollama' or 'production'")

def get_default_model() -> str:
    """
    Get the default model name based on provider
    
    Returns:
        Model name string
    """
    return DEFAULT_MODEL

 