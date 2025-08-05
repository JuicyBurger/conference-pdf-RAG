#!/usr/bin/env python3
"""
Test script to verify vLLM connection works with the sinon-RAG project
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.models.client_factory import get_llm_client, get_default_model
from src.models.LLM import LLM

def test_vllm_connection():
    """Test connection to vLLM server"""
    
    # Load environment
    load_dotenv()
    
    print("🔧 Testing vLLM Connection")
    print("=" * 40)
    
    # Get client and model
    try:
        llm_client = get_llm_client()
        default_model = get_default_model()
        
        print(f"✅ Client created: {type(llm_client).__name__}")
        print(f"✅ Default model: {default_model}")
        print(f"✅ Host: {llm_client.host}")
        print()
        
        # Test simple generation
        print("🚀 Testing text generation...")
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is the capital of France? Answer in one sentence."
        
        response = LLM(
            client=llm_client,
            model=default_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw=True  # Get raw text response
        )
        
        print(f"✅ Response: {response}")
        print()
        print("🎉 vLLM connection test successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("🔍 Troubleshooting:")
        print("1. Make sure vLLM server is running:")
        print("   CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --gpu-memory-utilization 0.90 --max-model-len 8192")
        print("2. Check your .env file has:")
        print("   LLM_PROVIDER=production")
        print("   PRODUCTION_HOST=http://localhost:8000") 
        print("   DEFAULT_MODEL=meta-llama/Llama-3.1-8B-Instruct")
        print("3. Test vLLM directly:")
        print("   curl http://localhost:8000/v1/models")

if __name__ == "__main__":
    test_vllm_connection()