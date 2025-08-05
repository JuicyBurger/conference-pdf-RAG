#!/usr/bin/env python3
"""
Quick script to check if vLLM server is running and responsive
"""

import requests
import json

def check_vllm_server():
    """Check vLLM server status"""
    
    server_url = "http://localhost:8000"
    
    print("🔍 Checking vLLM Server Status")
    print("=" * 40)
    
    try:
        # Test 1: Check models endpoint
        print("1. Testing /v1/models endpoint...")
        models_response = requests.get(f"{server_url}/v1/models", timeout=5)
        if models_response.status_code == 200:
            models_data = models_response.json()
            print(f"   ✅ Models endpoint working")
            print(f"   📋 Available models: {[m['id'] for m in models_data.get('data', [])]}")
        else:
            print(f"   ❌ Models endpoint failed: {models_response.status_code}")
            return False
            
        # Test 2: Simple chat completion
        print("\n2. Testing chat completion...")
        chat_payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",  # Match your served model
            "messages": [
                {"role": "user", "content": "Hello! Just say 'Hi' back."}
            ],
            "max_tokens": 10,
            "temperature": 0.7
        }
        
        chat_response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=chat_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            content = chat_data["choices"][0]["message"]["content"]
            print(f"   ✅ Chat completion working")
            print(f"   💬 Response: {content}")
            print(f"   📊 Tokens used: {chat_data.get('usage', {})}")
        else:
            print(f"   ❌ Chat completion failed: {chat_response.status_code}")
            print(f"   📄 Response: {chat_response.text}")
            return False
            
        print(f"\n🎉 vLLM server is working correctly!")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to vLLM server at {server_url}")
        print("💡 Make sure the server is running:")
        print("   CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --gpu-memory-utilization 0.90 --max-model-len 8192")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

if __name__ == "__main__":
    check_vllm_server()