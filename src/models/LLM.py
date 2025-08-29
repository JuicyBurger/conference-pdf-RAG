import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from openai import OpenAI
from ollama import Client  # For backward compatibility
import pandas as pd
from pydantic import BaseModel
import json
import requests
from typing import Union, Dict, Any, Optional, List

import os

try:
    # Load default decoding options
    from src.config import RESPONSE_MAX_TOKENS, RESPONSE_TEMPERATURE, DEFAULT_REASONING_EFFORT
except Exception:
    # Fallback values if config import fails
    RESPONSE_MAX_TOKENS = 512
    RESPONSE_TEMPERATURE = 0.0
    DEFAULT_REASONING_EFFORT = "low"

# JSON Schema
# You can add more key in this part, to fit different task.
class Result(BaseModel):
  response: str

class OllamaOpenAIClient:
    """Client for Ollama using OpenAI interface"""
    
    def __init__(self, host: str = "http://192.168.100.32:11434"):
        self.host = host.rstrip("/")
        self.client_type = "ollama_openai"
        self.client = OpenAI(
            base_url=f"{self.host}/v1",
            api_key="local"
        )
    
    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None, format: Dict = None) -> Dict:
        """
        Send a chat completion request to Ollama using OpenAI interface
        
        Args:
            model: Model name to use
            messages: List of message dicts with role and content
            options: Dictionary of options like temperature, max_tokens, reasoning_effort
            format: Format specification (ignored for Ollama)
            
        Returns:
            Dict with response content
        """
        # Default options
        if options is None:
            options = {}
            
        # Convert options to API parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": options.get("temperature", RESPONSE_TEMPERATURE),
            "max_tokens": options.get("max_tokens", RESPONSE_MAX_TOKENS),
            "top_p": options.get("top_p", 1.0),
            "stream": False,
        }
        
        # Handle reasoning_effort for GPT-OSS
        reasoning_effort = options.get("reasoning_effort") or options.get("effort") or DEFAULT_REASONING_EFFORT
        if reasoning_effort:
            # Send both shapes to maximize compatibility
            api_params["extra_body"] = {
                "reasoning_effort": reasoning_effort,
                "reasoning": {"effort": reasoning_effort},
            }
        
        # Make API request
        response = self.client.chat.completions.create(**api_params)
        
        # Extract content, falling back to reasoning text if present
        try:
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None) or ""
            reasoning_text = None
            if hasattr(message, "reasoning") and getattr(message, "reasoning"):
                reasoning_text = getattr(message, "reasoning")
            elif hasattr(choice, "reasoning") and getattr(choice, "reasoning"):
                reasoning_text = getattr(choice, "reasoning")
            else:
                try:
                    dumped = response.model_dump()
                    reasoning_text = (
                        dumped.get("choices", [{}])[0].get("message", {}).get("reasoning")
                        or dumped.get("choices", [{}])[0].get("reasoning")
                    )
                except Exception:
                    reasoning_text = None
            if not content and reasoning_text:
                content = reasoning_text
        except Exception:
            # As a last resort, stringify the whole response
            try:
                content = str(response)
            except Exception:
                content = ""
        
        return {"message": {"content": content}}

class ProductionAPIClient:
    """Client for the production API server"""
    
    def __init__(self, host: str = "http://localhost:8000"):
        self.host = host.rstrip("/")
        self.client_type = "production"
    
    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None, format: Dict = None) -> Dict:
        """
        Send a chat completion request to vLLM's OpenAI-compatible API
        
        Args:
            model: Model name to use (will be mapped to the served model)
            messages: List of message dicts with role and content
            options: Dictionary of options like temperature, max_tokens
            format: Format specification (ignored for vLLM API)
            
        Returns:
            Dict with response content
        """
        # Default options
        if options is None:
            options = {}
            
        # Convert options to API parameters
        api_params = {
            "model": model,  # vLLM will use whatever model is loaded
            "messages": messages,
            "temperature": options.get("temperature", RESPONSE_TEMPERATURE),
            "max_tokens": options.get("max_tokens", RESPONSE_MAX_TOKENS),
            "top_p": options.get("top_p", 1.0),
            "stream": False
        }
        
        # Make API request to vLLM's OpenAI-compatible chat endpoint
        response = requests.post(
            f"{self.host}/v1/chat/completions",
            json=api_params,
            headers={"Content-Type": "application/json"},
            timeout=60  # Add timeout for long generations
        )
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"vLLM API request failed with status {response.status_code}: {response.text}")
        
        # Parse response
        api_response = response.json()
        
        # Format response to match ollama client format
        return {
            "message": {
                "content": api_response["choices"][0]["message"]["content"]
            }
        }

# Get response from LLM
def LLM(
    client: Union[OllamaOpenAIClient, ProductionAPIClient],
    model: str,
    system_prompt: str,
    user_prompt: str,
    img: str = None,
    options: dict = None,
    timeout: float = None,
    raw: bool = False
) -> str:
    opts = {} if options is None else options.copy()
    # Default deterministic, fast, and bounded
    if not opts and not raw:
        opts = {"seed": 42, "temperature": RESPONSE_TEMPERATURE, "max_tokens": RESPONSE_MAX_TOKENS, "reasoning_effort": DEFAULT_REASONING_EFFORT}
    # If raw=True and no opts, default to concise creative
    if raw and not opts:
        opts = {"temperature": max(0.2, RESPONSE_TEMPERATURE), "max_tokens": RESPONSE_MAX_TOKENS, "reasoning_effort": DEFAULT_REASONING_EFFORT}

    # Normalize options to ensure expected keys are present
    # Map num_predict -> max_tokens if provided by caller
    if "max_tokens" not in opts and "num_predict" in opts:
        try:
            opts["max_tokens"] = int(opts.get("num_predict"))
        except Exception:
            pass
    # Ensure temperature is set
    if "temperature" not in opts:
        opts["temperature"] = RESPONSE_TEMPERATURE
    # Ensure reasoning_effort is propagated for GPT-OSS
    if "reasoning_effort" not in opts and "effort" not in opts:
        opts["reasoning_effort"] = DEFAULT_REASONING_EFFORT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    if img:
        messages[1]['images'] = [img]

    client_type = getattr(client, "client_type", "ollama")
    print(f">>> LLM call starting (model={model}, client={client_type})")
    print(f"    system_prompt: {len(system_prompt)} chars")
    print(f"      user_prompt: {len(user_prompt)} chars")
    print(f"        options: {opts}")
    if timeout:
        print(f"    timeout: {timeout}s")

    start = time.time()
    try:
        if timeout and client_type == "ollama":
            # run in a thread so we can time out (only for ollama client)
            with ThreadPoolExecutor(max_workers=1) as exec:
                future = exec.submit(
                    client.chat,
                    model=model,
                    messages=messages,
                    options=opts,
                    format=Result.model_json_schema() if not raw else None
                )
                response = future.result(timeout=timeout)
        else:
            if client_type == "production":
                # For production API, format is handled differently
                response = client.chat(
                    model=model,
                    messages=messages,
                    options=opts
                )
            else:
                # For ollama client
                response = client.chat(
                    model=model,
                    messages=messages,
                    options=opts,
                    format=Result.model_json_schema() if not raw else None
                )
    except FuturesTimeout:
        raise TimeoutError(f"LLM call timed out after {timeout} seconds")
    except Exception as e:
        dur = time.time() - start
        print(f"!!! LLM call ERROR after {dur:.1f}s: {e}")
        traceback.print_exc()
        raise
    else:
        dur = time.time() - start
        print(f"<<< LLM call completed in {dur:.1f}s")

    # Extract the assistant's content
    if raw:
        # Return raw text content
        # print(response["message"]["content"])
        return response["message"]["content"]
    else:
        # Return parsed JSON
        # print(response["message"]["content"])
        return extract_json(response["message"]["content"])

# Extract JSON from response
def extract_json(response: str):
    try:
        # First try to parse as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If it's not valid JSON, check if it looks like a plain text response
        # (especially Chinese text responses)
        if any('\u4e00' <= char <= '\u9fff' for char in response):
            # This contains Chinese characters, treat as plain text
            return json.dumps({"response": response})
        else:
            # For other non-JSON responses, return as error
            return json.dumps({"Error": f"'{response}' is not a valid JSON"})

def get_client(client_type: str = "ollama_openai", host: str = None) -> Union[OllamaOpenAIClient, ProductionAPIClient]:
    """
    Get the appropriate client based on type
    
    Args:
        client_type: "ollama_openai" (default), "ollama", or "production"
        host: Host URL or environment variable name
        
    Returns:
        Client instance
    """
    if client_type.lower() == "ollama_openai":
        # For Ollama using OpenAI interface
        host_url = host or "http://192.168.100.32:11434"
        return OllamaOpenAIClient(host=host_url)
    elif client_type.lower() == "ollama":
        # Legacy Ollama client (for backward compatibility)
        if host and host.startswith("M416_"):
            host_url = os.getenv(host)
            if not host_url:
                raise ValueError(f"Environment variable {host} not found")
            return Client(host=host_url)
        else:
            return Client(host=host)
    elif client_type.lower() == "production":
        # For production client
        host_url = host or "http://localhost:8000"
        return ProductionAPIClient(host=host_url)
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def main(client_type: str = "ollama_openai", host: str = None, model: str = "gpt-oss:20b"):
    data = pd.read_csv("demo.csv")
    num_files = len(data)
    result = []
    start_time = time.time()
    
    # Get appropriate client
    client = get_client(client_type, host)

    for i in range(num_files):
        # Change system prompt and user prompt here
        system_prompt = "You are a helpful assistant, name Llama."
        user_prompt = data['question'].iloc[i]
        # You can comment out the print statement once you've confirmed the code works as expected.
        print(user_prompt)
        
        # Generate LLM response
        response = LLM(client, model, system_prompt, user_prompt)
        
        # Extract JSON from response
        json_response = extract_json(response)

        # Append result and mark as successful
        result.append(json_response)
        
        # You can comment out the print statement once you've confirmed the code works as expected.
        print(json_response)

    df_result = pd.DataFrame(result)
    df_result.to_json('result.json', orient='records', indent=4)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    # Change client type, host and model here
    # Options:
    # 1. Ollama OpenAI: client_type="ollama_openai", host="http://192.168.100.32:11434" (default)
    # 2. Legacy Ollama: client_type="ollama", host="M416_3090" (env var) or direct URL
    # 3. Production: client_type="production", host="http://localhost:8000"
    client_type = "ollama_openai"  # Default to OpenAI interface
    host = "http://192.168.100.32:11434"  # Remote server
    model = "gpt-oss:20b"  # Default model
    
    main(client_type, host, model)
