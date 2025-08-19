"""
JSON utilities for QA generation components.

This module provides utilities for sanitizing and processing JSON data.
"""

import ast
import re
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

def sanitize_json_library(raw: str) -> str:
    """
    Sanitize a JSON string using Python's standard library.
    
    Args:
        raw: Raw JSON string to sanitize
        
    Returns:
        Sanitized JSON string
    """
    try:
        # Try to parse as JSON directly first
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from a larger text block
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON with more lenient pattern
        json_match = re.search(r'\[\s*\{.*\}\s*\]', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        # Try to parse as Python literal
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return json.dumps(parsed, ensure_ascii=False)
        except (SyntaxError, ValueError):
            pass
        
        # Last resort: try to fix common JSON errors
        fixed = raw
        # Replace single quotes with double quotes (but not within already double-quoted strings)
        # This is a simplified approach and may not work for all cases
        fixed = re.sub(r"(?<!\")('.*?')(?!\")", lambda m: m.group(0).replace("'", "\""), fixed)
        # Fix unquoted keys
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
        
        try:
            parsed = json.loads(fixed)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        # If all else fails, return the original string
        logger.warning("Failed to sanitize JSON using library methods")
        return raw
    except Exception as e:
        logger.error(f"Error sanitizing JSON: {e}")
        return raw

def extract_partial_qa(raw: str) -> list:
    """
    Extract partial QA pairs from a string.
    
    Args:
        raw: Raw string containing QA pairs
        
    Returns:
        List of QA pairs
    """
    try:
        # Try to parse as JSON directly
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Try to extract individual QA pairs
        qa_pairs = []
        # Pattern to match "question": "...", "answer": "..."
        pattern = r'"question"\s*:\s*"([^"]*)"[^"]*"answer"\s*:\s*"([^"]*)"'
        matches = re.finditer(pattern, raw, re.DOTALL)
        
        for match in matches:
            q, a = match.groups()
            qa_pairs.append({"question": q, "answer": a})
        
        if qa_pairs:
            return qa_pairs
        
        # If all else fails, return an empty list
        logger.warning("Failed to extract QA pairs")
        return []
    except Exception as e:
        logger.error(f"Error extracting QA pairs: {e}")
        return []

def sanitize_json_via_llm(raw: str, client, model: str, timeout: float = 15) -> str:
    """
    Sanitize a JSON string using an LLM.
    
    Args:
        raw: Raw JSON string to sanitize
        client: LLM client
        model: Model name
        timeout: Timeout in seconds
        
    Returns:
        Sanitized JSON string
    """
    try:
        system_prompt = (
            "You are a JSON repair assistant. Your task is to fix malformed JSON and return valid JSON.\n"
            "- Only output the fixed JSON, nothing else\n"
            "- Preserve all data in the original JSON\n"
            "- If the input is already valid JSON, return it as is\n"
            "- If the input is not JSON or cannot be repaired, return an empty array []\n"
        )
        
        user_prompt = f"Fix this JSON:\n\n{raw}"
        
        response = client.generate(
            model=model,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=4000,
            temperature=0.1,
            timeout=timeout
        )
        
        # Extract the JSON from the response
        content = response.get('content', '')
        
        # Try to parse the content as JSON
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # If the content is not valid JSON, try to extract JSON from it
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return json.dumps(parsed, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            
            logger.warning("LLM failed to produce valid JSON")
            return "[]"
    except Exception as e:
        logger.error(f"Error sanitizing JSON via LLM: {e}")
        return "[]"