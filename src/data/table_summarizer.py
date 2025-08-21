#!/usr/bin/env python3
"""
Table Summarizer

Simple module to summarize table data using GPT-OSS model
while preserving crucial information like numbers, names, and percentages.
"""

import json
import logging
from typing import Dict, Any
from src.models.LLM import LLM
from src.models.client_factory import get_llm_client, get_default_model

logger = logging.getLogger(__name__)

# Initialize clients using the factory
llm_client = get_llm_client()
default_model = get_default_model()

def summarize_table(table_data: Dict[str, Any]) -> str:
    """
    Summarize table data while preserving crucial information.
    
    Args:
        table_data: Dictionary containing table information (columns, records, notes)
    
    Returns:
        String containing the table summary
    """
    try:
        if not table_data.get("records"):
            return "表格無數據記錄"
        
        columns = table_data.get("columns", [])
        records = table_data.get("records", [])
        notes = table_data.get("notes", [])
        
        # Create prompt
        prompt = f"""請總結以下表格的關鍵信息，要求：
1. 保留所有重要的數字、百分比、金額等數值信息
2. 保留所有人名、職位、公司名稱等關鍵名稱
3. 保留所有日期、時間等時間信息
4. 突出顯示重要的趨勢、比較或異常數據
5. 總結要簡潔但完整，不遺漏關鍵信息
6. 如果有備註，請包含在總結中

表格欄位：{', '.join(columns)}
記錄數量：{len(records)} 筆

表格數據：
"""
        
        # Add table records (limit to first 10 for prompt length)
        for i, record in enumerate(records[:10]):
            prompt += f"記錄 {i+1}: {json.dumps(record, ensure_ascii=False)}\n"
        
        if len(records) > 10:
            prompt += f"... (還有 {len(records) - 10} 筆記錄)\n"
        
        if notes:
            # Handle both string and dict notes
            note_texts = []
            for note in notes:
                if isinstance(note, dict):
                    note_texts.append(note.get("text", str(note)))
                else:
                    note_texts.append(str(note))
            prompt += f"\n備註：{' '.join(note_texts)}\n"
        
        prompt += "\n請提供一個簡潔但完整的總結："
        
        # Generate summary using LLM
        response = LLM(
            llm_client,
            default_model,
            "",  # No system prompt needed for simple summarization
            prompt,
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                # IMPORTANT for Ollama-backed models:
                # prefer num_predict to cap generated tokens
                "num_predict": 256,     # ~ fast, ~ few seconds
                "num_ctx": 1024,        # enough for our tiny prompt
                "repeat_penalty": 1.0,
                # Optional: stop after a blank line to avoid drift
                # "stop": ["\n\n\n"]
                # If your wrapper only supports max_length, keep it small:
                "max_length": 512
            },
            timeout=60,
            raw=True
        )
        
        if not response:
            logger.warning("LLM 未返回總結，使用備用總結")
            return _create_fallback_summary(table_data)
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"表格總結生成失敗: {e}")
        return _create_fallback_summary(table_data)

def _create_fallback_summary(table_data: Dict[str, Any]) -> str:
    """Create a fallback summary when LLM fails."""
    columns = table_data.get("columns", [])
    records = table_data.get("records", [])
    notes = table_data.get("notes", [])
    
    summary = f"表格包含 {len(columns)} 個欄位：{', '.join(columns)}。"
    summary += f"共有 {len(records)} 筆記錄。"
    
    if notes:
        # Handle both string and dict notes
        note_texts = []
        for note in notes:
            if isinstance(note, dict):
                note_texts.append(note.get("text", str(note)))
            else:
                note_texts.append(str(note))
        summary += f" 備註：{' '.join(note_texts)}"
    
    # Add some key information from first few records
    if records:
        summary += " 前幾筆記錄包含："
        for i, record in enumerate(records[:3]):
            key_values = []
            for col, val in record.items():
                if val and str(val).strip():
                    key_values.append(f"{col}: {val}")
            if key_values:
                summary += f" 記錄{i+1}({', '.join(key_values[:3])})"
    
    return summary
