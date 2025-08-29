"""
Table summarization functionality.

This module handles the generation of human-readable summaries from
structured table data using LLMs while preserving crucial information.
"""

import json
import logging
from typing import Dict, Any, List
from ...models.LLM import LLM
from ...models.client_factory import get_llm_client, get_default_model
from ...config import LLM_CLEANUP_URL
import requests

logger = logging.getLogger(__name__)


class TableSummarizer:
    """
    Handles generation of summaries from structured table data.
    
    Uses LLM-based summarization to generate comprehensive summaries while
    preserving important numerical data, names, and trends.
    """
    
    def __init__(self):
        """Initialize the table summarizer."""
        # Initialize clients using the factory
        self.llm_client = get_llm_client()
        self.default_model = get_default_model()
        logger.info("Initialized table summarizer")
        self.cleanup_url = LLM_CLEANUP_URL or None
    
    def summarize_table(self, structured_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of structured table data.
        
        Args:
            structured_data: Structured table data dictionary
            
        Returns:
            Generated summary text
        """
        try:
            logger.debug("Generating table summary")
            
            # Use core summarization logic
            summary = self._summarize_table_impl(structured_data)
            
            if not summary or not summary.strip():
                logger.warning("Empty summary generated, creating fallback")
                summary = self._create_fallback_summary(structured_data)
            
            logger.debug(f"Generated summary ({len(summary)} characters)")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate table summary: {e}")
            return self._create_fallback_summary(structured_data)
    
    def _summarize_table_impl(self, table_data: Dict[str, Any]) -> str:
        """
        Core implementation for table summarization.
        
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
                # Handle both dict and list records
                if isinstance(record, dict):
                    prompt += f"記錄 {i+1}: {json.dumps(record, ensure_ascii=False)}\n"
                elif isinstance(record, list):
                    # Convert list to dict using column names
                    record_dict = {}
                    for j, val in enumerate(record):
                        col_name = columns[j] if j < len(columns) else f"欄位{j+1}"
                        record_dict[col_name] = val
                    prompt += f"記錄 {i+1}: {json.dumps(record_dict, ensure_ascii=False)}\n"
            
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
                self.llm_client,
                self.default_model,
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
                    "max_length": 512,
                    # Use low reasoning to reduce latency
                    "reasoning_effort": "low"
                },
                timeout=60,
                raw=True
            )
            
            if not response:
                logger.warning("LLM 未返回總結，使用備用總結")
                return self._create_fallback_summary(table_data)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"表格總結生成失敗: {e}")
            return self._create_fallback_summary(table_data)
    
    def _create_fallback_summary(self, structured_data: Dict[str, Any]) -> str:
        """
        Create a basic fallback summary when LLM summarization fails.
        
        Args:
            structured_data: Structured table data dictionary
            
        Returns:
            Basic summary text
        """
        try:
            table_id = structured_data.get('table_id', 'unknown')
            columns = structured_data.get('columns', [])
            records = structured_data.get('records', [])
            notes = structured_data.get('notes', [])
            
            summary_parts = [f"表格 {table_id}"]
            
            if columns:
                summary_parts.append(f"包含 {len(columns)} 欄位: {', '.join(columns[:5])}")
                if len(columns) > 5:
                    summary_parts[-1] += "等"
            
            if records:
                summary_parts.append(f"共 {len(records)} 筆記錄")
            
            if notes:
                summary_parts.append(f"附註 {len(notes)} 項")
            
            summary = "，".join(summary_parts) + "。"
            
            # Add some key information from first few records
            if records:
                summary += " 前幾筆記錄包含："
                for i, record in enumerate(records[:3]):
                    key_values = []
                    # Handle both dict and list records
                    if isinstance(record, dict):
                        for col, val in record.items():
                            if val and str(val).strip():
                                key_values.append(f"{col}: {val}")
                    elif isinstance(record, list):
                        # If record is a list, use column names if available
                        for j, val in enumerate(record):
                            if val and str(val).strip():
                                col_name = columns[j] if j < len(columns) else f"欄位{j+1}"
                                key_values.append(f"{col_name}: {val}")
                    
                    if key_values:
                        summary += f" 記錄{i+1}({', '.join(key_values[:3])})"
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create fallback summary: {e}")
            return f"表格資料摘要（生成失敗：{str(e)}）"
    
    def summarize_multiple_tables(self, tables_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate summaries for multiple tables.
        
        Args:
            tables_data: List of structured table data dictionaries
            
        Returns:
            Dictionary mapping table IDs to summary texts
        """
        summaries = {}
        
        for table_data in tables_data:
            table_id = table_data.get('table_id', 'unknown')
            try:
                summary = self.summarize_table(table_data)
                summaries[table_id] = summary
            except Exception as e:
                logger.error(f"Failed to summarize table {table_id}: {e}")
                summaries[table_id] = self._create_fallback_summary(table_data)
        
        return summaries

    def cleanup_llm(self) -> bool:
        """Optionally call server endpoint to unload/cleanup LLM."""
        if not self.cleanup_url:
            return False
        try:
            resp = requests.post(self.cleanup_url, timeout=15)
            if resp.status_code == 200:
                logger.info("✅ LLM cleanup successful")
                return True
            logger.warning(f"⚠️ LLM cleanup returned {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ LLM cleanup failed: {e}")
            return False


# Convenience function for backward compatibility
def summarize_table(table_data: Dict[str, Any]) -> str:
    """
    Convenience function to summarize table data.
    
    Args:
        table_data: Dictionary containing table information (columns, records, notes)
    
    Returns:
        String containing the table summary
    """
    summarizer = TableSummarizer()
    return summarizer.summarize_table(table_data)
