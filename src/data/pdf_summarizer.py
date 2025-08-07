#!/usr/bin/env python3
"""
PDF Summarization Module
Extracts text from PDFs and generates concise summaries using LLM
"""

import logging
import os
from typing import Dict, List, Any, Optional
from .pdf_ingestor import extract_text_pages
from ..models.LLM import LLM
from ..models.client_factory import get_llm_client, get_default_model

logger = logging.getLogger(__name__)

def summarize_pdf_content(
    pdf_path: str,
    max_pages: int = 50,
    summary_length: str = "medium",
    language: str = "zh-TW"
) -> Dict[str, Any]:
    """
    Extract text from PDF and generate a comprehensive summary using chunked approach.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (to handle large PDFs)
        summary_length: "short", "medium", or "long" summary
        language: Language for the summary (default: Traditional Chinese)
        
    Returns:
        Dict containing:
        - summary: The generated summary text
        - page_count: Number of pages processed
        - total_chars: Total characters in original text
        - summary_chars: Characters in summary
        - chunks_processed: Number of chunks processed
    """
    try:
        logger.info(f"📄 Starting PDF summarization for: {pdf_path}")
        
        # Extract text from PDF using existing function
        pages_data = extract_text_pages(pdf_path)
        total_pages = len(pages_data)
        
        if not pages_data:
            return {
                "summary": "",
                "page_count": 0,
                "total_chars": 0,
                "summary_chars": 0,
                "error": "No text content found in PDF"
            }
        
        # Limit pages for performance
        if total_pages > max_pages:
            logger.warning(f"⚠️ PDF has {total_pages} pages, limiting to {max_pages} for summarization")
            pages_data = pages_data[:max_pages]
        
        # Combine all text
        full_text = ""
        for page_data in pages_data:
            page_text = page_data.get("text", "").strip()
            if page_text:
                full_text += f"\n\n[第{page_data['page']}頁]\n{page_text}"
        
        total_chars = len(full_text)
        logger.info(f"📊 Extracted {total_chars} characters from {len(pages_data)} pages")
        
        # 🚨 NEW: Character threshold validation
        # Based on timing data: 70 pages ≈ 24 chunks ≈ 3 minutes
        # Target: ~1-1.5 minutes max processing time
        # 24 chunks = ~48,000 chars (2000 chars per chunk)
        # For 1.5 minutes: ~30 chunks max = ~60,000 chars
        MAX_CHARS_THRESHOLD = 60000  # ~30 chunks = ~1.5 minutes processing time
        
        if total_chars > MAX_CHARS_THRESHOLD:
            estimated_chunks = total_chars // 1800  # 2000 - 200 overlap
            estimated_time_minutes = estimated_chunks * 0.075  # ~4.5 seconds per chunk
            
            error_msg = (
                f"PDF too large for chat processing. "
                f"Content: {total_chars:,} characters ({estimated_chunks} estimated chunks). "
                f"Estimated processing time: {estimated_time_minutes:.1f} minutes. "
                f"Please use the RAG ingestion pipeline for large documents."
            )
            
            logger.warning(f"🚨 {error_msg}")
            return {
                "summary": "",
                "page_count": len(pages_data),
                "total_chars": total_chars,
                "summary_chars": 0,
                "error": error_msg,
                "estimated_chunks": estimated_chunks,
                "estimated_time_minutes": estimated_time_minutes
            }
        
        if not full_text.strip():
            return {
                "summary": "",
                "page_count": len(pages_data),
                "total_chars": 0,
                "summary_chars": 0,
                "error": "No meaningful text content found"
            }
        
        # Check if we need chunked summarization (rough estimate: 1 token ≈ 4 characters for Chinese)
        estimated_tokens = total_chars / 4
        logger.info(f"📊 Content analysis: {total_chars} chars ≈ {estimated_tokens:.0f} estimated tokens")
        
        if estimated_tokens > 6000:  # Leave buffer for prompts
            logger.info(f"📊 Large content ({estimated_tokens:.0f} estimated tokens), using chunked summarization")
            try:
                summary = _generate_chunked_summary(full_text, summary_length, language)
                logger.info(f"✅ Chunked summarization completed successfully")
            except Exception as chunk_error:
                logger.error(f"❌ Chunked summarization failed: {chunk_error}")
                # Fallback to direct summarization with truncated text
                logger.info(f"🔄 Falling back to direct summarization with truncated text")
                truncated_text = full_text[:12000]  # ~3000 tokens
                summary = _generate_summary_with_llm(truncated_text, summary_length, language)
        else:
            logger.info(f"📊 Small content ({estimated_tokens:.0f} estimated tokens), using direct summarization")
            summary = _generate_summary_with_llm(full_text, summary_length, language)
        
        result = {
            "summary": summary,
            "page_count": len(pages_data),
            "total_chars": total_chars,
            "summary_chars": len(summary),
            "compression_ratio": len(summary) / total_chars if total_chars > 0 else 0
        }
        
        logger.info(f"✅ Generated summary: {len(summary)} chars (compression: {result['compression_ratio']:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"❌ PDF summarization failed: {e}")
        return {
            "summary": "",
            "page_count": 0,
            "total_chars": 0,
            "summary_chars": 0,
            "error": str(e)
        }


def _generate_summary_with_llm(
    text: str,
    summary_length: str = "medium",
    language: str = "zh-TW"
) -> str:
    """
    Generate summary using the existing LLM infrastructure.
    
    Args:
        text: Full text to summarize
        summary_length: "short", "medium", or "long"
        language: Target language for summary
        
    Returns:
        Generated summary text
    """
    try:
        # Define summary parameters based on length
        length_configs = {
            "short": {
                "target_words": "200-300字",
                "focus": "最重要的關鍵點",
                "structure": "簡潔摘要"
            },
            "medium": {
                "target_words": "500-800字", 
                "focus": "主要內容和重點分析",
                "structure": "結構化摘要，包含主要段落"
            },
            "long": {
                "target_words": "1000-1500字",
                "focus": "詳細內容和深入分析",
                "structure": "完整摘要，包含所有重要細節"
            }
        }
        
        config = length_configs.get(summary_length, length_configs["medium"])
        
        # Create system prompt for summarization
        system_prompt = f"""你是一位專業的文件分析師，專門進行文件摘要。請根據提供的文件內容生成一個高質量的摘要。

要求：
1. 摘要長度：{config['target_words']}
2. 語言：繁體中文
3. 重點：{config['focus']}
4. 結構：{config['structure']}
5. 保持客觀、準確，不添加文件中沒有的資訊
6. 如果是財務或商業文件，請特別關注數字、趨勢、重要決策
7. 使用清晰的段落結構，便於閱讀

格式要求：
- 使用適當的段落分隔
- 重要數字和日期要保留
- 關鍵術語要準確
- 避免冗餘和重複"""

        user_prompt = f"""請為以下文件內容生成摘要：

{text}

請生成一個{config['target_words']}的{config['structure']}："""

        # Get LLM client and generate summary
        llm_client = get_llm_client()
        default_model = get_default_model()
        
        logger.info(f"🤖 Generating {summary_length} summary using LLM...")
        logger.info(f"📊 Input text length: {len(text)} chars")
        logger.info(f"📊 System prompt length: {len(system_prompt)} chars")
        logger.info(f"📊 User prompt length: {len(user_prompt)} chars")
        
        # Use higher temperature for more natural summarization
        options = {
            "temperature": 0.3,
            "max_length": 2048 if summary_length == "long" else 1024
        }
        
        logger.info(f"🔧 LLM options: {options}")
        
        try:
            summary = LLM(
                llm_client,
                default_model,
                system_prompt,
                user_prompt,
                options=options,
                timeout=60.0,  # Longer timeout for summarization
                raw=True
            )
            
            logger.info(f"✅ LLM call completed successfully: {len(summary)} chars returned")
            
        except Exception as llm_error:
            logger.error(f"❌ LLM call failed: {llm_error}")
            logger.error(f"📊 Failed with text length: {len(text)} chars")
            raise llm_error
        
        # Clean up the summary
        summary = summary.strip()
        if not summary:
            raise ValueError("LLM returned empty summary")
            
        logger.info(f"✅ Summary generated successfully: {len(summary)} characters")
        return summary
        
    except Exception as e:
        logger.error(f"❌ LLM summarization failed: {e}")
        # Return a basic fallback summary
        return f"摘要生成失敗：{str(e)}"


def _generate_chunked_summary(
    text: str,
    summary_length: str = "medium",
    language: str = "zh-TW"
) -> str:
    """
    Generate summary for large texts using two-stage summarization.
    Stage 1: Summarize small chunks with fast settings
    Stage 2: Summarize half-chunks, then combine
    
    Args:
        text: Full text to summarize
        summary_length: "short", "medium", or "long"
        language: Target language for summary
        
    Returns:
        Generated summary text
    """
    try:
        logger.info(f"🔄 Starting two-stage chunked summarization for {len(text)} characters")
        
        # Calculate safe chunk size (leaving room for prompts)
        max_chunk_chars = 2000
        overlap_chars = 200  # Overlap to maintain context between chunks
        
        # Split text into overlapping chunks
        chunks = []
        start = 0
        chunk_count = 0
        
        while start < len(text):
            end = start + max_chunk_chars
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            chunk_count += 1
            
            logger.info(f"📄 Created chunk {chunk_count}: {start}-{end} chars ({len(chunk)} chars)")
            
            # Move start position by chunk size minus overlap
            start = start + max_chunk_chars - overlap_chars
            
            # Safety check to prevent infinite loop
            if start >= len(text) or start >= end:
                break
        
        logger.info(f"📊 Split into {len(chunks)} chunks for processing")
        
        # Stage 1: Summarize each chunk with fast settings
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"🔄 Stage 1: Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            try:
                # Use ultra-fast settings for chunk summarization
                chunk_summary = _generate_summary_with_llm_fast(chunk, language)
                
                if chunk_summary and chunk_summary.strip():
                    chunk_summaries.append(chunk_summary)
                    logger.info(f"✅ Chunk {i+1} summarized: {len(chunk_summary)} chars")
                else:
                    logger.warning(f"⚠️ Chunk {i+1} produced empty summary")
                    
            except Exception as chunk_error:
                logger.error(f"❌ Failed to summarize chunk {i+1}: {chunk_error}")
                continue
        
        if not chunk_summaries:
            raise ValueError("No valid summaries generated from chunks")
        
        # Stage 2: Split chunk summaries into two halves and summarize each
        logger.info(f"📊 Stage 2: Processing {len(chunk_summaries)} chunk summaries")
        
        mid_point = len(chunk_summaries) // 2
        first_half = chunk_summaries[:mid_point]
        second_half = chunk_summaries[mid_point:]
        
        logger.info(f"📊 Split into two halves: {len(first_half)} + {len(second_half)} summaries")
        
        # Summarize first half
        first_half_text = "\n\n".join(first_half)
        logger.info(f"🔄 Summarizing first half ({len(first_half_text)} chars)")
        first_summary = _generate_summary_with_llm_fast(first_half_text, language)
        
        # Summarize second half as continuation
        second_half_text = "\n\n".join(second_half)
        logger.info(f"🔄 Summarizing second half as continuation ({len(second_half_text)} chars)")
        second_summary = _generate_summary_with_llm_fast_continuation(second_half_text, first_summary, language)
        
        # Combine the two half-summaries
        combined_summary = f"{first_summary}\n\n{second_summary}"
        logger.info(f"📊 Combined two half-summaries: {len(combined_summary)} chars")
        
        logger.info(f"✅ Two-stage summarization complete: {len(combined_summary)} chars")
        return combined_summary
        
    except Exception as e:
        logger.error(f"❌ Two-stage summarization failed: {e}")
        # Fallback: return a basic summary of the first chunk
        try:
            first_chunk = text[:3000]
            return _generate_summary_with_llm(first_chunk, "short", language)
        except:
            return f"摘要生成失敗：{str(e)}"


def _generate_summary_with_llm_fast(
    text: str,
    language: str = "zh-TW"
) -> str:
    """
    Generate ultra-fast summary for chunks using minimal settings.
    
    Args:
        text: Text to summarize
        language: Target language for summary
        
    Returns:
        Generated summary text
    """
    try:
        # Ultra-fast system prompt (minimal)
        system_prompt = """你是文件摘要專家。請用繁體中文生成簡潔摘要，重點突出關鍵信息。"""
        
        user_prompt = f"""請為以下內容生成簡潔摘要：

{text}

摘要："""

        # Get LLM client
        llm_client = get_llm_client()
        default_model = get_default_model()
        
        logger.info(f"⚡ Fast summarization: {len(text)} chars")
        
        # Ultra-fast settings
        options = {
            "temperature": 0.1,  # Very low for consistency
            "max_length": 512,   # Shorter for speed
            "top_p": 0.9,        # Faster generation
            "top_k": 40          # Faster generation
        }
        
        summary = LLM(
            llm_client,
            default_model,
            system_prompt,
            user_prompt,
            options=options,
            timeout=30.0,  # Shorter timeout
            raw=True
        )
        
        logger.info(f"⚡ Fast summary completed: {len(summary)} chars")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"❌ Fast summarization failed: {e}")
        return f"摘要失敗：{str(e)}"


def _generate_summary_with_llm_fast_continuation(
    text: str,
    previous_summary: str,
    language: str = "zh-TW"
) -> str:
    """
    Generate ultra-fast summary continuation that flows naturally from the previous summary.
    
    Args:
        text: Text to summarize
        previous_summary: The first half summary to continue from
        language: Target language for summary
        
    Returns:
        Generated continuation summary text
    """
    try:
        # Continuation-focused system prompt
        system_prompt = """你是文件摘要專家。請生成簡潔的摘要延續，要與前文自然銜接，避免重複開頭語句。"""
        
        user_prompt = f"""以下是第一部分的摘要：

{previous_summary}

請為以下內容生成摘要延續，要與上文自然銜接：

{text}

摘要延續："""

        # Get LLM client
        llm_client = get_llm_client()
        default_model = get_default_model()
        
        logger.info(f"⚡ Fast continuation summarization: {len(text)} chars")
        
        # Ultra-fast settings
        options = {
            "temperature": 0.1,  # Very low for consistency
            "max_length": 512,   # Shorter for speed
            "top_p": 0.9,        # Faster generation
            "top_k": 40          # Faster generation
        }
        
        summary = LLM(
            llm_client,
            default_model,
            system_prompt,
            user_prompt,
            options=options,
            timeout=30.0,  # Shorter timeout
            raw=True
        )
        
        logger.info(f"⚡ Fast continuation completed: {len(summary)} chars")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"❌ Fast continuation summarization failed: {e}")
        return f"摘要延續失敗：{str(e)}"


def extract_pdf_text_for_chat(pdf_path: str, max_chars: int = 50000) -> Dict[str, Any]:
    """
    Extract and prepare PDF text for immediate chat context (without summarization).
    Useful for smaller PDFs or when full text is needed.
    
    Args:
        pdf_path: Path to the PDF file
        max_chars: Maximum characters to extract (truncate if larger)
        
    Returns:
        Dict containing extracted text and metadata
    """
    try:
        logger.info(f"📄 Extracting PDF text for chat: {pdf_path}")
        
        pages_data = extract_text_pages(pdf_path)
        
        if not pages_data:
            return {
                "text": "",
                "page_count": 0,
                "total_chars": 0,
                "truncated": False,
                "error": "No text content found"
            }
        
        # Build formatted text
        formatted_text = ""
        for page_data in pages_data:
            page_text = page_data.get("text", "").strip()
            if page_text:
                formatted_text += f"\n\n[第{page_data['page']}頁]\n{page_text}"
        
        total_chars = len(formatted_text)
        truncated = False
        
        # Truncate if too long
        if total_chars > max_chars:
            formatted_text = formatted_text[:max_chars] + "\n\n[文件內容因長度限制而截斷...]"
            truncated = True
            logger.warning(f"⚠️ PDF text truncated from {total_chars} to {max_chars} characters")
        
        return {
            "text": formatted_text.strip(),
            "page_count": len(pages_data),
            "total_chars": total_chars,
            "extracted_chars": len(formatted_text),
            "truncated": truncated
        }
        
    except Exception as e:
        logger.error(f"❌ PDF text extraction failed: {e}")
        return {
            "text": "",
            "page_count": 0,
            "total_chars": 0,
            "truncated": False,
            "error": str(e)
        }
