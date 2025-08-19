"""
QA pair generation functionality.

This module provides functions for generating QA pairs from documents.
"""

import re
import json
import uuid
import logging
import random
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from src.rag.retrieval.retrieval_service import retrieval_service
from src.models.reranker import rerank
from src.models.LLM import LLM
from src.models.client_factory import get_llm_client, get_default_model
from .json_utils import sanitize_json_library, sanitize_json_via_llm, extract_partial_qa
from ..utils import handle_errors, GenerationError, DatabaseError, setup_logger

# Configure logging
logger = setup_logger(__name__)

# Initialize clients using the factory
llm_client = get_llm_client()
default_model = get_default_model()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@handle_errors(error_class=DatabaseError, fallback_return=[])
def fetch_doc_chunks(doc_id: str, limit: int = 1000):
    """
    Scroll Qdrant for all chunks where payload.doc_id == doc_id.
    Attach full payloads so you can see page/text.
    
    Args:
        doc_id: Document ID to fetch chunks for
        limit: Maximum number of chunks to fetch
        
    Returns:
        List of document chunks
    """
    # Build the filter condition
    scroll_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    # Note: use scroll_filter, and ask for payload
    records, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=scroll_filter,
        limit=limit,
        with_payload=True
    )
    return records

@handle_errors(error_class=GenerationError, fallback_return=[])
def generate_qa_pairs_for_doc(
    doc_id: str,
    num_pairs: int = 5,
    timeout: float = 120.0,
    context_top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Enhanced QA generation with multiple evidence sources and exact text snippets.
    1) Fetch all chunks for this doc_id
    2) Retrieve + rerank broader context using multiple seed queries
    3) Build rich context with detailed source tracking
    4) Generate QA with enhanced source information
    
    Args:
        doc_id: Document ID to generate QA pairs for
        num_pairs: Number of QA pairs to generate
        timeout: Timeout for LLM generation in seconds
        context_top_k: Maximum number of context chunks to use
        
    Returns:
        List of QA pairs
    """
    
    print(f"Number of pairs that will be generated: {num_pairs}")
    
    # 1) Fetch every chunk for this PDF
    records = fetch_doc_chunks(doc_id)
    if not records:
        return []

    # 2) Enhanced retrieval with multiple perspectives and more diverse sources
    
    # Use diverse queries to gather broader context
    query_templates = [
        "財務表現與獲利能力分析",
        "營運策略與業務發展", 
        "風險管理與公司治理",
        "市場展望與成長動能",
        "公司治理與內部控制",
        "產品技術與研發創新",
        "市場競爭與產業地位",
        "財務風險與資本結構"
    ]
    
    all_hits = []
    sources_meta = []
    seen_chunks = set()
    
    logger.info(f"🔍 Starting retrieval for {len(query_templates)} query templates")
    
    for i, query in enumerate(query_templates):
        logger.info(f"📋 Query {i+1}/{len(query_templates)}: {query}")
        
        # Get more results per query to increase diversity
        hits = retrieval_service.retrieve(query, top_k=context_top_k//3, score_threshold=0.2)  # Lower threshold, more results
        logger.info(f"   Retrieved {len(hits)} initial hits")
        
        # Rerank and keep more top results
        hits = rerank(query, hits)[:8]  # Top 8 per query (increased from 5)
        logger.info(f"   After reranking: {len(hits)} hits")
        
        for h in hits:
            if h.id not in seen_chunks:
                seen_chunks.add(h.id)
                all_hits.append(h)
                
                # Store detailed source metadata
                sources_meta.append({
                    "doc_id": doc_id,
                    "page": h.payload.get("page", "unknown"),
                    "text": h.payload.get("text", ""),
                    "chunk_id": h.id
                })
                logger.info(f"   ✅ Added chunk from page {h.payload.get('page', 'unknown')}")
    
    logger.info(f"🎯 Total unique chunks collected: {len(all_hits)}")
    
    # 3) Build context from diverse sources
    logger.info(f"📚 Building context from {len(all_hits)} hits (limiting to {context_top_k})")
    
    context_blocks = []
    for i, h in enumerate(all_hits[:context_top_k]):  # Limit total context
        page = h.payload.get("page", "unknown")
        text = h.payload.get("text", "")
        # Truncate for prompt efficiency but keep full text in metadata
        context_blocks.append(f"[第{page}頁] {text}")
        logger.info(f"   📄 Context block {i+1}: Page {page}, length {len(text)} chars")
    
    context = "\n\n".join(context_blocks)
    logger.info(f"📝 Final context length: {len(context)} characters")
    
    # 4) Enhanced system prompt for factual, specific QA with multiple sources
    system_prompt = f"""
    你是專業的投資簡報助手，擅長從公司法說會紀錄與年報中挖掘具吸引力的投資亮點。
    請根據以下文件內容，生成**恰好 {num_pairs} 組**高品質的問答對。

    重要要求：
    1. **嚴格控制數量**：必須生成恰好 {num_pairs} 組問答對，不能多也不能少
    2. **答案必須包含具體事實和數據**：提到具體數字、日期、人名、制度名稱等
    3. **自然流暢的語言**：使用自然的繁體中文，避免過於正式的公文用語
    4. **詳細的實施方式**：當提到制度或措施時，要說明具體如何實施
    5. **答案長度不限**：可以寫得詳細一些，確保包含所有重要事實
    6. **多個來源支援**：每個答案必須基於2-4個不同的來源，確保答案的完整性和準確性
    7. **詳細的來源資訊**：source 必須包含具體的頁碼和文件資訊

    輸出格式要求：
    - 必須是有效的 JSON 陣列
    - 每個物件包含：question、answer、source
    - source 必須是陣列格式，包含2-4個不同的來源資訊
    - 確保所有字串都有正確的引號包圍
    - 最後一個物件後不能有逗號

    範例格式：
    [
    {{
        "question": "公司近三年的營收表現如何？",
        "answer": "根據財務資料，近三年營收分別為：2021年1,521,567萬元、2022年1,420,199萬元、2023年1,325,971萬元。營收呈現穩健成長趨勢，主要受益於產品結構優化及市場擴展策略。",
        "source": [
            {{
                "doc_id": "112年報 20240531",
                "page": 73,
                "text": "近三年營收分別為：2021年1,521,567萬元、2022年1,420,199萬元、2023年1,325,971萬元..."
            }},
            {{
                "doc_id": "112年報 20240531", 
                "page": 67,
                "text": "營收分析顯示穩健成長趨勢，主要受益於產品結構優化..."
            }},
            {{
                "doc_id": "112年報 20240531",
                "page": 75,
                "text": "市場擴展策略有效提升營收表現..."
            }}
        ]
    }},
    {{
        "question": "公司的內部稽核制度如何運作？",
        "answer": "我們建立了完整的內部稽核制度，包括：訂有詳細的內部稽核制度及各項管理辦法，規範對外商業活動、金錢往來、利益衝突迴避及機密資料管理。每年執行一次內部自評，評估期間為112年1月1日至112年12月31日。",
        "source": [
            {{
                "doc_id": "112年報 20240531",
                "page": 49,
                "text": "內部稽核制度包括訂定誠信經營政策及方案、建立良好公司治理及風險控管機制..."
            }},
            {{
                "doc_id": "112年報 20240531",
                "page": 21,
                "text": "內部自評：每年執行一次 評估期間：112 年 1 月 1 日至 112 年 12 月 31 日..."
            }}
        ]
    }}
    ]

    請確保輸出的是完整的、有效的 JSON 陣列，包含恰好 {num_pairs} 個物件。
    """
    user_prompt = f"""基於以下文件內容，生成恰好 {num_pairs} 組高品質的問答對。

**重要提醒**：
- 必須生成恰好 {num_pairs} 組問答對
- 答案必須包含文件中的具體數字、日期、人名、制度名稱等事實
- **每個答案必須基於2-4個不同的來源**，確保答案的完整性和準確性
- 確保 JSON 格式正確，所有字串都有引號包圍
- 使用自然的語言，避免過於正式的公文用語
- 答案可以寫得詳細一些，整合多個來源的資訊

文件內容：
{context}

請輸出完整的 JSON 陣列："""

    # 5) Call LLM with enhanced settings for longer, more detailed answers
    logger.info(f"🤖 Starting LLM generation with {len(context_blocks)} context blocks")
    logger.info(f"📊 Context summary: {len(context_blocks)} blocks, {len(context)} total chars")
    
    raw_response = LLM(
        llm_client, 
        default_model, 
        system_prompt, 
        user_prompt, 
        options={"temperature": 0.3, "max_length": 4096}, 
        timeout=timeout, 
        raw=True
    )
    
    logger.info(f"📝 Generated response length: {len(raw_response)} characters")
    print("LLM raw QA output:", raw_response)
    
    # 6) Parse JSON with robust sanitization
    logger.info("🔍 Parsing JSON response...")
    qa_list = []
    
    # Strategy 1: Try direct JSON parsing
    try:
        qa_list = json.loads(raw_response)
        logger.info(f"✅ Successfully parsed {len(qa_list)} QA pairs")
    except json.JSONDecodeError as e:
        logger.warning(f"❌ Direct JSON parsing failed: {e}")
        print(f"Raw response: {raw_response}")
        
        # Strategy 2: Try library-based sanitization
        try:
            cleaned_response = sanitize_json_library(raw_response)
            qa_list = json.loads(cleaned_response)
            logger.info("✅ Fixed JSON parsing issues using library sanitization")
        except Exception as e:
            logger.warning(f"❌ Library-based JSON fixing failed: {e}")
            
            # Strategy 3: Try LLM-based JSON repair (as last resort)
            try:
                logger.info("🔄 Attempting LLM-based JSON repair...")
                repaired_response = sanitize_json_via_llm(raw_response, llm_client, default_model, timeout=10)
                qa_list = json.loads(repaired_response)
                logger.info("✅ Fixed JSON using LLM repair")
            except Exception as e2:
                logger.warning(f"❌ LLM-based JSON repair failed: {e2}")
                
                # Strategy 4: Extract partial QA pairs
                try:
                    partial_qa = extract_partial_qa(raw_response)
                    if partial_qa:
                        qa_list = partial_qa
                        logger.info(f"✅ Extracted {len(partial_qa)} partial QA pairs")
                    else:
                        raise ValueError("Could not extract any valid QA pairs")
                except Exception as e3:
                    logger.error(f"❌ All JSON fixing methods failed: {e3}")
                    return []
    
    # Validate the parsed JSON structure
    if not isinstance(qa_list, list):
        logger.error("❌ Parsed JSON is not a list")
        return []
    
    # Validate each QA pair
    valid_qa_list = []
    for i, item in enumerate(qa_list):
        if not isinstance(item, dict):
            logger.warning(f"⚠️ QA pair {i+1} is not a dictionary, skipping")
            continue
            
        # Check required fields
        if "question" not in item or "answer" not in item:
            logger.warning(f"⚠️ QA pair {i+1} missing required fields, skipping")
            continue
            
        # Ensure fields are strings
        if not isinstance(item["question"], str) or not isinstance(item["answer"], str):
            logger.warning(f"⚠️ QA pair {i+1} has non-string question/answer, skipping")
            continue
            
        # Add ID if missing
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
            
        valid_qa_list.append(item)
    
    qa_list = valid_qa_list
    logger.info(f"✅ Validated {len(qa_list)} QA pairs")
    
    # 7) Enhance each QA pair with detailed source information
    for item in qa_list:
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
        
        # Transform simple source list to detailed source objects (if needed)
        old_source = item.get("source", [])
        
        # Check if source is already in detailed format
        if old_source and isinstance(old_source[0], dict):
            # Already in detailed format, skip transformation
            continue
            
        detailed_sources = []
        
        # Handle both string and list formats
        if isinstance(old_source, str):
            old_source = [old_source]
        
        # Map page references to actual source text
        for page_ref in old_source:
            if isinstance(page_ref, str):
                # Extract page number from reference like "第18頁"
                page_match = re.search(r'第(\d+)頁', page_ref)
                if page_match:
                    page_num = int(page_match.group(1))
                    # Find matching sources
                    matching_sources = [s for s in sources_meta if s["page"] == page_num]
                    if matching_sources:
                        # Take first match, prepare detailed source
                        source = matching_sources[0]
                        detailed_source = {
                            "doc_id": source["doc_id"],
                            "page": source["page"],
                            "text": source["text"]
                        }
                        detailed_sources.append(detailed_source)
        
        # If no matches found, add some sources anyway (fallback)
        if not detailed_sources and sources_meta:
            # Use first few sources as fallback
            for source in sources_meta[:min(3, len(sources_meta))]:
                detailed_source = {
                    "doc_id": source["doc_id"],
                    "page": source["page"],
                    "text": source["text"]
                }
                detailed_sources.append(detailed_source)
        
        # Replace source field with enhanced data
        item["source"] = detailed_sources
    
    # 8) Quality checks and warnings
    logger.info("🔍 Quality checking QA pairs...")
    for i, item in enumerate(qa_list):
        num_sources = len(item.get("source", []))
        if num_sources < 2:
            logger.warning(f"⚠️ QA pair {i+1} has thin evidence: only {num_sources} source(s)")
        elif num_sources >= 3:
            logger.info(f"✅ QA pair {i+1} has rich evidence: {num_sources} sources")
    
    # enforce exact count
    if len(qa_list) != num_pairs:
        logger.warning(f"⚠️ Expected {num_pairs} items but got {len(qa_list)}")
        # Truncate or pad to exact count
        if len(qa_list) > num_pairs:
            qa_list = qa_list[:num_pairs]
            logger.info(f"📝 Truncated to {num_pairs} items")
        else:
            logger.warning(f"⚠️ Cannot generate enough QA pairs, returning {len(qa_list)} items")
    
    logger.info(f"🎉 Final result: {len(qa_list)} QA pairs generated")
    return qa_list
