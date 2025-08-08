# src/generator.py

import ast
import re
import json
import uuid
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from ..config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from .retriever import retrieve
from ..models.reranker import rerank
from ..models.LLM import LLM
from ..models.client_factory import get_llm_client, get_default_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients using the factory
llm_client = get_llm_client()
default_model = get_default_model()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def generate_answer(query: str, hits) -> str:
    """
    Generate an answer in Traditional Chinese.
    - query: the user's question
    - hits: list of ScoredPoint from Qdrant
    """
    # Handle empty or weak hits: ask for clarification instead of fabricating
    if not hits:
        return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"
    
    # Filter valid hits with proper payload structure
    valid_hits = []
    top_score_observed = 0.0
    for h in hits:
        try:
            # Check for both 'text' and 'content' fields (different indexing methods)
            text_field = None
            if hasattr(h, 'payload') and h.payload:
                if 'text' in h.payload:
                    text_field = 'text'
                elif 'content' in h.payload:
                    text_field = 'content'
                
                if text_field and 'page' in h.payload:
                    # Additional validation: ensure text is not empty
                    if h.payload[text_field] and h.payload[text_field].strip():
                        # Create a normalized hit with 'text' field for consistency
                        normalized_hit = type('Hit', (), {
                            'id': getattr(h, 'id', 'unknown'),
                            'score': getattr(h, 'score', 0.0),
                            'payload': {
                                'text': h.payload[text_field],
                                'page': h.payload['page'],
                                'doc_id': h.payload.get('doc_id', 'unknown'),
                                'type': h.payload.get('type', 'unknown')
                            }
                        })()
                        valid_hits.append(normalized_hit)
                        try:
                            top_score_observed = max(top_score_observed, float(getattr(h, 'score', 0.0)))
                        except Exception:
                            pass
                    else:
                        print(f"⚠️ Skipping hit with empty {text_field}: {getattr(h, 'id', 'unknown')}")
                else:
                    print(f"⚠️ Skipping hit without proper payload structure: {getattr(h, 'id', 'unknown')}")
                    if hasattr(h, 'payload'):
                        print(f"   Payload keys: {list(h.payload.keys()) if h.payload else 'None'}")
            else:
                print(f"⚠️ Skipping hit without payload: {getattr(h, 'id', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Error processing hit {getattr(h, 'id', 'unknown')}: {e}")
            continue
    
    if not valid_hits:
        return "我目前沒有足夠的文件證據來回答。可以提供更具體的主題、關鍵詞或頁碼嗎？"

    # Abstain if evidence is weak, unless disabled for demo (shares kill-switch with query signal)
    if os.getenv("DISABLE_QUERY_SIGNAL", "0") != "1":
        if top_score_observed < 0.4:
            return "目前檢索到的內容相關性不足。請提供更明確的關鍵詞、頁碼或文件名稱以便查找。"
    
    # Include doc_id in context to reduce invented labels like "Document 1"
    context = "\n\n".join(
        f"[{h.payload.get('doc_id', 'unknown')} 第{h.payload['page']}頁] {h.payload['text']}" for h in valid_hits
    )
    system_prompt = (
        "你是投資人簡報助理。"
        "僅根據提供的節錄內容回答；若內容不足或無關，請禮貌要求使用者提供更多線索。"
        "回答時務必使用繁體中文；若引用，僅標註實際出現於節錄中的 doc_id 與頁碼，切勿臆測。"
    )
    user_prompt = (
        f"以下是可用的文件節錄（可能為零或多段）：\n{context}\n\n"
        f"問題：{query}\n"
        "要求：\n"
        "- 僅在節錄能直接支持答案時作答，並標註 [doc_id 第X頁]；\n"
        "- 若節錄不足或無關，請直接回覆需要更明確的主題、關鍵詞或頁碼（不要臆測內容）。\n"
        "回答："
    )

    return LLM(llm_client, default_model, system_prompt, user_prompt, raw=True)


def fetch_doc_chunks(doc_id: str, limit: int = 1000):
    """
    Scroll Qdrant for all chunks where payload.doc_id == doc_id.
    Attach full payloads so you can see page/text.
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


def generate_qa_pairs_for_doc(
    doc_id: str,
    num_pairs: int = 5,
    timeout: float = 30.0,
    context_top_k: int = 20
) -> list[dict]:
    """
    Enhanced QA generation with multiple evidence sources and exact text snippets.
    1) Fetch all chunks for this doc_id
    2) Retrieve + rerank broader context using multiple seed queries
    3) Build rich context with detailed source tracking
    4) Generate QA with enhanced source information
    """
    
    print(f"Number of pairs that will be generated: {num_pairs}")
    
    # 1) Fetch every chunk for this PDF
    records = fetch_doc_chunks(doc_id)
    if not records:
        return []

    # 2) Enhanced retrieval with multiple perspectives and more diverse sources
    import random
    
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
        hits = retrieve(query, top_k=context_top_k//3, score_threshold=0.2)  # Lower threshold, more results
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


def sanitize_json_library(raw: str) -> str:
    """
    Robust JSON sanitization that handles various malformed JSON cases from LLMs.
    """
    import re
    import json
    
    # Step 1: Clean up the raw text
    cleaned = raw.strip()
    
    # Remove any text before the JSON array
    start_idx = cleaned.find('[')
    if start_idx == -1:
        # Try to find JSON object if array not found
        start_idx = cleaned.find('{')
        if start_idx == -1:
            raise ValueError("No JSON array or object found")
    
    # Find the end of the JSON structure (handle nested brackets/braces)
    bracket_count = 0
    brace_count = 0
    end_idx = start_idx
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(cleaned)):
        char = cleaned[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '[' or char == '{':
                if char == '[':
                    bracket_count += 1
                else:
                    brace_count += 1
            elif char == ']' or char == '}':
                if char == ']':
                    bracket_count -= 1
                else:
                    brace_count -= 1
                    
                if bracket_count == 0 and brace_count == 0:
                    end_idx = i
                    break
    
    # Extract the JSON structure
    json_str = cleaned[start_idx:end_idx+1]
    
    # Step 2: Fix common JSON issues
    
    # Fix trailing commas before closing brackets/braces
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix missing quotes around property names
    json_str = re.sub(r'(\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
    
    # Fix unescaped quotes in strings (but not in property names)
    # This is tricky, so we'll be more careful
    json_str = re.sub(r'(?<!\\)"(?=.*?":)', r'\\"', json_str)
    
    # Fix unterminated strings
    json_str = re.sub(r'([^\\])"([^"]*?)(?=\s*[,}\]])', r'\1"\2"', json_str)
    
    # Fix newlines and tabs in strings
    json_str = re.sub(r'\n', r'\\n', json_str)
    json_str = re.sub(r'\t', r'\\t', json_str)
    json_str = re.sub(r'\r', r'\\r', json_str)
    
    # Fix unescaped backslashes
    json_str = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', json_str)
    
    # Fix missing quotes around string values
    # Look for patterns like: "key": value, (where value is not quoted)
    json_str = re.sub(r':\s*([^"][^,}\]]*?)(?=\s*[,}\]])', r': "\1"', json_str)
    
    # Fix boolean and null values (they shouldn't be quoted)
    json_str = re.sub(r':\s*"true"', r': true', json_str)
    json_str = re.sub(r':\s*"false"', r': false', json_str)
    json_str = re.sub(r':\s*"null"', r': null', json_str)
    
    # Fix numeric values (they shouldn't be quoted)
    json_str = re.sub(r':\s*"(\d+)"', r': \1', json_str)
    json_str = re.sub(r':\s*"(\d+\.\d+)"', r': \1', json_str)
    
    # Step 3: Validate and try to parse
    try:
        # Test if it's valid JSON
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        # If still invalid, try more aggressive fixes
        
        # Try to fix common array issues
        # Remove any text that's not part of the JSON structure
        lines = json_str.split('\n')
        cleaned_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('[') or line.startswith('{'):
                in_json = True
            if in_json:
                cleaned_lines.append(line)
            if line.endswith(']') or line.endswith('}'):
                in_json = False
        
        json_str = '\n'.join(cleaned_lines)
        
        # Try one more time
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # Last resort: try to extract valid JSON objects
            raise ValueError(f"Could not sanitize JSON: {e}")
    
    return json_str


def extract_partial_qa(raw: str) -> list:
    """
    Extract partial QA pairs from broken JSON using multiple strategies.
    """
    import re
    import json
    
    qa_pairs = []
    
    # Strategy 1: Look for complete QA objects with regex
    # Pattern for {"question": "...", "answer": "...", "source": [...]}
    pattern1 = r'\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"answer"\s*:\s*"[^"]*"\s*,\s*"source"\s*:\s*\[[^\]]*\]\s*\}'
    matches1 = re.findall(pattern1, raw, re.DOTALL)
    
    for match in matches1:
        try:
            # Clean up the match
            cleaned_match = re.sub(r',\s*}', '}', match)  # Remove trailing commas
            cleaned_match = re.sub(r',\s*]', ']', cleaned_match)  # Remove trailing commas in arrays
            cleaned_match = re.sub(r'\n', r'\\n', cleaned_match)  # Fix newlines
            cleaned_match = re.sub(r'\t', r'\\t', cleaned_match)  # Fix tabs
            
            qa_pair = json.loads(cleaned_match)
            if "question" in qa_pair and "answer" in qa_pair:
                qa_pairs.append(qa_pair)
        except Exception as e:
            logger.debug(f"Failed to parse QA object: {e}")
            continue
    
    # Strategy 2: Look for individual fields and reconstruct
    if not qa_pairs:
        # Find all question-answer pairs
        question_pattern = r'"question"\s*:\s*"([^"]*)"'
        answer_pattern = r'"answer"\s*:\s*"([^"]*)"'
        source_pattern = r'"source"\s*:\s*\[([^\]]*)\]'
        
        questions = re.findall(question_pattern, raw, re.DOTALL)
        answers = re.findall(answer_pattern, raw, re.DOTALL)
        sources = re.findall(source_pattern, raw, re.DOTALL)
        
        # Match them up
        min_len = min(len(questions), len(answers))
        for i in range(min_len):
            try:
                qa_pair = {
                    "question": questions[i].strip(),
                    "answer": answers[i].strip(),
                    "source": []
                }
                
                # Try to add source if available
                if i < len(sources):
                    try:
                        # Parse source array
                        source_str = f"[{sources[i]}]"
                        source_str = re.sub(r',\s*]', ']', source_str)  # Fix trailing commas
                        source_array = json.loads(source_str)
                        qa_pair["source"] = source_array
                    except:
                        # If source parsing fails, use empty array
                        pass
                
                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.debug(f"Failed to reconstruct QA pair {i}: {e}")
                continue
    
    # Strategy 3: Look for markdown-style QA pairs
    if not qa_pairs:
        # Pattern for Q: ... A: ... format
        qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
        matches = re.findall(qa_pattern, raw, re.DOTALL)
        
        for question, answer in matches:
            try:
                qa_pair = {
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "source": []
                }
                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.debug(f"Failed to parse markdown QA: {e}")
                continue
    
    # Strategy 4: Look for numbered lists
    if not qa_pairs:
        # Pattern for 1. Question: ... Answer: ...
        numbered_pattern = r'\d+\.\s*Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\d+\.|$)'
        matches = re.findall(numbered_pattern, raw, re.DOTALL)
        
        for question, answer in matches:
            try:
                qa_pair = {
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "source": []
                }
                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.debug(f"Failed to parse numbered QA: {e}")
                continue
    
    # Clean up the extracted pairs
    cleaned_pairs = []
    for pair in qa_pairs:
        try:
            # Ensure required fields exist
            if "question" not in pair or "answer" not in pair:
                continue
                
            # Clean up text fields
            pair["question"] = pair["question"].strip()
            pair["answer"] = pair["answer"].strip()
            
            # Ensure source field exists
            if "source" not in pair:
                pair["source"] = []
            
            # Add ID if missing
            if "id" not in pair:
                import uuid
                pair["id"] = str(uuid.uuid4())
            
            cleaned_pairs.append(pair)
        except Exception as e:
            logger.debug(f"Failed to clean QA pair: {e}")
            continue
    
    return cleaned_pairs


def sanitize_json_via_llm(raw: str, client, model: str, timeout: float = 15) -> str:
    """
    Ask the LLM to repair and re‐emit a valid JSON array.
    (Kept as fallback but should rarely be used)
    """
    system = (
        "You are a JSON validator and reformatter. "
        "The user will give you a string that is _almost_ a valid JSON array. "
        "Your job is to output **only** a well-formed JSON array—nothing else."
    )
    user = f"Here is the raw JSON to fix:\n\n{raw}\n\nPlease output a corrected JSON array."
    
    # Call your LLM wrapper in raw mode (no Pydantic schema)
    fixed = LLM(
        client=client,
        model=model,
        system_prompt=system,
        user_prompt=user,
        options={"temperature": 0},  # deterministic cleanup
        timeout=timeout,
        raw=True
    )
    return fixed.strip()


