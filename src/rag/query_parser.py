"""
Advanced query parser for extracting constraints from natural language queries.

Supports:
- Document ID extraction (e.g., "112Q4興農合併財報", "2023Q3_AAPL_10Q")
- Page number extraction (English: "page 31", "pages 12-15"; Chinese: "第三十一頁", "第43-45頁") 
- Content type extraction (table, paragraph)
- Fuzzy matching for document IDs using RapidFuzz
"""

import re
import jieba
from typing import Optional, List, Dict, Tuple, Any
from rapidfuzz import fuzz, process
from cn2an import cn2an
import logging

logger = logging.getLogger(__name__)

class QueryParser:
    def __init__(self, known_doc_ids: Optional[List[str]] = None):
        """
        Initialize query parser with optional list of known document IDs for fuzzy matching.
        
        Args:
            known_doc_ids: List of known document IDs for fuzzy matching
        """
        self.known_doc_ids = known_doc_ids or []
        
        # Add essential domain words to jieba dictionary for better segmentation
        # These are core concepts that should always be available
        essential_words = [
            "合併財報", "財務報表", "年報", "季報", "月報",
            "第一頁", "第二頁", "第三頁", "第四頁", "第五頁",
            "表格", "段落", "附註", "備註",
            "議事手冊", "手冊", "議事"
        ]
        
        for word in essential_words:
            jieba.add_word(word)
        
        # Dynamically add document IDs to jieba if provided
        if known_doc_ids:
            self._add_doc_ids_to_jieba(known_doc_ids)
    
    def _add_doc_ids_to_jieba(self, doc_ids: List[str]):
        """Add document IDs to jieba dictionary for better segmentation."""
        added_count = 0
        for doc_id in doc_ids:
            # Only add reasonable-length document IDs
            if doc_id and len(doc_id) < 100 and doc_id.strip():
                jieba.add_word(doc_id.strip())
                added_count += 1
                logger.debug(f"Added to jieba: '{doc_id}'")
        
        if added_count > 0:
            logger.info(f"Added {added_count} document IDs to jieba dictionary")
    
    def add_new_doc_id(self, doc_id: str):
        """Add a new document ID to both jieba and known_doc_ids."""
        if doc_id and len(doc_id) < 100 and doc_id.strip():
            clean_doc_id = doc_id.strip()
            jieba.add_word(clean_doc_id)
            if clean_doc_id not in self.known_doc_ids:
                self.known_doc_ids.append(clean_doc_id)
                logger.info(f"Added new doc_id to parser: '{clean_doc_id}'")
            return True
        else:
            logger.warning(f"Invalid doc_id for jieba: '{doc_id}'")
            return False
    
    def update_known_doc_ids(self, doc_ids: List[str]):
        """Update the list of known document IDs for fuzzy matching."""
        self.known_doc_ids = doc_ids
    
    def _extract_chinese_numbers(self, text: str) -> List[int]:
        """Extract Chinese numbers and convert to integers."""
        numbers = []
        
        # Pattern for Chinese numbers in page context
        patterns = [
            r'第([一二三四五六七八九十百千零壹貳參肆伍陸柒捌玖拾佰仟萬億兆\d]+)頁',
            r'第([一二三四五六七八九十百千零壹貳參肆伍陸柒捌玖拾佰仟萬億兆\d]+)到第([一二三四五六七八九十百千零壹貳參肆伍陸柒捌玖拾佰仟萬億兆\d]+)頁',
            r'([一二三四五六七八九十百千零壹貳參肆伍陸柒捌玖拾佰仟萬億兆]+)頁',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    if len(match.groups()) == 2:  # Range pattern
                        start_num = cn2an(match.group(1), "smart")
                        end_num = cn2an(match.group(2), "smart")
                        if isinstance(start_num, int) and isinstance(end_num, int):
                            numbers.extend(range(start_num, end_num + 1))
                    else:  # Single number
                        num = cn2an(match.group(1), "smart")
                        if isinstance(num, int):
                            numbers.append(num)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to convert Chinese number '{match.group(1)}': {e}")
                    continue
        
        return numbers
    
    def _extract_arabic_numbers(self, text: str) -> List[int]:
        """Extract Arabic page numbers."""
        numbers = []
        
        # Patterns for Arabic numbers
        patterns = [
            r'(?:page|页|頁)[s]?\s*(\d+)(?:\s*[-到至]\s*(\d+))?',
            r'(?:p|P)\.?\s*(\d+)(?:\s*[-到至]\s*(\d+))?',
            r'第\s*(\d+)\s*[-到至]\s*(\d+)\s*[页頁]',
            r'第\s*(\d+)\s*[页頁]',
            r'(\d+)\s*[-到至]\s*(\d+)\s*[页頁]',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Check if group 2 exists before accessing it
                    if match.groups() and len(match.groups()) > 1 and match.group(2):
                        start = int(match.group(1))
                        end = int(match.group(2))
                        numbers.extend(range(start, end + 1))
                    else:  # Single number
                        numbers.append(int(match.group(1)))
                except (ValueError, TypeError, IndexError):
                    continue
        
        return numbers
    
    def _extract_doc_ids(self, text: str) -> List[str]:
        """Extract document IDs with fuzzy matching support."""
        doc_ids = []
        
        # Direct pattern matching for document ID formats
        # 1) Full PDF filename with Chinese/ASCII, allowing spaces, hyphens, parentheses, and Chinese parentheses
        patterns = [
            r'([\u4e00-\u9fffA-Za-z0-9_\s\-\(\)（）]+\.pdf)(?=[,\s]|$)',
            # 2) Same as above but without requiring .pdf (some users omit extension)
            r'([\u4e00-\u9fffA-Za-z0-9_\s\-\(\)（）]+)(?=\s*文件|\s*檔案|\s*$)',
            # 3) Common quarterly formats
            r'([A-Za-z0-9]+Q[1-4][A-Za-z0-9]*)',
            r'(\d{4}年[Q第]?[一二三四1-4][季度]?[A-Za-z0-9（）\-\s]*)',
            # 4) Mixed alphanumeric codes
            r'([A-Za-z]{2,}[0-9]{2,}[A-Za-z0-9]*)',
        ]

        # Additional pattern for better PDF filename extraction (include parentheses variants)
        pdf_pattern = r'([\u4e00-\u9fffA-Za-z0-9_\s\-\(\)（）]+\.pdf)'
        pdf_matches = re.findall(pdf_pattern, text)
        for match in pdf_matches:
            # Clean up the match to remove trailing text
            clean_match = match.strip()
            if clean_match and clean_match not in doc_ids:
                doc_ids.append(clean_match)
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            doc_ids.extend(matches)
        
        # Use jieba to segment and look for potential document IDs
        words = jieba.lcut(text)
        for word in words:
            # Check if word looks like a document ID
            if len(word) > 3 and re.search(r'[A-Za-z0-9\u4e00-\u9fff]', word):
                # Look for PDF files with Chinese characters
                if word.endswith('.pdf') or any(char in word for char in ['Q', '財報', '報表', '年報', '季報', '手冊', '議事']):
                    doc_ids.append(word)
                    # Auto-add PDF filenames to jieba for future queries
                    if word.endswith('.pdf') and word not in self.known_doc_ids:
                        jieba.add_word(word)
                        self.known_doc_ids.append(word)
                        logger.debug(f"Auto-added PDF filename to jieba: '{word}'")
        
        # Fuzzy matching against known document IDs
        if self.known_doc_ids:
            for potential_id in doc_ids[:]:  # Copy to avoid modification during iteration
                best_match, score, _ = process.extractOne(
                    potential_id, 
                    self.known_doc_ids,
                    scorer=fuzz.ratio
                )
                if score > 75:  # 75% similarity threshold
                    if best_match not in doc_ids:
                        doc_ids.append(best_match)
                        logger.debug(f"Fuzzy matched '{potential_id}' -> '{best_match}' (score: {score})")
        
        return list(set(doc_ids))  # Remove duplicates
    
    def _extract_content_types(self, text: str) -> List[str]:
        """Extract content type constraints."""
        content_types = []
        
        # Table indicators
        table_patterns = [
            r'表[格]?[一二三四五六七八九十\d]*',
            r'table[s]?\s*[\d]*',
            r'圖表',
            r'列表',
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                content_types.append('table_record')
                break
        
        # Paragraph indicators
        paragraph_patterns = [
            r'段落',
            r'paragraph[s]?',
            r'內文',
            r'正文',
        ]
        
        for pattern in paragraph_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                content_types.append('paragraph')
                break
        
        return content_types
    
    def _clean_query(self, original_query: str, extracted_items: Dict[str, Any]) -> str:
        """Remove extracted constraint items from the original query."""
        cleaned = original_query
        
        # Remove document IDs
        for doc_id in extracted_items.get('doc_ids', []):
            cleaned = re.sub(re.escape(doc_id), '', cleaned, flags=re.IGNORECASE)
        
        # Remove page references
        page_patterns = [
            r'(?:page|页|頁)[s]?\s*\d+(?:\s*[-到至]\s*\d+)?',
            r'(?:p|P)\.?\s*\d+(?:\s*[-到至]\s*\d+)?',
            r'第\s*[一二三四五六七八九十百千零壹貳參肆伍陸柒捌玖拾佰仟萬億兆\d]+\s*(?:[-到至]\s*[一二三四五六七八九十百千零壹貳參肆伍陸柒捌玖拾佰仟萬億兆\d]+\s*)?[页頁]',
        ]
        
        for pattern in page_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove content type indicators
        type_patterns = [
            r'表[格]?[一二三四五六七八九十\d]*',
            r'table[s]?\s*[\d]*',
            r'段落',
            r'paragraph[s]?',
        ]
        
        for pattern in type_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation
        cleaned = re.sub(r'[，。、\s]+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a natural language query and extract constraints.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (cleaned_query, constraints_dict)
            
        constraints_dict format:
        {
            'doc_ids': List[str],      # Document IDs found
            'pages': List[int],        # Page numbers found  
            'content_types': List[str], # Content types found
        }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Extract all constraint types
        doc_ids = self._extract_doc_ids(query)
        
        # Combine Chinese and Arabic page number extraction
        pages = []
        pages.extend(self._extract_chinese_numbers(query))
        pages.extend(self._extract_arabic_numbers(query))
        pages = sorted(list(set(pages)))  # Remove duplicates and sort
        
        content_types = self._extract_content_types(query)
        
        # Build constraints dict
        constraints = {}
        if doc_ids:
            constraints['doc_ids'] = doc_ids
        if pages:
            constraints['pages'] = pages
        if content_types:
            constraints['content_types'] = content_types
        
        # Clean the query
        extracted_items = {
            'doc_ids': doc_ids,
            'pages': pages,
            'content_types': content_types
        }
        cleaned_query = self._clean_query(query, extracted_items)
        
        logger.debug(f"Extracted constraints: {constraints}")
        logger.debug(f"Cleaned query: {cleaned_query}")
        
        # Add validation for doc_ids
        if 'doc_ids' in constraints:
            logger.info(f"Extracted doc_ids: {constraints['doc_ids']}")
            # Validate each doc_id
            valid_doc_ids = []
            for doc_id in constraints['doc_ids']:
                if doc_id and doc_id.strip() and len(doc_id.strip()) > 0:
                    # Clean up any trailing punctuation or extra text
                    cleaned_id = doc_id.strip().rstrip(',.!?;:')
                    
                    # Additional cleaning: remove text after the last .pdf
                    if '.pdf' in cleaned_id:
                        # Keep up to the exact '.pdf'
                        pdf_end = cleaned_id.rfind('.pdf') + 4
                        cleaned_id = cleaned_id[:pdf_end]
                    
                    # Remove any remaining quotes or extra punctuation
                    cleaned_id = cleaned_id.strip('"\'')
                    
                    # Remove .pdf extension for matching against Qdrant doc_ids
                    if cleaned_id.endswith('.pdf'):
                        cleaned_id = cleaned_id[:-4]

                    # Final trim of stray ')' or '）' if dangling at end only with 1 char
                    cleaned_id = cleaned_id.rstrip()
                    if cleaned_id.endswith(').pdf') or cleaned_id.endswith('）.pdf'):
                        # Already handled in pdf strip; nothing extra
                        pass
                    elif cleaned_id.endswith(')') or cleaned_id.endswith('）'):
                        # Allow closing parenthesis in doc_id (Qdrant stores without '.pdf' but with parentheses)
                        pass
                    
                    if cleaned_id and len(cleaned_id) > 4:  # Minimum reasonable length
                        valid_doc_ids.append(cleaned_id)
                        logger.debug(f"Valid doc_id (without .pdf): '{cleaned_id}'")
                    else:
                        logger.warning(f"Cleaned doc_id too short: '{cleaned_id}'")
                else:
                    logger.warning(f"Invalid doc_id found: '{doc_id}'")
            constraints['doc_ids'] = valid_doc_ids
            logger.info(f"Final valid doc_ids (for Qdrant matching): {valid_doc_ids}")
        
        return cleaned_query, constraints

# Convenience function for backward compatibility
def parse_query(query: str, known_doc_ids: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to parse a query without creating a parser instance.
    
    Args:
        query: Natural language query string
        known_doc_ids: Optional list of known document IDs for fuzzy matching
        
    Returns:
        Tuple of (cleaned_query, constraints_dict)
    """
    parser = QueryParser(known_doc_ids)
    return parser.parse_query(query)