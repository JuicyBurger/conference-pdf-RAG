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
        
        # Add custom words to jieba dictionary for better segmentation
        custom_words = [
            "合併財報", "財務報表", "年報", "季報", "月報",
            "第一頁", "第二頁", "第三頁", "第四頁", "第五頁",
            "表格", "段落", "附註", "備註"
        ]
        
        for word in custom_words:
            jieba.add_word(word)
    
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
        
        # Direct pattern matching for common document ID formats
        patterns = [
            r'([A-Za-z0-9]+Q[1-4][A-Za-z0-9]*[^\\s]*)',  # Quarterly reports
            r'(\d{4}年?[第]?[一二三四1-4][季度]?[A-Za-z0-9]*[^\\s]*)',  # Year-based reports
            r'([A-Za-z0-9_]+\.pdf)',  # PDF filenames
            r'([A-Za-z]{2,}[0-9]{2,}[A-Za-z0-9]*)',  # Mixed alphanumeric codes
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            doc_ids.extend(matches)
        
        # Use jieba to segment and look for potential document IDs
        words = jieba.lcut(text)
        for word in words:
            # Check if word looks like a document ID
            if len(word) > 3 and re.search(r'[A-Za-z0-9]', word):
                if any(char in word for char in ['Q', '財報', '報表', '年報', '季報']):
                    doc_ids.append(word)
        
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