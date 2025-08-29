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
        """Extract document IDs from text using fuzzy matching."""
        doc_ids = []
        
        # First, try exact matches
        for doc_id in self.known_doc_ids:
            if doc_id.lower() in text.lower():
                doc_ids.append(doc_id)
        
        # If no exact matches, try fuzzy matching
        if not doc_ids and self.known_doc_ids:
            # Use jieba to segment the text and look for potential doc_ids
            words = jieba.lcut(text)
            
            for word in words:
                # Skip very short words or common words that shouldn't be doc_ids
                if len(word) < 5 or word.lower() in ['show', 'me', 'complete', 'tables', 'from', 'page', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                    continue
                
                # Try fuzzy matching
                best_match = process.extractOne(word, self.known_doc_ids, scorer=fuzz.ratio)
                if best_match and best_match[1] > 80:  # 80% similarity threshold
                    doc_ids.append(best_match[0])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_doc_ids = []
        for doc_id in doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_doc_ids.append(doc_id)
        
        return unique_doc_ids
    
    def _extract_content_types(self, text: str) -> List[str]:
        """Extract content types from text."""
        content_types = []
        
        # Table indicators - expanded patterns
        table_patterns = [
            r'表[格]?[一二三四五六七八九十\d]*',
            r'table[s]?\s*[\d]*',
            r'圖表',
            r'列表',
            r'complete\s+table[s]?',  # "complete tables"
            r'show\s+me\s+table[s]?',  # "show me tables"
            r'get\s+table[s]?',        # "get tables"
            r'find\s+table[s]?',       # "find tables"
            r'display\s+table[s]?',    # "display tables"
            r'列出\s*表[格]?',          # "列出表格"
            r'顯示\s*表[格]?',          # "顯示表格"
            r'查詢\s*表[格]?',          # "查詢表格"
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # For table queries, include both table_row and table_summary
                content_types.extend(['table_row', 'table_summary'])
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