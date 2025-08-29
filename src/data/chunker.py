# src/chunker.py

import re
import jieba
from typing import List, Tuple
from ..config import CHUNK_MAX_CHARS, CHUNK_OVERLAP

# Chinese sentence boundary regex (period, question, exclamation, semicolon, ellipsis)
ZH_BOUNDARY = re.compile(r'(?<=[。！？；…])')

def split_sentences_zh(text: str) -> List[str]:
    """
    Split Chinese on 。！？；… punctuation.
    """
    parts = ZH_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, content_type: str = "general") -> List[str]:
    """
    Return list of Chinese text chunks with adaptive sizing.
    Each chunk contains up to CHUNK_MAX_CHARS tokens.
    Uses overlap of CHUNK_OVERLAP tokens between chunks.
    
    Args:
        text: Text to chunk
        content_type: Type of content ("financial", "table", "general")
    """
    chunks: List[str] = []
    
    # Adaptive chunk sizes based on content type
    if content_type == "financial":
        # Financial documents need larger chunks for context
        max_chars = int(CHUNK_MAX_CHARS * 1.5)
        overlap = int(CHUNK_OVERLAP * 1.5)
    elif content_type == "table":
        # Tables can use smaller chunks since they're structured; use same base size for consistency
        max_chars = CHUNK_MAX_CHARS
        overlap = CHUNK_OVERLAP
    else:
        # General content uses default settings
        max_chars = CHUNK_MAX_CHARS             # 3000 chars
        overlap = CHUNK_OVERLAP                 # 200 chars
    
    # Coarse paragraphs/pages
    segments = re.split(r'\n\s*\n', text)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Chinese: sentence → tokens → chunks by token count
        sentences = split_sentences_zh(seg)
        buf_tokens: List[str] = []
        for sent in sentences:
            toks = jieba.lcut(sent)
            # if adding this sentence would exceed limit, flush
            if len(buf_tokens) + len(toks) > max_chars:
                # flush current buffer
                chunk_str = "".join(buf_tokens)
                if chunk_str.strip():  # Only add non-empty chunks
                    chunks.append(chunk_str)
                # overlap last overlap tokens, then add this sentence
                buf_tokens = buf_tokens[-overlap:] + toks
            else:
                buf_tokens.extend(toks)
        if buf_tokens:
            chunk_str = "".join(buf_tokens)
            if chunk_str.strip():  # Only add non-empty chunks
                chunks.append(chunk_str)

    return chunks


if __name__ == "__main__":
    sample = (
        "這是一句中文。這是第二句！這是第三句？\n\n"
        "這是第四句。這是第五句！這是第六句？"
    )
    print("Max chars/tokens:", CHUNK_MAX_CHARS, "Overlap:", CHUNK_OVERLAP)
    out = chunk_text(sample)
    print(f"→ produced {len(out)} chunks:")
    for c in out:
        display = c if len(c) < 50 else c[:47] + "..."
        print(f"[zh] {display}\n---")
