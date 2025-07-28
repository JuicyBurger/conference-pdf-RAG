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

def chunk_text(text: str) -> List[str]:
    """
    Return list of Chinese text chunks.
    Each chunk contains up to CHUNK_MAX_CHARS tokens.
    Uses overlap of CHUNK_OVERLAP tokens between chunks.
    """
    chunks: List[str] = []
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
            if len(buf_tokens) + len(toks) > CHUNK_MAX_CHARS:
                # flush current buffer
                chunk_str = "".join(buf_tokens)
                chunks.append(chunk_str)
                # overlap last CHUNK_OVERLAP tokens, then add this sentence
                buf_tokens = buf_tokens[-CHUNK_OVERLAP:] + toks
            else:
                buf_tokens.extend(toks)
        if buf_tokens:
            chunks.append("".join(buf_tokens))

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
