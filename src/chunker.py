# src/chunker.py

import re
import jieba
from typing import List, Tuple
from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP

# Chinese sentence boundary regex (period, question, exclamation, semicolon, ellipsis)
ZH_BOUNDARY = re.compile(r'(?<=[。！？；…])')

def is_chinese_text(text: str) -> bool:
    """
    Heuristic: True if >50% of chars are Chinese CJK.
    """
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk > len(text) / 2

def split_sentences_en(text: str) -> List[str]:
    """
    Split English on .!? plus whitespace.
    """
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def split_sentences_zh(text: str) -> List[str]:
    """
    Split Chinese on 。！？；… punctuation.
    """
    parts = ZH_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str) -> List[Tuple[str, str]]:
    """
    Return list of (lang, chunk) where:
      - lang: 'zh' or 'en'
      - chunk: string of up to CHUNK_MAX_CHARS tokens/characters
    Uses overlap of CHUNK_OVERLAP tokens/chars between chunks.
    """
    chunks: List[Tuple[str, str]] = []
    # Coarse paragraphs/pages
    segments = re.split(r'\n\s*\n', text)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        if is_chinese_text(seg):
            # Chinese: sentence → tokens → chunks by token count
            sentences = split_sentences_zh(seg)
            buf_tokens: List[str] = []
            for sent in sentences:
                toks = jieba.lcut(sent)
                # if adding this sentence would exceed limit, flush
                if len(buf_tokens) + len(toks) > CHUNK_MAX_CHARS:
                    # flush current buffer
                    chunk_str = "".join(buf_tokens)
                    chunks.append(("zh", chunk_str))
                    # overlap last CHUNK_OVERLAP tokens, then add this sentence
                    buf_tokens = buf_tokens[-CHUNK_OVERLAP:] + toks
                else:
                    buf_tokens.extend(toks)
            if buf_tokens:
                chunks.append(("zh", "".join(buf_tokens)))

        else:
            # English/mixed: sentence → chunks by character count
            sentences = split_sentences_en(seg)
            buf = ""
            for sent in sentences:
                sent = sent.strip()
                sep = " "
                # if adding this sentence would exceed limit, flush
                if len(buf) + len(sent) + len(sep) > CHUNK_MAX_CHARS:
                    chunks.append(("en", buf.strip()))
                    # overlap last chars
                    buf = buf[-CHUNK_OVERLAP:] + sep + sent
                else:
                    buf += sep + sent
            if buf:
                chunks.append(("en", buf.strip()))

    return chunks


if __name__ == "__main__":
    sample = (
        "This is the first sentence. Here's the second! And the third?\n\n"
        "這是一句中文。這是第二句！這是第三句？"
    )
    print("Max chars/tokens:", CHUNK_MAX_CHARS, "Overlap:", CHUNK_OVERLAP)
    out = chunk_text(sample)
    print(f"→ produced {len(out)} chunks:")
    for lang, c in out:
        display = c if len(c) < 50 else c[:47] + "..."
        print(f"[{lang}] {display}\n---")
