from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Sequence, Optional, Union
import hashlib
import re

from .table_processing.change_html_to_json import parse_html_tables


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    table_index: int
    row_start: int
    row_end: int
    rows: int
    text: str
    meta: Dict[str, Union[str, int, float, bool, dict, list]]

    def to_dict(self) -> Dict[str, Union[str, int, float, bool, dict, list]]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "table_index": self.table_index,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "rows": self.rows,
            "text": self.text,
            "meta": self.meta,
        }


def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def _row_to_text(row: Dict[str, Union[str, int, float]], *, joiner: str = " | ", kv_sep: str = ": ", sort_keys: bool = False) -> str:
    keys = [k for k in row.keys() if not k.startswith("__")]
    if sort_keys:
        keys = sorted(keys)
    parts = []
    for k in keys:
        v = row[k]
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        parts.append(f"{k}{kv_sep}{s}")
    return joiner.join(parts)


def _norm_text_for_hash(t: str) -> str:
    return " ".join(t.split())


def _text_hash(t: str) -> str:
    return hashlib.sha1(_norm_text_for_hash(t).encode("utf-8", errors="ignore")).hexdigest()[:12]


def _extract_page_number(file_name: str) -> Optional[int]:
    name_lower = file_name.lower()
    m = re.search(r'(?:^|[_\-\s])(?:page|p|pg|頁)0*(\d{1,4})(?=[_\-\s\.]|$)', name_lower)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r'[_\-]0*(\d{1,4})(?=\.[a-z0-9]+$)', name_lower)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            pass
    return None


def _extract_table_number(file_name: str) -> Optional[int]:
    name_lower = file_name.lower()
    m = re.search(r'(?:^|[_\-\s])(?:table|tbl|表)0*(\d{1,4})(?=[_\-\s\.]|$)', name_lower)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def extract_table_chunks(
    html: str,
    *,
    source_id: str = "memory",
    max_rows_per_chunk: int = 15,
    max_chars: int = 1500,
    row_level: bool = False,
    include_notes: bool = False,
    joiner: str = " | ",
    kv_sep: str = ": ",
    sort_keys: bool = False,
    keep_raw_rows_json: bool = False,
    char_chunk: bool = False,
    max_chars_per_chunk: int = 1200,
    overlap_chars: int = 150,
    char_only_if_long: bool = True,
) -> List[Chunk]:
    """
    將 HTML 中所有 <table> 拆為 chunk 列表。
    - row_level=True: 每列成一塊
    - group: 預設依行/字元雙閾值合併
    - char_chunk=True: 對整張表的合併全文再做滑動視窗
      (會避開與 group 完全相同的重複片段)
    """
    tables = parse_html_tables(html)
    chunks: List[Chunk] = []
    file_name = Path(source_id).name
    filename_prefix = f"{file_name} "
    page_number = _extract_page_number(file_name)
    table_number = _extract_table_number(file_name)

    for t_idx, table_rows in enumerate(tables):
        if not isinstance(table_rows, list):
            continue

        if table_number is not None:
            table_uuid = _hash_id(file_name.lower(), f"table{table_number:04d}")
        else:
            table_uuid = _hash_id(source_id, str(t_idx), "table")

        text_rows: List[str] = []
        row_map: List[Dict[str, Union[str, int, float]]] = []
        for r_idx, row in enumerate(table_rows):
            if not isinstance(row, dict):
                continue
            if not include_notes and row.get("__note__"):
                continue
            line = _row_to_text(row, joiner=joiner, kv_sep=kv_sep, sort_keys=sort_keys)
            if line.strip():
                text_rows.append(line)
                row_map.append(row)

        if not text_rows:
            continue

        if row_level:
            for local_idx, line in enumerate(text_rows):
                cid = _hash_id(source_id, str(t_idx), str(local_idx), line[:50])
                meta = {
                    "mode": "row",
                    "total_rows_in_table": len(text_rows),
                    "row_index": local_idx,
                    "table_index": t_idx,
                    "file_name": file_name,
                    "table_uuid": table_uuid,
                }
                if page_number is not None:
                    meta["page_number"] = page_number
                if table_number is not None:
                    meta["table_number"] = table_number
                if keep_raw_rows_json:
                    meta["row_json"] = row_map[local_idx]
                c = Chunk(
                    chunk_id=cid,
                    source_id=source_id,
                    table_index=t_idx,
                    row_start=local_idx,
                    row_end=local_idx,
                    rows=1,
                    text=filename_prefix + line,
                    meta=meta,
                )
                chunks.append(c)
            return chunks

        group_start = 0
        group_chunks: List[Chunk] = []
        group_text_hashes: set[str] = set()

        while group_start < len(text_rows):
            end = group_start
            acc_chars = 0
            while end < len(text_rows):
                line_len = len(text_rows[end])
                new_rows = end - group_start + 1
                new_chars = acc_chars + line_len + (0 if end == group_start else 1)
                if new_rows > max_rows_per_chunk or new_chars > max_chars:
                    if end == group_start:
                        end += 1
                    break
                acc_chars = new_chars
                end += 1

            slice_rows = text_rows[group_start:end]
            cid = _hash_id(source_id, str(t_idx), str(group_start), str(end), slice_rows[0][:40])
            group_text = filename_prefix + "\n".join(slice_rows)
            meta = {
                "mode": "group",
                "total_rows_in_table": len(text_rows),
                "row_start": group_start,
                "row_end": end - 1,
                "table_index": t_idx,
                "rows_in_chunk": len(slice_rows),
                "file_name": file_name,
                "table_uuid": table_uuid,
            }
            if page_number is not None:
                meta["page_number"] = page_number
            if table_number is not None:
                meta["table_number"] = table_number
            if keep_raw_rows_json:
                meta["rows_json"] = row_map[group_start:end]
            c = Chunk(
                chunk_id=cid,
                source_id=source_id,
                table_index=t_idx,
                row_start=group_start,
                row_end=end - 1,
                rows=len(slice_rows),
                text=group_text,
                meta=meta,
            )
            group_chunks.append(c)
            group_text_hashes.add(_text_hash(group_text))
            group_start = end

        chunks.extend(group_chunks)

        if char_chunk:
            joined = " ".join(text_rows)
            joined = " ".join(joined.split())
            n = len(joined)
            if not joined:
                continue
            if char_only_if_long and n <= max_chars_per_chunk:
                continue
            if overlap_chars >= max_chars_per_chunk:
                overlap_chars = max_chars_per_chunk // 3 or 1
            step = max(max_chars_per_chunk - overlap_chars, 1)

            pos = 0
            idx = 0
            while pos < n:
                end = min(pos + max_chars_per_chunk, n)
                segment = joined[pos:end]
                text_full = filename_prefix + segment
                seg_hash = _text_hash(text_full)
                if seg_hash in group_text_hashes:
                    if end >= n:
                        break
                    pos += step
                    idx += 1
                    continue

                cid = _hash_id(source_id, str(t_idx), "char", str(pos), str(end))
                meta = {
                    "mode": "char",
                    "table_index": t_idx,
                    "char_start": pos,
                    "char_end": end - 1,
                    "total_chars_table_joined": n,
                    "overlap_chars": overlap_chars,
                    "chunk_index": idx,
                    "file_name": file_name,
                    "table_uuid": table_uuid,
                }
                if page_number is not None:
                    meta["page_number"] = page_number
                if table_number is not None:
                    meta["table_number"] = table_number
                c = Chunk(
                    chunk_id=cid,
                    source_id=source_id,
                    table_index=t_idx,
                    row_start=0,
                    row_end=0,
                    rows=0,
                    text=text_full,
                    meta=meta,
                )
                chunks.append(c)

                if end >= n:
                    break
                pos += step
                idx += 1

    return chunks


__all__ = [
    "Chunk",
    "extract_table_chunks",
]


