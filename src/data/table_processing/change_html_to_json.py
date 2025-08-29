import json
import re
from collections import defaultdict
from typing import List, Dict, Optional, Union, Sequence
from bs4 import BeautifulSoup, Tag

# -------------- Public API ----------------

def parse_html_tables(
    html: str,
    header_rows: Optional[int] = None,
    *,
    level_join: str = " / ",
    drop_empty_levels: bool = True,
    parse_numbers: bool = True,
    separate_notes: bool = False,
    note_min_colspan_ratio: float = 0.55,
    note_min_text_len: int = 20,
    keep_note_rows_in_rows: bool = True,   # 新增：是否把註解列也放回 rows
) -> List[Union[List[Dict[str, Union[str, int, float]]], Dict[str, object]]]:
    """
    Parse every <table> in HTML into JSON rows.
    Enhancements:
      - Robust header detection
      - Footnote / 說明長文字列抽出
      - keep_note_rows_in_rows: 仍將說明列插入 rows，並標記 __note__=True
    When separate_notes=True:
        返回每個表: {"rows":[...], "notes":[{"label":..., "text":...}, ...]}
    """
    soup = BeautifulSoup(html, "html.parser")
    out: List[Union[List[Dict[str, Union[str, int, float]]], Dict[str, object]]] = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue

        if header_rows is None:
            hdr_n = _infer_header_rows(trs)
        else:
            hdr_n = min(header_rows, len(trs) - 1)

        header_trs = trs[:hdr_n]
        data_trs = trs[hdr_n:]

        header_grid = _build_header_grid(header_trs)
        if not header_grid:
            continue
        header_paths = _flatten_header_paths(
            header_grid,
            level_join=level_join,
            drop_empty=drop_empty_levels
        )
        header_paths = _dedup_headers(header_paths)
        col_count = len(header_paths)

        # --- 探測說明列 ---
        note_indices = set()
        notes: List[Dict[str, str]] = []
        raw_note_rows: List[tuple] = []  # (row_index, label, text)
        for idx, tr in enumerate(data_trs):
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue
            cell_meta = []
            long_span_found = False
            for c in tds:
                cs = int(c.get("colspan", 1))
                txt = _cell_text(c)
                cell_meta.append((txt, cs))
            for i, (txt, cs) in enumerate(cell_meta):
                ratio = cs / max(1, col_count)
                if (i == 0 and len(cell_meta) == 1 and cs >= col_count and len(txt) >= note_min_text_len):
                    long_span_found = True
                elif i > 0 and ratio >= note_min_colspan_ratio and len(txt) >= note_min_text_len:
                    long_span_found = True
            if long_span_found:
                note_indices.add(idx)
                if len(cell_meta) == 1:
                    label = ""
                    body = cell_meta[0][0]
                else:
                    first = cell_meta[0][0].strip()
                    rest_text = " ".join(m[0] for m in cell_meta[1:]).strip()
                    if len(first) <= 12 and rest_text:
                        label, body = first, rest_text
                    else:
                        label, body = "", " ".join(m[0] for m in cell_meta).strip()
                notes.append({"label": label, "text": body})
                raw_note_rows.append((idx, label, body))

        # --- 展開資料列（去除註解列） ---
        filtered_trs = [tr for i, tr in enumerate(data_trs) if i not in note_indices]

        # 重新估算資料實際可能欄寬 (某些內部子表頭會有 colspan > 原表頭欄數)
        data_col_count = col_count
        for tr in filtered_trs:
            w = 0
            for cell in tr.find_all(["td", "th"]):
                w += int(cell.get("colspan", 1))
            if w > data_col_count:
                data_col_count = w

        data_matrix = _expand_rows(filtered_trs, total_cols=data_col_count)

        # --- 針對首欄表頭為 None 的對齊修正 (解決最後一列職稱被左移) ---
        if header_paths and header_paths[0] is None and data_matrix:
            # 找出最常出現的首欄值 (視為群組標籤，例如 "經理人")
            freq: Dict[str, int] = {}
            for r in data_matrix:
                v = r[0]
                if v:
                    freq[v] = freq.get(v, 0) + 1
            if freq:
                group_label = max(freq.items(), key=lambda x: x[1])[0]
                fixed_matrix = []
                for r in data_matrix:
                    if r and r[0] and r[0] != group_label:
                        r = [None] + r
                        if len(r) > col_count:
                            r = r[:col_count]
                        else:
                            r += [None] * (col_count - len(r))
                    fixed_matrix.append(r)
                data_matrix = fixed_matrix

        # --- 單一常數欄位 forward-fill (解決最後一列 0 遺失) ---
        if data_matrix:
            unique_cols: Dict[int, str] = {}
            for c_idx in range(data_col_count):
                vals = [row[c_idx] for row in data_matrix if c_idx < len(row) and row[c_idx] not in (None, "")]
                if vals and len(set(vals)) == 1:
                    unique_cols[c_idx] = vals[0]
            if unique_cols:
                for row in data_matrix:
                    for c_idx, val in unique_cols.items():
                        if c_idx < len(row) and row[c_idx] in (None, ""):
                            row[c_idx] = val

        # --- 內部子表頭偵測 ---
        internal_segments = []
        internal_header_index = None
        if data_matrix and data_col_count >= 4:
            for i in range(len(data_matrix)-1):
                r1 = data_matrix[i]
                r2 = data_matrix[i+1]
                if not r1 or not r2:
                    continue
                # 比對前兩欄是否與主表頭相同
                if (len(r1) >= 2 and len(header_paths) >= 2 and
                    r1[0] and r1[1] and r1[0] == header_paths[0] and r1[1] == header_paths[1]):
                    tail1 = [x for x in r1[2:] if x]
                    # 需要至少 2 個以上且全相同 (重複標籤) -> 可能是第一列子表頭
                    if tail1 and len(set(tail1)) == 1 and len(tail1) >= 2:
                        span_len = len(tail1)
                        tail2 = r2[2:2+span_len]
                        # 第二列子表頭：全部非空且互異
                        if tail2 and all(t for t in tail2) and len(set(tail2)) == len(tail2):
                            internal_header_index = i
                            internal_segments.append({
                                "type": "segment1",
                                "header": header_paths,
                                "start": 0,
                                "end": i  # 不含 i
                            })
                            seg2_header = [
                                header_paths[0],
                                header_paths[1],
                                *tail2
                            ]
                            internal_segments.append({
                                "type": "segment2",
                                "header": seg2_header,
                                "start": i+2,
                                "end": len(data_matrix)
                            })
                            break

        # --- 產生 rows_json (考慮分段) ---
        rows_json: List[Dict[str, Union[str, int, float]]] = []
        if internal_segments:
            # 第一段
            for seg in internal_segments:
                seg_header: List[str] = [h for h in seg["header"] if h is not None]
                start = seg["start"]; end = seg["end"]
                for row_idx in range(start, end):
                    row = data_matrix[row_idx]
                    if not row: 
                        continue
                    # 跳過內部 header 行 (i, i+1)
                    if internal_header_index is not None and row_idx in (internal_header_index, internal_header_index+1):
                        continue
                    rec: Dict[str, Union[str, int, float]] = {}
                    # 對應欄位
                    if seg["type"] == "segment1":
                        # 使用原 header_paths mapping
                        for h_idx, (h, v) in enumerate(zip(header_paths, row)):
                            if h is None:
                                continue
                            cell_text = _clean_cell(v)
                            rec[h] = _maybe_parse_number(cell_text) if parse_numbers else cell_text
                    else:
                        # segment2
                        for ci, h in enumerate(seg["header"]):
                            if h is None:
                                continue
                            v = row[ci] if ci < len(row) else ""
                            cell_text = _clean_cell(v)
                            rec[h] = _maybe_parse_number(cell_text) if parse_numbers else cell_text
                    rows_json.append(rec)
        else:
            # 原單段處理
            for row in data_matrix:
                if not row or all((c is None or str(c).strip() == "") for c in row):
                    continue
                obj: Dict[str, Union[str, int, float]] = {}
                for h, v in zip(header_paths, row):
                    if h is None:
                        continue
                    cell_text = _clean_cell(v)
                    obj[h] = _maybe_parse_number(cell_text) if parse_numbers else cell_text
                rows_json.append(obj)

        # --- 選擇性把 note row 也放回 rows 末尾 ---
        if keep_note_rows_in_rows and notes:
            for note in notes:
                # 依照最後一段的 header 來建立空欄
                effective_headers = (internal_segments[-1]["header"] if internal_segments 
                                     else header_paths)
                row_obj: Dict[str, Union[str, int, float]] = {h: "" for h in effective_headers if h}
                first_header = effective_headers[0] if effective_headers and effective_headers[0] else "label"
                if note["label"]:
                    row_obj[first_header] = f"{note['label']}".strip()
                row_obj["__note__"] = True
                row_obj["note_label"] = note["label"]
                row_obj["note_text"] = note["text"]
                rows_json.append(row_obj)

        if separate_notes:
            out.append({"rows": rows_json, "notes": notes})
        else:
            out.append(rows_json)
    return out


def html_table_to_json(html: str, **kwargs) -> Union[List[Dict[str, Union[str, int, float]]], Dict[str, object]]:
    """
    Convenience wrapper: returns first table.
    Pass separate_notes=True to obtain dict with rows + notes.
    """
    tables = parse_html_tables(html, **kwargs)
    return tables[0] if tables else ([] if not kwargs.get("separate_notes") else {"rows": [], "notes": []})


# -------------- Header construction ----------------

def _build_header_grid(tr_tags: List[Tag]) -> List[List[Optional[str]]]:
    """
    Build a full header grid (rows x columns) with all rowspans/colspans expanded.
    修正：計算 total_cols 時納入上一層尚在作用中的 rowspan 欄位，避免少算 leaf 欄數。
    """
    if not tr_tags:
        return []

    # 第一階段：模擬各列寬度（含延續中的 rowspan），取得正確 leaf 欄數
    total_cols = 0
    # active_spans: list of (remaining_rows, colspan)
    active_spans: List[tuple[int, int]] = []
    for tr in tr_tags:
        # 步驟1: 累計目前仍覆蓋本列的 rowspan 欄位寬度
        inherited_width = sum(cs for rem, cs in active_spans if rem > 0)
        # 步驟2: 本列實際 cell colspan 總和
        row_width = 0
        for cell in tr.find_all(["td", "th"]):
            row_width += int(cell.get("colspan", 1))
        # 本列 leaf 寬度 = 繼承 + 當列
        leaf_width = inherited_width + row_width
        total_cols = max(total_cols, leaf_width)

        # 更新 active_spans（遞減尚未結束的 rowspan）
        new_active: List[tuple[int, int]] = []
        for rem, cs in active_spans:
            if rem - 1 > 0:
                new_active.append((rem - 1, cs))
        active_spans = new_active

        # 新增本列新出現的 rowspan
        for cell in tr.find_all(["td", "th"]):
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            if rs > 1:
                active_spans.append((rs - 1, cs))

    rows_n = len(tr_tags)
    grid: List[List[Optional[str]]] = [[None] * total_cols for _ in range(rows_n)]
    pending: Dict[int, List[tuple]] = defaultdict(list)

    # 第二階段：正式展開到 grid
    for r, tr in enumerate(tr_tags):
        # 套用前面列的 rowspan
        if r in pending:
            for col_idx, text in pending[r]:
                if 0 <= col_idx < total_cols:
                    grid[r][col_idx] = text
        cells = tr.find_all(["td", "th"])
        c = 0
        for cell in cells:
            # 找到下一個空位
            while c < total_cols and grid[r][c] is not None:
                c += 1
            if c >= total_cols:
                break
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            text = _cell_text(cell)
            # 填滿當列
            for i in range(cs):
                if c + i < total_cols:
                    grid[r][c + i] = text
            # 排程後續列的 rowspan
            if rs > 1:
                for rr in range(r + 1, r + rs):
                    for i in range(cs):
                        if c + i < total_cols:
                            pending[rr].append((c + i, text))
            c += cs
    return grid


def _flatten_header_paths(
    grid: List[List[Optional[str]]],
    *,
    level_join: str,
    drop_empty: bool
) -> List[Optional[str]]:
    """
    For each column collect vertical path of texts; remove duplicates in sequence.
    """
    if not grid:
        return []
    rows_n = len(grid)
    cols_n = len(grid[0])
    headers: List[Optional[str]] = []
    for col in range(cols_n):
        parts: List[str] = []
        last = None
        for row in range(rows_n):
            val = grid[row][col]
            if not val:
                continue
            v = val.strip()
            if drop_empty and v == "":
                continue
            if v != last:
                parts.append(v)
                last = v
        headers.append(level_join.join(parts) if parts else None)
    return headers


# -------------- Data row expansion ----------------

def _expand_rows(tr_tags: List[Tag], total_cols: int) -> List[List[Optional[str]]]:
    """
    Expand data rows (handles rowspan & colspan).
    """
    if not tr_tags:
        return []
    pending: Dict[int, List[tuple]] = defaultdict(list)
    matrix: List[List[Optional[str]]] = []
    for r, tr in enumerate(tr_tags):
        row = [None] * total_cols
        # apply pending
        if r in pending:
            for col_idx, text in pending[r]:
                if 0 <= col_idx < total_cols:
                    row[col_idx] = text
        cells = tr.find_all(["td", "th"])
        c = 0
        for cell in cells:
            while c < total_cols and row[c] is not None:
                c += 1
            if c >= total_cols:
                break
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            text = _cell_text(cell)
            for i in range(cs):
                if c + i < total_cols:
                    row[c + i] = text
            if rs > 1:
                for rr in range(r + 1, r + rs):
                    for i in range(cs):
                        if c + i < total_cols:
                            pending[rr].append((c + i, text))
            c += cs
        matrix.append(row)
    return matrix


# -------------- Heuristics & helpers ----------------

_num_re = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$")

# 追加：更寬鬆的「數值樣式」判斷 (含百分比 / 括號 / 千分位)
_numeric_like_re = re.compile(r"[0-9０-９]")

def _maybe_parse_number(s: str):
    if _num_re.match(s):
        s2 = s.replace(",", "")
        try:
            return int(s2) if "." not in s2 else float(s2)
        except:
            return s
    return s


def _infer_header_rows(tr_tags: List[Tag]) -> int:
    """
    強化版表頭列推斷：
      1. 若第一列表頭本身不含任何數值，遇到後續列出現數值 (任一) 即視為資料列 -> 停止。
         (解決：「只有第一列是真表頭，第二列開始就是資料」的情況)
      2. 否則沿用舊規則：數值樣式比例 >= 0.5 視為資料列
      3. 仍保證至少 1 列表頭，且不吃掉全部
    """
    def is_numeric_like(text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        if _numeric_like_re.search(t):
            core = t.replace(" ", "")
            if re.fullmatch(r"[()％%0-9０-９.\-+,/]+", core):
                return True
            if '%' in t or '％' in t:
                return True
        return False

    count = 0
    first_row_numeric = 0
    for tr in tr_tags:
        cells = tr.find_all(["td", "th"])
        if not cells:
            break
        texts = [c.get_text(strip=True) for c in cells]
        numeric_like = sum(1 for t in texts if is_numeric_like(t))
        total = len(cells)
        numeric_ratio = numeric_like / total if total else 0.0

        if count == 0:
            first_row_numeric = numeric_like
            # 第一行一定算表頭
            count += 1
            continue

        # 規則1：第一列無數字 -> 後面只要出現任何數字，就停止
        if first_row_numeric == 0 and numeric_like > 0:
            break

        # 規則2：數值比例高 -> 停止
        if numeric_ratio >= 0.5:
            break

        # 否則仍視為表頭
        count += 1

    return max(1, min(count, len(tr_tags) - 1))


def _cell_text(cell: Tag) -> str:
    for br in cell.find_all("br"):
        br.replace_with("\n")
    txt = cell.get_text(separator=" ").strip()
    return re.sub(r"\s+", " ", txt)


def _dedup_headers(headers: List[Optional[str]]) -> List[Optional[str]]:
    seen: Dict[str, int] = {}
    out: List[Optional[str]] = []
    for h in headers:
        if h is None:
            out.append(None)
            continue
        if h not in seen:
            seen[h] = 1
            out.append(h)
        else:
            seen[h] += 1
            out.append(f"{h}_{seen[h]}")
    return out


def _clean_cell(val: Optional[str]) -> str:
    return "" if val is None else val.strip()


# -------------- Demo --------------
if __name__ == "__main__":
    sample_html = """
評估週期：每年執行一次
評估期間：113年1月1日至113年12月31日
評估範圍：董事會、個別董事成員及功能性委員會之績效評估

<table>
  <tr>
    <td>評估內容及方式</td>
    <td>評估項目</td>
    <td>平均分數</td>
    <td>平均總分</td>
  </tr>
  <tr>
    <td rowspan="5">整體董事會<br>(由提名委員會評估)</td>
    <td>1.對公司營運之參與程度</td>
    <td>19.43</td>
    <td rowspan="5">99.19</td>
  </tr>
  <tr>
    <td>2.提升董事會決策品質</td>
    <td>29.76</td>
  </tr>
  <tr>
    <td>3.董事會組成與結構</td>
    <td>20.00</td>
  </tr>
  <tr>
    <td>4.董事的選任及持續進修</td>
    <td>10.00</td>
  </tr>
  <tr>
    <td>5.內部控制</td>
    <td>20.00</td>
  </tr>
  <tr>
    <td rowspan="4">個別董事成員<br>(由董事成員自評)</td>
    <td>1.掌握公司目標與任務及董事職責認知</td>
    <td>40.00</td>
    <td rowspan="4">99.70</td>
  </tr>
  <tr>
    <td>2.對公司營運之參與程度及內部關係經營與溝通</td>
    <td>39.70</td>
  </tr>
  <tr>
    <td>3.董事之專業及持續進修</td>
    <td>10.00</td>
  </tr>
  <tr>
    <td>4.內部控制</td>
    <td>10.00</td>
  </tr>
  <tr>
    <td rowspan="2">評估內容及方式</td>
    <td rowspan="2">評估項目</td>
    <td colspan="3">平均分數</td>
  </tr>
  <tr>
    <td>薪資報酬委員會</td>
    <td>審計委員會</td>
    <td>提名委員會</td>
  </tr>
  <tr>
    <td rowspan="6">功能性委員會<br>(由委員自評)</td>
    <td>1.對公司營運之參與程度</td>
    <td>10.00</td>
    <td>9.33</td>
    <td>10.00</td>
  </tr>
  <tr>
    <td>2.功能性委員會職責認知</td>
    <td>29.60</td>
    <td>30.00</td>
    <td>29.52</td>
  </tr>
  <tr>
    <td>3.提升功能性委員會決策品質</td>
    <td>40.00</td>
    <td>40.00</td>
    <td>40.00</td>
  </tr>
  <tr>
    <td>4.功能性委員會組成及成員選任</td>
    <td>10.00</td>
    <td>10.00</td>
    <td>10.00</td>
  </tr>
  <tr>
    <td>5.內部控制</td>
    <td>10.00</td>
    <td>10.00</td>
    <td>10.00</td>
  </tr>
  <tr>
    <td>平均總分</td>
    <td>99.60</td>
    <td>99.33</td>
    <td>99.52</td>
  </tr>
  <tr>
    <td>評估結果</td>
    <td colspan="4">本公司已完成113年度整體董事會績效評估、個別董事成員自我績效評估及各功能性委員會績效評估內部自評，評估結果 已提報113年11月8日各委員會及董事會報告。整體平均總分皆達99分以上(滿分100分)，顯示整體運作情形良好。</td>
  </tr>
</table>
    """
    data = html_table_to_json(sample_html)
    print(json.dumps(data, ensure_ascii=False, indent=2))


