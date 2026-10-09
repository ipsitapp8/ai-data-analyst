"""Map a step's "Data Used" rows back to their line numbers in the uploaded CSV.

The data slice an analysis step prints is {columns, rows}. When every one of
those columns exists in the CSV, each slice row is (or should be) a verbatim
CSV row, so we can say exactly which line it came from. When the slice
carries derived columns (a sum, a ratio...) its rows are aggregates, not
CSV rows, and we report no line numbers rather than a misleading guess.
"""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict, deque
from functools import lru_cache

import pandas as pd

_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00(\.0+)?$")


def _norm(value) -> str:
    """Canonical text for comparing a JSON value against a raw CSV cell
    ("1500", "1500.0" and 1500 all match; a date matches its midnight timestamp)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    text = str(value).strip()
    try:
        number = float(text.replace(",", "")) if text else None
    except ValueError:
        number = None
    if number is not None and math.isfinite(number):
        return f"{number:.10g}"
    m = _MIDNIGHT.match(text)
    return m.group(1) if m else text


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> pd.DataFrame:  # mtime: cache key only
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def match_csv_lines(csv_path: str | None, columns: list, rows: list) -> list[int | None]:
    if not rows:
        return []
    unmatched = [None] * len(rows)
    if not csv_path or not columns or not os.path.exists(csv_path):
        return unmatched
    try:
        csv = _load(csv_path, os.path.getmtime(csv_path))
    except Exception:  # noqa: BLE001 - line numbers are a nice-to-have, never fatal
        return unmatched
    if any(c not in csv.columns for c in columns):
        return unmatched

    lookup: dict[tuple, deque] = defaultdict(deque)
    for offset, values in enumerate(zip(*(csv[c] for c in columns))):
        # header is line 1, so dataframe row 0 is line 2
        lookup[tuple(_norm(v) for v in values)].append(offset + 2)

    out: list[int | None] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(columns):
            out.append(None)
            continue
        hits = lookup.get(tuple(_norm(v) for v in row))
        # duplicates: hand out each matching line once, in file order
        out.append(hits.popleft() if hits else None)
    return out
