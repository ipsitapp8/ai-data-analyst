"""Files that hold several logical tables in one sheet ("Table" = Sales Transactions / Customers / Expenses ...).

Such a file has a column whose values say which table a record belongs to, and every table fills in its own columns
and leaves the others empty. Analysing it as one table mixes records of different kinds (the revenue of 72 marketing
campaigns added to 12,000 sales) and offers empty columns of other tables as filters. Here the question is answered
on ONE table: the one it names, else the one the conversation is already on, else the only table that has every column
the question uses. Silt asks only when several tables fit. Nothing here knows any table name: the table column and
its tables are read from the data (which columns are empty for which value).
"""
from __future__ import annotations

import re

import pandas as pd

from app.analyst.data import kind_of

_NAME_HINT = re.compile(r"table|source|entity|record.?type|sheet|dataset|kind|type", re.I)
_GENERIC = {"table", "data", "records", "record", "list", "details", "snapshots", "snapshot", "log", "logs"}


def table_column(df: pd.DataFrame) -> str | None:
    """The column that splits the records into logical tables, or None for an ordinary table.

    A column qualifies when it is a complete text column with a few values, and each of its values leaves at least
    two other columns entirely empty, a different set for each value, with at least a fifth of all (value, column)
    cells empty -- the shape of tables stacked into one file, not of a category with a few missing values."""
    if df is None or len(df.columns) < 4 or len(df) < 4:
        return None
    key = (df.shape, tuple(map(str, df.columns)))
    cached = df.attrs.get("_table_column")
    if isinstance(cached, tuple) and cached[0] == key:
        return cached[1]
    best, best_score = None, 0.0
    for c in df.columns:
        s = df[c]
        if kind_of(s) != "categorical" or s.isna().any():
            continue
        n = s.nunique()
        if not 2 <= n <= 30 or s.value_counts().min() < 2:
            continue
        filled = df.drop(columns=[c]).notna().groupby(s.astype(str)).any()
        empty = ~filled
        if (empty.sum(axis=1) < 2).any() or filled.duplicated().any():
            continue
        score = float(empty.to_numpy().mean())
        if score < 0.2:
            continue
        score += 0.05 if _NAME_HINT.search(str(c)) else 0.0
        if score > best_score:
            best, best_score = c, score
    df.attrs["_table_column"] = (key, best)
    return best


def tables(df: pd.DataFrame, col: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[col].astype(str).value_counts().items()}


def _phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def named_tables(message: str, names: list[str]) -> list[str]:
    """Tables the question names: the whole name ("Sales Transactions", "customers"), or with its last word singular
    ("sales transaction")."""
    low = f" {_phrase(message)} "
    out = []
    for t in names:
        p = _phrase(t)
        forms = {p}
        if " " in p and p.endswith("s"):
            forms.add(p[:-1])
        if any(f" {f} " in low for f in forms if f):
            out.append(t)
    # "Sales Transactions" also contains no other name; but "Customer Reviews" contains "reviews": keep the longest
    return [t for t in out if not any(t != u and f" {_phrase(t)} " in f" {_phrase(u)} " for u in out)]


def _hinted(message: str, names: list[str]) -> list[str]:
    """Tables a word of whose name the question uses ("transactions", "campaign"): a hint, used only to choose among
    tables that all have the columns asked about."""
    words = {w.rstrip("s") for w in _phrase(message).split() if len(w) >= 4}
    out = []
    for t in names:
        tw = {w.rstrip("s") for w in _phrase(t).split() if len(w) >= 4 and w not in _GENERIC}
        if tw & words:
            out.append(t)
    return out


def _referenced(message: str, df: pd.DataFrame, col: str) -> list[str]:
    """Columns the question uses: named outright, or through a value of theirs written as in the data."""
    from app.analyst.calc import named_values
    from app.analyst.data import find_column
    from app.analyst.scope import _lowercase_word_only
    cols = [c for c in find_column(df, message) if c != col]
    cols += [c for c, v in named_values(df, message) if c != col and not _lowercase_word_only(message, c, v)]
    return list(dict.fromkeys(cols))


def subset(df: pd.DataFrame, col: str, table: str) -> pd.DataFrame:
    """The records of one table, with the columns it leaves empty (other tables' columns) and the table column
    removed, and dates / numbers stored as text read as they would be in a file of this table alone. Row labels are
    kept, so record lines still name lines of the uploaded file."""
    sub = df[df[col].astype(str) == table].dropna(axis=1, how="all").drop(columns=[col]).copy()
    for c in sub.columns:
        if sub[c].dtype != object:
            continue
        vals = sub[c].dropna()
        num = pd.to_numeric(vals, errors="coerce")
        if len(vals) and num.notna().all():
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        elif re.search(r"date|month|week|period|time", str(c), re.I):
            parsed = pd.to_datetime(sub[c], errors="coerce")
            if len(vals) and parsed.notna().sum() > 0.9 * len(vals):
                sub[c] = parsed
    sub.attrs = {k: v for k, v in df.attrs.items() if not k.startswith("_table")}
    sub.attrs["_table_column"] = ((sub.shape, tuple(map(str, sub.columns))), None)
    sub.attrs["file_lines"] = True             # row label + 2 is the record's line in the uploaded file
    sub.attrs["logical_table"] = {"column": col, "table": table, "records": int(len(sub)), "of": int(len(df))}
    return sub


def outside(message: str, df: pd.DataFrame, col: str, table: str) -> list[dict]:
    """Columns the question names, or values of theirs it writes as in the data, that are empty in `table` (they
    belong to other tables): [{"column", "named", "values", "tables"}]. A named one cannot be used on this table; a
    value met in passing ("a Business overview") is not applied."""
    from app.analyst.calc import named_values
    from app.analyst.scope import _lowercase_word_only, column_named
    here = df[col].astype(str) == table
    empty = [c for c in df.columns if c != col and df.loc[here, c].isna().all() and df[c].notna().any()]
    vals = {}
    for c, v in named_values(df, message):
        if c in empty and not _lowercase_word_only(message, c, v):
            vals.setdefault(c, []).append(v)
    out = []
    for c in empty:
        if column_named(message, c) or c in vals:
            where = sorted(df.loc[df[c].notna(), col].astype(str).unique())
            out.append({"column": c, "named": column_named(message, c), "values": vals.get(c, []), "tables": where})
    return out


def select(message: str, df: pd.DataFrame, memory: dict) -> dict | None:
    """Which table the question is about, for a file of stacked tables; None for an ordinary file.

    Returns {"column", "table", "how", "note"} with the chosen table, or {"column", "clarify", "options"} when several
    tables fit and nothing says which, or {"column", "table": None} when the question uses no column of any one table
    (a question about the file as a whole)."""
    col = table_column(df)
    if col is None:
        return None
    sizes = tables(df, col)
    names = list(sizes)
    named = named_tables(message, names)
    refs = _referenced(message, df, col)
    filled = {t: set(df.columns[df[df[col].astype(str) == t].notna().any()]) for t in names}
    fits = [t for t in names if refs and all(r in filled[t] for r in refs)]
    prev = (memory.get("logical_table") or {}).get("table")
    if len(named) == 1:
        return {"column": col, "table": named[0], "how": "named", "outside": outside(message, df, col, named[0])}
    if len(named) > 1:
        both = [t for t in named if t in fits] if refs else []
        if len(both) == 1:
            return {"column": col, "table": both[0], "how": "named"}
        return {"column": col, "clarify": f"Your question names more than one table of this file ({', '.join(named)}). "
                "Each is analysed on its own — which one should I use?", "options": named}
    if prev in names and (not refs or prev in fits):
        return {"column": col, "table": prev, "how": "previous", "outside": outside(message, df, col, prev)}
    if len(fits) == 1:
        return {"column": col, "table": fits[0], "how": "columns"}
    if len(fits) > 1:
        hint = [t for t in _hinted(message, fits)]
        if len(hint) == 1:
            return {"column": col, "table": hint[0], "how": "columns"}
        what = ", ".join(f"`{r}`" for r in refs)
        return {"column": col, "options": fits,
                "clarify": f"This file holds {len(names)} separate tables (column `{col}`), and {what} "
                           f"{'is' if len(refs) == 1 else 'are'} recorded in more than one of them: "
                           + ", ".join(f"**{t}** ({sizes[t]:,} records)" for t in fits)
                           + ". Which table should I use? I won't add them together, because they are different "
                             "kinds of records."}
    if refs:
        where = {r: [t for t in names if r in filled[t]] for r in refs}
        split = "; ".join(f"`{r}` is in {', '.join(ts) if ts else 'no table'}" for r, ts in where.items())
        return {"column": col, "options": [],
                "clarify": f"This file holds {len(names)} separate tables (column `{col}`), and no single table has "
                           f"everything your question uses: {split}. Silt analyses one table at a time — it does not "
                           "join tables — so please ask about one table."}
    return {"column": col, "table": None, "how": "none"}
