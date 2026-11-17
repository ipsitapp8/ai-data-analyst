"""One shared reading of a table's columns, applied once when a file is loaded (app/analyst/data.load_table), so the
overview, the calculator, rankings, statistics, model-written programs and the verifier all see the same columns.

What it does, from the data alone (no column name or value of any particular file is known here):
- rows whose text values sit one column to the left of where they belong (a field missing in the export, every later
  field moved up) are put back where their values fit, and the move is reported;
- counts written with a unit and a multiplier ("(26.3k Reviews)", "1.2M downloads", "3 Lakh", "53 years old") are
  read as numbers, and the unit is kept;
- a column that is only the row number saved with the file (0, 1, 2, ... under no real name) is set aside, not
  analysed as a measure.
Every change is written into the read report (df.attrs["ingestion"]["notes"]) and df.attrs["schema"]; nothing is
silently dropped or invented.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9,
         "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "cr": 1e7, "crore": 1e7, "crores": 1e7}
# "(26.3k Reviews)", "1,234 votes", "2.5M", "53 years old": a number, a multiplier, a unit. Not a range ("1k-5k").
_COUNT = re.compile(r"^\(?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<mult>k|m|mn|b|bn|thousand|million|"
                    r"billion|lakhs?|lacs?|cr|crores?)?\b\+?\s*(?P<unit>[A-Za-z][A-Za-z ]{0,30}?)?\s*\)?$", re.I)
_GENERIC_INDEX_NAME = re.compile(r"^(column_\d+|unnamed(:?\s*\d+)?|index|idx|row|row_?(no|num|number|id)?|#|sr_?no|s_?no)$",
                                 re.I)


def _sig(v) -> str:
    """The shape of a text value: words -> a, numbers -> 9, punctuation kept ("Mumbai,Maharashtra + 203 more" ->
    "a,a + 9 a", "53 years old" -> "9 a")."""
    t = str(v).strip()
    t = re.sub(r"\d+(?:\.\d+)?", "9", t)
    t = re.sub(r"[A-Za-z]+(?:[ ./&'-]+[A-Za-z]+)*", "a", t)
    return re.sub(r"\s+", " ", t)


def _typical(s: pd.Series) -> set[str]:
    sig = s.dropna().map(_sig)
    if sig.empty:
        return set()
    share = sig.value_counts(normalize=True)
    return set(share[share >= 0.01].index)


def realign_shifted_rows(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Rows whose trailing text fields sit to the left of where they belong (one or more fields missing in the
    export, every later field moved up) are put back. For a row whose last column is empty, its filled text values
    are matched, in order, to the earliest column to their right whose values usually have that shape (learned from
    the rows whose last column is filled). The row is moved only when every value finds such a column and most of
    them did not fit where they were. Applied only when this is systematic (at least 1% of rows and 5 rows);
    otherwise reported and left as recorded. Columns a moved row has no value for are missing -- never guessed."""
    if len(df) < 20 or df.shape[1] < 3:
        return df
    cols = list(df.columns)
    last = cols[-1]
    if df[last].dtype != object:
        return df
    start = len(cols) - 1                         # the trailing block of text columns
    while start > 0 and df[cols[start - 1]].dtype == object:
        start -= 1
    block = cols[start:]
    if len(block) < 3:
        return df
    complete = df[df[last].notna()]
    if len(complete) < 10:
        return df
    typ = {c: _typical(complete[c]) for c in block}
    fixes = {}
    for r in df.index[df[last].isna()]:
        filled = [(j, df.at[r, c]) for j, c in enumerate(block) if pd.notna(df.at[r, c])]
        # values before the first one that does not fit its column are where they belong; the rest are matched on
        first_bad = next((i for i, (j, v) in enumerate(filled) if _sig(v) not in typ[block[j]]), None)
        if first_bad is None:
            continue
        keep, filled = filled[:first_bad], filled[first_bad:]
        own = sum(_sig(v) in typ[block[j]] for j, v in filled)
        if own > len(filled) // 2:
            continue
        target, nxt, ok = [], (keep[-1][0] + 1 if keep else 0), True
        for j, v in filled:
            k = next((t for t in range(max(nxt, j), len(block)) if _sig(v) in typ[block[t]]), None)
            if k is None:
                ok = False
                break
            target.append(k)
            nxt = k + 1
        if ok and any(k != j for k, (j, _) in zip(target, filled)):
            fixes[r] = [(block[j], v) for j, v in keep] + [(block[k], v) for k, (_, v) in zip(target, filled)]
    if fixes and len(fixes) >= max(5, 0.01 * len(df)):
        # rows still holding a value shaped like another column's values, which could not be placed: reported
        moved = set(fixes)
        left = [r for r in df.index if r not in moved and any(
            pd.notna(df.at[r, c]) and _sig(df.at[r, c]) not in typ[c]
            and any(_sig(df.at[r, c]) in typ[o] for o in block if o != c) for c in block)]
        left_note = None
        if left:
            left_note = (f"{len(left)} more row(s) still have a value that looks like it belongs in another column "
                         f"(lines {', '.join(str(int(r) + 2) for r in left[:8])}{' …' if len(left) > 8 else ''}); they "
                         "could not be placed and were left as recorded, so a few category values may be misplaced.")
    else:
        left_note = None
    if not fixes:
        return df
    if len(fixes) < max(5, 0.01 * len(df)):
        notes.append(f"{len(fixes)} row(s) look like their values are in the wrong columns (to the left of where "
                     "they belong); they were left as recorded.")
        return df
    df = df.copy()
    emptied = {}
    for r, placed in fixes.items():
        df.loc[r, block] = np.nan
        for c, v in placed:
            df.at[r, c] = v
        for c in block:
            if pd.isna(df.at[r, c]):
                emptied[c] = emptied.get(c, 0) + 1
    rows = list(fixes)
    lines = ", ".join(str(int(r) + 2) for r in rows[:8])
    notes.append(f"{len(rows)} row(s) had values to the left of the column they belong in (a field was missing and "
                 f"the later fields had moved up; lines {lines}{' …' if len(rows) > 8 else ''}). Each value was put "
                 "back in the column whose values it matches; the fields those rows never had are missing ("
                 + ", ".join(f"{c}: {n}" for c, n in emptied.items()) + ").")
    if left_note:
        notes.append(left_note)
    df.attrs.setdefault("schema", {})["realigned"] = {"rows": [int(r) for r in rows]}
    return df


def parse_unit_counts(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Text columns whose values are numbers with a multiplier and a unit ("(26.3k Reviews)") become numbers. Needs
    at least 95% of the filled values to read this way with one unit, and a multiplier or a unit word to be present;
    the values that do not read are listed as missing, not guessed."""
    for c in list(df.columns):
        s = df[c]
        if s.dtype != object:
            continue
        v = s.dropna().astype(str).str.strip()
        v = v[v != ""]
        if len(v) < 5:
            continue
        m = v.str.extract(_COUNT.pattern, flags=re.I)
        ok = m["num"].notna()
        if ok.mean() < 0.95:
            continue
        units = m.loc[ok, "unit"].fillna("").str.strip().str.lower().str.rstrip("s")
        mult = m.loc[ok, "mult"].fillna("").str.lower()
        if not ((units != "").any() or (mult != "").any()):
            continue                              # plain numbers are read by app/ingest.py
        top = units.value_counts(normalize=True)
        if top.iloc[0] < 0.95:
            continue                              # several units ("5 km", "3 miles"): not one quantity
        num = m["num"].str.replace(",", "", regex=False).astype(float) * mult.map(lambda x: _MULT.get(x, 1.0)).reindex(
            m.index).fillna(1.0)
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[num.index[ok]] = num[ok]
        unit = m.loc[ok, "unit"].dropna().str.strip()
        unit = unit.value_counts().index[0] if len(unit) else ""
        bad = int(len(v) - ok.sum())
        df[c] = out
        has_mult = mult != ""
        ex_i = has_mult.idxmax() if has_mult.any() else v.index[0]
        notes.append(f"'{c}' held numbers written as text with a unit (e.g. '{v[ex_i]}' → {out[ex_i]:,.0f}"
                     + ("; k = 1,000, M = 1,000,000, Lakh = 100,000" if has_mult.any() else "") + "); read as numbers"
                     + (f" (unit: {unit})" if unit else "") + (f". {bad} value(s) did not read and are missing" if bad else "")
                     + ".")
        df.attrs.setdefault("schema", {}).setdefault("units", {})[c] = unit
    return df


def set_aside_row_index(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """A column with no real name whose values are the row positions (0, 1, 2, ... or 1, 2, 3, ...) is the index a
    spreadsheet or pandas saved with the file. It carries no information about the records, so it is set aside
    (kept in df.attrs["schema"]["row_index"]) instead of being averaged, correlated or offered as a measure."""
    out = []
    for c in list(df.columns):
        if not _GENERIC_INDEX_NAME.match(str(c).strip()):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().any() or len(x) < 3:
            continue
        pos = np.arange(len(x))
        if (x.to_numpy() == pos).mean() >= 0.99 or (x.to_numpy() == pos + 1).mean() >= 0.99:
            out.append(c)
    for c in out:
        df = df.drop(columns=[c])
        notes.append(f"'{c}' is the row number saved with the file (0, 1, 2, …), not data about the records; it was "
                     "set aside and is not analysed.")
    if out:
        df.attrs.setdefault("schema", {})["row_index"] = out
    return df


def apply(df: pd.DataFrame) -> pd.DataFrame:
    """The schema reading above, in order; notes go into the read report."""
    attrs = dict(df.attrs)
    report = attrs.get("ingestion") or {}
    notes = report.setdefault("notes", [])
    df = set_aside_row_index(df, notes)
    df = realign_shifted_rows(df, notes)
    df = parse_unit_counts(df, notes)
    schema = df.attrs.get("schema") or {}
    df.attrs = {**attrs, "ingestion": report, "schema": {**(attrs.get("schema") or {}), **schema}}
    return df


# ------------------------------------------------------------------ entities --
_NAME_TOKENS = {"name", "title", "label"}


def _tokens(col) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(col)).lower()) if t]


# words for the same kind of thing: "which firm" asks about the companies a table of companies names
_SYNONYMS = [{"company", "firm", "organisation", "organization", "employer", "corporation", "enterprise"}]


def entity_column(df: pd.DataFrame, noun: str, _alias: bool = True) -> str | None:
    """The column that names the things a question counts or ranks ("which company", "number of companies").

    A column called after the noun itself ("company", "company_name", "company_id") is it. Otherwise, when the
    noun appears only as the first word of other columns ("company_type", "Company_Age" -- attributes OF the
    company) and the table has one text column of names ("name", "title"), each record is one such thing and that
    column names it. None when nothing says so: the caller asks."""
    w = noun.lower().strip()
    sing = re.sub(r"(ies)$", "y", w) if w.endswith("ies") else (w[:-2] if w.endswith("ses") else w.rstrip("s"))
    forms = {w, sing}
    own = [c for c in df.columns if (lambda t: t and set(t) <= forms | {"name", "id"} and t[0] in forms)(_tokens(c))
           and not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if own:                               # a text column named after the things: its name before its id
        own.sort(key=lambda c: (0 if "name" in _tokens(c) else 1 if "id" not in _tokens(c) else 2))
        return own[0]
    attrs = [c for c in df.columns if len(_tokens(c)) > 1 and _tokens(c)[0] in forms]
    if not attrs:
        if _alias:
            for group in _SYNONYMS:
                if sing in group:
                    for other in sorted(group - {sing}):
                        hit = entity_column(df, other, _alias=False)
                        if hit is not None:
                            return hit
        return None
    names = [c for c in df.columns if set(_tokens(c)) & _NAME_TOKENS and df[c].dtype == object
             and df[c].nunique(dropna=True) >= 0.5 * max(1, df[c].notna().sum())]
    return names[0] if len(names) == 1 else None
