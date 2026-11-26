"""Reading an uploaded table correctly before anything analyses it.

`read_table(path)` detects, rather than assumes:
  encoding      UTF-8 (with or without BOM), UTF-16 (BOM), else Windows-1252 / Latin-1 -- reported when not UTF-8
  delimiter     , ; tab | -- the one that splits the lines into the same number of fields most consistently
  decimal mark  "3,5" in a ;-separated file is 3.5 -- detected from the values, reported
  header        a first row of numbers is data, not column names -- names are generated and reported
  bad lines     rows with a different number of fields are skipped and counted, never silently merged
  numbers stored as text ("$1,200", "12%", "1 200") -- converted when nearly all values parse, reported
  dates         text columns whose values are dates are parsed

and then validates the result: a single column whose name still contains another delimiter, no data rows, or
mostly-unparseable rows are problems, raised as IngestError with a plain explanation -- never a dataframe that only
looks loaded. The report (encoding, delimiter, notes) travels with the dataset profile so the user can see how the
file was read. Excel workbooks are read per sheet; a sheet whose header is not on the first row is located.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DELIMITERS = [",", ";", "\t", "|"]
_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
_NUMERIC_TEXT = re.compile(r"^\s*[-+(]?\s*[$€£₹¥]?\s*[-+]?\d[\d,.\s']*\s*%?\s*[)]?\s*$")


class IngestError(ValueError):
    """The file cannot be read reliably; the message says why, in plain words."""


def _decode(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8 (with BOM)"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "windows-1252"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _field_counts(lines: list[str], delim: str) -> list[int]:
    try:
        return [len(r) for r in csv.reader(lines, delimiter=delim, quotechar='"') if r]
    except csv.Error:
        return []


def sniff_delimiter(lines: list[str]) -> tuple[str, int, float]:
    """(delimiter, fields per row, share of rows with that many fields). Highest consistency wins; a delimiter that
    does not split the header into at least two fields is not a candidate."""
    best = (",", 1, 0.0)
    for d in DELIMITERS:
        counts = _field_counts(lines, d)
        if not counts or counts[0] < 2:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        share = freq / len(counts)
        if mode >= 2 and (share > best[2] + 1e-9 or (abs(share - best[2]) < 1e-9 and mode > best[1])):
            best = (d, mode, share)
    return best


def _is_number(s: str) -> bool:
    return bool(re.fullmatch(r"\s*[-+]?(\d+([.,]\d+)?|\.\d+)\s*", s))


def _to_number(series: pd.Series) -> tuple[pd.Series | None, str]:
    """Numbers written as text -> numbers, when nearly every value parses. (series or None, note)."""
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0 or not s.head(500).map(lambda v: bool(_NUMERIC_TEXT.match(v))).mean() >= 0.95 \
            or not s.map(lambda v: bool(_NUMERIC_TEXT.match(v))).mean() >= 0.95:
        return None, ""
    sample = s.iloc[0]

    def conv(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        t = str(v).strip()
        if t == "":
            return None
        neg = t.startswith("(") and t.endswith(")")
        t = re.sub(r"[()$€£₹¥%\s']", "", t)
        if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", t):        # 1.234,56 (European thousands)
            t = t.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", t):       # 1,234.56
            t = t.replace(",", "")
        elif re.fullmatch(r"[-+]?\d+,\d+", t):                        # 3,5 as a decimal
            t = t.replace(",", ".")
        try:
            x = float(t)
        except ValueError:
            return None
        return -x if neg else x
    out = series.map(conv)
    if out.notna().sum() < 0.95 * len(s):
        return None, ""
    # A value that does not parse but holds a digit (ICD codes V57, E909; '12 kg') is data, not a missing
    # marker like 'n/a' or '-': converting would silently blank it, so the column stays text.
    if s[out.loc[s.index].isna()].str.contains(r"\d").any():
        return None, ""
    kinds = []
    if re.search(r"[$€£₹¥]", sample):
        kinds.append("currency symbols removed")
    if "%" in sample:
        kinds.append("% kept as the number written (12% → 12)")
    if re.search(r"\d[,.\s']\d{3}\b", sample):
        kinds.append("thousands separators removed")
    return pd.to_numeric(out, errors="coerce"), (f"e.g. '{sample}'" + (f" — {', '.join(kinds)}" if kinds else ""))


def _parse_dates(s: pd.Series, name: str, notes: list[str]) -> pd.Series:
    """Dates as written. For a/b/yyyy the order is decided from the values (a part above 12 is the day); when every
    value fits both orders the ambiguity is reported, not hidden."""
    v = s.dropna().astype(str).str.strip()
    parts = v.str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
    dayfirst = False
    if parts[0].notna().mean() > 0.9:
        a, b = parts[0].astype(float), parts[1].astype(float)
        if (a > 12).any() and not (b > 12).any():
            dayfirst = True
        elif not (a > 12).any() and not (b > 12).any():
            notes.append(f"'{name}' dates such as '{v.iloc[0]}' fit both day/month and month/day order; read as "
                         "month/day. Say if they are day/month.")
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")


def _looks_like_dates(s: pd.Series) -> bool:
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""].head(50)
    if len(v) < 2 or not v.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}", regex=True).mean() > 0.9:
        return False
    return max(pd.to_datetime(v, errors="coerce", format="mixed").notna().mean(),
               pd.to_datetime(v, errors="coerce", format="mixed", dayfirst=True).notna().mean()) > 0.9


def _clean_columns(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    empty = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        notes.append(f"{len(empty)} empty column(s) with no name (e.g. from a trailing delimiter) were dropped.")
    names, seen = [], Counter()
    for i, c in enumerate(df.columns, 1):
        n = re.sub(r"\s+", " ", str(c)).strip()
        if not n or n.startswith("Unnamed:"):
            n = f"column_{i}"
            notes.append(f"Column {i} had no name; it is called {n}.")
        seen[n] += 1
        if seen[n] > 1:
            notes.append(f"Two columns were named '{n}'; the second is called '{n}_{seen[n]}'.")
            n = f"{n}_{seen[n]}"
        names.append(n)
    df.columns = names
    blank_rows = df.isna().all(axis=1)
    if blank_rows.any():
        df = df[~blank_rows]
        notes.append(f"{int(blank_rows.sum())} completely empty row(s) were ignored.")
    return df


def _types(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        num, how = _to_number(s)
        if num is not None and not re.search(r"(^|_)(id|code|zip|postcode|phone)$", str(c), re.I):
            df[c] = num
            if how and not s.dropna().astype(str).str.fullmatch(r"\s*[-+]?\d+(\.\d+)?\s*").all():
                notes.append(f"'{c}' held numbers written as text ({how}); read as numbers.")
            continue
        if _looks_like_dates(s) or (re.search(r"date|month|week|period|time|day|year", str(c), re.I)
                                    and pd.to_datetime(s.dropna(), errors="coerce", format="mixed").notna().mean() > 0.9
                                    and not s.dropna().astype(str).str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?").any()):
            parsed = _parse_dates(s, str(c), notes)
            filled = s.notna() & (s.astype(str).str.strip() != "")
            if filled.any() and parsed[filled].notna().mean() > 0.9:        # blanks are missing dates, not failures
                df[c] = parsed
    return df


def read_csv_text(text: str, encoding: str, name: str = "file") -> tuple[pd.DataFrame, dict]:
    report = {"format": "csv", "encoding": encoding, "notes": [], "skipped_lines": 0}
    notes = report["notes"]
    lines = [ln for ln in text.splitlines() if ln.strip()][:400]
    if not lines:
        raise IngestError(f"{name} is empty: it has no rows.")
    delim, fields, share = sniff_delimiter(lines)
    report.update(delimiter=_NAMES[delim], columns_per_row=fields, consistency=round(share, 3))
    if fields < 2:
        other = [n for d, n in _NAMES.items() if d in lines[0]]
        if other:
            raise IngestError(f"{name}: the header contains {', '.join(other)} characters, but no delimiter splits the "
                              "rows consistently, so the columns can't be told apart reliably. Check the file's "
                              "separator (comma, semicolon, tab or pipe) and quoting.")
        notes.append("The file has a single column.")
    elif share < 0.6:
        raise IngestError(f"{name}: the rows do not have a consistent number of fields with any common delimiter "
                          f"(best: {_NAMES[delim]}, only {share:.0%} of rows agree). It may not be a table.")
    if encoding not in ("utf-8", "utf-8 (with BOM)"):
        notes.append(f"Text encoding detected as {encoding} (not UTF-8).")
    if delim != ",":
        notes.append(f"Columns are separated by {_NAMES[delim]}s, not commas.")
    body = list(csv.reader(lines[1:60], delimiter=delim))
    cells = [x.strip() for r in body for x in r]
    decimal = "."
    if delim != "," and cells and sum(bool(re.fullmatch(r"-?\d+,\d+", x)) for x in cells) >= max(2, 0.1 * len(cells)) \
            and not any(re.fullmatch(r"-?\d+\.\d+", x) for x in cells):
        decimal = ","
        notes.append("Numbers use a decimal comma (3,5 = 3.5).")
    report["decimal"] = decimal
    head = next(csv.reader([lines[0]], delimiter=delim))
    header = 0
    if head and all(_is_number(x) or not x.strip() for x in head) and any(_is_number(x) for x in head):
        header = None
        notes.append("The first row holds numbers, not column names, so it was read as data and the columns were "
                     "named column_1, column_2, …")
    bad: list = []
    df = pd.read_csv(io.StringIO(text), sep=delim, decimal=decimal, header=header, engine="python",
                     skipinitialspace=True, on_bad_lines=lambda row: bad.append(row) and None)
    if header is None:
        df.columns = [f"column_{i}" for i in range(1, df.shape[1] + 1)]
    if bad:
        report["skipped_lines"] = len(bad)
        notes.append(f"{len(bad)} row(s) had a different number of fields from the header and were skipped (not "
                     "guessed at).")
        if len(bad) > 0.2 * max(1, len(df) + len(bad)):
            raise IngestError(f"{name}: {len(bad)} of {len(df) + len(bad)} rows don't match the header's columns, "
                              "so the table can't be read reliably.")
    if fields >= 2:
        short = [i for i, r in enumerate(csv.reader(io.StringIO(text), delimiter=delim), 1)
                 if r and any(x.strip() for x in r) and i > (1 if header == 0 else 0) and len(r) < fields]
        if short:
            report["short_lines"] = short[:20]
            notes.append(f"{len(short)} row(s) had fewer fields than the header (line{'s' if len(short) > 1 else ''} "
                         f"{', '.join(map(str, short[:10]))}{' …' if len(short) > 10 else ''}); their missing fields "
                         "are left empty, not filled in. Check them if they look malformed.")
    df = _clean_columns(df, notes)
    if df.empty:
        raise IngestError(f"{name} has column names but no data rows.")
    df = _types(df, notes)
    report.update(rows=int(len(df)), columns=int(df.shape[1]))
    return df, report


def read_table(path: str, sheet: str | int | None = None) -> tuple[pd.DataFrame, dict]:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        report = {"format": "excel", "notes": [], "skipped_lines": 0}
        xf = pd.ExcelFile(p)
        report["sheets"] = xf.sheet_names
        df = pd.read_excel(xf, sheet_name=sheet if sheet is not None else 0)
        unnamed = sum(str(c).startswith("Unnamed:") for c in df.columns)
        if len(df.columns) and unnamed / len(df.columns) > 0.5:
            # a title or blank rows above the real header: use the first row that is mostly filled with text
            raw = pd.read_excel(xf, sheet_name=sheet if sheet is not None else 0, header=None)
            for i in range(min(20, len(raw))):
                row = raw.iloc[i]
                if row.notna().mean() > 0.6 and row.dropna().map(lambda v: isinstance(v, str)).mean() > 0.8:
                    df = pd.read_excel(xf, sheet_name=sheet if sheet is not None else 0, header=i)
                    report["notes"].append(f"The header is on row {i + 1} of the sheet (rows above it were titles or "
                                           "blank).")
                    break
        if len(xf.sheet_names) > 1:
            report["notes"].append(f"The workbook has {len(xf.sheet_names)} sheets; this dataset is "
                                   f"'{sheet if isinstance(sheet, str) else xf.sheet_names[sheet or 0]}'.")
        df = _clean_columns(df, report["notes"])
        if df.empty:
            raise IngestError(f"{p.name}: the sheet has no data rows.")
        df = _types(df, report["notes"])
        report.update(rows=int(len(df)), columns=int(df.shape[1]))
        return df, report
    text, enc = _decode(p.read_bytes())
    return read_csv_text(text, enc, p.name.split("_", 1)[-1] if re.match(r"[0-9a-f]{32}_", p.name) else p.name)
