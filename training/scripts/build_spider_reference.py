"""Build the offline reference set for Spider's single-table validation questions.

For every validation question whose reference SQL reads one table and has no
nested query, run that SQL read-only against the official SQLite database,
record the result, and decide whether the case can be scored fairly:

  - does the query run, twice, with the same result?
  - is the result empty?
  - with ORDER BY ... LIMIT, is there a tie at the cut-off?
  - with GROUP BY, does it select a column that is neither grouped nor aggregated?
  - does the result survive exporting the table to CSV and loading it back?
  - is the answer the same when the table is typed the way pandas reads it?
    (Spider declares many numeric columns as text; SQLite then sorts and
    compares them as strings, and a tool that reads them as numbers is right
    to disagree.)

Nothing here calls a model, imports application code, or writes outside
training/data/eval/. Standard library plus pandas (already a project
dependency, used only for the last check).

    python training/scripts/build_spider_reference.py

Inputs (see training/data/raw/spider_official/MANIFEST.json):
    training/data/raw/spider_official/spider_data/{dev.json,tables.json,database/<db>/<db>.sqlite}
Outputs, in training/data/eval/spider_dev_single_table/:
    tables/<db>__<table>.csv     one CSV per table a question reads   (ignored by Git)
    reference_results.jsonl      one line per distinct reference query (ignored by Git)
    cases.jsonl                  one line per question                 (ignored by Git)
    BUILD_MANIFEST.json          counts, input hashes, tool versions   (tracked)
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import platform
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPIDER = ROOT / "training/data/raw/spider_official/spider_data"
OUT = ROOT / "training/data/eval/spider_dev_single_table"
QUERY_SECONDS = 20.0          # a reference query that runs longer is recorded as a timeout
MAX_ROWS = 50_000             # rows fetched per query; more is recorded as truncated
ROWS_STORED = 200             # rows written to reference_results.jsonl (the hash covers all of them)
FLOAT_DIGITS = 6
PANDAS_NA_STRINGS = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                     "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}
SELECT_ONLY = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


# ------------------------------------------------------------ classification --

def classify(toks: list[str]) -> dict:
    """What the reference SQL needs, read from its tokens."""
    toks = [t.lower() for t in toks]
    q = " ".join(toks)
    tables = [toks[i + 1] for i, t in enumerate(toks) if t in ("from", "join") and i + 1 < len(toks) and toks[i + 1] != "("]
    setops = sum(1 for t in toks if t in ("intersect", "union", "except"))
    join = "join" in toks or bool(re.search(r"\bfrom \w+ (?:as \w+ )?, ", q))
    out = {"tables": sorted(set(tables)), "join": join, "setop": setops > 0, "subquery": toks.count("select") > 1 + setops,
           "where": "where" in toks, "group_by": "group by" in q, "having": "having" in toks, "order_by": "order by" in q,
           "limit": "limit" in toks, "distinct": "distinct" in toks}
    if len(out["tables"]) != 1 or join:
        out["category"] = "3_unsupported" if out["tables"] else "4_uncertain"
    elif out["subquery"] or out["setop"]:
        out["category"] = "2_after_conversion"
    else:
        out["category"] = "1_supported"
    return out


def official_single_table(sql: dict) -> bool:
    """The same question asked of Spider's own parse of the query."""
    def nested(node) -> bool:
        if isinstance(node, dict):
            return True
        return isinstance(node, (list, tuple)) and any(nested(x) for x in node)
    units = sql["from"]["table_units"]
    return (len(units) == 1 and units[0][0] == "table_unit" and not any(sql[k] for k in ("intersect", "union", "except"))
            and not nested(sql["where"]) and not nested(sql["having"]) and not nested(sql["from"]["conds"]))


def answer_shape(sql: dict) -> str:
    items = sql["select"][1]
    all_aggregate = bool(items) and all(agg != 0 or unit[1][0] != 0 for agg, unit in items)
    if all_aggregate and not sql["groupBy"]:
        return "scalar" if len(items) == 1 else "scalar_row"
    if sql["orderBy"] and sql["limit"] is not None:
        return "top_n"
    return "grouped_table" if sql["groupBy"] else "row_list"


# ------------------------------------------------------------------ sqlite --

def _text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def connect(path: Path, select_only: bool = True) -> sqlite3.Connection:
    """Read-only. Reference SQL is third-party text, so the connection it runs
    on is also limited to SELECT; `select_only=False` is for this script's own
    schema reads (PRAGMA table_info), on a connection no reference SQL touches."""
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.text_factory = _text
    if select_only:
        con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in SELECT_ONLY else sqlite3.SQLITE_DENY)
    return con


def run(con: sqlite3.Connection, sql: str) -> dict:
    deadline = time.monotonic() + QUERY_SECONDS
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS + 1)
        return {"ok": True, "columns": [d[0] for d in cur.description or []], "rows": [tuple(r) for r in rows[:MAX_ROWS]],
                "truncated": len(rows) > MAX_ROWS}
    except sqlite3.OperationalError as e:
        return {"ok": False, "error": "timeout" if "interrupted" in str(e) else f"OperationalError: {e}"}
    except sqlite3.Error as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        con.set_progress_handler(None, 0)


def canon(value):
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS) + 0.0
    if isinstance(value, bytes):
        return "0x" + value.hex()
    return value


def canon_rows(rows, ordered: bool) -> list[list]:
    out = [[canon(v) for v in r] for r in rows]
    if not ordered:
        out.sort(key=lambda r: json.dumps(r, sort_keys=True, default=str))
    return out


def digest(rows: list[list]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str).encode()).hexdigest()


# -------------------------------------------------------------- ambiguity --

_ORDER = re.compile(r"\s+order\s+by\s+(?P<keys>.+?)(?:\s+limit\s+(?P<limit>\d+))?\s*;?\s*$", re.IGNORECASE | re.DOTALL)


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return [p.strip() for p in parts + [text[start:]]]


def order_ties(con: sqlite3.Connection, query: str, n_select: int) -> dict:
    """With ORDER BY: is the result decided by the sort keys alone?

    Re-runs the query without LIMIT and with its sort keys added to the select
    list, then looks at rows whose keys are equal. A tie that straddles the
    LIMIT cut-off, or (without LIMIT) any tie between rows that differ, means
    SQLite's answer is one of several correct ones."""
    m = _ORDER.search(query)
    if not m:
        return {"checked": False, "why": "the ORDER BY clause could not be read"}
    if re.match(r"\s*select\s+distinct\b", query, re.IGNORECASE):
        return {"checked": False, "why": "SELECT DISTINCT: adding the sort keys would change which rows are distinct"}
    keys = [re.sub(r"\s+(?:asc|desc)\s*$", "", k, flags=re.IGNORECASE) for k in _split_top_level(m.group("keys"))]
    core = query[:m.start()]
    cut = re.search(r"\sfrom\s", core, re.IGNORECASE)
    if not cut:
        return {"checked": False, "why": "no FROM clause found"}
    keyed = (core[:cut.start()] + ", " + ", ".join(f"({k}) AS __key{i}" for i, k in enumerate(keys)) + core[cut.start():]
             + " ORDER BY " + m.group("keys"))
    out = run(con, keyed)
    if not out["ok"]:
        return {"checked": False, "why": f"the keyed query failed ({out['error']})"}
    rows = [([canon(v) for v in r[:n_select]], [canon(v) for v in r[n_select:]]) for r in out["rows"]]
    limit = int(m.group("limit")) if m.group("limit") else None
    if limit is not None:
        if limit == 0 or len(rows) <= limit:
            return {"checked": True, "tie_at_limit": False, "order_ties": False}
        boundary = rows[limit - 1][1]
        if rows[limit][1] != boundary:
            return {"checked": True, "tie_at_limit": False, "order_ties": False}
        tied = {json.dumps(sel, default=str) for sel, key in rows if key == boundary}
        return {"checked": True, "tie_at_limit": len(tied) > 1, "order_ties": False,
                "tied_rows": sum(1 for _, key in rows if key == boundary), "distinct_answers_in_tie": len(tied)}
    ties = any(a[1] == b[1] and a[0] != b[0] for a, b in zip(rows, rows[1:]))
    return {"checked": True, "tie_at_limit": False, "order_ties": ties}


def bare_group_columns(con: sqlite3.Connection, sql: dict, schema: dict) -> list[str]:
    """Columns selected next to a GROUP BY that are neither grouped nor
    aggregated, and that take more than one value inside some group. SQLite
    returns one of those values arbitrarily; another engine may return another."""
    if not sql["groupBy"]:
        return []
    grouped = {unit[1] for unit in sql["groupBy"]}
    names = schema["column_names_original"]
    table = schema["table_names_original"][sql["from"]["table_units"][0][1]]
    group_sql = ", ".join(f'"{names[c][1]}"' for c in sorted(grouped) if c > 0)
    out = []
    for agg, unit in sql["select"][1]:
        inner_agg, col, _ = unit[1]
        if agg != 0 or inner_agg != 0 or col in grouped or col <= 0 or not group_sql:
            continue
        probe = run(con, f'SELECT 1 FROM "{table}" GROUP BY {group_sql} HAVING count(DISTINCT "{names[col][1]}") > 1 LIMIT 1')
        if not probe["ok"] or probe["rows"]:
            out.append(names[col][1])
    return out


# --------------------------------------------------------------- csv check --

def export_table(con: sqlite3.Connection, db: str, table: str) -> dict:
    """One table to CSV. NULL and the empty string both become an empty field:
    CSV has no way to tell them apart, which is one of the things checked below."""
    info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    cols = [(r[1], r[2] or "") for r in info]
    path = OUT / "tables" / f"{db}__{table}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
    stats = {c: collections.Counter() for c, _ in cols}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c for c, _ in cols])
        for row in rows:
            out = []
            for (name, _), v in zip(cols, row):
                s = stats[name]
                if v is None:
                    s["null"] += 1
                    out.append("")
                    continue
                text = v.hex() if isinstance(v, bytes) else repr(v) if isinstance(v, float) else str(v)
                if isinstance(v, str):
                    s["text"] += 1
                    s["empty_string"] += v == ""
                    s["pandas_would_read_as_missing"] += v in PANDAS_NA_STRINGS and v != ""
                    s["number_like_text_that_changes"] += bool(re.fullmatch(r"\s*[+-]?(?:0\d+|\d+\.\d*0|\.\d+|\d+\.)\s*", v))
                    s["undecodable_bytes"] += "�" in v
                out.append(text)
            writer.writerow(out)
    risky = {c: {k: n for k, n in s.items() if n and k in ("empty_string", "pandas_would_read_as_missing",
                                                              "number_like_text_that_changes", "undecodable_bytes")}
             for c, s in stats.items()}
    return {"db": db, "table": table, "path": str(path.relative_to(ROOT)), "rows": len(rows), "columns": cols,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "risky_columns": {c: r for c, r in risky.items() if r}}


def rebuild_from_csv(table_info: dict) -> sqlite3.Connection:
    """An in-memory database holding only this table, loaded back from its CSV
    with the original declared column types. Empty fields become NULL."""
    con = sqlite3.connect(":memory:")
    con.text_factory = _text
    cols = table_info["columns"]
    con.execute(f'CREATE TABLE "{table_info["table"]}" (' + ", ".join(f'"{c}" {t}' for c, t in cols) + ")")
    with open(ROOT / table_info["path"], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        con.executemany(f'INSERT INTO "{table_info["table"]}" VALUES (' + ", ".join("?" * len(cols)) + ")",
                        ([None if v == "" else v for v in row] for row in reader))
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in SELECT_ONLY else sqlite3.SQLITE_DENY)
    return con


def pandas_view(table_info: dict) -> dict:
    """What pandas.read_csv makes of the CSV with default settings, which is
    how the application loads an upload: per column, the dtype it infers."""
    try:
        import pandas as pd
    except ImportError:
        return {"checked": False}
    df = pd.read_csv(ROOT / table_info["path"])
    declared = dict(table_info["columns"])
    changed = {}
    for col in df.columns:
        kind = "number" if pd.api.types.is_numeric_dtype(df[col]) else "text"
        was_text = bool(re.search(r"char|text|clob", declared.get(col, ""), re.IGNORECASE))
        if was_text and kind == "number":
            changed[col] = "declared text, read as a number"
    return {"checked": True, "rows": int(len(df)), "columns_read_differently": changed}


def rebuild_as_pandas(table_info: dict) -> sqlite3.Connection | None:
    """The table as pandas.read_csv types it, back in SQLite: a column pandas
    reads as numbers becomes numeric, and the strings it treats as missing
    ("NA", "null", ...) become NULL. Running the reference SQL here shows
    whether its answer depends on Spider's text typing."""
    try:
        import pandas as pd
    except ImportError:
        return None
    df = pd.read_csv(ROOT / table_info["path"])
    kinds = ["INTEGER" if pd.api.types.is_integer_dtype(df[c]) else "REAL" if pd.api.types.is_numeric_dtype(df[c]) else "TEXT"
             for c in df.columns]
    con = sqlite3.connect(":memory:")
    con.execute(f'CREATE TABLE "{table_info["table"]}" (' + ", ".join(f'"{c}" {k}' for c, k in zip(df.columns, kinds)) + ")")
    values = df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)
    con.executemany(f'INSERT INTO "{table_info["table"]}" VALUES (' + ", ".join("?" * len(kinds)) + ")",
                    ([v.item() if hasattr(v, "item") else v for v in row] for row in values))
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in SELECT_ONLY else sqlite3.SQLITE_DENY)
    return con


def loose(rows, ordered: bool) -> list[list]:
    """Rows with representation differences removed: "2015", 2015 and 2015.0
    are one value, and so are NULL and the empty string. What is left when two
    results still differ is a difference in the answer."""
    def value(v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return round(float(v), FLOAT_DIGITS) + 0.0
        if isinstance(v, str):
            try:
                return round(float(v.strip()), FLOAT_DIGITS) + 0.0
            except ValueError:
                return v.strip()
        return canon(v)
    out = [[value(v) for v in r] for r in rows]
    if not ordered:
        out.sort(key=lambda r: json.dumps(r, sort_keys=True, default=str))
    return out


# ------------------------------------------------------------------- build --

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def train_overlap() -> tuple[set[str], set[str]] | None:
    """Training questions and database names, to mark validation cases that
    are not cleanly held out. Read from the Hugging Face training file when it
    and pyarrow are present; the markers are omitted otherwise."""
    path = ROOT / "training/data/raw/huggingface/spider/train-00000-of-00001.parquet"
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=["db_id", "question"]).to_pydict()
    except Exception:  # noqa: BLE001 - an optional input
        return None
    family = lambda name: re.sub(r"_\d+$", "", name.lower())
    return ({re.sub(r"\s+", " ", q.strip().lower()) for q in table["question"]}, {family(d) for d in table["db_id"]})


def main() -> int:
    dev = json.loads((SPIDER / "dev.json").read_text())
    schemas = {t["db_id"]: t for t in json.loads((SPIDER / "tables.json").read_text())}
    OUT.mkdir(parents=True, exist_ok=True)

    cases, by_query, coverage, disagreements = [], collections.OrderedDict(), collections.Counter(), []
    for index, ex in enumerate(dev):
        c = classify(ex["query_toks"])
        coverage[c["category"]] += 1
        official = official_single_table(ex["sql"])
        if official != (c["category"] == "1_supported"):
            disagreements.append(index)
        case = {"case_id": f"spider-dev-{index:04d}", "dev_index": index, "db_id": ex["db_id"], "question": ex["question"],
                "query": ex["query"], "category": c["category"], "official_parse_single_table": official}
        if c["category"] == "1_supported" and official:
            key = (ex["db_id"], norm_sql(ex["query"]))
            ref = by_query.setdefault(key, {"reference_id": f"ref-{len(by_query):04d}", "db_id": ex["db_id"], "query": ex["query"],
                                            "table": c["tables"][0], "sql": ex["sql"], "features": c, "case_ids": []})
            ref["case_ids"].append(case["case_id"])
            case["reference_id"] = ref["reference_id"]
        cases.append(case)

    tables, connections = {}, {}
    for ref in by_query.values():
        db = ref["db_id"]
        if db not in connections:
            connections[db] = connect(SPIDER / "database" / db / f"{db}.sqlite")
        con = connections[db]
        real = {r[0].lower(): r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        ref["table"] = real.get(ref["table"].lower(), ref["table"])
        if (db, ref["table"]) not in tables:
            info = export_table(connect(SPIDER / "database" / db / f"{db}.sqlite", select_only=False), db, ref["table"])
            info["pandas"] = pandas_view(info)
            tables[(db, ref["table"])] = info

    typed: dict[tuple, sqlite3.Connection | None] = {}
    verdicts = collections.Counter()
    with open(OUT / "reference_results.jsonl", "w", encoding="utf-8") as out:
        for ref in by_query.values():
            db, sql, con = ref["db_id"], ref["sql"], connections[ref["db_id"]]
            ordered = bool(sql["orderBy"])
            first = run(con, ref["query"])
            again = run(connect(SPIDER / "database" / db / f"{db}.sqlite"), ref["query"])
            record = {"reference_id": ref["reference_id"], "db_id": db, "table": ref["table"], "query": ref["query"],
                      "case_ids": ref["case_ids"], "paraphrases": len(ref["case_ids"]), "answer_shape": answer_shape(sql),
                      "ordered": ordered, "features": {k: v for k, v in ref["features"].items() if k not in ("tables", "category")}}
            flags: list[str] = []
            if not first["ok"]:
                record["error"] = first["error"]
                flags.append("timeout" if first["error"] == "timeout" else "execution_error")
            else:
                rows = canon_rows(first["rows"], ordered)
                record.update(columns=first["columns"], row_count=len(rows), rows=rows[:ROWS_STORED],
                              rows_stored=min(len(rows), ROWS_STORED), result_sha256=digest(rows))
                if first["truncated"]:
                    flags.append("truncated")
                if not again["ok"] or digest(canon_rows(again["rows"], ordered)) != record["result_sha256"]:
                    flags.append("not_reproducible")
                if not rows:
                    flags.append("empty_result")
                if ordered:
                    ties = order_ties(con, ref["query"], len(first["columns"]))
                    record["order_check"] = ties
                    if not ties["checked"]:
                        flags.append("order_not_checked")
                    elif ties["tie_at_limit"]:
                        flags.append("tie_at_limit")
                    # Ties without a LIMIT leave the set of rows fixed and only their order open:
                    # such a case is scored as a set, not excluded.
                    record["compare"] = "ordered" if ties.get("checked") and not ties.get("order_ties") else "set"
                else:
                    record["compare"] = "set"
                bare = bare_group_columns(con, sql, schemas[db])
                if bare:
                    record["bare_group_columns"] = bare
                    flags.append("ambiguous_group_column")
                # the same query on a database rebuilt from the exported CSV
                rebuilt = run(rebuild_from_csv(tables[(db, ref["table"])]), ref["query"])
                same = rebuilt["ok"] and digest(canon_rows(rebuilt["rows"], ordered)) == record["result_sha256"]
                if not same and rebuilt["ok"] and ordered and record["compare"] == "set":
                    same = digest(canon_rows(rebuilt["rows"], False)) == digest(canon_rows(first["rows"], False))
                record["csv_round_trip"] = "same" if same else ("error: " + rebuilt["error"] if not rebuilt["ok"] else "different")
                if not same:
                    flags.append("csv_round_trip_differs")
                # ... and on the table typed as pandas reads it
                if (db, ref["table"]) not in typed:
                    typed[(db, ref["table"])] = rebuild_as_pandas(tables[(db, ref["table"])])
                if typed[(db, ref["table"])] is None:
                    record["pandas_typed"] = "not checked"
                else:
                    strict = record["compare"] == "ordered"
                    as_pandas = run(typed[(db, ref["table"])], ref["query"])
                    agrees = as_pandas["ok"] and loose(as_pandas["rows"], strict) == loose(first["rows"], strict)
                    record["pandas_typed"] = "same" if agrees else ("error: " + as_pandas["error"] if not as_pandas["ok"] else "different")
                    if not agrees and same:
                        flags.append("answer_depends_on_text_typing")
                info = tables[(db, ref["table"])]
                used = [c for c in list(info["risky_columns"]) + list(info["pandas"].get("columns_read_differently", {}))
                        if re.search(rf"(?<![\w]){re.escape(c)}(?![\w])", ref["query"], re.IGNORECASE)]
                if used:
                    record["data_fidelity_risk"] = sorted(set(used))
            record["flags"] = flags
            record["valid_for_scoring"] = not flags
            verdicts["valid" if not flags else flags[0]] += 1
            ref["record"] = record
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    by_id = {ref["reference_id"]: ref["record"] for ref in by_query.values()}
    case_verdicts, shapes, fidelity = collections.Counter(), collections.Counter(), 0
    overlap, held_out = train_overlap(), collections.Counter()
    with open(OUT / "cases.jsonl", "w", encoding="utf-8") as out:
        for case in cases:
            rec = by_id.get(case.get("reference_id"))
            if rec is not None:
                case.update(valid_for_scoring=rec["valid_for_scoring"], flags=rec["flags"], answer_shape=rec["answer_shape"],
                            compare=rec.get("compare"), csv=tables[(rec["db_id"], rec["table"])]["path"],
                            data_fidelity_risk=rec.get("data_fidelity_risk", []))
                case_verdicts["valid" if rec["valid_for_scoring"] else rec["flags"][0]] += 1
                if overlap is not None:
                    questions, families = overlap
                    case["leakage"] = [name for name, hit in (
                        ("question_also_in_train", re.sub(r"\s+", " ", case["question"].strip().lower()) in questions),
                        ("database_family_in_train", re.sub(r"_\d+$", "", case["db_id"].lower()) in families)) if hit]
                    if rec["valid_for_scoring"]:
                        for name in case["leakage"]:
                            held_out[name] += 1
                        held_out["cleanly_held_out"] += not case["leakage"]
                if rec["valid_for_scoring"]:
                    shapes[rec["answer_shape"]] += 1
                    fidelity += bool(rec.get("data_fidelity_risk"))
            out.write(json.dumps(case, ensure_ascii=False) + "\n")

    valid_refs = [r for r in by_id.values() if r["valid_for_scoring"]]
    manifest = {
        "built_by": "training/scripts/build_spider_reference.py",
        "tools": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version},
        "limits": {"query_seconds": QUERY_SECONDS, "max_rows": MAX_ROWS, "rows_stored": ROWS_STORED, "float_digits": FLOAT_DIGITS},
        "inputs": {name: sha256(SPIDER / name) for name in ("dev.json", "tables.json")},
        "databases": {db: sha256(SPIDER / "database" / db / f"{db}.sqlite") for db in sorted(connections)},
        "validation_questions": len(dev), "coverage": dict(sorted(coverage.items())),
        "classifier_disagrees_with_official_parse": disagreements,
        "reference_queries": {"executed": len(by_id), "by_verdict": dict(verdicts.most_common()), "valid": len(valid_refs),
                              "valid_compared_in_order": sum(1 for r in valid_refs if r["compare"] == "ordered"),
                              "valid_compared_as_set": sum(1 for r in valid_refs if r["compare"] == "set")},
        "questions": {"single_table": sum(case_verdicts.values()), "by_verdict": dict(case_verdicts.most_common()),
                      "valid": case_verdicts["valid"], "valid_by_answer_shape": dict(shapes.most_common()),
                      "valid_with_data_fidelity_risk": fidelity,
                      "valid_by_leakage_marker": dict(held_out) if overlap is not None else "not computed"},
        "tables": [{k: v for k, v in info.items()} for info in sorted(tables.values(), key=lambda i: (i["db"], i["table"]))],
        "outputs": {name: sha256(OUT / name) for name in ("reference_results.jsonl", "cases.jsonl")},
    }
    (OUT / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("validation_questions", "coverage", "reference_queries", "questions")}, indent=1))
    print("classifier vs official parse disagreements:", len(disagreements))
    print("tables exported:", len(tables), "| total rows:", sum(t["rows"] for t in tables.values()),
          "| largest:", max(((t["rows"], f"{t['db']}.{t['table']}") for t in tables.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
