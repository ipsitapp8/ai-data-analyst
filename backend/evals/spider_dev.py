"""Development set for the copilot's question reading, from Spider's TRAINING split.

The held-out evaluation (backend/evals/cases/spider_numeric.json, Spider's
validation split) is never read here. Spider's training and validation splits
share no database.

The training databases are not available locally (only the validation ones
are), so each training table is rebuilt as a small synthetic CSV from its
schema in tables.json: the real column names and types, with every literal the
selected training queries compare against included as a real value. The data
is invented, so a development case is scored on the *reading*, not the number:
does the copilot turn the question into the same formula(s) and filters that
the reference SQL states?

    spec_match  the copilot produced the reference spec
    abstain     the copilot declined (not understood, clarification, no metric)
    wrong_spec  the copilot answered with a different spec   <- must stay rare
    error       something raised

Split: databases are assigned to "tune" (about 70%) or "check" (about 30%) by a
hash of their name. Rules are developed by looking at "tune" cases only;
"check" is read only as aggregate figures, to see whether a change generalises.

    cd backend
    python evals/spider_dev.py build
    python evals/spider_dev.py score --label NAME [--split tune|check|all] [--show N] [--outcome abstain]

Writes only under training/data/eval/spider_train_dev/. No model is called.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
TRAIN = ROOT / "training/data/raw/huggingface/spider/train-00000-of-00001.parquet"
TABLES = ROOT / "training/data/raw/spider_official/spider_data/tables.json"
OUT = ROOT / "training/data/eval/spider_train_dev"
ROWS = 30
AGGS = {"count", "sum", "avg", "min", "max"}
TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|>=|<=|!=|<>|[A-Za-z_][\w.]*|-?\d+(?:\.\d+)?|[(),*=<>;%]")


class Unsupported(Exception):
    pass


# --------------------------------------------------------------- SQL subset --

def parse(query: str) -> dict:
    """SELECT agg(..)[, agg(..)] FROM t [WHERE cond ((AND|OR) cond)*] -> spec
    with column names as written (resolved against the schema later)."""
    toks = TOKEN.findall(query.strip())
    low = [t.lower() for t in toks]
    i = 0

    def need(word):
        nonlocal i
        if i >= len(low) or low[i] != word:
            raise Unsupported(f"expected {word!r} at {i}")
        i += 1

    need("select")
    formulas = []
    while True:
        if i >= len(low) or low[i] not in AGGS:
            raise Unsupported("select item is not an aggregation")
        fn = low[i]
        i += 1
        need("(")
        distinct = False
        if low[i] == "distinct":
            distinct, i = True, i + 1
        col = toks[i]
        i += 1
        need(")")
        if col == "*":
            if fn != "count" or distinct:
                raise Unsupported("aggregate over *")
            formulas.append(("count", None))
        else:
            if "." in col:
                col = col.split(".", 1)[1]
            formulas.append(("count_distinct" if distinct else fn, col))
            if distinct and fn != "count":
                raise Unsupported("distinct inside a non-count aggregate")
        if i < len(low) and low[i] == ",":
            i += 1
            continue
        break
    need("from")
    table = toks[i]
    i += 1
    if i < len(low) and low[i] == "as":
        i += 2
    conds, joins = [], []
    if i < len(low) and low[i] == "where":
        i += 1
        while True:
            negate = False
            if low[i] == "not":
                negate, i = True, i + 1
            col = toks[i].split(".", 1)[-1]
            i += 1
            op = low[i]
            i += 1
            if op == "not":
                negate, op, i = True, low[i], i + 1
            if op == "between":
                a, _, b = toks[i], need_and(low, i + 1), toks[i + 2]
                i += 3
                conds.append({"column": col, "op": "between", "value": [lit(a), lit(b)], "negate": negate})
            elif op == "like":
                conds.append({"column": col, "op": "like", "value": lit(toks[i]), "negate": negate})
                i += 1
            elif op in ("=", ">", "<", ">=", "<=", "!=", "<>"):
                conds.append({"column": col, "op": {"=": "==", "<>": "!="}.get(op, op), "value": lit(toks[i]), "negate": negate})
                i += 1
            else:
                raise Unsupported(f"condition operator {op}")
            if i < len(low) and low[i] in ("and", "or"):
                joins.append(low[i])
                i += 1
                continue
            break
    if i < len(low) and low[i] == ";":
        i += 1
    if i != len(low):
        raise Unsupported("trailing clauses")
    return {"table": table, "formulas": formulas, "conds": conds, "joins": joins}


def need_and(low, j):
    if low[j] != "and":
        raise Unsupported("BETWEEN without AND")
    return "and"


def lit(tok: str):
    if tok[:1] in "'\"":
        return tok[1:-1]
    try:
        return float(tok)
    except ValueError:
        raise Unsupported(f"value {tok!r} is not a literal") from None


_NEG = {"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}


def to_spec(p: dict, columns: dict[str, str]) -> dict:
    """Engine formulas and filters, with schema column names."""
    def col(name):
        if name.lower() not in columns:
            raise Unsupported(f"column {name} not in table")
        return columns[name.lower()]
    formulas = ["count()" if c is None else f'{fn}("{col(c)}")' for fn, c in p["formulas"]]
    filters = []
    for c in p["conds"]:
        name, op, value = col(c["column"]), c["op"], c["value"]
        if op == "like":
            if c["negate"] or not (isinstance(value, str) and value.startswith("%") and value.endswith("%") and "%" not in value[1:-1]):
                raise Unsupported("LIKE that is not a plain contains")
            filters.append({"column": name, "op": "contains", "value": value[1:-1]})
        elif op == "between":
            if c["negate"]:
                raise Unsupported("NOT BETWEEN")
            filters.append({"column": name, "op": "between", "value": value})
        else:
            filters.append({"column": name, "op": _NEG[op] if c["negate"] else op, "value": value})
    if "or" in p["joins"]:
        if set(p["joins"]) != {"or"} or len({f["column"] for f in filters}) != 1 or any(f["op"] != "==" for f in filters):
            raise Unsupported("OR that is not a list of equalities on one column")
        filters = [{"column": filters[0]["column"], "op": "in", "value": [f["value"] for f in filters]}]
    return {"formulas": formulas, "filters": filters}


# ------------------------------------------------------------------- build --

def split_of(db: str) -> str:
    return "tune" if int(hashlib.sha256(db.encode()).hexdigest(), 16) % 10 < 7 else "check"


def _token(rng: random.Random) -> str:
    return "".join(rng.choice("bcdfghjkmnpqrstvwxz") for _ in range(2)) + str(rng.randint(10, 99))


def build() -> dict:
    import pyarrow.parquet as pq

    train = pq.read_table(TRAIN, columns=["db_id", "query", "question"]).to_pydict()
    schemas = {t["db_id"]: t for t in json.loads(TABLES.read_text(encoding="utf-8"))}
    cases, skipped = [], collections.Counter()
    literals: dict[tuple, dict[str, set]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for index, (db, query, question) in enumerate(zip(train["db_id"], train["query"], train["question"])):
        try:
            p = parse(query)
        except (Unsupported, IndexError) as e:
            skipped[type(e).__name__] += 1
            continue
        schema = schemas[db]
        tables = {n.lower(): k for k, n in enumerate(schema["table_names_original"])}
        if p["table"].lower() not in tables:
            skipped["table_not_in_schema"] += 1
            continue
        t_idx = tables[p["table"].lower()]
        cols = {n.lower(): n for ti, n in schema["column_names_original"] if ti == t_idx}
        try:
            spec = to_spec(p, cols)
        except Unsupported:
            skipped["not_expressible"] += 1
            continue
        table = schema["table_names_original"][t_idx]
        for f in spec["filters"]:
            for v in (f["value"] if isinstance(f["value"], list) else [f["value"]]):
                literals[(db, table)][f["column"]].add(v)
        cases.append({"id": f"spider-train-{index:04d}", "db_id": db, "table": table, "question": question, "query": query,
                      "spec": spec, "split": split_of(db), "csv": f"training/data/eval/spider_train_dev/tables/{db}__{table}.csv"})

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for (db, table) in sorted({(c["db_id"], c["table"]) for c in cases}):
        schema = schemas[db]
        t_idx = [n.lower() for n in schema["table_names_original"]].index(table.lower())
        cols = [(n, schema["column_types"][k]) for k, (ti, n) in enumerate(schema["column_names_original"]) if ti == t_idx]
        rng = random.Random(f"{db}/{table}")
        lit = literals[(db, table)]
        data = {}
        for name, kind in cols:
            given = sorted(lit.get(name, set()), key=str)
            numeric_given = all(isinstance(v, float) for v in given)
            if kind == "number" and numeric_given:
                lo = min(given) if given else 1
                hi = max(given) if given else 100
                if "year" in name.lower():
                    lo, hi = min(lo, 1990), max(hi, 2020)
                pool = [v for v in given] + [rng.randint(int(lo) - 5, int(hi) + 5) for _ in range(ROWS)]
                data[name] = [int(v) if float(v).is_integer() else v for v in pool[:ROWS]]
            elif kind == "time":
                data[name] = [f"20{rng.randint(0, 22):02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" for _ in range(ROWS)]
            else:
                pool = [str(v) for v in given] + [_token(rng) for _ in range(ROWS)]
                data[name] = pool[:ROWS]
            rng.shuffle(data[name])
        path = OUT / "tables" / f"{db}__{table}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([n for n, _ in cols])
            for r in range(ROWS):
                w.writerow([data[n][r] for n, _ in cols])
    with open(OUT / "cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"source": "Spider training split (xlangai/spider train parquet) + tables.json schemas",
               "train_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
               "cases": len(cases), "by_split": dict(collections.Counter(c["split"] for c in cases)),
               "databases": len({c["db_id"] for c in cases}), "skipped": dict(skipped),
               "cases_sha256": hashlib.sha256((OUT / "cases.jsonl").read_bytes()).hexdigest()}
    (OUT / "BUILD_MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


# ------------------------------------------------------------------- score --

def _norm_value(v):
    if isinstance(v, list):
        return tuple(sorted(_norm_value(x) for x in v))
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _norm_formula(f: str) -> str:
    # count("col") counts non-empty cells; the synthetic tables have none empty, so it equals count().
    return "count()" if re.fullmatch(r'count\("[^"]*"\)', f) else f


def same_spec(got: dict, want: dict) -> bool:
    if [_norm_formula(f) for f in got["formulas"]] != [_norm_formula(f) for f in want["formulas"]]:
        return False
    norm = lambda fs: sorted((f["column"], f["op"], _norm_value(f["value"])) for f in fs)
    return norm(got["filters"]) == norm(want["filters"])


def score(split: str) -> list[dict]:
    from app import metric_engine as me
    from app.evals import spider
    from app.profiling import load_and_profile_csv

    rows = []
    for c in map(json.loads, (OUT / "cases.jsonl").open(encoding="utf-8")):
        if split != "all" and c["split"] != split:
            continue
        path = str(ROOT / c["csv"])
        try:
            profile = load_and_profile_csv(path)[1]
            out = spider.ask_copilot(c["question"], path, profile, me.load_frame(path), dataset_name=f"{c['table']}.csv")
        except Exception as e:  # noqa: BLE001
            rows.append({**c, "outcome": "error", "copilot": {"status": "error", "reason": f"{type(e).__name__}: {e}"}})
            continue
        if out["status"] in ("answered", "answered_multi", "answered_table"):
            spec = out.get("spec") or {}
            got = {"formulas": spec.get("formulas") or [spec.get("formula")], "filters": spec.get("filters") or []}
            ok = out["status"] != "answered_table" and same_spec(got, c["spec"])
            outcome = "spec_match" if ok else "wrong_spec"
        else:
            outcome = "abstain"
        rows.append({**c, "outcome": outcome, "copilot": out})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--label", required=True)
    s.add_argument("--split", choices=["tune", "check", "all"], default="tune")
    s.add_argument("--show", type=int, default=0, help="print N cases (tune split only)")
    s.add_argument("--outcome", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(), indent=1))
        return 0
    rows = score(args.split)
    counts = collections.Counter(r["outcome"] for r in rows)
    n = len(rows)
    summary = {"label": args.label, "split": args.split, "cases": n,
               **{k: counts[k] for k in ("spec_match", "abstain", "wrong_spec", "error")},
               "spec_match_rate": round(counts["spec_match"] / n, 4), "wrong_spec_rate": round(counts["wrong_spec"] / n, 4)}
    print(json.dumps(summary))
    with open(OUT / f"score_{args.label}_{args.split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    if args.show and args.split == "tune":
        shown = [r for r in rows if args.outcome is None or r["outcome"] == args.outcome]
        random.Random(0).shuffle(shown)
        for r in shown[:args.show]:
            o = r["copilot"]
            print(f"- [{r['outcome']}] {r['question']}\n    want {r['spec']}\n    got  {o.get('status')} {o.get('spec')} {o.get('reason', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
