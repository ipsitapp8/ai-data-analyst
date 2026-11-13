"""Import Spider's number-answer validation questions into the evaluation lab.

Reads the reference set built by training/scripts/build_spider_reference.py
(it is not rerun, and nothing under training/ is written) and writes one case
file the evaluation runner loads as the separate suite "spider-numeric":

    backend/evals/cases/spider_numeric.json

Selection: every question that the reference build marked valid for scoring and
whose answer is a single row of aggregates (answer shapes "scalar" and
"scalar_row"). That is the 165 questions the reference report names.

Before anything is written, the inputs are verified:
  1. cases.jsonl and reference_results.jsonl have the SHA-256 recorded in
     BUILD_MANIFEST.json;
  2. every selected question is still valid for scoring, with no flags, and its
     reference has exactly one row;
  3. every CSV a case reads has the SHA-256 recorded in the manifest;
  4. when the official SQLite databases are present, every reference query is
     executed again, read-only, and must return the stored row.
Any failure stops the import with nothing written.

Idempotent: the output contains no timestamps and is written with sorted keys,
so a second run produces the same bytes and leaves the file untouched. If the
file exists with different content, the import refuses to replace it unless
--replace is given (bump SUITE_VERSION when the selection or format changes).

No model is called and no application code is imported.

    cd backend && python evals/build_spider_cases.py [--check] [--replace]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training/data/eval/spider_dev_single_table"
SPIDER = ROOT / "training/data/raw/spider_official/spider_data"
OUT = Path(__file__).resolve().parent / "cases" / "spider_numeric.json"
SUITE_VERSION = "spider-numeric.1"
SHAPES = ("scalar", "scalar_row")
EXPECTED_CASES = 165
FLOAT_DIGITS = 6
# Scoring tolerance, recorded in every case: a value is "within tolerance" when it
# is within 0.1% of the reference or 0.005 of it in absolute terms.
TOLERANCE = {"rel": 1e-3, "abs": 5e-3}
SELECT_ONLY = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


class ImportFailure(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def canon(value):
    """As the reference build canonicalises: floats to 6 decimals."""
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS) + 0.0
    if isinstance(value, bytes):
        return "0x" + value.hex()
    return value


def as_number(value) -> float | None:
    """The reference value as a number, the way the reference build's pandas
    check reads it: numbers stay numbers, number-like text ("9") becomes a
    number, anything else (dates, "44.7%", NULL) has no numeric value."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def rerun(db: str, query: str) -> list | None:
    path = SPIDER / "database" / db / f"{db}.sqlite"
    if not path.exists():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda raw: raw.decode("utf-8", "replace")
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in SELECT_ONLY else sqlite3.SQLITE_DENY)
    try:
        return [[canon(v) for v in row] for row in con.execute(query).fetchall()]
    finally:
        con.close()


def build() -> tuple[dict, dict]:
    manifest = json.loads((SOURCE / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    for name in ("cases.jsonl", "reference_results.jsonl"):
        got, want = sha256(SOURCE / name), manifest["outputs"][name]
        if got != want:
            raise ImportFailure(f"{name} has SHA-256 {got}, the manifest records {want}: the reference set changed")
    table_hashes = {t["path"]: t["sha256"] for t in manifest["tables"]}
    refs = {r["reference_id"]: r for r in map(json.loads, (SOURCE / "reference_results.jsonl").open(encoding="utf-8"))}
    all_cases = [json.loads(line) for line in (SOURCE / "cases.jsonl").open(encoding="utf-8")]
    selected = [c for c in all_cases if c.get("valid_for_scoring") and c.get("answer_shape") in SHAPES]

    # Spider's own parse of each query and its schema, for the evaluation's SQL-to-engine oracle.
    dev = json.loads((SPIDER / "dev.json").read_text(encoding="utf-8")) if (SPIDER / "dev.json").exists() else None
    schemas = ({t["db_id"]: t for t in json.loads((SPIDER / "tables.json").read_text(encoding="utf-8"))}
               if (SPIDER / "tables.json").exists() else None)
    if dev is None or schemas is None:
        raise ImportFailure("training/data/raw/spider_official/spider_data/{dev.json,tables.json} are needed for the query parse")
    if manifest["inputs"]["dev.json"] != sha256(SPIDER / "dev.json") or manifest["inputs"]["tables.json"] != sha256(SPIDER / "tables.json"):
        raise ImportFailure("dev.json or tables.json differ from the files the reference set was built from")

    verification = collections.Counter()
    rerun_results: dict[str, list | None] = {}
    cases = []
    for c in selected:
        ref = refs.get(c["reference_id"])
        if ref is None or not ref["valid_for_scoring"] or ref["flags"] or c["flags"]:
            raise ImportFailure(f"{c['case_id']}: reference {c.get('reference_id')} is missing or no longer valid")
        if ref["row_count"] != 1 or len(ref["rows"]) != 1:
            raise ImportFailure(f"{c['case_id']}: a single-row answer was expected, the reference has {ref['row_count']} rows")
        if c["question"] != dev[c["dev_index"]]["question"] or c["query"] != dev[c["dev_index"]]["query"]:
            raise ImportFailure(f"{c['case_id']}: question or query differs from dev.json")
        csv_rel = c["csv"]
        if table_hashes.get(csv_rel) != sha256(ROOT / csv_rel):
            raise ImportFailure(f"{c['case_id']}: {csv_rel} differs from the manifest")
        verification["csv_hash_ok"] += 1

        if ref["reference_id"] not in rerun_results:
            rerun_results[ref["reference_id"]] = rerun(ref["db_id"], ref["query"])
            again = rerun_results[ref["reference_id"]]
            if again is None:
                verification["reference_not_reexecuted"] += 1
            elif again != ref["rows"]:
                raise ImportFailure(f"{ref['reference_id']}: re-executing the reference SQL returned {again}, stored {ref['rows']}")
            else:
                verification["reference_reexecuted_identical"] += 1

        values = ref["rows"][0]
        numbers = [as_number(v) for v in values]
        if any(v is None for v in values):
            note = "null_reference"        # the reference SQL returns NULL: no number to score against
        elif any(n is None for n in numbers):
            note = "non_numeric_reference"  # a date or text such as "44.7%"
        elif any(isinstance(v, str) for v in values):
            note = "numeric_text_reference"  # e.g. "9": scored as the number it denotes
        else:
            note = None
        schema = schemas[c["db_id"]]
        cases.append({
            "id": c["case_id"],
            "component": "spider_copilot",
            "question": c["question"],
            "db_id": c["db_id"],
            "table": ref["table"],
            "csv": csv_rel,
            "csv_sha256": table_hashes[csv_rel],
            "answer_shape": c["answer_shape"],
            "reference_id": ref["reference_id"],
            "query": ref["query"],
            "sql_parse": dev[c["dev_index"]]["sql"],
            "column_names": [name for _, name in schema["column_names_original"]],
            "reference": {"columns": ref["columns"], "values": values, "numbers": numbers,
                          "scorable": note in (None, "numeric_text_reference"), "note": note},
            "tolerance": TOLERANCE,
            "leakage": c.get("leakage", []),
            "data_fidelity_risk": c.get("data_fidelity_risk", []),
        })

    if len(cases) != EXPECTED_CASES:
        raise ImportFailure(f"selected {len(cases)} questions, the milestone is defined on {EXPECTED_CASES}")
    cases.sort(key=lambda x: x["id"])
    shapes = collections.Counter(x["answer_shape"] for x in cases)
    notes = collections.Counter(x["reference"]["note"] or "numeric" for x in cases)
    doc = {
        "suite_version": SUITE_VERSION,
        "description": ("Spider validation questions with a single-row numeric answer, scored against the reference SQL "
                        "result. Derived from Spider (CC BY-SA 4.0); regenerate with backend/evals/build_spider_cases.py."),
        "source": {"reference_build": "training/scripts/build_spider_reference.py",
                   "manifest_outputs": manifest["outputs"], "inputs": manifest["inputs"]},
        "selection": {"rule": "valid_for_scoring and answer_shape in (scalar, scalar_row)", "cases": len(cases),
                      "by_answer_shape": dict(sorted(shapes.items())), "by_reference_kind": dict(sorted(notes.items())),
                      "distinct_reference_queries": len({x["reference_id"] for x in cases}),
                      "scorable_as_numbers": sum(1 for x in cases if x["reference"]["scorable"])},
        "cases": cases,
    }
    return doc, dict(verification)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify and compare with the existing file; write nothing")
    ap.add_argument("--replace", action="store_true", help="replace an existing case file whose content differs")
    args = ap.parse_args(argv)
    try:
        doc, verification = build()
    except ImportFailure as e:
        print(f"import stopped, nothing written: {e}", file=sys.stderr)
        return 2
    text = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    print(json.dumps({"verification": verification, "selection": doc["selection"]}, indent=1))
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
    if current == text:
        print(f"{OUT.relative_to(ROOT)} is up to date (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
        return 0
    if args.check:
        print(f"{OUT.relative_to(ROOT)} {'differs from' if current else 'does not exist; it would be built from'} the inputs")
        return 1
    if current is not None and not args.replace:
        print(f"{OUT.relative_to(ROOT)} exists with different content; not replaced (use --replace and bump SUITE_VERSION)",
              file=sys.stderr)
        return 3
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
