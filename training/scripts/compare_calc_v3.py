"""Summarise eval_calc_v3.py results for several models side by side (markdown on stdout)."""
from __future__ import annotations

import json
import math
import sys


def rate(rows, key):
    xs = [r.get(key) for r in rows if r.get(key) is not None]
    return (sum(bool(x) for x in xs), len(xs))


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)


def gold_kinds(per_kind=30):
    """The gold kind of every evaluated example, in evaluation order (eval_calc_v3.py stores the model's own plan
    type under "kind"). Rebuilt with the same seed and order the evaluation used."""
    import random
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent))
    import eval_calc_v3 as ev
    meta = [json.loads(x) for x in open(ev.V3 / "test_meta.jsonl")]
    rng = random.Random(20261115)
    kinds = []
    for kind in ("plan", "followup", "decline", "clarify"):
        for held in (True, False):
            pool = [m for m in meta if m["kind"] == kind and m["heldout_domain"] == held]
            kinds += [kind] * min(len(pool), per_kind if kind in ("plan", "followup") else 12)
            rng.sample(pool, min(len(pool), per_kind if kind in ("plan", "followup") else 12))
    tables = json.load(open(ev.V3 / "test_tables.json"))
    for it in ev.set_b(tables, meta):
        kinds.append("followup" if it[4] is not None else ("refuse" if it[3] == "refuse" else "plan"))
    return kinds


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    kinds = gold_kinds()
    for run in runs:
        assert len(run["rows"]) == len(kinds), (len(run["rows"]), len(kinds))
        for r, k in zip(run["rows"], kinds):
            r["model_kind"], r["kind"] = r.get("kind"), k
    labels = [r["label"] for r in runs]
    groups = [
        ("A plans, train domains (unseen tables)", lambda r: r["set"] == "A" and r["kind"] == "plan" and not r["heldout"], "exec_ok"),
        ("A plans, held-out domains", lambda r: r["set"] == "A" and r["kind"] == "plan" and r["heldout"], "exec_ok"),
        ("A follow-ups, train domains", lambda r: r["set"] == "A" and r["kind"] == "followup" and not r["heldout"], "exec_ok"),
        ("A follow-ups, held-out domains", lambda r: r["set"] == "A" and r["kind"] == "followup" and r["heldout"], "exec_ok"),
        ("A refusals (decline/clarify), all", lambda r: r["set"] == "A" and r["kind"] in ("decline", "clarify"), "refused_ok"),
        ("B hand-written plans", lambda r: r["set"] == "B" and r["kind"] == "plan", "exec_ok"),
        ("B hand-written follow-ups", lambda r: r["set"] == "B" and r["kind"] == "followup", "exec_ok"),
        ("B adversarial (must refuse)", lambda r: r["set"] == "B" and r["kind"] == "refuse", "refused_ok"),
        ("plan type correct (all)", lambda r: True, "kind_ok"),
        ("valid plan (all plan/follow-up)", lambda r: r["kind"] in ("plan", "followup"), "valid"),
        ("first pass, no rejection or repair (all plan/follow-up)", lambda r: r["kind"] in ("plan", "followup"), "first_pass"),
    ]
    print("| measure | n | " + " | ".join(labels) + " |")
    print("|---|---:|" + "---:|" * len(labels))
    for name, pred, key in groups:
        cells, n = [], None
        for run in runs:
            k, m = rate([r for r in run["rows"] if pred(r)], key)
            n = m
            lo, hi = wilson(k, m)
            cells.append(f"{k}/{m} = {k / m:.2f} [{lo:.2f}–{hi:.2f}]" if m else "—")
        print(f"| {name} | {n} | " + " | ".join(cells) + " |")
    print()
    print("| seconds per example (median) | " + " | ".join(
        f"{sorted(r['seconds'] for r in run['rows'] if 'seconds' in r)[len(run['rows']) // 2]}" for run in runs) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
