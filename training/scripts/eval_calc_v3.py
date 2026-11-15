"""Calculator-planning evaluation: base vs v1 vs v3 on identical held-out examples, scored by execution.

Every model runs through the application's own planner (app.analyst.calc.model_plan: same prompts, same section
checks and repairs), with greedy decoding, in this process. A plan is correct when the calculator's result for it
equals the result of the gold plan -- the numbers, not the text (names of derived columns may differ).

Test sets
  A  the v3 test split (template wording; tables never seen in training, including 4 domains never seen at all),
     a fixed stratified sample
  B  hand-written questions in different wording on held-out-domain tables and on public datasets bundled with
     installed libraries (plotly tips / gapminder / election / carshare), with adversarial cases (a condition the
     data cannot express, a column that does not exist) and hand-written follow-ups

Measures (per example): kind_ok (plan vs clarify/decline), valid (a plan that passes the checks), first_pass (valid
with no section rejected or repaired), exec_ok (same numbers as the gold), refused_ok (clarify/decline when the gold
refuses, no invented proxy).

    .venv313/bin/python training/scripts/eval_calc_v3.py --label v3 --adapter training/adapters/analyst_qwen3_4b_lora_v3 \
        --out <path>.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.analyst import calc  # noqa: E402

BASE = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
V3 = ROOT / "training/data/analyst_sft_v3"


def table(tables, gid):
    t = tables[str(gid)]
    df = pd.read_csv(io.StringIO(t["csv"]))
    for c in t["dates"]:
        df[c] = pd.to_datetime(df[c])
    return df


def result_vectors(res: dict) -> list[tuple]:
    """Each aggregate as (totals value, {group: value}) -- compared regardless of the names chosen."""
    out = []
    for name in res["totals"]:
        groups = tuple(sorted((g["group"], round(g[name], 6) if isinstance(g.get(name), (int, float)) else None)
                              for g in res.get("groups", []) if name in g))
        tot = res["totals"][name]
        out.append((round(tot, 6) if isinstance(tot, (int, float)) else None, groups))
    return out


def same(gold: dict, got: dict) -> bool:
    """Every gold aggregate appears among the model's aggregates (equal totals and equal per-group values)."""
    def close(a, b):
        if a is None or b is None:
            return a is b
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    gv, mv = result_vectors(gold), result_vectors(got)
    for gt, gg in gv:
        if not any(close(gt, mt) and len(gg) == len(mg) and all(k1 == k2 and close(v1, v2) for (k1, v1), (k2, v2) in zip(gg, mg))
                   for mt, mg in mv):
            return False
    return True


def run_gold(gold, df, q):
    spec = {**gold, "sort": None, "explain": ""}
    r = calc.run(spec, df, question=q)
    return {"totals": r["totals"], "groups": r["groups"]}


# ---------------------------------------------------------------- set B --
def public(name):
    import plotly
    p = os.path.join(os.path.dirname(plotly.__file__), "package_data", "datasets", name + ".csv.gz")
    return pd.read_csv(p)


def plan(derive=None, exclude=None, group=None, aggregate=None):
    return {"type": "calc", "derive": derive or {}, "exclude": exclude or [], "group_by": group, "aggregate": aggregate or {}}


def set_b(tables_test, meta_test):
    """Hand-written. Gold plans are checked by running them; refusals have type clarify/decline."""
    items = []
    tips, gap, elec, cars = public("tips"), public("gapminder"), public("election"), public("carshare")
    R = "refuse"
    items += [
        ("tips", tips, "roughly what did people tip as a fraction of the bill, on average, each day?",
         plan({"tip_share": "tip / total_bill"}, [], "day", {"avg_tip_share": "mean(tip_share)"}), None),
        ("tips", tips, "bill total per head - avg by time of day", plan({"per_head": "total_bill / size"}, [], "time",
                                                                         {"avg_per_head": "mean(per_head)"}), None),
        ("tips", tips, "how much did smokers spend altogether vs the rest",
         plan({}, [], "smoker", {"total_bill_sum": "sum(total_bill)"}), None),
        ("tips", tips, "sum of tips for parties bigger than 4", plan({}, [{"reason": "party of 4 or fewer", "where": "size <= 4"}],
                                                                     None, {"tips": "sum(tip)"}), None),
        ("tips", tips, "total bill by day, leaving out the bills that were paid by card", R, None),
        ("tips", tips, "average tip from vegetarian tables", R, None),
        ("gapminder", gap, "average life expectancy per continent in 2007",
         plan({}, [{"reason": "not 2007", "where": "year != 2007"}], "continent", {"avg_life": "mean(lifeExp)"}), None),
        ("gapminder", gap, "total GDP (gdp per cap times population) by continent for 1952",
         plan({"gdp": "gdpPercap * pop"}, [{"reason": "not 1952", "where": "year != 1952"}], "continent", {"gdp": "sum(gdp)"}), None),
        ("gapminder", gap, "how many countries had a life expectancy under 50 in 2007",
         plan({}, [{"reason": "not 2007", "where": "year != 2007"}, {"reason": "50 or more", "where": "lifeExp >= 50"}],
              None, {"countries": "count(*)"}), None),
        ("gapminder", gap, "average literacy rate by continent", R, None),
        ("election", elec, "share of the vote Coderre got in each district, averaged",
         plan({"coderre_share": "Coderre / total"}, [], None, {"avg_share": "mean(coderre_share)"}), None),
        ("election", elec, "total votes per winner", plan({}, [], "winner", {"votes": "sum(total)"}), None),
        ("election", elec, "turnout by district excluding spoiled ballots", R, None),
        ("carshare", cars, "mean car hours across peak hours 7 to 9 only",
         plan({}, [{"reason": "outside 7-9", "where": "peak_hour < 7 or peak_hour > 9"}], None, {"avg_hours": "mean(car_hours)"}), None),
    ]
    # held-out domain tables (generated, never trained): hand-written wording, gold built from the table's roles
    rng = random.Random(7)
    seen = set()
    for m in meta_test:
        if not m["heldout_domain"] or m["group"] in seen or m["kind"] != "plan":
            continue
        seen.add(m["group"])
        df = table(tables_test, m["group"])
        cols = list(df.columns)
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in cols if df[c].dtype == object and 2 <= df[c].nunique() <= 6]
        ent = next(c for c in cols if df[c].dtype == object and df[c].nunique() > 3)
        who = df[ent].iloc[0]
        q = lambda c: c if c.isidentifier() else f"`{c}`"  # noqa: E731
        lab = lambda c: c.replace("_", " ").lower()  # noqa: E731
        items += [
            (m["domain"], df, f"for {who.split()[0]}'s rows only, whats the combined {lab(num[2])}",
             plan({}, [{"reason": "only", "where": f"{q(ent)} != '{who}'"}], None, {"x": f"sum({q(num[2])})"}), None),
            (m["domain"], df, f"{lab(num[0])} multiplied by {lab(num[1])}, totalled for every {lab(cat[0])}",
             plan({"v": f"{q(num[0])} * {q(num[1])}"}, [], cat[0], {"v": "sum(v)"}), None),
            (m["domain"], df, f"tell me the biggest single {lab(num[2])} per {lab(cat[-1])}",
             plan({}, [], cat[-1], {"m": f"max({q(num[2])})"}), None),
            (m["domain"], df, f"average {lab(num[2])} by {lab(cat[0])}, but skip anything flagged as a test record", R, None),
        ]
        if len(seen) >= 8:
            break
    # hand-written follow-ups (turn 2 wording differs from the training templates)
    follow = [("tips", tips, "total tip by day", plan({}, [], "day", {"tips": "sum(tip)"}),
               ("ok now just the dinner ones", plan({}, [{"reason": "only dinner", "where": "time != 'Dinner'"}], "day", {"tips": "sum(tip)"}))),
              ("tips", tips, "average bill by sex", plan({}, [], "sex", {"avg": "mean(total_bill)"}),
               ("and split by smoker rather than sex", plan({}, [], "smoker", {"avg": "mean(total_bill)"}))),
              ("gapminder", gap, "total population by continent in 2007",
               plan({}, [{"reason": "not 2007", "where": "year != 2007"}], "continent", {"pop": "sum(pop)"}),
               ("same for 1952", plan({}, [{"reason": "not 1952", "where": "year != 1952"}], "continent", {"pop": "sum(pop)"}))),
              ("election", elec, "votes for Joly by winner", plan({}, [], "winner", {"joly": "sum(Joly)"}),
               ("what about Bergeron", plan({}, [], "winner", {"b": "sum(Bergeron)"})))]
    for f in follow:
        items.append(f)
    return items


# --------------------------------------------------------------- running --
def make_completer(adapter):
    from mlx_lm import generate, load
    model, tok = load(BASE, adapter_path=adapter)

    def complete(prompt: str, max_tokens: int) -> str:
        return generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return complete


def score(question, df, gold, memory, earlier=""):
    t0 = time.time()
    try:
        spec, trace = calc.model_plan(question, df, memory)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200], "seconds": round(time.time() - t0, 1)}
    refused_gold = gold == "refuse" or (isinstance(gold, dict) and gold.get("type") in ("clarify", "decline"))
    kind = spec.get("type")
    out = {"kind": kind, "seconds": round(time.time() - t0, 1),
           "rejected_or_repaired": sum(1 for t in trace if t.get("accepted") is False or t.get("repaired"))}
    if refused_gold:
        out["refused_ok"] = kind in ("clarify", "decline")
        out["kind_ok"] = out["refused_ok"]
        return out
    out["kind_ok"] = kind == "calc"
    out["valid"] = kind == "calc"
    out["first_pass"] = out["valid"] and out["rejected_or_repaired"] == 0
    if out["valid"]:
        try:
            got = calc.run(spec, df, question=question)
            out["exec_ok"] = same(run_gold(gold, df, question), got)
        except Exception as e:  # noqa: BLE001
            out["exec_ok"] = False
            out["run_error"] = str(e)[:160]
        out["plan"] = {k: spec.get(k) for k in ("derive", "exclude", "group_by", "aggregate")}
    else:
        out["exec_ok"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=45)
    args = ap.parse_args()
    meta = [json.loads(x) for x in open(V3 / "test_meta.jsonl")]
    tables = json.load(open(V3 / "test_tables.json"))
    rng = random.Random(20261115)
    sample = []
    for kind in ("plan", "followup", "decline", "clarify"):
        for held in (True, False):
            pool = [m for m in meta if m["kind"] == kind and m["heldout_domain"] == held]
            sample += rng.sample(pool, min(len(pool), args.per_kind if kind in ("plan", "followup") else 12))
    calc._complete = make_completer(args.adapter)
    rows = []
    for m in sample:
        df = table(tables, m["group"])
        memory = {}
        if m["kind"] == "followup":
            first = next(x for x in meta if x["group"] == m["group"] and x["question"] == m["first"] and x["kind"] == "plan")
            g1 = first["gold"]
            spec1 = {**g1, "sort": None, "explain": ""}
            memory = {"calc_history": [(m["first"], {k: g1[k] for k in ("type", "derive", "exclude", "group_by", "aggregate")})],
                      "last_calc": {"question": m["first"], "spec": spec1, "result": calc.run(spec1, df, question=m["first"])}}
        gold = m["gold"] if m["gold"]["type"] == "calc" else "refuse"
        r = score(m["question"], df, gold, memory)
        rows.append({"set": "A", "domain": m["domain"], "heldout": m["heldout_domain"], "kind": m["kind"], **r})
        print(args.label, "A", m["kind"], m["domain"], {k: r.get(k) for k in ("kind_ok", "exec_ok", "refused_ok", "first_pass")}, flush=True)
    for it in set_b(tables, meta):
        name, df, q, gold, follow = it
        if follow is None:
            r = score(q, df, gold, {})
            rows.append({"set": "B", "domain": name, "heldout": True, "kind": "refuse" if gold == "refuse" else "plan",
                         "question": q, **r})
        else:
            spec1 = {**gold, "sort": None, "explain": ""}
            memory = {"calc_history": [(q, {k: gold[k] for k in ("type", "derive", "exclude", "group_by", "aggregate")})],
                      "last_calc": {"question": q, "spec": spec1, "result": calc.run(spec1, df, question=q)}}
            q2, g2 = follow
            r = score(q2, df, g2, memory)
            rows.append({"set": "B", "domain": name, "heldout": True, "kind": "followup", "question": q2, **r})
        print(args.label, "B", rows[-1]["kind"], name, {k: rows[-1].get(k) for k in ("kind_ok", "exec_ok", "refused_ok")}, flush=True)
    Path(args.out).write_text(json.dumps({"label": args.label, "adapter": args.adapter, "base": BASE,
                                          "decoding": "greedy (mlx_lm.generate default), app calculator max tokens per section",
                                          "rows": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
