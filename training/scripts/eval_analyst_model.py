"""Base model vs fine-tuned adapter on the analyst SFT test split.

    .venv313/bin/python training/scripts/eval_analyst_model.py --label v1 \
        --adapter training/adapters/analyst_qwen3_4b_lora [--per-intent 20]

Both models see exactly the prompts the application sends. Greedy decoding.
The test split holds two domains never seen in training (education,
real_estate) and held-out phrasings; see build_analyst_sft.py.

Plan metrics (per example):
    json_valid   the reply parses as a plan (app.analyst.planner.parse_plan)
    type_ok      plan type equals the reference (chat/clarify/decline/tools/explain_previous)
    tools_ok     same tool names in the same order (tools plans)
    args_ok      same tools and identical arguments (column names, aggregation, filters, ...)
    executable   every tool call passes the backend's validation against the schema
Narration metric:
    grounded     the summary is non-empty and states no number absent from the tool results

Writes training/reports/analyst_finetune_eval_<label>.{json,md}; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.analyst import explain, planner, tools  # noqa: E402

DATA = ROOT / "training/data/analyst_sft"
REPORTS = ROOT / "training/reports"
BASE = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
UNSEEN = {"education", "real_estate"}
# The ten measures asked for, as (intents, metric) over the per-example scores.
MEASURES = {
    "1 intent recognition (plan type)": (None, "type_ok"),
    "2 tool selection": (None, "tools_ok"),
    "3 structured plan validity (parses and passes backend validation)": (None, "valid_and_executable"),
    "4 column selection (exact arguments)": (None, "args_ok"),
    "5 clarification behaviour": (("clarify",), "type_ok"),
    "6 unsupported-request handling": (("decline",), "type_ok"),
    "7 business recommendation planning": (("recommendation", "performance"), "args_ok"),
    "8 statistical method selection": (("statistics", "comparison", "correlation", "regression", "group_comparison"), "tools_ok"),
    "9 Random Forest planning (target and task)": (("prediction", "explain_prediction"), "args_ok"),
    "10 general conversation": (("chat",), "type_ok"),
    "narration grounded in tool results": (("narrate_metric", "narrate_which", "narrate_compare"), "grounded"),
}


def frame_from_prompt(user: str) -> pd.DataFrame:
    """A stand-in frame with the schema's columns and kinds, for validating calls."""
    cols = {}
    for name, kind in re.findall(r"^- (.+?) \((numeric|categorical|datetime)", user, re.M):
        cols[name] = (np.arange(3, dtype=float) if kind == "numeric" else
                      pd.date_range("2025-01-01", periods=3) if kind == "datetime" else ["a", "b", "c"])
    return pd.DataFrame(cols)


def norm_args(a):
    if isinstance(a, dict):
        return {k: norm_args(v) for k, v in sorted(a.items()) if v not in (None, "", [])}
    if isinstance(a, list):
        return [norm_args(v) for v in a]
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return float(a)
    return a


def score_plan(reply: str, ref: dict, user: str) -> dict:
    got = planner.parse_plan(reply)
    s = {"json_valid": got is not None, "type_ok": False, "tools_ok": None, "args_ok": None, "executable": None,
         "valid_and_executable": False}
    if got is None:
        return s
    s["type_ok"] = got["type"] == ref["type"]
    if ref["type"] == "tools":
        g_calls = got.get("calls") or [] if got["type"] == "tools" else []
        s["tools_ok"] = [c.get("tool") for c in g_calls] == [c["tool"] for c in ref["calls"]]
        s["args_ok"] = s["tools_ok"] and all(norm_args(a.get("args") or {}) == norm_args(b["args"])
                                             for a, b in zip(g_calls, ref["calls"]))
        df = frame_from_prompt(user)
        try:
            for c in g_calls:
                tools.validate_call(c, df, user.split("Question: ")[-1])
            s["executable"] = bool(g_calls)
        except tools.InvalidCall:
            s["executable"] = False
        s["valid_and_executable"] = bool(s["executable"])
    elif ref["type"] == "clarify" and got["type"] == "clarify":
        s["args_ok"] = set(got.get("options") or []) <= set(ref.get("options") or []) and bool(got.get("options"))
    if ref["type"] != "tools":
        s["valid_and_executable"] = got["type"] in ("chat", "clarify", "decline", "explain_previous")
    return s


def measures(rows: list[dict]) -> dict:
    out = {}
    for name, (intents, key) in MEASURES.items():
        vals = [r[key] for r in rows if (intents is None or r["intent"] in intents) and isinstance(r.get(key), bool)
                and (intents is not None or not r["intent"].startswith("narrate"))]
        out[name] = {"rate": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}
    return out


def score_narration(reply: str, user: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    results = user.split("Results:", 1)[-1]
    q = user.split("\n", 1)[0]
    bad = explain.ungrounded(text, results, q)
    return {"grounded": bool(text) and not bad, "ungrounded": bad}


def sample(per_intent: int, seed: int = 7) -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "test.jsonl").open(encoding="utf-8")]
    meta = [json.loads(l) for l in (DATA / "test_meta.jsonl").open(encoding="utf-8")]
    by = collections.defaultdict(list)
    for r, m in zip(rows, meta):
        by[m.get("intent", "?")].append({**r, **m})
    rnd = random.Random(seed)
    out = []
    for intent in sorted(by):
        items = by[intent]
        rnd.shuffle(items)
        out += items[:per_intent]
    return out


def run(model_path: str, adapter: str | None, examples: list[dict]) -> list[dict]:
    from mlx_lm import generate, load
    model, tok = load(model_path, adapter_path=adapter)
    out = []
    for ex in examples:
        msgs = ex["messages"][:-1]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0 = time.monotonic()
        reply = generate(model, tok, prompt=prompt, max_tokens=320, verbose=False)
        out.append({"reply": reply, "seconds": time.monotonic() - t0})
    return out


def summarize(examples, outputs):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    rows = []
    for ex, o in zip(examples, outputs):
        user = ex["messages"][1]["content"]
        if ex["intent"].startswith("narrate"):
            s = score_narration(o["reply"], user)
        else:
            s = score_plan(o["reply"], json.loads(ex["messages"][-1]["content"]), user)
        rows.append({"intent": ex["intent"], "source": ex["source"], **s, "reply": o["reply"][:600], "seconds": o["seconds"]})
        domain = ex["source"].split(":")[-1] if ":" in ex["source"] else ex["source"]
        split = "unseen:" + domain if domain in UNSEEN else "seen_domains_heldout_phrasing"
        for k, v in s.items():
            if isinstance(v, bool):
                per[ex["intent"]][k].append(v)
                per["_all"][k].append(v)
                per[split][k].append(v)
    rate = {i: {k: round(sum(v) / len(v), 4) for k, v in d.items()} | {"n": max(len(v) for v in d.values())}
            for i, d in per.items()}
    return rate, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--per-intent", type=int, default=20)
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    jpath, mpath = REPORTS / f"analyst_finetune_eval_{args.label}.json", REPORTS / f"analyst_finetune_eval_{args.label}.md"
    if jpath.exists() or mpath.exists():
        print(f"refusing to overwrite {jpath.name}")
        return 1
    examples = sample(args.per_intent)
    result = {"label": args.label, "base_model": BASE, "adapter": args.adapter, "examples": len(examples),
              "test_split_sha256": json.loads((DATA / "MANIFEST.json").read_text())["splits"]["test"]["sha256"],
              "decoding": "greedy, max 320 tokens", "models": {}}
    import inference_record
    decoding = {"sampler": "greedy (mlx_lm.generate default: temperature 0, argmax)", "max_tokens": 320,
                "prompt": "tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)"}
    for name, adapter in (("base", None), ("fine_tuned", args.adapter)):
        result.setdefault("inference", {})[name] = inference_record.record(adapter, decoding)
        t0 = time.monotonic()
        outputs = run(BASE, adapter, examples)
        rate, rows = summarize(examples, outputs)
        result["models"][name] = {"rates": rate, "measures": measures(rows),
                                  "measures_unseen": {d: measures([r for r in rows if r["source"].endswith(d)]) for d in sorted(UNSEEN)},
                                  "seconds": round(time.monotonic() - t0, 1),
                                  "mean_seconds_per_example": round(float(np.mean([o["seconds"] for o in outputs])), 2),
                                  "rows": rows}
        print(name, json.dumps(rate["_all"]))
    jpath.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    keys = ["json_valid", "type_ok", "tools_ok", "args_ok", "executable", "grounded"]
    lines = [f"# Analyst model evaluation {args.label}", "",
             f"Base `{BASE}` vs the same model with LoRA adapter `{args.adapter}`. {len(examples)} test examples "
             f"(up to {args.per_intent} per intent), greedy decoding, prompts identical to the application's.", "",
             "| intent | n | " + " | ".join(f"{k} base → tuned" for k in keys) + " |", "|---|---:|" + "---|" * len(keys)]
    b, t = result["models"]["base"]["rates"], result["models"]["fine_tuned"]["rates"]
    for intent in ["_all"] + sorted(k for k in b if k != "_all"):
        cells = []
        for k in keys:
            if k in b.get(intent, {}):
                cells.append(f"{b[intent][k]:.2f} → {t[intent][k]:.2f}")
            else:
                cells.append("")
        lines.append(f"| {intent} | {b[intent]['n']} | " + " | ".join(cells) + " |")
    for title, getter in (("All test examples", lambda m: m["measures"]),
                          *((f"Unseen domain: {d}", (lambda d: lambda m: m["measures_unseen"][d])(d)) for d in sorted(UNSEEN))):
        lines += ["", f"## {title}", "", "| measure | n | base | fine-tuned |", "|---|---:|---:|---:|"]
        mb, mt = getter(result["models"]["base"]), getter(result["models"]["fine_tuned"])
        for k in MEASURES:
            if mb[k]["n"]:
                lines.append(f"| {k} | {mb[k]['n']} | {mb[k]['rate']:.3f} | {mt[k]['rate']:.3f} |")
    lines += ["", f"Time: base {result['models']['base']['mean_seconds_per_example']} s/example, fine-tuned "
                  f"{result['models']['fine_tuned']['mean_seconds_per_example']} s/example."]
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", jpath, mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
