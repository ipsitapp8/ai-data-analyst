"""The Silt Copilot's route into the universal what-if (app/whatif/service.py), beside the business route in scenario_bridge.py.

The Copilot reads a sentence, the shared service builds a structured scenario from it, the shared engine calculates, and the
Copilot reports the structured result. It has no formulas of its own: the standalone What-If page calls the same functions, so the
same question gives the same numbers from either, and the standalone page never needs the Copilot to run.

It is deliberately narrow:

* It answers only when the sentence matches a scenario template (education, healthcare, travel, government, marketing, hiring...). The
  price / cost / break-even business sentences stay with scenario_bridge, which runs first; anything this adapter does not match
  returns None and the Copilot's other paths answer exactly as before.
* A value the sentence and the data do not give (seats, faculty, the current pass mark) is asked for and remembered for the next message,
  never assumed.
* A question the data cannot support is said to be unsupported, not answered.
* "Suggest some what-ifs" returns the discovery list with each question's feasibility.
"""
from __future__ import annotations

import re

import pandas as pd

from app.whatif import discovery, engine, service
from app.whatif.explain import as_text
from app.whatif.parse import NUM, _num

BRAINSTORM = re.compile(r"\b(?:suggest|brainstorm|ideas?|recommend)\b.*\b(?:what[\s-]?ifs?|scenarios?|questions?|simulat\w+)\b|"
                        r"\bwhat\s+(?:scenarios?|what[\s-]?ifs?|questions?)\b.*\b(?:ask|try|explore|simulate|evaluate)\b|"
                        r"\bwhich\s+scenarios?\b|\bwhat\s+should\s+i\s+(?:explore|simulate|ask)\b", re.I)
COMPARE = re.compile(r"\bcompare\b.*\b(?:previous|earlier|last|before|those|them|scenarios?)\b|\b(?:previous|earlier)\s+scenarios?\b", re.I)
EXPLAIN = re.compile(r"^\s*(?:why|how reliable|how sure|what (?:are|were) the assumptions|what assumptions|what can'?t|what cannot|explain)\b|\bassumptions\b|\bhow reliable\b", re.I)
WHATIF = re.compile(r"\bwhat if\b|\bwhat (?:happens|would happen)\b|\b(?:increas|rais|reduc|decreas|cut|lower|grow|improv|extend|add|hire|enrol|enroll|study|change)\w*\b", re.I)
MAX_HISTORY = 5
# The business sentences stay with the Copilot's existing paths (scenario_bridge for price/cost/break-even; the earlier follow-up and
# column-change paths for marketing and budget sentences, which carry scoping such as "in the lowest-performing area"). The marketing
# template is still available on the What-If page and through the API.
BUSINESS_HANDLED = {"biz_price_change", "biz_cost_change", "biz_break_even", "biz_sensitivity", "biz_marketing_spend"}


def _bullets(result: dict) -> str:
    lines = []
    for m in result["metrics"][:6]:
        pct = f" ({m['pct_change']:+.1f}%)" if m.get("pct_change") is not None else ""
        lines.append(f"- {m['label']}: {m['baseline']:,.2f} → {m['scenario']:,.2f}{pct}" if m["baseline"] is not None and m["scenario"] is not None else f"- {m['label']}: —")
    return "\n".join(lines)


def _reply(answer: str, plan: dict, notes: list[str], details: str = "", calls: list | None = None) -> dict:
    return {"answer": answer, "plan": plan, "calls": calls or [], "notes": notes, "details": details}


def _compact(s: dict) -> dict:
    """What the conversation keeps of an executed scenario (no row data: a profile and result only)."""
    return {k: s[k] for k in ("id", "name", "domain", "template_id", "question", "method", "inputs", "columns", "results", "validation", "explanation", "dataset") if k in s}


def _brainstorm(message: str, df: pd.DataFrame, filename: str | None) -> dict:
    objective = message if re.search(r"\b(?:i am|i'm|as a|we are|my goal|objective|before the)\b", message, re.I) else None
    out = service.suggest(df, None, filename=filename, objective=objective)
    items = out["suggestions"][:8]
    label = {"ready": "ready to run", "needs_assumptions": "needs a value from you", "needs_external_data": "needs data this file does not have", "unsupported": "cannot be answered reliably"}
    lines = [f"- {s['question']} — *{label[s['feasibility']]}*" + (f" ({'; '.join(s['missing_inputs'])})" if s["missing_inputs"] and s["feasibility"] != "ready" else "") for s in items]
    det = out["domain"].get("detection", {})
    intro = f"Here are what-if questions that fit this dataset ({out['domain']['title']}" + ("; " + det["explanation"].split(" You can")[0].lower() if det.get("source") == "detected" else "") + "):"
    return _reply(intro + "\n\n" + "\n".join(lines) + "\n\nSay any of them with your numbers, for example *what if passing marks increase from 40 to 50?*",
                  {"type": "whatif_suggestions", "domain": out["domain"]["key"]}, [f"Suggestions come from the dataset's columns and the {out['domain']['title']} templates; none was invented. Feasibility is checked against the data."])


def _fill_pending(message: str, scenario: dict, missing: list[dict]) -> tuple[dict, list[str]]:
    """Numbers (and column names) the user gave for what was asked. Only values that name the thing asked for are taken."""
    inputs, notes = dict(scenario["inputs"]), []
    cols = dict(scenario.get("columns") or {})
    for m in missing:
        if m["kind"] == "number":
            words = {w for w in re.split(r"[^a-z]+", (m["name"] + " " + m["label"]).lower()) if len(w) > 3 and w not in ("current", "available", "number", "your", "data", "from")}
            for w in sorted(words):
                hit = re.search(rf"\b{re.escape(w)}\w*\b\W{{0,3}}(?:(?:is|are|equals?|of|at|to|be|about)\W{{0,3}})*{NUM}", message, re.I)
                if hit:
                    inputs[m["name"]] = _num(hit.group(1))
                    notes.append(f"{m['label']} = {inputs[m['name']]:g}, as you told me.")
                    break
        elif m["kind"] == "column":
            concept = m["name"].split(":", 1)[1]
            pick = [c for c in m["candidates"] if re.search(rf"\b{re.escape(c.lower())}\b", message.lower())]
            if len(pick) == 1:
                cols[concept] = pick
    return {**scenario, "inputs": inputs, "columns": cols}, notes


def _ask_for(missing: list[dict], profile_note: str = "") -> str:
    parts = []
    for m in missing:
        if m["kind"] == "column":
            parts.append(f"which column to use for **{m['name'].split(':', 1)[1].replace('_', ' ')}**: {', '.join(m['candidates'])}")
        elif m["kind"] == "distribution":
            parts.append("how uncertain it is (for example *normal with a 15% spread*)")
        elif m["kind"] == "target":
            parts.append("the target you want to reach")
        else:
            parts.append(f"**{m['label'].lower()}**" + (f" ({m['why'].lower()})" if m.get("why") else ""))
    return "I can set that up, but I need " + "; ".join(parts) + ". I will not assume a value. Tell me, for example *faculty is 10, seats are 200*."


def _run_and_report(s: dict, df: pd.DataFrame, memory: dict, filename: str | None, extra_notes: list[str], scope_note: str | None) -> dict:
    out = service.run(s, df)
    val = out["validation"]
    plan = {"type": "whatif", "domain": out["domain"], "template": out["template_id"], "method": out["method"]}
    if val["status"] == "needs_input":
        memory["whatif_pending"] = {"scenario": {**out, "results": None, "explanation": None}, "missing": val["missing_inputs"]}
        return _reply(_ask_for(val["missing_inputs"]), {**plan, "type": "clarify"}, extra_notes + ["The scenario is set up; the values above are needed because they are not in the data."])
    memory.pop("whatif_pending", None)
    if val["status"] in ("invalid", "unsupported"):
        return _reply("I could not run that what-if: " + " ".join(val["errors"]), {**plan, "type": "clarify"}, extra_notes)
    r, e = out["results"], out["explanation"]
    hist = memory.setdefault("whatif_history", [])
    hist.append(_compact(out))
    del hist[:-MAX_HISTORY]
    notes = ([f"Computed over the records in scope only: {scope_note}."] if scope_note else []) + extra_notes + [
        f"Answered by the shared what-if engine ({engine.ENGINE_VERSION}); the same one the What-If page uses. No number came from a language model.",
        f"Kind of result: {e['evidence_label']}."]
    if r["status"] != "ok":
        return _reply(f"**{e['evidence_label']}.** {r['reason']}", plan, notes)
    sections = e["sections"]
    det = as_text(e).split("\n", 1)[1] if "\n" in as_text(e) else ""
    answer = f"**{e['evidence_label']}.** {e['headline']}\n\n{_bullets(r)}"
    if sections["reliability"]:
        answer += "\n\n" + sections["reliability"][-1] if len(sections["reliability"]) > 1 else ""
    compact = {k: r[k] for k in ("status", "evidence_kind", "primary") if k in r}
    return _reply(answer, plan, notes, details=det, calls=[{"tool": "whatif_engine", "status": "ok", "result": compact}])


def try_answer(message: str, df: pd.DataFrame, memory: dict, filename: str | None = None, scope_note: str | None = None) -> dict | None:
    """A reply for a universal what-if sentence, or None to let the Copilot's other paths answer."""
    pending = memory.get("whatif_pending")
    if pending and not BRAINSTORM.search(message):
        s, notes = _fill_pending(message, pending["scenario"], pending["missing"])
        if notes or s["columns"] != pending["scenario"].get("columns"):
            return _run_and_report(s, df, memory, filename, notes, scope_note)
    if BRAINSTORM.search(message):
        return _brainstorm(message, df, filename)
    hist = memory.get("whatif_history") or []
    if COMPARE.search(message) and len(hist) >= 2:
        try:
            cmp = service.compare(hist[-2:])
        except ValueError as e:
            return _reply(f"I could not compare them: {e}", {"type": "clarify"}, [])
        rows = "\n".join(f"- {m['label']}: " + " vs ".join(f"{(m['scenario'][s['id']] if m['scenario'][s['id']] is not None else float('nan')):,.2f}" for s in cmp["scenarios"]) for m in cmp["metrics"][:6])
        names = " vs ".join(f"“{s['question'] or s['name']}”" for s in cmp["scenarios"])
        return _reply(f"Comparing {names}:\n\n{rows}" + ("\n\n" + " ".join(cmp["notes"]) if cmp["notes"] else ""), {"type": "whatif_compare"},
                      [f"Compared by the shared what-if engine ({engine.ENGINE_VERSION}); the numbers are the saved results of each scenario."])
    if hist and EXPLAIN.search(message) and not WHATIF.search(message.replace("assumptions", "")):
        last = hist[-1]
        text = as_text(last["explanation"]) if last.get("explanation") else "There is no explanation for that scenario."
        return _reply(f"About “{last['question'] or last['name']}”:\n\n{text}", {"type": "whatif_explain"}, ["Explained from the stored result of the last scenario; nothing was recalculated."])
    if not WHATIF.search(message):
        return None
    hit = discovery.classify_question(message, None, None)
    if hit is None or hit["template"].id in BUSINESS_HANDLED:
        return None
    s_out = service.from_question(message, df, filename=filename)
    if s_out["status"] != "ready":
        return None
    s = s_out["scenario"]
    t = hit["template"]
    from app.whatif.profile import build_profile
    prof = build_profile(df, filename=filename)
    own = discovery.assess(t, s["domain"], prof, df)
    if own["feasibility"] == "needs_external_data" and t.operation in ("threshold", "shift_metric", "association", "forecast_trend"):
        if s["domain"] == "business":
            return None
        return _reply(f"I can't answer that from this dataset: {'; '.join(own['missing'])}. "
                      f"It is a {s['domain']} question, so it needs those columns. Ask me *what what-if questions fit this data?* for ones it can answer.",
                      {"type": "clarify", "domain": s["domain"]}, ["The dataset was checked for the columns this scenario needs; they are not there, so nothing was calculated."])
    if own["feasibility"] == "unsupported" and t.operation != "none":
        return _reply("I won't estimate that: " + " ".join(own["limitations"][:1]) + " The data does not support a reliable answer.", {"type": "clarify", "domain": s["domain"]},
                      ["Declined rather than guessed: the relationship is too weak or the history too short."])
    return _run_and_report(s, df, memory, filename, [], scope_note)
