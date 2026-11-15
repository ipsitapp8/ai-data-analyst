"""The conversation's ordered result register: every answered turn, in order, with what it computed.

It lives inside the conversation's own memory (orchestrator.memory_for: one dict per team, user, dataset, document
and conversation), so nothing in it is visible to another user, dataset or conversation, and a new dataset starts a
new register. Entries hold the question, the kind of answer, its headline figures (as computed, with their labels),
whether they are hypothetical, and -- for target / scenario objectives -- the objective itself, so a later turn can
refer to it reliably:

    "compare that with the previous result"     latest entry vs the one before it
    "set that next to the original ask"         latest entry vs the first one
    "compare the first and third analyses"      entries 1 and 3
    "the second result" / "result 2"            entry 2

References are resolved here, deterministically; the comparison uses the stored figures and recalculates nothing.
"""
from __future__ import annotations

import re

MAX = 20

_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
        "ninth": 9, "tenth": 10, "original": 1, "initial": 1, "last": -1, "latest": -1, "previous": -2, "earlier": -2,
        "prior": -2, "before": -2}
_NOUN = r"(?:results?|answers?|analys[ie]s|calculations?|figures?|numbers?|scenarios?|asks?|questions?|ones?|targets?|" \
        r"versions?|cases?|options?)"
_ORD_RE = re.compile(r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|original|initial|"
                     r"last|latest|previous|earlier|prior|(\d+)(?:st|nd|rd|th))\s+(?:\w+\s+)?" + _NOUN + r"\b|\b"
                     + r"(?:result|answer|analysis|calculation|scenario)\s+(?:number\s+|#\s*)?(\d+)\b", re.I)
_PRONOUN = re.compile(r"\b(?:that|this|it|these|the latest|the new one|the last one)\b", re.I)
_COMPARE = re.compile(r"\b(?:compare[sd]?|comparing|comparison|set\b[^?]{0,20}\bnext to|side by side|against|versus|"
                      r"vs\.?|difference between|differ\w* from|what (?:moved|changed))\b", re.I)


def _headline(out: dict, memory: dict) -> tuple[dict | None, str, bool]:
    """(figures, label, hypothetical) of an answer, from its computed results -- never from its text."""
    for c in out.get("calls", []):
        if c.get("status") != "ok":
            continue
        res = c.get("result") or {}
        if c.get("tool") == "analysis_program" and res.get("objective") in ("rule_planned", "model_planned"):
            steps = (res.get("programs") or {}).get("") or []
            vals = (res.get("results") or {}).get("") or {}
            pick = [s for s in steps if s["op"] == "ratio"] or [s for s in steps if s["op"] == "aggregate"]
            if pick:
                s = pick[-1]
                v = vals.get(s["id"]) or {}
                if v.get("kind") in ("scalar", "series"):
                    lab = (f"{s['fn']} of {s.get('column') or 'records'}" if s["op"] == "aggregate" else
                           "ratio " + " ÷ ".join(f"{(next((x for x in steps if x['id'] == r), {}) or {}).get('fn', '')} "
                                                 f"of {(next((x for x in steps if x['id'] == r), {}) or {}).get('column', r)}"
                                                 for r in (s["a"], s["b"])))
                    by = next((x.get("by") for x in steps if x["op"] == "aggregate" and x.get("by")), None)
                    fig = v["value"] if isinstance(v["value"], dict) else {"value": v["value"]}
                    return fig, lab + (f" by {by}" if by else ""), bool(v.get("hypothetical"))
        if c.get("tool") == "analysis_program" and res.get("objective") in ("feasibility", "absence", "closure",
                                                                             "vs_rest"):
            from app.analyst.objectives import _key  # noqa: PLC0415
            o = memory.get("objective")
            if o:
                k = {a: b for a, b in _key(o).items() if isinstance(b, (int, float)) and not isinstance(b, bool)}
                return k, o["type"], o["type"] in ("absence", "closure") or bool(o.get("scenario_of"))
    for c in out.get("calls", []):
        res = c.get("result") or {}
        if c.get("status") != "ok" or not isinstance(res, dict):
            continue
        if c.get("tool") == "compute_metric" and res.get("column"):
            lab = f"{res.get('aggregation')} of {res['column']}"
            if res.get("groups"):
                return ({str(g["group"]): g["value"] for g in res["groups"]}, f"{lab} by {res.get('group_by')}", False)
            if isinstance(res.get("value"), (int, float)):
                return {"value": res["value"]}, lab, False
        if c.get("tool") == "rank_groups" and res.get("rows"):
            return ({str(x["group"]): x["value"] for x in res["rows"]},
                    f"{res.get('aggregation', 'total')} of {res.get('measure', '')}".strip(), False)
    lc = memory.get("last_calc")
    if out.get("plan_source") in ("metric", "model_calc", "refine") and lc:
        aggs = list((lc.get("spec") or {}).get("aggregate") or {})
        r, prim = lc["result"], (lc.get("meta") or {}).get("primary") or (aggs[0] if len(aggs) == 1 else None)
        m = re.match(r"^\s*(\w+)\((\w+)\)\s*$", str(((lc.get("spec") or {}).get("aggregate") or {}).get(prim, "")))
        name = f"{'avg' if m.group(1) == 'mean' else m.group(1)} of {m.group(2)}" if m else prim   # as compute_metric names it
        if prim and r.get("groups"):
            return ({str(g.get("group")): g.get(prim) for g in r["groups"]}, f"{name} by {r.get('group_by')}",
                    r.get("answer_type") == "scenario")
        if prim and isinstance(r.get("totals"), dict) and prim in r["totals"]:
            return {"value": r["totals"][prim]}, str(name), r.get("answer_type") == "scenario"
    return None, "", False


def record(memory: dict, message: str, out: dict) -> None:
    """Append an answered turn. Comparisons of earlier entries are recorded too (they are answers), without figures."""
    if not any(c.get("status") == "ok" for c in out.get("calls", [])):
        return
    if (out.get("plan") or {}).get("type") in ("clarify", "decline", "chat", "error"):
        return
    fig, label, hyp = _headline(out, memory)
    obj = out.get("plan", {}).get("objective")
    reg = memory.setdefault("register", [])
    reg.append({"n": (reg[-1]["n"] + 1) if reg else 1, "question": message, "source": out.get("plan_source"),
                "kind": obj or out.get("plan_source"), "figures": fig, "label": label, "hypothetical": hyp,
                "objective": memory.get("objective") if obj in ("feasibility", "absence", "closure", "vs_rest") else None})
    del reg[:-MAX]


_COMPARISONS = {"compare_results", "compare", "recall"}


def _results(reg: list) -> list[int]:
    """Positions of computed results: 'the first result', 'the previous result' never mean an earlier comparison."""
    return [k for k, e in enumerate(reg) if e.get("kind") not in _COMPARISONS]


def _index(word: str, num: str | None, reg: list, recall: bool = False) -> int | None:
    """A position in the register (0-based), or None when it is out of range. A number names the register entry as
    shown ("result 3"); an ordinal or relative word counts computed results only."""
    if num:
        return next((k for k, e in enumerate(reg) if e["n"] == int(num)), None)
    i = _ORD[word.lower()]
    if recall and i == -2:
        i = -1          # "show me the previous result again": the most recent one, not the one before it
    res = _results(reg)
    j = len(res) + i if i < 0 else i - 1
    return res[j] if 0 <= j < len(res) else None


def refs(message: str, memory: dict, recall: bool = False) -> list[int] | None:
    """The register entries a message points at (0-based, in the order named), [] when it names references that do
    not exist, None when it names none."""
    reg = memory.get("register") or []
    found = []
    for m in _ORD_RE.finditer(message):
        word, num1, num2 = m.group(1), m.group(2), m.group(3)
        found.append(_index(word or "", num1 or num2, reg, recall))
    if not found:
        return None
    if None in found:
        return []
    return found


def compare_request(message: str, memory: dict) -> tuple[int, int] | None | str:
    """(a, b) register positions for "compare that with the previous result" and the like; a message (str) when the
    reference cannot be resolved; None when the message is not such a comparison."""
    if not _COMPARE.search(message):
        return None
    r = refs(message, memory)
    if r is None:
        return None
    reg = memory.get("register") or []
    if r == []:
        return (f"This conversation has {len(reg)} answered question(s) so far, so I can't find the one you mean. "
                "Say which, for example *compare the first and second results*.") if reg else \
            "There is nothing earlier in this conversation to compare with yet."
    if len(r) >= 2:
        if r[0] == r[1]:
            return "Both references point at the same result; say which two to compare."
        return tuple(sorted(r[:2]))           # earlier first: shown as "previous" and "latest"
    res = _results(reg)
    if (_PRONOUN.search(message) or len(res) >= 2) and res:
        latest = res[-1]
        if r[0] == latest:
            return ("That refers to the latest result itself; say which earlier result to compare it with."
                    if len(res) < 2 else (res[-2], latest))
        return tuple(sorted((r[0], latest)))
    return None


_RECALL = re.compile(r"\b(?:go back to|back to|show (?:me )?(?:again )?|remind me (?:of|what)|recall|bring (?:back|up)|"
                     r"return to|repeat|what (?:was|were|is))\b", re.I)


def recall_request(message: str, memory: dict) -> int | None | str:
    """The register position of "go back to the first result", "what was the previous answer"; a message when the
    reference does not exist; None when the message is not such a request."""
    if not _RECALL.search(message) or _COMPARE.search(message):
        return None
    r = refs(message, memory, recall=True)
    if r is None:
        return None
    reg = memory.get("register") or []
    if r == [] or not reg:
        return (f"This conversation has {len(_results(reg))} computed result(s) so far, so I can't find the one you "
                "mean.") if reg else "There are no earlier results in this conversation yet."
    return r[0]


def recall(e: dict) -> dict:
    """An earlier result as it was computed -- its question, what it measured, and its figures; nothing recalculated."""
    from app.analyst.explain import f as fmt  # noqa: PLC0415
    lines = [f"**Result {e['n']}** — *{e['question']}*" + (" (hypothetical)" if e.get("hypothetical") else ""), "",
             f"{e.get('label') or e.get('kind')}:", ""]
    fig = e.get("figures") or {}
    if fig:
        lines += ["| | value |", "|---|---:|"] + [f"| {k} | {fmt(v) if isinstance(v, (int, float)) else v} |"
                                                for k, v in list(fig.items())[:25]]
    else:
        lines.append("_That answer has no single figure to repeat (for example a list of findings)._")
    return {"answer": "\n".join(lines), "calls": [{"tool": "recall_result", "status": "ok",
                                                   "result": {"n": e["n"], "figures": fig}}],
            "charts": [], "notes": ["Shown from this conversation's result register, as it was computed; nothing was "
                                    "recalculated."], "plan": {"type": "program", "objective": "recall"}}


_TARGET_Q = re.compile(r"\b(?:original|first|initial|starting|earlier|previous|old|current|latest|new)\s+(?:target|goal)\b|"
                       r"\bwhat (?:was|is|were) (?:my|the|our) (?:target|goal)s?\b", re.I)


def target_request(message: str, memory: dict) -> dict | None:
    """'What was my original target?': the target chain of this conversation -- the first target asked and any later
    changes -- from its own memory; never invented."""
    if not _TARGET_Q.search(message):
        return None
    chain = [o for o in memory.get("objectives") or [] if o.get("type") == "feasibility"]
    if not chain:
        return {"answer": "No target has been set in this conversation yet, so there is no original target. Ask for "
                          "one first — for example *Can Retail reach 35000 revenue this month?*", "calls": [],
                "charts": [], "notes": ["Read from this conversation's memory; nothing was calculated."],
                "plan": {"type": "chat"}}
    start = max((i for i, o in enumerate(chain) if not o.get("changed")), default=0)
    seq = chain[start:]
    from app.analyst.explain import f as fmt  # noqa: PLC0415
    first, last = seq[0]["params"], seq[-1]["params"]
    unit = first.get("unit") or ""
    who = first.get("entity") or "the combined total"
    lines = [f"**Original target: {(fmt(first['target']) + ' ' + unit).strip()}** of {first['measure']['label']} for {who}, "
             f"{first['period']}."]
    if len(seq) > 1:
        lines.append("")
        lines.append("Changes since then: " + "; ".join(o.get("changed", "") for o in seq[1:]) +
                     f". The current target is **{(fmt(last['target']) + ' ' + unit).strip()}**; the original is kept unchanged.")
    return {"answer": "\n".join(lines), "calls": [{"tool": "recall_result", "status": "ok",
                                                   "result": {"original": first["target"], "current": last["target"]}}],
            "charts": [], "notes": ["Read from this conversation's memory of its targets; nothing was recalculated."],
            "plan": {"type": "program", "objective": "recall"}}


def compare(a: dict, b: dict) -> dict:
    """Two stored entries side by side; a difference only where both measure the same thing."""
    lines = [f"**Comparing result {a['n']} with result {b['n']}** (both computed earlier in this conversation; nothing "
             "was recalculated).", "",
             f"- **{a['n']}.** *{a['question']}* — {a['label'] or a['kind']}" + (" (hypothetical)" if a["hypothetical"] else ""),
             f"- **{b['n']}.** *{b['question']}* — {b['label'] or b['kind']}" + (" (hypothetical)" if b["hypothetical"] else ""),
             ""]
    fa, fb = a.get("figures") or {}, b.get("figures") or {}
    same = a["label"] == b["label"] and a["label"]
    keys = [k for k in fa if k in fb]
    if not fa or not fb:
        lines.append("_One of them has no single figure to set against the other (for example a list of associations "
                     "or findings), so they are listed, not subtracted._")
    elif not keys:
        lines.append("_They report different things, with nothing in common to set side by side._")
    else:
        lines += [f"| | result {a['n']} | result {b['n']} |" + (" change |" if same else ""),
                  "|---|---:|---:|" + ("---:|" if same else "")]
        from app.analyst.explain import f as fmt  # noqa: PLC0415
        for k in keys:
            x, y = fa[k], fb[k]
            num = isinstance(x, (int, float)) and isinstance(y, (int, float))
            lines.append(f"| {k} | {fmt(x) if num else x} | {fmt(y) if num else y} |"
                         + (f" {fmt(y - x)} |" if same and num else ""))
        if not same:
            lines.append(f"\n_They measure different things ({a['label']} vs {b['label']}), so no difference is computed._")
    fig_rows = {str(k): {"a": fa.get(k), "b": fb.get(k)} for k in keys}
    return {"answer": "\n".join(lines), "calls": [{"tool": "compare_results", "status": "ok",
                                                    "result": {"a": a["n"], "b": b["n"], "figures": fig_rows,
                                                               "same_measure": bool(same)}}],
            "charts": [], "notes": ["Both results come from this conversation's result register; the references were "
                                    "resolved by Silt, and nothing was recalculated."],
            "plan": {"type": "program", "objective": "compare_results"}}
