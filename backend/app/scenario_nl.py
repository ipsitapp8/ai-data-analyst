"""Natural-language what-if requests: a sentence becomes a validated, structured request, and nothing else.

    "What happens if I increase the price by 10%?"      -> evaluate: price +10%
    "What price would give me a 25% profit margin?"     -> goal seek: margin_pct = 25, solve for price
    "Compare a 10% price increase with 15% more volume" -> two scenarios, side by side
    "And also cut fixed costs by 5%"                    -> the previous scenario plus one more change

This module only *understands wording*. It never produces a result and never supplies a number: every number in a
request is one the user wrote (checked), every input and output is one the model really has (checked again by
`validate_intent` before the engines see it), and anything it cannot resolve becomes a question for the user instead of a
guess. The rules below always run first. A language model may be asked to translate a sentence the rules could not read,
but its answer goes through exactly the same checks, plus one more: it may not use a number the user did not write.

Conversation context is explicit: the caller passes back what it was given last time. It is dropped when the model or the
baseline changed, and the previous scenario is carried over only when the sentence says so ("also", "too", "on top of
that"). A sentence that does not is a fresh scenario and says so.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable

from app.scenarios import ScenarioError

NL_VERSION = "scenario-nl.1"
MAX_MESSAGE = 500
MAX_SCENARIOS = 4
MAX_ADJUSTMENTS = 8
MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
               "m": 1e6, "mn": 1e6, "million": 1e6, "billion": 1e9, "bn": 1e9}
UP_WORDS = r"increas\w*|rais\w*|grow\w*|boost\w*|improv\w*|rise\w*|expand\w*|bump\w*|add\w*|higher|more|extra|up|gain\w*"
DOWN_WORDS = r"reduc\w*|decreas\w*|cut\w*|lower\w*|drop\w*|fall\w*|declin\w*|less|trim\w*|shrink\w*|sav\w*|slash\w*|fewer|down|dip\w*"
SET_WORDS = r"set|make|change|fix|hold|peg"
UNCHANGED = r"(?:remain\w*|stay\w*|keep\w*|is|are|be)?\s*(?:unchanged|the same|constant|flat|fixed at today'?s?)|doesn'?t change|do(?:es)? not change|remain\w* unchanged"
DIMENSIONLESS_UNITS = {"", "1", "fraction", "ratio", "percent", "%", "rate", "share"}
STOP = {"the", "a", "an", "of", "per", "my", "our", "your", "in", "on", "for", "to", "by", "and", "i", "we", "it", "if", "is"}
SYNONYMS = {
    "price": ["price", "pricing", "selling price", "unit price", "prices"],
    "volume": ["volume", "units sold", "units", "quantity", "sales volume", "number of units"],
    "unit_cost": ["unit cost", "cost per unit", "variable cost", "variable costs", "cost of goods", "cogs", "production cost"],
    "fixed_cost": ["fixed cost", "fixed costs", "overhead", "overheads"],
    "visitors": ["visitors", "traffic", "visits", "leads"],
    "conversion_rate": ["conversion", "conversion rate", "conversions"],
    "average_order_value": ["order value", "average order value", "aov", "basket size"],
    "cost_per_visitor": ["cost per visitor", "acquisition cost per visitor"],
}
OUTPUT_SYNONYMS = {
    "margin_pct": ["profit margin", "margin", "margins"], "profit": ["profit", "profits", "earnings", "net income"],
    "revenue": ["revenue", "sales", "turnover", "income"], "orders": ["orders"], "gross_profit": ["gross profit"],
}
EXAMPLES = ["What happens if I increase the price by 10%?", "What if I reduce unit cost by 5%?", "What price would give me a 25% profit margin?",
            "How many units do I need to break even?", "Compare a 10% price increase with a 15% volume increase."]


# ---------------------------------------------------------------- text helpers --

def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[₹$€£]|\brs\.?\s*(?=\d)|\binr\b|\busd\b", " ", t)
    t = re.sub(r"(?<=\d),(?=\d{2,3}(?!\d))", "", t)            # 1,20,000 and 50,000 are single numbers
    t = re.sub(r"percentage\s*points?|\bppts?\b|\bpts?\b", " pp ", t)        # before "percent", which is a prefix of "percentage"
    t = re.sub(r"per\s*cent|percent", "%", t)

    def expand(m: re.Match) -> str:
        v = float(m.group(1)) * MULTIPLIERS[m.group(2)]
        return f" {v:.10g} "
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(" + "|".join(sorted(MULTIPLIERS, key=len, reverse=True)) + r")\b(?!\s*%)", expand, t)
    t = re.sub(r"(\d)\s+%", r"\1%", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def stated_numbers(message: str) -> set[float]:
    """Every number the user wrote, as written and as expanded (50k, 2 lakh), with their /100 forms for percentages."""
    out: set[float] = set()
    for src in (message or "", _normalize(message)):
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)", src.replace(",", "")):
            v = float(m.group(1))
            out |= {v, -v, v / 100, -v / 100}
    return out


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


# ------------------------------------------------------------- name resolution --

class Index:
    """Maps phrases to a model's inputs and outputs: its names and labels, plus a few everyday synonyms."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.inputs = self._build(spec["inputs"], lambda k, r: r["label"], SYNONYMS)
        self.outputs = self._build({k: {} for k in spec["outputs"]}, lambda k, r: spec["outputs"][k], OUTPUT_SYNONYMS)

    @staticmethod
    def _build(items: dict, label_of: Callable, synonyms: dict) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for key in items:
            names = {key.replace("_", " "), label_of(key, items[key]).lower(), re.sub(r"\s*\(.*?\)", "", label_of(key, items[key]).lower()).strip()}
            names |= {s for s in synonyms.get(key, [])}
            out[key] = {_clean(n) for n in names if n}
        return out

    def find(self, text: str, table: dict[str, set[str]]) -> list[tuple[str, str, int, int]]:
        """Matches in `text`: (key, alias, start, end), longest alias first, overlaps dropped."""
        norm = " " + _clean(text) + " "
        hits = []
        for key, aliases in table.items():
            for alias in aliases:
                for m in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?:s|es)?(?![a-z0-9])", norm):
                    hits.append((key, alias, m.start(), m.end()))
        hits.sort(key=lambda h: (-(h[3] - h[2]), h[2]))
        chosen: list[tuple[str, str, int, int]] = []
        for h in hits:
            if not any(h[2] < c[3] and c[2] < h[3] for c in chosen):
                chosen.append(h)
        return sorted(chosen, key=lambda h: h[2])

    def generic_cost_candidates(self, text: str) -> list[str]:
        """The word 'cost' alone matches several inputs; those inputs are the options to ask about."""
        return [k for k, a in self.inputs.items() if any(w == "cost" or w == "costs" for al in a for w in al.split())]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9% ]+", " ", (text or "").lower())).strip()


def _is_rate(rule: dict) -> bool:
    """A 0-to-1 rate (a conversion rate, a churn rate): where 'percentage points' and 'to 5%' make sense."""
    unit = (rule.get("unit") or "").strip().lower()
    return (rule.get("max") is not None and float(rule["max"]) <= 1.0) or any(w in unit for w in ("fraction", "ratio", "rate", "percent"))


def _noun(clause: str) -> str:
    """The thing a clause talks about, without the lead-in and the verb, for a question to the user."""
    c = re.sub(r"^\s*(?:what (?:happens|would happen) if|what if|if)\s+(?:i|we)?\s*", "", clause.strip())
    c = re.sub(rf"^(?:{UP_WORDS}|{DOWN_WORDS}|{SET_WORDS})\s+(?:the\s+|my\s+|our\s+)?", "", c)
    return re.sub(r"\s+(?:by|to)\s+\d.*$", "", c).strip() or clause.strip()


# ------------------------------------------------------------------ the parse --

def _result(status: str, **kw: Any) -> dict:
    return {"status": status, "engine_version": NL_VERSION, **kw}


def _clarify(question: str, options: list[dict], missing: str, interpretation: list[str] | None = None) -> dict:
    return _result("clarify", clarification={"question": question, "options": options, "missing": missing},
                   interpretation=interpretation or [])


def _unsupported(why: str) -> dict:
    return _result("unsupported", message=why, examples=EXAMPLES, interpretation=[])


_AMOUNT = [
    ("pp", re.compile(r"(\d+(?:\.\d+)?)\s*pp\b")),
    ("pct", re.compile(r"(\d+(?:\.\d+)?)%")),
]


def _direction(clause: str) -> int:
    up = re.search(rf"\b(?:{UP_WORDS})\b", clause)
    down = re.search(rf"\b(?:{DOWN_WORDS})\b", clause)
    if up and down:
        return 1 if up.start() < down.start() else -1
    return 1 if up else -1 if down else 0


def _amount(clause: str) -> dict | None:
    """The change a clause asks for: {'kind': pct|pp|abs|set, 'value': signed number}, or None."""
    sign = _direction(clause)
    m = re.search(r"\b(?:to|at|equals?|=)\s*(\d+(?:\.\d+)?)\s*(%?)", clause)
    if m and not sign:
        return {"kind": "set", "value": float(m.group(1)), "unsigned": float(m.group(1)), "explicit_direction": False}
    for kind, rx in _AMOUNT:
        m = rx.search(clause)
        if m:
            return {"kind": kind, "value": float(m.group(1)) * (sign or 1), "unsigned": float(m.group(1)), "explicit_direction": bool(sign)}
    m = re.search(r"\b(?:to|at|=|equals?)\s*(\d+(?:\.\d+)?)", clause)
    if m and (re.search(rf"\b(?:{SET_WORDS})\b|\bto\b", clause)) and not sign:
        return {"kind": "set", "value": float(m.group(1)), "unsigned": float(m.group(1)), "explicit_direction": False}
    m = (re.search(r"\bby\s*(\d+(?:\.\d+)?)", clause) or re.search(r"(\d+(?:\.\d+)?)\s*(?:more|higher|extra|less|lower|fewer)\b", clause)
         or re.search(rf"\b(?:{UP_WORDS}|{DOWN_WORDS})\s+(\d+(?:\.\d+)?)(?!\s*%)", clause))
    if m and sign:
        return {"kind": "abs", "value": float(m.group(1)) * sign, "unsigned": float(m.group(1)), "explicit_direction": True}
    m = re.search(r"\bto\s*(\d+(?:\.\d+)?)", clause)
    if m:
        return {"kind": "set", "value": float(m.group(1)), "unsigned": float(m.group(1)), "explicit_direction": False}
    return None


def _adjustment(amount: dict, rule: dict, label: str) -> tuple[dict | None, str | None]:
    """An adjustment for an input from a parsed amount, with the user-facing wording of what it means."""
    kind, v = amount["kind"], amount["value"]
    rate = _is_rate(rule)
    if kind == "pct":
        if v <= -100:
            return None, f"{label} cannot fall by {abs(v):g}% or more"
        return {"type": "pct", "value": v}, f"{label}: {'up' if v >= 0 else 'down'} {abs(v):g}%"
    if kind == "pp":
        if not rate:
            return {"type": "pct", "value": v}, f"{label}: {'up' if v >= 0 else 'down'} {abs(v):g}% (percentage points apply to rates; {label.lower()} is not one, so this is a % change)"
        return {"type": "abs", "value": v / 100}, f"{label}: {'up' if v >= 0 else 'down'} {abs(v):g} percentage points"
    if kind == "abs":
        return {"type": "abs", "value": v}, f"{label}: {'up' if v >= 0 else 'down'} by {abs(v):,.10g}"
    value = v / 100 if rate and v > 1 else v      # "conversion to 5" on a 0-1 rate is read as 5%
    return {"type": "set", "value": value}, f"{label}: set to {value:,.10g}"


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"\b(?:and then|but|then|while|plus|as well as)\b|[;,]|\band\b(?=\s+(?:also\s+)?(?:" + UP_WORDS + "|" + DOWN_WORDS + "|" + SET_WORDS + r")\b)", text)
    return [p.strip() for p in parts if p and p.strip()]


def _clauses_to_adjustments(text: str, index: Index) -> tuple[dict, list[str], dict | None]:
    """Adjustments from the clauses of one scenario. Returns (adjustments, interpretation, clarification_or_None)."""
    spec = index.spec
    adjustments: dict[str, dict] = {}
    notes: list[str] = []
    clauses = _split_clauses(text)
    amounts = [_amount(c) for c in clauses]
    for i, c in enumerate(clauses):                           # "increase price and volume by 10%": the amount belongs to both
        if amounts[i] is None and index.find(c, index.inputs):
            nxt = next((a for a in amounts[i + 1:] if a), None)
            if nxt and not _direction(c) and re.search(r"\band\b", text):
                amounts[i] = nxt
    for c, amount in zip(clauses, amounts):
        found = index.find(c, index.inputs)
        if re.search(UNCHANGED, c) and found:
            notes.append(f"{spec['inputs'][found[0][0]]['label']}: left unchanged")
            continue
        if not found:
            generic = index.generic_cost_candidates(c)
            if len(generic) > 1 and re.search(r"\bcosts?\b", _clean(c)) and amount:
                return {}, notes, {"question": "Which cost do you mean?", "missing": "input",
                                   "options": [{"label": spec["inputs"][k]["label"], "value": k} for k in generic]}
            if amount or _direction(c):
                return {}, notes, {"question": f"I could not match “{_noun(c)}” to an input of this model. Which input do you mean?",
                                   "missing": "input", "options": [{"label": r["label"], "value": k} for k, r in spec["inputs"].items()]}
            continue
        keys = list(dict.fromkeys(f[0] for f in found))
        if len(keys) > 1 and amount is None:
            continue
        if amount is None:
            return {}, notes, {"question": f"By how much should {spec['inputs'][keys[0]]['label'].lower()} change?", "missing": "amount",
                               "options": [], "input": keys[0]}
        for k in keys:
            adj, note = _adjustment(amount, spec["inputs"][k], spec["inputs"][k]["label"])
            if adj is None:
                return {}, notes, {"question": note + ". Choose a smaller change.", "missing": "amount", "options": []}
            adjustments[k] = adj
            notes.append(note)
    return adjustments, notes, None


def _wants_break_even(t: str) -> bool:
    return bool(re.search(r"break[\s-]?even", t))


def _goal(text: str, index: Index) -> dict | None:
    spec = index.spec
    if not re.search(rf"\b(?:what|which|how (?:many|much))\b.*\b(?:would|should|do i need|do we need|must|need|to (?:achieve|reach|hit|get|earn|make))\b|break[\s-]?even|\bneed to\b", text):
        return None
    out_hits = index.find(text, index.outputs)
    target_output, target_value, interp = None, None, []
    primary = spec["primary_output"]
    if _wants_break_even(text):
        target_output, target_value = primary, 0.0
        interp.append(f"Goal: {spec['outputs'][primary].lower()} reaches 0 (break-even)")
    else:
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        num = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\s*%)", text)
        if out_hits:
            target_output = out_hits[0][0]
            for h in out_hits:                              # prefer the output the number is next to
                target_output = h[0]
                break
        if pct and out_hits and any("margin" in h[1] or h[0].endswith("_pct") for h in out_hits):
            target_output = next(h[0] for h in out_hits if "margin" in h[1] or h[0].endswith("_pct"))
            v = float(pct.group(1))
            unit = (spec.get("output_units") or {}).get(target_output, "")
            target_value = v if target_output.endswith("_pct") or unit.strip().lower() == "percent" else v / 100
        elif out_hits and num:
            target_value = float(num.group(1))
        elif num and not out_hits:
            target_output, target_value = primary, float(num.group(1))
        if target_output is None or target_value is None:
            return {"clarify": {"question": "What result do you want to reach, and what value should it have?", "missing": "target",
                                "options": [{"label": lbl, "value": k} for k, lbl in spec["outputs"].items()]}}
        interp.append(f"Goal: {spec['outputs'][target_output].lower()} reaches {target_value:,.10g}")
    scrub = text
    for h in out_hits:
        scrub = scrub.replace(h[1], " ")
    cands = list(dict.fromkeys(f[0] for f in index.find(scrub, index.inputs)))
    if not cands:
        return {"clarify": {"question": "Which input should I change to reach that?", "missing": "input",
                            "options": [{"label": r["label"], "value": k} for k, r in spec["inputs"].items()]}}
    if len(cands) > 1:
        return {"clarify": {"question": "Which one input should I solve for?", "missing": "input",
                            "options": [{"label": spec["inputs"][k]["label"], "value": k} for k in cands]}}
    interp.append(f"Solve for: {spec['inputs'][cands[0]]['label'].lower()}")
    return {"goal": {"target_output": target_output, "target_value": target_value, "solve_for": cands[0]}, "interpretation": interp}


def baseline_sha(spec: dict, baseline: dict, assumptions: dict | None = None) -> str:
    """A fingerprint of the numbers a conversation is about. Assumptions are filled in with the model's defaults first, so
    leaving one out and stating its default are the same thing."""
    full = {k: float((assumptions or {}).get(k, r["default"])) for k, r in spec["assumptions"].items()}
    body = {"b": {k: float(v) for k, v in baseline.items()}, "a": full}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def parse(message: str, spec: dict, context: dict | None = None, *, baseline: dict | None = None, assumptions: dict | None = None) -> dict:
    """Read one sentence. Returns {status: ok|clarify|unsupported|explain, intent?, interpretation, notes, context_used}."""
    if not isinstance(message, str) or not message.strip():
        return _unsupported("Type what you want to try.")
    if len(message) > MAX_MESSAGE:
        return _unsupported(f"Keep the request under {MAX_MESSAGE} characters.")
    t = _normalize(message)
    index = Index(spec)
    ctx, ctx_note = validate_context(context, spec, baseline, assumptions)
    notes = [ctx_note] if ctx_note else []
    last = (ctx or {}).get("last")

    if re.search(r"\bwhy\b.*\b(?:decreas|increas|fall|fell|rise|rose|drop|chang|go down|go up|lower|higher|worse|better)\w*", t):
        if not last:
            return _clarify("Run a scenario first, then I can explain how its result changed.", [], "context")
        return _result("explain", question="why", interpretation=["Explain why the last result changed, from its computed attribution"], notes=notes)
    if re.search(r"\bwhich\b.*\b(?:variable|input|factor|driver|lever|assumption)\b.*\b(?:impact|matter|biggest|most|largest|affect|influence)\w*|\bbiggest (?:impact|driver|effect)\b|\bmost (?:important|sensitive)\b", t):
        if not last:
            return _clarify("Run a scenario first, then I can say which input matters most.", [], "context")
        return _result("explain", question="drivers", interpretation=["Say which input accounts for the most of the last change"], notes=notes)

    cmp_prev = re.search(r"\bcompare\b.*\b(?:that|this|it)\b.*\b(?:previous|last|earlier|before|prior)\b|\bcompare\b.*\bprevious\b", t)
    if cmp_prev:
        hist = (ctx or {}).get("history") or []
        if len(hist) < 2:
            return _clarify("I need two scenarios to compare. Run another one first.", [], "context")
        a, b = hist[-2], hist[-1]
        intent = {"action": "evaluate", "scenarios": {a["name"]: a["adjustments"], b["name"]: b["adjustments"]}}
        return _result("ok", intent=intent, interpretation=[f"Compare “{a['name']}” with “{b['name']}”"], notes=notes + ["Both scenarios come from this conversation."])

    if re.search(r"^\s*compare\b", t):
        body = re.sub(r"^\s*compare\s+", "", t)
        sides = re.split(r"\s+(?:with|to|against|versus|vs\.?|and)\s+(?=(?:a |an |the )?(?:\d|[a-z]))", body, maxsplit=1)
        if len(sides) == 2:
            scns, interp = {}, []
            for side in sides:
                adj, note, clar = _clauses_to_adjustments(_as_clause(side, index), index)
                if clar:
                    return _clarify(clar["question"], clar["options"], clar["missing"], interp)
                if not adj:
                    return _clarify(f"I could not find a change in “{side.strip()}”. What should it change, and by how much?", [], "amount", interp)
                name = "; ".join(note)
                scns[name[:60]] = adj
                interp.append(name)
            return _result("ok", intent={"action": "evaluate", "scenarios": scns}, interpretation=["Compare: " + "  vs  ".join(interp)], notes=notes)

    if _wants_break_even(t) and not re.search(r"\b(?:what|which|how)\b.*\b(?:would|should|need|do)\b", t):
        return _result("ok", intent={"action": "break_even"}, interpretation=["Show the break-even point of the baseline"], notes=notes)

    goal = _goal(t, index)
    if goal is not None:
        if "clarify" in goal:
            c = goal["clarify"]
            return _clarify(c["question"], c["options"], c["missing"])
        return _result("ok", intent={"action": "goal_seek", "goal": goal["goal"]}, interpretation=goal["interpretation"], notes=notes)

    carry = bool(re.search(r"\b(?:also|too|as well|in addition|additionally|on top(?: of that)?|and also|plus|keep that)\b|^\s*(?:and|but)\b", t))
    adj, interp, clar = _clauses_to_adjustments(t, index)
    if clar:
        return _clarify(clar["question"], clar["options"], clar["missing"], interp)
    unchanged = [i for i in interp if i.endswith("left unchanged")]
    if not adj:
        if unchanged:
            return _unsupported("That only says an input stays as it is. Say what should change.")
        return _unsupported("I could not find a change or a goal in that sentence.")
    n_prev = len((ctx or {}).get("history") or [])
    merged, name = dict(adj), ("your scenario" if n_prev == 0 else f"scenario {n_prev + 1}")      # unique within a conversation
    if carry and last:
        base = dict(last["adjustments"])
        clash = [k for k in adj if k in base]
        merged = {**base, **adj}
        interp = [f"{spec['inputs'][k]['label']}: {_phrase(v)} (from the previous scenario)" for k, v in base.items() if k not in adj] + interp
        notes.append("Added to the previous scenario" + (f"; the new value replaces the earlier one for {', '.join(spec['inputs'][k]['label'] for k in clash)}" if clash else ""))
    elif last:
        notes.append("This is a fresh scenario: the earlier changes were not carried over. Say “also” to build on them.")
    if len(merged) > MAX_ADJUSTMENTS:
        return _unsupported(f"At most {MAX_ADJUSTMENTS} inputs can change at once.")
    return _result("ok", intent={"action": "evaluate", "scenarios": {name: merged}, "unchanged": [f[0] for c in _split_clauses(t) if re.search(UNCHANGED, c) for f in index.find(c, index.inputs)]},
                   interpretation=interp, notes=notes)


def _as_clause(side: str, index: Index) -> str:
    """'a 10% price increase' / '15% more volume' -> a clause the amount and input readers understand."""
    s = side.strip()
    m = re.match(r"^(?:a |an |the )?(\d+(?:\.\d+)?%?)\s+([a-z ]+?)\s+(increase|decrease|reduction|rise|cut|drop|boost|growth|improvement|decline)\b", s)
    if m:
        verb = "reduce" if m.group(3) in ("decrease", "reduction", "cut", "drop", "decline") else "increase"
        return f"{verb} {m.group(2)} by {m.group(1)}"
    m = re.match(r"^(?:a |an |the )?(\d+(?:\.\d+)?%?)\s+(more|less|higher|lower|fewer)\s+([a-z ]+)$", s)
    if m:
        verb = "reduce" if m.group(2) in ("less", "lower", "fewer") else "increase"
        return f"{verb} {m.group(3)} by {m.group(1)}"
    return s


def _phrase(a: dict) -> str:
    v = a["value"]
    return {"pct": f"{'up' if v >= 0 else 'down'} {abs(v):g}%", "abs": f"{'up' if v >= 0 else 'down'} by {abs(v):g}", "set": f"set to {v:g}"}[a["type"]]


# -------------------------------------------------------- validation and context --

def validate_intent(intent: Any, spec: dict) -> dict:
    """A request is accepted only if every input, output and number in it is real. Returns a cleaned copy."""
    if not isinstance(intent, dict):
        raise ScenarioError("A request must be an object")
    extra = set(intent) - {"action", "scenarios", "goal", "adjustments", "assumptions", "unchanged", "note"}
    if extra:
        raise ScenarioError(f"Unknown field(s) in the request: {', '.join(sorted(extra))}")
    action = intent.get("action")
    if action not in ("evaluate", "goal_seek", "break_even"):
        raise ScenarioError("action must be evaluate, goal_seek or break_even")
    out: dict[str, Any] = {"action": action}
    if action == "evaluate":
        scns = intent.get("scenarios")
        if not isinstance(scns, dict) or not 1 <= len(scns) <= MAX_SCENARIOS:
            raise ScenarioError(f"evaluate needs 1 to {MAX_SCENARIOS} scenarios")
        out["scenarios"] = {str(k)[:60]: _clean_adjustments(v, spec) for k, v in scns.items()}
    if action == "goal_seek":
        g = intent.get("goal")
        if not isinstance(g, dict) or set(g) - {"target_output", "target_value", "solve_for"}:
            raise ScenarioError("goal_seek needs {target_output, target_value, solve_for}")
        if g.get("target_output") not in spec["outputs"]:
            raise ScenarioError(f"Unknown output '{g.get('target_output')}'. Available: {', '.join(spec['outputs'])}")
        if g.get("solve_for") not in spec["inputs"]:
            raise ScenarioError(f"Unknown input '{g.get('solve_for')}'. Available: {', '.join(spec['inputs'])}")
        v = g.get("target_value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ScenarioError("The target value must be a finite number")
        out["goal"] = {"target_output": g["target_output"], "target_value": float(v), "solve_for": g["solve_for"]}
        if intent.get("adjustments"):
            out["adjustments"] = _clean_adjustments(intent["adjustments"], spec)
    return out


def _clean_adjustments(adj: Any, spec: dict) -> dict:
    if not isinstance(adj, dict) or not 1 <= len(adj) <= MAX_ADJUSTMENTS:
        raise ScenarioError(f"A scenario changes 1 to {MAX_ADJUSTMENTS} inputs")
    out = {}
    for k, a in adj.items():
        if k not in spec["inputs"]:
            raise ScenarioError(f"Unknown input '{k}'. Available: {', '.join(spec['inputs'])}")
        if not isinstance(a, dict) or a.get("type") not in ("pct", "abs", "set") or set(a) - {"type", "value"}:
            raise ScenarioError(f"The change to '{k}' needs a type (pct, abs or set) and a value")
        v = a.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ScenarioError(f"The change to '{k}' needs a finite number")
        out[k] = {"type": a["type"], "value": float(v)}
    return out


def check_numbers_stated(intent: dict, message: str) -> str | None:
    """None if every number in a request is one the user wrote (as written, expanded, or as a percentage); otherwise why not."""
    allowed = stated_numbers(message)
    values: list[float] = []
    for adj in (intent.get("scenarios") or {}).values():
        values += [a["value"] for a in adj.values()]
    if intent.get("goal"):
        values.append(intent["goal"]["target_value"])
    for adj in [intent.get("adjustments") or {}]:
        values += [a["value"] for a in adj.values()]
    for v in values:
        if v == 0 and intent.get("goal"):
            continue
        if not any(math.isclose(abs(v), abs(x), rel_tol=1e-9, abs_tol=1e-12) for x in allowed):
            return f"the number {v:g} is not one that was written in the request"
    return None


def validate_context(context: Any, spec: dict, baseline: dict | None, assumptions: dict | None) -> tuple[dict | None, str | None]:
    """The conversation so far, or None with the reason it was dropped. Context is never reused across a different model or
    baseline, and only well-formed entries that name real inputs survive."""
    if not context:
        return None, None
    if not isinstance(context, dict) or len(json.dumps(context, default=str)) > 20_000:
        return None, "The earlier conversation was ignored (it was not in the expected form)."
    if context.get("model") != spec["key"]:
        return None, "The earlier conversation was dropped because the model changed."
    if baseline is not None and context.get("baseline_sha") != baseline_sha(spec, baseline, assumptions):
        return None, "The earlier conversation was dropped because the baseline or assumptions changed."
    hist = []
    for h in (context.get("history") or [])[-6:]:
        try:
            hist.append({"name": str(h["name"])[:60], "adjustments": _clean_adjustments(h["adjustments"], spec)})
        except (KeyError, TypeError, ScenarioError):
            continue
    last = context.get("last")
    clean_last = None
    if isinstance(last, dict):
        try:
            clean_last = {"name": str(last.get("name", "your scenario"))[:60], "adjustments": _clean_adjustments(last["adjustments"], spec),
                          "summary": last.get("summary") if isinstance(last.get("summary"), dict) else {}}
        except (KeyError, TypeError, ScenarioError):
            clean_last = None
    return {"history": hist, "last": clean_last}, None


def next_context(spec: dict, baseline: dict, assumptions: dict | None, ctx: dict | None, name: str, adjustments: dict, summary: dict) -> dict:
    hist = list((ctx or {}).get("history") or [])
    hist.append({"name": name, "adjustments": adjustments})
    return {"model": spec["key"], "baseline_sha": baseline_sha(spec, baseline, assumptions), "history": hist[-6:],
            "last": {"name": name, "adjustments": adjustments, "summary": summary}}


def summarize_for_context(result: dict, name: str) -> dict:
    """What a follow-up question ('why did profit decrease?') needs from the last result, taken from the result itself."""
    p = result["primary_output"]
    s = result["scenarios"][name]
    return {"output": p, "label": result["output_labels"][p], "baseline_value": result["baseline"]["outputs"][p],
            "scenario_value": s["outputs"][p], "change_abs": (s["delta_vs_baseline"].get(p) or {}).get("abs"),
            "contributions": s["contributions"]["by_input"], "input_labels": result.get("input_labels", {})}


# ------------------------------------------------------------ optional translator --

TOOL_NAME = "submit_scenario_request"


def tool_schema(spec: dict) -> dict:
    adj = {"type": "object", "properties": {"input": {"type": "string", "enum": list(spec["inputs"])},
                                            "type": {"type": "string", "enum": ["pct", "abs", "set"]}, "value": {"type": "number"}},
           "required": ["input", "type", "value"]}
    return {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["evaluate", "goal_seek", "break_even", "cannot_translate"]},
        "scenarios": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "adjustments": {"type": "array", "items": adj}}}},
        "goal": {"type": "object", "properties": {"target_output": {"type": "string", "enum": list(spec["outputs"])},
                                                  "target_value": {"type": "number"}, "solve_for": {"type": "string", "enum": list(spec["inputs"])}}},
        "reason": {"type": "string"}}, "required": ["action"]}


def translate_with_model(message: str, spec: dict, call_tool: Callable | None = None) -> dict:
    """Ask a language model to put a sentence the rules could not read into the structured form. It may only use inputs and
    outputs of this model and numbers the user wrote; the answer is validated like any other, and refused otherwise."""
    if call_tool is None:
        from app.agents import llm_client
        if not llm_client.provider_available(TOOL_NAME):
            return _unsupported("No language model is available to read that sentence, and the rules could not.")
        call_tool = llm_client.call_tool
    system = ("You translate a business user's sentence into a structured what-if request. You do not compute anything and you never "
              "invent a number: use only numbers the user wrote. Use only these inputs: "
              + "; ".join(f"{k} ({r['label']}, {r['unit']})" for k, r in spec["inputs"].items()) + ". Outputs: "
              + "; ".join(f"{k} ({v})" for k, v in spec["outputs"].items()) + ". A percentage change is type pct with the number as written "
              "(10% is 10); an amount is abs; 'to X' is set. If the sentence is not a what-if, answer cannot_translate.")
    try:
        raw = call_tool(system, message, TOOL_NAME, tool_schema(spec), "Translate a sentence into a what-if request.", 800)
    except Exception as e:  # noqa: BLE001 - the model is optional
        return _unsupported(f"The language model could not be reached ({type(e).__name__}), and the rules could not read the sentence.")
    if not isinstance(raw, dict) or raw.get("action") == "cannot_translate" or "action" not in raw:
        reason = raw.get("reason") if isinstance(raw, dict) else None
        return _unsupported(str(reason or "That does not look like a what-if request."))
    try:
        intent: dict[str, Any] = {"action": raw["action"]}
        if raw["action"] == "evaluate":
            intent["scenarios"] = {str(s.get("name") or f"scenario {i + 1}")[:60]: {a["input"]: {"type": a["type"], "value": a["value"]} for a in s.get("adjustments", [])}
                                   for i, s in enumerate(raw.get("scenarios") or [])}
        if raw["action"] == "goal_seek":
            intent["goal"] = raw.get("goal")
        clean = validate_intent(intent, spec)
    except (ScenarioError, KeyError, TypeError, AttributeError) as e:
        return _unsupported(f"The model's reading was not usable ({e}).")
    why = check_numbers_stated(clean, message)
    if why:
        return _unsupported(f"The model's reading was refused: {why}.")
    interp = [f"{spec['inputs'][k]['label']}: {_phrase(a)}" for adj in (clean.get("scenarios") or {}).values() for k, a in adj.items()]
    if clean.get("goal"):
        g = clean["goal"]
        interp = [f"Goal: {spec['outputs'][g['target_output']].lower()} reaches {g['target_value']:g}", f"Solve for: {spec['inputs'][g['solve_for']]['label'].lower()}"]
    return _result("ok", intent=clean, interpretation=interp, notes=["Read by the language model; checked against this model's inputs and the numbers you wrote."],
                   translated_by="model")
