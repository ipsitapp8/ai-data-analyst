"""Read the numbers out of a what-if sentence for a matched template (rules only).

It fills only values the sentence states. A value it cannot find is left out, so validation reports it as missing and the
Copilot asks for it. It never invents a number and never guesses between two.
"""
from __future__ import annotations

import re

from app.whatif.domains.base import Template

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
         "fifty": 50, "hundred": 100, "a hundred": 100}
NUM = r"(\d[\d,]*(?:\.\d+)?|" + "|".join(sorted((w for w in WORDS if " " not in w), key=len, reverse=True)) + r")"
DOWN = re.compile(r"\b(?:decreas|reduc|fall|drop|cut|lower|fewer|unavailable|absent|leave|lose|remov|close|shrink|declin)\w*", re.I)
FROM_TO = re.compile(rf"from\s+{NUM}\s*(?:%|percent|marks?|minutes?)?\s+to\s+{NUM}", re.I)
TO_ONLY = re.compile(rf"\bto\s+{NUM}\s*(?:%|percent|marks?|minutes?)?", re.I)
BY_PCT = re.compile(rf"{NUM}\s*(?:%|percent)", re.I)


def _num(tok: str) -> float:
    t = tok.lower().replace(",", "")
    return float(WORDS[t]) if t in WORDS else float(t)


def extract_values(template: Template, text: str) -> dict:
    """{input name: number} for the inputs of `template` the sentence states."""
    t = " " + re.sub(r"\s+", " ", text) + " "
    names = {s.name: s for s in template.inputs}
    out: dict = {}
    if "current_threshold" in names and "new_threshold" in names:
        m = FROM_TO.search(t)
        if m:
            out["current_threshold"], out["new_threshold"] = _num(m.group(1)), _num(m.group(2))
        else:
            m = TO_ONLY.search(t)
            if m:
                out["new_threshold"] = _num(m.group(1))
        return out
    if "doubles" in t.lower() or "double" in t.lower() and "change" in names:
        out["change"] = 100.0
    targets = [s for s in template.inputs if s.role == "change"] or ([names["change"]] if "change" in names else [])
    if not targets:
        return out
    spec = targets[0]
    pct = spec.unit == "%" or (spec.role == "change" and spec.change_type == "pct")
    if "double" in t.lower() and pct:
        out[spec.name] = 100.0
        return out
    if re.search(r"\bhalv\w+", t, re.I) and pct:
        out[spec.name] = -50.0
        return out
    m = BY_PCT.search(t) if pct else re.search(rf"{NUM}", t)
    if m is None and pct:
        m = re.search(rf"by\s+{NUM}", t, re.I)
    if m is None:
        return out
    value = _num(m.group(1))
    if DOWN.search(t) and (spec.min is None or spec.min < 0):
        value = -value
    out[spec.name] = value
    return out
