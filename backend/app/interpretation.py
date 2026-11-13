"""The copilot's reading of one message, as data: what it understood, from
which words of the message, by which rule -- and what it could not read.

This is an additive record next to the operations `copilot.interpret` already
produces. It changes nothing about how a message is read, applied or computed:

* `Trace` observes the rule passes. A pass reports the words it reads, and
  every rewrite of its working text, so each reading can be traced back to the
  exact characters of the original message. The trace never alters the text a
  pass sees, and an internal failure only marks the record incomplete.
* `build` turns a reading (operations, clarification or "not understood") and
  its trace into an Interpretation.
* `to_operations` is the deterministic conversion back to operations. For a
  rules reading it reproduces the operations exactly.
* `validate` checks a record: whitelisted aggregations and operators, spans
  inside the message that say what the rule read, literal values that occur in
  the message, and a status that agrees with the content.

An Interpretation never holds a computed number. A number in it is a literal
copied from the message (a threshold, a year) or a value the dataset itself
contains; every result still comes from the metric engine.

Confidence is a label, never a score, and it never chooses between readings:
it is the weakest kind of evidence a resolved reading rests on.

    definitional  an approved team definition
    lexical       the words of the message name it directly: a column's name,
                  a value in the data, an aggregation or comparison word
    derived       a rule inferred it: a rephrased column name, the dataset's
                  file name, a place adjective, the only column of a kind, or
                  words the rules rewrote before reading them
"""
from __future__ import annotations

import copy
import difflib
import functools
import logging
import re
from typing import Any, Callable, TypedDict

from app import metric_engine as me

logger = logging.getLogger(__name__)

SCHEMA = "interp/1"
STATUSES = ("resolved", "needs_clarification", "unmapped")
TIERS = ("definitional", "lexical", "derived")   # strongest first
SOURCE_TIER = {
    "approved_metric": "definitional",
    "column_name": "lexical",
    "category_value": "lexical",
    "aggregation_word": "lexical",
    "comparison_phrase": "lexical",
    "explicit_operator": "lexical",
    "literal": "lexical",
    "keyword": "lexical",
    "column_phrasing": "derived",
    "column_inference": "derived",
    "dataset_name": "derived",
    "place_adjective": "derived",
}
SLOTS = ("measure", "filter", "group", "entity", "value")
INTERPRETERS = ("rules", "model", "semantic")
# Rewrites that only normalise spelling (case, punctuation, separators): the
# characters still say what the message said, so they do not make a reading derived.
NORMALIZING = ("normalize",)
_STATE_OPS = ("reset", "rebase", "drill_up", "clear_filters", "remove_filter")


# -------------------------------------------------------------------- types --

class Evidence(TypedDict, total=False):
    source: str            # a key of SOURCE_TIER
    rule: str              # stable id of the rule that read it, e.g. "copilot.read_comparisons"
    role: str              # what it supports: column, aggregation, operator, value, metric, rows, ...
    spans: list[list[int]]  # [start, end) in the original message, in message order
    text: str              # the message's own characters at those spans, joined by a space
    matched: str           # what the rule matched in its (normalised, possibly rewritten) text
    via: list[str]         # rewrites the matched characters went through, if any
    ref: dict              # what it points at: column, value, metric id and version, dataset name


class Interpretation(TypedDict, total=False):
    schema: str
    status: str                 # resolved | needs_clarification | unmapped
    interpreted_by: str         # rules | model | semantic
    entity: dict | None         # {"phrase", "meaning": rows|context, "evidence"}
    measures: list[dict]        # {"op_index", "part", "formula", "label", "aggregation", "column", "columns",
    #                              "metric_ref", "definition_filters" (part 0), "evidence"}
    filters: list[dict]         # {"op_index", "column", "operator", "value", "time_range": False, "evidence"}
    #                             {"op_index", "column", "time_range": True, "start", "end", "evidence"}
    group_by: list[dict]        # {"op_index", "column", "evidence"}
    state_ops: list[dict]       # {"op_index", "op", "column"?, "evidence"}
    ambiguities: list[dict]     # {"slot", "phrase", "spans", "reason", "candidates": [{"label", "binding"}]}
    clarification: dict | None  # {"reason", "options"}: what the user was asked, as sent
    unread: list[dict]          # content words nothing accounted for: {"spans", "text", "matched"}
    reason: str | None          # why a message was not mapped, when a rule said so
    confidence: str | None      # weakest evidence tier of a resolved rules reading
    trace: dict                 # {"complete", "realigned", "error"}


# ------------------------------------------------------------------- trace --

def _safe(method: Callable) -> Callable:
    """An observer must never change the reading: any failure inside the trace
    is recorded and the trace goes quiet."""
    @functools.wraps(method)
    def wrapper(self: "Trace", *args, **kwargs):
        if self.failed:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 - see docstring
            self.failed = f"{method.__name__}: {type(e).__name__}: {e}"
            logger.warning("Interpretation trace stopped", exc_info=True)
            return None
    return wrapper


class _NullTrace:
    """Stands in when no trace is wanted: every call does nothing."""

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


NULL_TRACE = _NullTrace()


def _cover(origins: list[tuple[int, int, str | None]], anchor: int) -> tuple[int, int]:
    spans = [(s, e) for s, e, _ in origins if e > s]
    if not spans:
        return anchor, anchor
    return min(s for s, _ in spans), max(e for _, e in spans)


class Trace:
    """Where every character of a pass's working text came from.

    `text` mirrors the working text; `_org[i]` is (start, end, via) for its
    i-th character: the original message characters it stands for, and the
    rewrite that produced it (None: the message's own character)."""

    def __init__(self, message: str):
        self.message = message
        self.text = message.lower()
        org = [(i, i + 1, None) for i, ch in enumerate(message) for _ in ch.lower()]
        self.realigned: list[str] = []
        if len(org) != len(self.text):       # a lower-casing that changes length: align instead
            org = [(min(i, len(message)), min(i + 1, len(message)), "lowercase") for i in range(len(self.text))]
            self.realigned.append("lowercase")
        self._org = org
        self._init_records()
        self.column_source: Callable[[str, str], str] | None = None
        self.dataset_name: str | None = None

    def _init_records(self) -> None:
        self._evidence: dict[Any, list[dict]] = {}
        self._held: list[Any] = []            # keeps targets alive so their ids stay unique
        self._stash: dict[str, Any] = {}
        self.ambiguities: list[dict] = []
        self.unread: list[dict] = []
        self.entity: dict | None = None
        self.reason: str | None = None
        self.failed: str | None = None

    def __bool__(self) -> bool:
        return True

    # -- keeping the mirror in step with the working text --

    def _key(self, target: Any) -> Any:
        if isinstance(target, (dict, list)):
            self._held.append(target)
            return ("obj", id(target))
        return target

    def _ensure(self, text: str, rule: str = "unlabelled") -> None:
        """Bring the mirror to `text`. Used when a pass's text is not the one
        the trace last saw: only a rewrite the instrumentation did not label."""
        if text == self.text:
            return
        if len(text) == len(self.text) and all(b == " " for a, b in zip(self.text, text) if a != b):
            # words blanked out once read (the fast path's way of consuming them): no new content
            self._align(text, "consumed")
            return
        self._align(text, rule)
        if rule == "unlabelled":
            self.realigned.append(rule)

    def _align(self, new: str, via: str) -> None:
        old, org = self.text, self._org
        if len(new) == len(old):              # same length: a character-for-character rewrite
            self._org = [o if a == b else (o[0], o[1], o[2] or via) for o, a, b in zip(org, old, new)]
            self.text = new
            return
        if self._project(new, via):
            return
        out: list[tuple[int, int, str | None]] = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag == "equal":
                out += org[i1:i2]
            elif tag in ("replace", "insert"):
                anchor = org[i1 - 1][1] if i1 > 0 and org else (org[0][0] if org else 0)
                s, e = _cover(org[i1:i2], anchor)
                out += [(s, e, via)] * (j2 - j1)
        self.text, self._org = new, out

    def _project(self, new: str, via: str) -> bool:
        """A rewrite that only turns separators into spaces and drops spaces
        (normalisation) maps character by character; False if it is not one."""
        old, org = self.text, self._org
        out, i = [], 0
        for ch in new:
            while i < len(old) and not (old[i] == ch or (ch == " " and not old[i].isalnum())):
                if old[i].isalnum():
                    return False
                i += 1
            if i == len(old):
                return False
            s, e, v = org[i]
            out.append((s, e, v if old[i] == ch else (v or via)))
            i += 1
        if any(c.isalnum() for c in old[i:]):
            return False
        self.text, self._org = new, out
        return True

    @_safe
    def sync(self, text: str, rule: str) -> None:
        """After a rewrite by `rule` (several places at once, or normalisation)."""
        if text != self.text:
            self._align(text, rule)

    def _replace(self, start: int, end: int, replacement: str, rule: str) -> None:
        """`text[start:end]` becomes `replacement`. Characters the replacement
        keeps at its start and end ("average number of" -> "average") stay the
        message's own; the rest stands for the characters it replaced."""
        old = self.text[start:end]
        p = 0
        while p < len(old) and p < len(replacement) and old[p] == replacement[p]:
            p += 1
        q = 0
        while q < len(old) - p and q < len(replacement) - p and old[-1 - q] == replacement[-1 - q]:
            q += 1
        a, b = start + p, end - q
        anchor = self._org[a - 1][1] if a > 0 and self._org else 0
        # what the replacement stands for: the words it replaced, not the blanks
        # left where earlier rules consumed other words
        words = [o for o, ch in zip(self._org[a:b], self.text[a:b]) if not ch.isspace()]
        s, e = _cover(words or self._org[a:b], anchor)
        self._org = self._org[:a] + [(s, e, rule)] * (len(replacement) - p - q) + self._org[b:]
        self.text = self.text[:start] + replacement + self.text[end:]

    @_safe
    def edit(self, before: str, start: int, end: int, replacement: str, rule: str) -> None:
        """After `before[start:end]` was replaced by `replacement`: exact."""
        self._ensure(before)
        self._replace(start, end, replacement, rule)

    @_safe
    def resub(self, before: str, after: str, pattern: str, repl: str | Callable, rule: str, *,
              start: int = 0, count: int = 0) -> None:
        """After `after = before[:start] + re.sub(pattern, repl, before[start:], count)`:
        the same substitutions, replayed one by one so each stays exact. Should
        the replay not give `after`, the mirror is aligned to it instead."""
        self._ensure(before)
        edits, n = [], 0
        for m in re.finditer(pattern, before[start:]):
            if count and n >= count:
                break
            edits.append((start + m.start(), start + m.end(), repl(m) if callable(repl) else m.expand(repl)))
            n += 1
        rebuilt, last = [], 0
        for s, e, r in edits:
            rebuilt += [before[last:s], r]
            last = e
        rebuilt.append(before[last:])
        if "".join(rebuilt) != after:
            self._align(after, rule)
            return
        for s, e, r in reversed(edits):
            self._replace(s, e, r, rule)

    # -- reading evidence off the mirror --

    def _fragments(self, start: int, end: int) -> tuple[list[list[int]], list[str]]:
        ranges, via = [], []
        for k in range(start, end):
            if self.text[k].isspace():
                continue
            s, e, v = self._org[k]
            if v and v not in NORMALIZING and v not in via:
                via.append(v)
            if e > s:
                ranges.append([s, e])
        merged: list[list[int]] = []
        for s, e in sorted(ranges):
            # adjacent, or separated only by spaces and punctuation: one fragment
            if merged and (s <= merged[-1][1] or not any(c.isalnum() for c in self.message[merged[-1][1]:s])):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged, via

    @_safe
    def evidence(self, text: str, start: int, end: int, *, source: str, rule: str, role: str,
                 ref: dict | None = None) -> dict | None:
        """Evidence for `text[start:end]`, where `text` is the pass's current text."""
        self._ensure(text)
        spans, via = self._fragments(start, end)
        ev: dict = {"source": source, "rule": rule, "role": role, "spans": spans,
                    "text": " ".join(self.message[s:e] for s, e in spans),
                    "matched": re.sub(r"\s+", " ", text[start:end]).strip()}
        if via:
            ev["via"] = via
        if ref:
            ev["ref"] = ref
        return ev

    @_safe
    def add(self, target: Any, *evidence: dict | None) -> None:
        bucket = self._evidence.setdefault(self._key(target), [])
        bucket.extend(e for e in evidence if e)

    @_safe
    def note(self, target: Any, text: str, start: int, end: int, **kwargs) -> None:
        self.add(target, self.evidence(text, start, end, **kwargs))

    def for_target(self, target: Any) -> list[dict]:
        key = ("obj", id(target)) if isinstance(target, (dict, list)) else target
        return list(self._evidence.get(key, []))

    @_safe
    def ambiguity(self, *, slot: str, reason: str, candidates: list[dict], text: str | None = None,
                  start: int | None = None, end: int | None = None, spans: list | None = None) -> None:
        """A clarification about to be asked. `candidates` are the structured
        bindings, in the order of the options the user is shown. The phrase is
        `text[start:end]`, or already-known original `spans`."""
        phrase = ""
        if text is not None and start is not None and end is not None:
            self._ensure(text)
            spans, _ = self._fragments(start, end)
            phrase = re.sub(r"\s+", " ", text[start:end]).strip()
        else:
            spans = [list(s) for s in spans or []]
            phrase = " ".join(self.message[s:e] for s, e in spans)
        self.ambiguities.append({"slot": slot, "phrase": phrase, "spans": spans, "reason": reason,
                                 "candidates": [dict(c) for c in candidates]})

    @_safe
    def unread_words(self, text: str, stop: set[str] | frozenset[str]) -> None:
        """The content words left in `text` that nothing accounted for."""
        self._ensure(text)
        for m in re.finditer(r"\S+", text):
            if m.group(0) not in stop:
                spans, _ = self._fragments(m.start(), m.end())
                self.unread.append({"spans": spans, "text": " ".join(self.message[s:e] for s, e in spans),
                                    "matched": m.group(0)})

    @_safe
    def classify(self, column: str, matched: str) -> str | None:
        """How `matched` names `column`: "column_name" or "column_phrasing"."""
        return self.column_source(column, matched) if self.column_source else "column_name"

    @_safe
    def stash(self, name: str, value: Any) -> None:
        self._stash[name] = value

    @_safe
    def take(self, name: str) -> Any:
        return self._stash.pop(name, None)

    @_safe
    def attach_pending(self, target: Any, key: str) -> None:
        """Evidence recorded before its operation existed, now given to it."""
        self.add(target, *self._evidence.pop(key, []))

    @_safe
    def set_entity(self, text: str, start: int, end: int, *, meaning: str, rule: str, ref: dict) -> None:
        ref = {"dataset_name": self.dataset_name, **ref}
        ev = self.evidence(text, start, end, source="dataset_name", rule=rule, role="entity", ref=ref)
        if ev is None:
            return
        if self.entity is None:
            self.entity = {"phrase": ev["matched"], "meaning": meaning, "evidence": [ev]}
        else:
            if meaning == "rows":
                self.entity["meaning"] = "rows"
            self.entity["evidence"].append(ev)

    @_safe
    def set_reason(self, reason: str) -> None:
        self.reason = reason

    # -- passes that read a piece of the text on their own --

    @_safe
    def child(self, pieces: list[tuple]) -> "Trace | None":
        """A trace for a text assembled from pieces: ("slice", text, start, end)
        of this trace's current text, or ("synth", string, spans, rule) -- words
        a rule wrote, standing for the given original spans."""
        c = Trace.__new__(Trace)
        c.message, c.column_source, c.dataset_name = self.message, self.column_source, self.dataset_name
        c._init_records()
        c.realigned = []
        text, org = [], []
        for piece in pieces:
            if piece[0] == "slice":
                _, cur, s, e = piece
                self._ensure(cur)
                text.append(cur[s:e])
                org += self._org[s:e]
            else:
                _, string, spans, rule = piece
                lo, hi = (min(s for s, _ in spans), max(e for _, e in spans)) if spans else (0, 0)
                text.append(string)
                org += [(lo, hi, rule)] * len(string)
        c.text, c._org = "".join(text), org
        return c

    @_safe
    def adopt(self, child: "Trace | None", mapping: dict[Any, Any]) -> None:
        """Take a child's evidence (child key -> this trace's target), and its
        ambiguities, unread words and failures."""
        if child is None:
            return
        if child.failed:
            self.failed = child.failed
            return
        for key, target in mapping.items():
            self.add(target, *child._evidence.get(key, []))
        self.ambiguities += child.ambiguities
        self.unread += child.unread
        self.realigned += child.realigned

    def evidence_of(self, key: Any) -> list[dict]:
        return list(self._evidence.get(key, []))


def restate(ev: dict | None, **changes) -> dict | None:
    """The same words read as evidence of something else (e.g. a time phrase
    that also chose the only date column). None stays None."""
    return {**ev, **changes} if isinstance(ev, dict) else None


def spans_of(ev: dict | None) -> list:
    return list(ev.get("spans") or []) if isinstance(ev, dict) else []


# ------------------------------------------------------------------- build --

def _tier(ev: dict) -> str:
    return "derived" if ev.get("via") else SOURCE_TIER.get(ev.get("source", ""), "derived")


def _measure(formula: str, label: Any, part: int, op_index: int, evidence: list[dict]) -> dict:
    aggregation, column, columns = None, None, []
    try:
        parsed = me.parse_formula(formula)
    except me.FormulaError:
        pass
    else:
        columns = parsed.columns
        if parsed.tree[0] == "agg":
            aggregation, column = parsed.tree[1], parsed.tree[2]
        else:
            aggregation = "formula"
    metric_ref = next((dict(e["ref"]) for e in evidence if e.get("source") == "approved_metric" and e.get("ref")), None)
    return {"op_index": op_index, "part": part, "formula": formula, "label": label, "aggregation": aggregation,
            "column": column, "columns": columns, "metric_ref": metric_ref, "evidence": evidence}


_KNOWN_KEYS = {
    "set_metric": {"op", "formula", "label", "filters", "metrics"},
    "add_filter": {"op", "column", "operator", "value"},
    "set_time_range": {"op", "column", "start", "end"},
    "drill_down": {"op", "column"},
    **{k: {"op", "column"} for k in _STATE_OPS},
}


def _extra(op: dict) -> dict:
    known = _KNOWN_KEYS.get(op.get("op"), {"op"})
    return {k: copy.deepcopy(v) for k, v in op.items() if k not in known}


def build(message: str, result: dict, trace: Trace | None = None, *, interpreted_by: str = "rules") -> dict:
    """The Interpretation of `result` (what `copilot.interpret`, the model
    fallback or the semantic layer returned for `message`)."""
    t = trace if isinstance(trace, Trace) else None
    ops = list(result.get("operations") or [])
    if "clarification" in result:
        status = "needs_clarification"
    elif result.get("unmapped") or not ops:
        status = "unmapped"
    else:
        status = "resolved"

    def ev(target: Any) -> list[dict]:
        return copy.deepcopy(t.for_target(target)) if t is not None else []

    measures, filters, group_by, state_ops = [], [], [], []
    for i, op in enumerate(ops):
        kind = op.get("op")
        extra = _extra(op)
        if kind == "set_metric":
            parts = [{"formula": op.get("formula"), "label": op.get("label")}]
            if op.get("metrics"):
                parts = [{"formula": m.get("formula"), "label": m.get("label")} for m in op["metrics"]]
            for p, m in enumerate(parts):
                evidence = (ev(op) if p == 0 else []) + (copy.deepcopy(t.evidence_of(("measure", id(op), p)))
                                                         if t is not None else [])
                item = _measure(m["formula"], m["label"], p, i, evidence)
                if p == 0:
                    item["definition_filters"] = copy.deepcopy(op["filters"]) if "filters" in op else None
                    if extra:
                        item["extra"] = extra
                measures.append(item)
        elif kind == "add_filter":
            item = {"op_index": i, "column": op.get("column"), "operator": op.get("operator"),
                    "value": copy.deepcopy(op.get("value")), "time_range": False, "evidence": ev(op)}
            filters.append({**item, **({"extra": extra} if extra else {})})
        elif kind == "set_time_range":
            item = {"op_index": i, "column": op.get("column"), "time_range": True, "start": op.get("start"),
                    "end": op.get("end"), "evidence": ev(op)}
            filters.append({**item, **({"extra": extra} if extra else {})})
        elif kind == "drill_down":
            group_by.append({"op_index": i, "column": op.get("column"), "evidence": ev(op),
                             **({"extra": extra} if extra else {})})
        else:
            item = {"op_index": i, "op": kind, "evidence": ev(op)}
            if "column" in op:
                item["column"] = op["column"]
            state_ops.append({**item, **({"extra": extra} if extra else {})})

    clarification = None
    ambiguities = copy.deepcopy(t.ambiguities) if t is not None else []
    if status == "needs_clarification":
        c = result["clarification"]
        clarification = {"reason": c.get("reason"), "options": copy.deepcopy(c.get("options") or [])}
        if ambiguities:
            # The structured bindings were recorded in the order of the options shown.
            last = ambiguities[-1]
            labels = [o.get("label") for o in clarification["options"]]
            last["candidates"] = [{"label": labels[k] if k < len(labels) else None, "binding": b}
                                  for k, b in enumerate(last["candidates"])]
            ambiguities = [last]
    else:
        ambiguities = []     # an ambiguity is a question to the user, or it is not one

    items = measures + filters + group_by + state_ops
    confidence = None
    if status == "resolved" and interpreted_by == "rules" and t is not None and not t.failed \
            and items and all(item["evidence"] for item in items):
        confidence = max((_tier(e) for item in items for e in item["evidence"]), key=TIERS.index)

    return {
        "schema": SCHEMA,
        "status": status,
        "interpreted_by": interpreted_by,
        "entity": copy.deepcopy(t.entity) if t is not None else None,
        "measures": measures,
        "filters": filters,
        "group_by": group_by,
        "state_ops": state_ops,
        "ambiguities": ambiguities,
        "clarification": clarification,
        "unread": copy.deepcopy(t.unread) if t is not None else [],
        "reason": t.reason if t is not None else None,
        "confidence": confidence,
        "trace": {"complete": t is not None and not t.failed and not t.realigned,
                  "realigned": list(t.realigned) if t is not None else [],
                  "error": t.failed if t is not None else None},
    }


# --------------------------------------------------------- back to operations --

def to_operations(interp: dict) -> list[dict]:
    """The operations an Interpretation stands for, in their original order.
    Deterministic; for a rules reading, identical to the operations it was built from."""
    rules = interp.get("interpreted_by") == "rules"
    by_index: dict[int, dict] = {}
    measures: dict[int, list[dict]] = {}
    for m in interp.get("measures") or []:
        measures.setdefault(m["op_index"], []).append(m)
    for i, parts in measures.items():
        parts = sorted(parts, key=lambda p: p["part"])
        first = parts[0]
        op = {"op": "set_metric", "formula": first["formula"]}
        if first.get("label") is not None:
            op["label"] = first["label"]
        if first.get("definition_filters") is not None:
            op["filters"] = copy.deepcopy(first["definition_filters"])
        if len(parts) > 1:
            op["metrics"] = [{"formula": p["formula"], "label": p["label"]} for p in parts]
        op.update(copy.deepcopy(first.get("extra") or {}))
        by_index[i] = op
    for f in interp.get("filters") or []:
        if f.get("time_range"):
            op = {"op": "set_time_range", "column": f["column"]}
            for key in ("start", "end"):
                if rules or f.get(key) is not None:
                    op[key] = f.get(key)
        else:
            op = {"op": "add_filter"}
            for key in ("column", "operator", "value"):
                if f.get(key) is not None:
                    op[key] = copy.deepcopy(f[key])
        op.update(copy.deepcopy(f.get("extra") or {}))
        by_index[f["op_index"]] = op
    for g in interp.get("group_by") or []:
        by_index[g["op_index"]] = {"op": "drill_down", "column": g["column"], **copy.deepcopy(g.get("extra") or {})}
    for s in interp.get("state_ops") or []:
        op = {"op": s["op"]}
        if "column" in s:
            op["column"] = s["column"]
        op.update(copy.deepcopy(s.get("extra") or {}))
        by_index[s["op_index"]] = op
    return [by_index[i] for i in sorted(by_index)]


# ---------------------------------------------------------------- validate --

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z%<>=!]+", " ", text.lower())).strip()


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _check_evidence(ev: Any, message: str, where: str) -> list[str]:
    if not isinstance(ev, dict):
        return [f"{where}: evidence is not an object"]
    problems = []
    if ev.get("source") not in SOURCE_TIER:
        problems.append(f"{where}: unknown evidence source {ev.get('source')!r}")
    if not isinstance(ev.get("rule"), str) or not ev.get("rule"):
        problems.append(f"{where}: evidence names no rule")
    spans = ev.get("spans")
    if not isinstance(spans, list) or not spans:
        return problems + [f"{where}: evidence has no span in the message"]
    last = -1
    for span in spans:
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)
                and 0 <= span[0] < span[1] <= len(message) and span[0] >= last):
            return problems + [f"{where}: span {span!r} is not an ordered range inside the message"]
        last = span[1]
    text = " ".join(message[s:e] for s, e in spans)
    if ev.get("text") != text:
        problems.append(f"{where}: evidence text {ev.get('text')!r} is not the message at its spans ({text!r})")
    if not ev.get("via") and _norm(text) != _norm(ev.get("matched") or ""):
        problems.append(f"{where}: the message at {spans} says {text!r}, the rule read {ev.get('matched')!r}")
    return problems


def _literal_ok(value: Any, evidence: list[dict]) -> bool:
    """A filter value is a literal of the message: a number written there, or
    the words themselves."""
    literals = [e for e in evidence if e.get("role") == "value" and e.get("source") == "literal"]
    if not literals:
        return True                       # a value read from the data, checked through its reference
    written = " ".join(e.get("matched") or "" for e in literals)
    numbers = {float(n) for e in literals for n in _NUMBER.findall(e.get("text") or "")}
    values = value if isinstance(value, list) else [value]
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if float(v) not in numbers:
                return False
        elif _norm(str(v)) not in _norm(written):
            return False
    return True


def validate(interp: dict, message: str, operations: list[dict] | None = None) -> list[str]:
    """Problems with an Interpretation of `message`. Empty means valid. With
    `operations`, also checks that the record converts back to exactly them."""
    problems: list[str] = []
    if interp.get("schema") != SCHEMA:
        problems.append(f"schema is {interp.get('schema')!r}, expected {SCHEMA!r}")
    status = interp.get("status")
    if status not in STATUSES:
        problems.append(f"status {status!r} is not one of {STATUSES}")
    if interp.get("interpreted_by") not in INTERPRETERS:
        problems.append(f"interpreted_by {interp.get('interpreted_by')!r} is not one of {INTERPRETERS}")
    rules = interp.get("interpreted_by") == "rules"
    items = [*(("measure", m) for m in interp.get("measures") or []),
             *(("filter", f) for f in interp.get("filters") or []),
             *(("group", g) for g in interp.get("group_by") or []),
             *(("state", s) for s in interp.get("state_ops") or [])]
    if status == "resolved" and not items:
        problems.append("a resolved reading has no operations")
    if status in ("needs_clarification", "unmapped") and items:
        problems.append(f"a reading that is {status} carries operations")
    if status == "needs_clarification" and not interp.get("clarification"):
        problems.append("a clarification is needed but none was recorded")

    for kind, item in items:
        where = f"{kind} #{item.get('op_index')}"
        evidence = item.get("evidence") or []
        if rules and status == "resolved" and not evidence:
            problems.append(f"{where}: no evidence for a rules reading")
        for e in evidence:
            problems += _check_evidence(e, message, where)
        if kind == "measure":
            if "value" in item or "result" in item:
                problems.append(f"{where}: a measure carries a value; numbers come only from the engine")
            agg = item.get("aggregation")
            if agg not in (*me.AGGREGATIONS, "formula"):
                problems.append(f"{where}: aggregation {agg!r} is not supported")
            try:
                parsed = me.parse_formula(item.get("formula") or "")
                if parsed.tree[0] == "agg" and (parsed.tree[1], parsed.tree[2]) != (agg, item.get("column")):
                    problems.append(f"{where}: formula {item.get('formula')!r} is not {agg}({item.get('column')})")
            except me.FormulaError as e:
                problems.append(f"{where}: formula is not valid: {e}")
        elif kind == "filter" and not item.get("time_range"):
            if item.get("operator") is not None and item["operator"] not in me.FILTER_OPS:
                problems.append(f"{where}: operator {item['operator']!r} is not supported")
            if rules and not _literal_ok(item.get("value"), evidence):
                problems.append(f"{where}: value {item.get('value')!r} is not written in the message")
            for e in evidence:
                ref = e.get("ref") or {}
                if e.get("role") == "value" and "value" in ref:
                    values = item.get("value") if isinstance(item.get("value"), list) else [item.get("value")]
                    if ref["value"] not in values:
                        problems.append(f"{where}: evidence reads value {ref['value']!r}, the filter has {item.get('value')!r}")
        elif kind == "filter":
            if item.get("start") is None and item.get("end") is None:
                problems.append(f"{where}: a time range with neither start nor end")
            written = " ".join(e.get("text") or "" for e in evidence if e.get("role") == "time_range")
            for bound in (item.get("start"), item.get("end")):
                if rules and bound is not None and str(bound)[:4] not in written:
                    problems.append(f"{where}: the year of {bound!r} is not written in the message")

    if status == "needs_clarification":
        for a in interp.get("ambiguities") or []:
            if a.get("slot") not in SLOTS:
                problems.append(f"ambiguity slot {a.get('slot')!r} is not one of {SLOTS}")
            for span in a.get("spans") or []:
                if not (0 <= span[0] < span[1] <= len(message)):
                    problems.append(f"ambiguity span {span!r} is outside the message")
    for u in interp.get("unread") or []:
        for span in u.get("spans") or []:
            if not (0 <= span[0] < span[1] <= len(message)):
                problems.append(f"unread span {span!r} is outside the message")

    confidence = interp.get("confidence")
    if confidence is not None:
        if confidence not in TIERS:
            problems.append(f"confidence {confidence!r} is not one of {TIERS}")
        elif status == "resolved":
            weakest = max((_tier(e) for _, item in items for e in item.get("evidence") or []),
                          key=TIERS.index, default=None)
            if weakest != confidence:
                problems.append(f"confidence is {confidence!r}, the weakest evidence is {weakest!r}")
    elif status == "resolved" and rules and (interp.get("trace") or {}).get("complete"):
        problems.append("a complete rules reading has no confidence")

    if operations is not None:
        rebuilt = to_operations(interp)
        if rebuilt != operations:
            problems.append(f"the record converts to {rebuilt!r}, not to the operations {operations!r}")
    return problems


def semantic_ambiguity(message: str, ambiguous: dict, normalized: str) -> dict:
    """The Interpretation of a message the semantic layer stopped on: one
    phrase that names more than one approved definition. `normalized` is the
    message as the semantic layer read it (`semantic.norm_term`)."""
    t = Trace(message)
    t.sync(normalized, "normalize")
    term = ambiguous["term"]
    m = re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", t.text)
    candidates = [{"metric_id": c["metric_id"], "formula": c["formula"]} for c in ambiguous["candidates"]]
    reason = f"'{term}' has more than one approved definition."
    if m is not None:
        t.ambiguity(slot="measure", reason=reason, candidates=candidates, text=t.text, start=m.start(), end=m.end())
    else:
        t.ambiguity(slot="measure", reason=reason, candidates=candidates)
    options = [{"label": f"{c['name']} = {c['formula']}", "message": c["formula"]} for c in ambiguous["candidates"]]
    return build(message, {"clarification": {"reason": reason + " Which one do you mean?", "options": options}},
                 t, interpreted_by="semantic")
