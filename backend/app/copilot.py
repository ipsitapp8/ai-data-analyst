"""Stateful analytical copilot: a multi-turn session over one dataset version.

The session holds an explicit analytical state -- metric, filters, grouping --
and every user message is turned into controlled operations on that state:

    set_metric · add_filter · remove_filter · clear_filters · set_time_range
    drill_down · drill_up · rebase (move to the latest data) · reset

After each turn the answer is recomputed from the dataset by the deterministic
metric engine. The conversation is a record of decisions, never the source of
a number: nothing is answered from memory, and when a message refers to
something the state does not contain, the copilot says so instead of inventing
it.

A message is interpreted by rules first. Only when no rule applies, and a model
is configured, is a model asked to translate the message into operations --
and its output is validated against the same whitelist and the dataset's real
columns before anything runs. A model never computes the answer.

Ambiguity produces a question back to the user (which column? which metric?),
with the options listed, and leaves the state unchanged.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy.orm import Session

from app import fastpath, provenance, semantic
from app import metric_engine as me
from app.dataset_versions import latest_version
from app.models import CopilotSession, CopilotTurn, Dataset, DatasetVersion

logger = logging.getLogger(__name__)

OPERATIONS = ("set_metric", "add_filter", "remove_filter", "clear_filters", "set_time_range",
              "drill_down", "drill_up", "rebase", "reset")
MAX_TURNS = 200
MAX_MESSAGE_CHARS = 500
MAX_CATEGORY_VALUES = 300
RESULT_LIMIT = 25
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
_NEEDS_ANALYSIS = re.compile(r"\bwhy\b|root cause|forecast|predict|trend|correlat|outlier|anomal|explain", re.IGNORECASE)

ACTIONS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OPERATIONS)},
                    "formula": {"type": "string", "description": "For set_metric, e.g. sum(revenue)"},
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "in", "contains"]},
                    "value": {"type": "string"},
                    "start": {"type": "string"}, "end": {"type": "string"},
                },
                "required": ["op"],
            },
        },
        "cannot_map": {"type": "boolean", "description": "True if the message is not one of these operations."},
    },
    "required": ["operations", "cannot_map"],
}
ACTIONS_SYSTEM = (
    "You translate one user message into operations on an analytical state (metric, filters, group-by). "
    "Use ONLY the listed operations and ONLY the column names given. Never answer the question yourself and "
    "never invent a column, a value or a number. If the message is not one of these operations, set cannot_map."
)


class CopilotError(ValueError):
    pass


def empty_state() -> dict:
    return {"formula": None, "metric_label": None, "filters": [], "group_by": [], "history_depth": 0}


# ------------------------------------------------------------ interpretation --

def _phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(text).strip().lower()))


def category_index(df: pd.DataFrame, profile: dict) -> dict[str, list[tuple[str, str]]]:
    """lower-cased value -> [(column, original value)] for categorical columns."""
    index: dict[str, list[tuple[str, str]]] = {}
    for c in profile.get("columns", []):
        if c.get("kind") != "categorical" or c["name"] not in df.columns:
            continue
        values = df[c["name"]].dropna().astype(str).unique()
        if len(values) > MAX_CATEGORY_VALUES:
            continue
        for v in values:
            index.setdefault(_phrase(v), []).append((c["name"], v))
    return index


def _time_columns(profile: dict) -> list[str]:
    return [c["name"] for c in profile.get("columns", []) if c.get("kind") == "datetime"]


def _parse_time_range(text: str) -> tuple[str | None, str | None, tuple[int, int]] | None:
    """(start, end, span of the matched phrase) from the raw, lower-cased
    message -- before punctuation is normalised, so ISO dates are still intact."""
    iso = r"(\d{4}-\d{2}-\d{2})"
    m = re.search(rf"\b(?:from|between)\s+{iso}\s+(?:to|and|until|through)\s+{iso}", text)
    if m:
        return m.group(1), m.group(2), m.span()
    m = re.search(rf"\b(?:since|after|from)\s+{iso}", text)
    if m:
        return m.group(1), None, m.span()
    m = re.search(rf"\b(?:before|until|up to)\s+{iso}", text)
    if m:
        return None, m.group(1), m.span()
    m = re.search(r"\b(?:in|during|for)\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", text)
    if m:
        month, year = _MONTHS[m.group(1)], int(m.group(2))
        end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return f"{year:04d}-{month:02d}-01", end.strftime("%Y-%m-%d"), m.span()
    m = re.search(r"\b(?:in|during|for)\s+(?:the year\s+)?((?:19|20)\d{2})\b", text)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(1)}-12-31", m.span()
    return None


def interpret(message: str, state: dict, profile: dict, cat_index: dict, resolved_metrics: list[dict]) -> dict:
    """Rules only. Returns {"operations": [...]} or {"clarification": {...}} or
    {"unmapped": True}."""
    raw = message.lower()
    time_range = _parse_time_range(raw)
    if time_range is not None:
        start, end = time_range[2]
        raw = raw[:start] + " " + raw[end:]
    text = _phrase(re.sub(r"[^\w\s%.<>=!\-:/]", " ", raw))
    ops: list[dict] = []
    columns = {c["name"]: _phrase(c["name"]) for c in profile.get("columns", [])}

    if re.search(r"\b(reset|start over|start again|clear everything)\b", text):
        return {"operations": [{"op": "reset"}]}
    if re.search(r"\b(use|switch to|move to|update to|rebase)\b.*\b(latest|newest|new|current)\b.*\b(data|version|dataset)\b", text) \
            or text in ("rebase", "use latest data"):
        return {"operations": [{"op": "rebase"}]}
    if re.search(r"\b(clear|remove|drop)\s+(all\s+)?(the\s+)?filters?\b", text):
        named = [name for name, phrase in columns.items() if re.search(rf"\b{re.escape(phrase)}\b", text)]
        if named:
            return {"operations": [{"op": "remove_filter", "column": n} for n in named]}
        return {"operations": [{"op": "clear_filters"}]}
    if re.search(r"\b(overall|in total|drill up|remove (the )?(breakdown|grouping)|no breakdown|ungroup)\b", text):
        ops.append({"op": "drill_up"})

    if time_range is not None:
        time_cols = _time_columns(profile)
        if not time_cols:
            return {"clarification": {"reason": "This dataset has no date column, so a time range cannot be applied.",
                                      "options": []}}
        if len(time_cols) > 1:
            named = [c for c in time_cols if re.search(rf"\b{re.escape(_phrase(c))}\b", text)]
            if len(named) != 1:
                return {"clarification": {
                    "reason": "This dataset has more than one date column. Which one should the time range use?",
                    "options": [{"label": c, "message": f"{message} (by {c})"} for c in time_cols]}}
            time_cols = named
        ops.append({"op": "set_time_range", "column": time_cols[0], "start": time_range[0], "end": time_range[1]})

    # explicit comparison: "where price > 10", "quantity >= 5"
    for name, phrase in sorted(columns.items(), key=lambda kv: -len(kv[1])):
        m = re.search(rf"\b{re.escape(phrase)}\s*(>=|<=|!=|==|=|>|<|is not|is|equals)\s*([\w.\-]+)((?: [\w.\-]+){{0,2}})", text)
        if m:
            op = {"=": "==", "is": "==", "equals": "==", "is not": "!="}.get(m.group(1), m.group(1))
            # The value is one word, unless a longer run of words is a real
            # value of this column ("region = new south wales").
            words = [m.group(2), *m.group(3).split()]
            value, used = words[0], 1
            for n in range(len(words), 0, -1):
                hit = [v for c, v in cat_index.get(" ".join(words[:n]), []) if c == name]
                if hit:
                    value, used = hit[0], n
                    break
            end = m.start(2) + len(" ".join(words[:used]))
            ops.append({"op": "add_filter", "column": name, "operator": op, "value": value})
            text = text[:m.start()] + " " + text[end:]

    # a category value named on its own: "only west", "for enterprise", "excluding returns"
    negate = bool(re.search(r"\b(exclud\w*|without|except|not)\b", text))
    for value_phrase in sorted(cat_index, key=len, reverse=True):
        if len(value_phrase) < 2 or not re.search(rf"(?<![a-z0-9]){re.escape(value_phrase)}(?![a-z0-9])", text):
            continue
        owners = cat_index[value_phrase]
        already = {o.get("column") for o in ops if o["op"] == "add_filter"}
        owners = [o for o in owners if o[0] not in already]
        if not owners:
            continue
        if len({c for c, _ in owners}) > 1:
            return {"clarification": {
                "reason": f"'{owners[0][1]}' is a value in more than one column. Which do you mean?",
                "options": [{"label": f"{c} = {v}", "message": f"{c} = {v}"} for c, v in owners]}}
        column, original = owners[0]
        ops.append({"op": "add_filter", "column": column, "operator": "!=" if negate else "==", "value": original})
        text = re.sub(rf"(?<![a-z0-9]){re.escape(value_phrase)}(?![a-z0-9])", " ", text, count=1)

    # grouping: "by region", "break down by product", "drill into channel"
    m = re.search(r"\b(?:by|per|for each|break(?:ing)? (?:it |this |that )?down by|split by|drill (?:down )?(?:into|by|on))\s+(?:the\s+)?", text)
    if m:
        rest = text[m.end():]
        for name, phrase in sorted(columns.items(), key=lambda kv: -len(kv[1])):
            if re.match(rf"{re.escape(phrase)}s?(?![a-z0-9])", rest):
                ops.append({"op": "drill_down", "column": name})
                text = text[:m.start()] + " " + rest[len(phrase):]
                break

    # a metric: an approved definition, or "<aggregation> <column>"
    parsed = fastpath.parse(text, profile, resolved_metrics) if text.strip() else None
    if parsed is not None and parsed["kind"] == "ambiguous":
        return {"clarification": {"reason": parsed["reason"] + " Which one do you mean?",
                                  "options": [{"label": o["label"], "message": o["question"]} for o in parsed["options"]]}}
    if parsed is not None:
        ops.insert(0, {"op": "set_metric", "formula": parsed["formula"], "label": parsed["label"],
                       "filters": parsed["filters"]})
        for g in parsed["group_by"]:
            if not any(o["op"] == "drill_down" and o["column"] == g for o in ops):
                ops.append({"op": "drill_down", "column": g})
    if not ops:
        return {"unmapped": True}
    return {"operations": ops}


def interpret_with_model(message: str, state: dict, profile: dict) -> dict:
    """Ask a model to map the message onto operations. Its answer is data: it is
    validated by `apply_operations` like any other input."""
    from app.agents import llm_client, prompts

    cols = [f"{c['name']} ({c.get('kind')})" for c in profile.get("columns", [])][:120]
    user = (f"Columns: {', '.join(cols)}\nCurrent state: metric={state.get('formula')}, "
            f"filters={state.get('filters')}, group_by={state.get('group_by')}\n"
            f"Operations: {', '.join(OPERATIONS)}\nUser message: {message!r}")
    try:
        out = llm_client.call_tool(system=ACTIONS_SYSTEM + prompts.UNTRUSTED_DATA_NOTICE, user_content=user,
                                   tool_name="submit_operations", tool_schema=ACTIONS_TOOL_SCHEMA,
                                   tool_description="Submit the operations for this message.", max_tokens=600)
    except Exception:  # noqa: BLE001 - no model, no quota: fall back to "not understood"
        logger.info("Copilot model interpretation unavailable", exc_info=True)
        return {"unmapped": True}
    if out.get("cannot_map") or not out.get("operations"):
        return {"unmapped": True}
    ops = []
    for raw in list(out["operations"])[:5]:
        if not isinstance(raw, dict) or raw.get("op") not in OPERATIONS:
            return {"unmapped": True}
        op = {"op": raw["op"]}
        for key in ("formula", "column", "operator", "value", "start", "end"):
            if raw.get(key) not in (None, ""):
                op[key] = raw[key]
        ops.append(op)
    return {"operations": ops, "interpreted_by": "model"}


# -------------------------------------------------------------------- state --

def apply_operations(state: dict, operations: list[dict], profile: dict) -> tuple[dict, list[str]]:
    """Apply validated operations. Returns (new state, human-readable decisions).
    Raises CopilotError on anything that is not a valid operation on this dataset."""
    columns = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    new = {"formula": state.get("formula"), "metric_label": state.get("metric_label"),
           "filters": [dict(f) for f in state.get("filters") or []], "group_by": list(state.get("group_by") or []),
           "history_depth": int(state.get("history_depth") or 0) + 1}
    decisions: list[str] = []

    def need_column(name) -> str:
        if not isinstance(name, str) or name not in columns:
            raise CopilotError(f"'{name}' is not a column of this dataset")
        return name

    for op in operations:
        kind = op.get("op")
        if kind == "reset":
            new = {**empty_state(), "history_depth": new["history_depth"]}
            decisions.append("Cleared the metric, filters and grouping.")
        elif kind == "set_metric":
            try:
                parsed = me.parse_formula(str(op.get("formula", "")))
            except me.FormulaError as e:
                raise CopilotError(str(e)) from e
            problems = me.validate_against_profile(parsed, profile)
            if problems:
                raise CopilotError("; ".join(problems))
            new["formula"], new["metric_label"] = parsed.text, str(op.get("label") or parsed.text)[:120]
            for f in op.get("filters") or []:
                new["filters"] = [x for x in new["filters"] if x["column"] != f["column"]] + [dict(f)]
            decisions.append(f"Metric set to {new['metric_label']} ({parsed.text}).")
        elif kind == "add_filter":
            column = need_column(op.get("column"))
            operator = op.get("operator", "==")
            value = op.get("value")
            try:
                clean = me.normalize_filters([{"column": column, "op": operator, "value": value}], list(columns))[0]
            except me.FormulaError as e:
                raise CopilotError(str(e)) from e
            new["filters"] = [f for f in new["filters"] if not (f["column"] == column and not f.get("time_range"))]
            new["filters"].append(clean)
            decisions.append(f"Filter added: {column} {clean['op']} {clean['value']}.")
        elif kind == "remove_filter":
            column = need_column(op.get("column"))
            before = len(new["filters"])
            new["filters"] = [f for f in new["filters"] if f["column"] != column]
            decisions.append(f"Filter on {column} removed." if len(new["filters"]) < before
                             else f"There was no filter on {column}.")
        elif kind == "clear_filters":
            new["filters"] = []
            decisions.append("All filters removed.")
        elif kind == "set_time_range":
            column = need_column(op.get("column"))
            if columns[column] != "datetime":
                raise CopilotError(f"'{column}' is not a date column")
            start, end = op.get("start"), op.get("end")
            for bound in (start, end):
                if bound is not None and pd.isna(pd.to_datetime(bound, errors="coerce")):
                    raise CopilotError(f"'{bound}' is not a date")
            if start is None and end is None:
                raise CopilotError("A time range needs a start or an end")
            new["filters"] = [f for f in new["filters"] if not f.get("time_range")]
            if start is not None:
                new["filters"].append({"column": column, "op": ">=", "value": start, "time_range": True})
            if end is not None:
                new["filters"].append({"column": column, "op": "<=", "value": f"{end} 23:59:59", "time_range": True})
            decisions.append(f"Time range on {column}: {start or 'the beginning'} to {end or 'the end'}.")
        elif kind == "drill_down":
            column = need_column(op.get("column"))
            if column not in new["group_by"]:
                if len(new["group_by"]) >= 2:
                    new["group_by"] = new["group_by"][1:]
                new["group_by"].append(column)
            decisions.append(f"Broken down by {', '.join(new['group_by'])}.")
        elif kind == "drill_up":
            if new["group_by"]:
                new["group_by"] = new["group_by"][:-1]
                decisions.append("Removed one level of breakdown." if new["group_by"] else "Showing the overall figure.")
            else:
                decisions.append("There is no breakdown to remove.")
        elif kind == "rebase":
            continue  # handled by the caller: it changes the session's dataset version
        else:
            raise CopilotError(f"Unsupported operation '{kind}'")
    return new, decisions


def prune_state(state: dict, profile: dict) -> tuple[dict, list[str]]:
    """After a dataset change: drop whatever no longer exists, and say so."""
    columns = {c["name"] for c in profile.get("columns", [])}
    notes = []
    new = dict(state)
    kept_filters = [f for f in state.get("filters") or [] if f["column"] in columns]
    if len(kept_filters) != len(state.get("filters") or []):
        notes.append("Removed filters on columns that the new data no longer has.")
    new["filters"] = kept_filters
    kept_groups = [g for g in state.get("group_by") or [] if g in columns]
    if len(kept_groups) != len(state.get("group_by") or []):
        notes.append("Removed a breakdown by a column that the new data no longer has.")
    new["group_by"] = kept_groups
    if state.get("formula"):
        try:
            if me.validate_against_profile(me.parse_formula(state["formula"]), profile):
                raise me.FormulaError("columns missing")
        except me.FormulaError:
            new["formula"], new["metric_label"] = None, None
            notes.append("The metric used a column the new data no longer has, so it was cleared.")
    return new, notes


# ------------------------------------------------------------------ compute --

def _engine_filters(state: dict) -> list[dict]:
    return [{"column": f["column"], "op": f["op"], "value": f["value"]} for f in state.get("filters") or []]


def compute(csv_path: str, state: dict) -> dict:
    """The current state, evaluated on the dataset."""
    df = me.load_frame(csv_path)
    filters, group_by = _engine_filters(state), list(state.get("group_by") or [])
    engine = me.evaluate(df, state["formula"], filters, group_by or None)
    try:
        ref = me.reference_evaluate(csv_path, state["formula"], filters, group_by or None)
        agrees = None
        if not group_by:
            agrees = me.values_agree(engine["value"], ref["value"], rel_tol=1e-9, abs_tol=1e-9)
        elif len(group_by) == 1:
            ref_map = {str(g["key"][group_by[0]]): g["value"] for g in ref["groups"]}
            eng_map = {str(g["key"][group_by[0]]): g["value"] for g in engine["groups"]}
            if set(ref_map) == set(eng_map):
                agrees = all(me.values_agree(eng_map[k], ref_map[k], rel_tol=1e-9, abs_tol=1e-9) for k in eng_map)
        reference_check = {True: "agrees", False: "disagrees", None: "not_comparable"}[agrees]
    except (me.ReferenceNotApplicable, me.FormulaError, OSError, UnicodeDecodeError):
        reference_check = "not_run"
    label = state.get("metric_label") or state["formula"]
    result: dict = {"rows": engine["rows"], "label": label}
    if not group_by:
        result.update({"type": "scalar", "value": engine["value"], "display": fastpath.format_value(engine["value"])})
    else:
        groups = [g for g in engine["groups"] if g["value"] is not None]
        groups.sort(key=lambda g: -g["value"])
        shown = groups[:RESULT_LIMIT]
        result.update({
            "type": "table", "group_by": group_by, "total_groups": len(groups),
            "rows_table": [{**{k: v for k, v in g["key"].items()}, "value": g["value"], "rows": g["rows"]} for g in shown],
            "chart": {"data": [{"type": "bar",
                                "x": [" / ".join(str(g["key"][c]) for c in group_by) for g in shown],
                                "y": [g["value"] for g in shown]}],
                      "layout": {"title": {"text": f"{label} by {', '.join(group_by)}"}}},
        })
    result["reference_check"] = reference_check
    result["coerced_non_numeric"] = engine.get("coerced_non_numeric") or {}
    return result


def describe(state: dict, result: dict) -> str:
    scope = ""
    filters = state.get("filters") or []
    if filters:
        scope = " where " + " and ".join(f"{f['column']} {f['op']} {f['value']}" for f in filters)
    if result["type"] == "scalar":
        return f"{result['label']}{scope}: {result['display']} (from {result['rows']:,} rows)."
    top = result["rows_table"][:3]
    lead = "; ".join(f"{' / '.join(str(r[c]) for c in result['group_by'])}: {fastpath.format_value(r['value'])}" for r in top)
    more = f" Showing the top {len(result['rows_table'])} of {result['total_groups']}." \
        if result["total_groups"] > len(result["rows_table"]) else ""
    return f"{result['label']} by {', '.join(result['group_by'])}{scope}. Highest: {lead}.{more}"


# ------------------------------------------------------------------ session --

def get_session(db: Session, team_id: int, user_id: int, session_id: int) -> CopilotSession:
    s = db.get(CopilotSession, session_id)
    # Same answer for "not found", "another team's" and "another user's".
    if s is None or s.team_id != team_id or s.user_id != user_id:
        raise CopilotError("Session not found")
    return s


def create_session(db: Session, *, team_id: int, user_id: int, dataset: Dataset, title: str = "") -> CopilotSession:
    version = latest_version(db, dataset)
    s = CopilotSession(team_id=team_id, user_id=user_id, dataset_id=dataset.id,
                       dataset_version_id=version.id if version else None,
                       title=(title or f"Session on {dataset.filename}")[:120], state_json=empty_state())
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _add_turn(db: Session, session: CopilotSession, role: str, content: str, **extra) -> CopilotTurn:
    seq = (db.query(CopilotTurn).filter_by(session_id=session.id).count()) + 1
    turn = CopilotTurn(session_id=session.id, seq=seq, role=role, content=content,
                       action_json=extra.get("action"), result_json=extra.get("result"),
                       evidence_json=extra.get("evidence"))
    db.add(turn)
    return turn


def turn_out(t: CopilotTurn) -> dict:
    return {"seq": t.seq, "role": t.role, "content": t.content, "action": t.action_json,
            "result": t.result_json, "evidence": t.evidence_json, "created_at": t.created_at}


def session_out(db: Session, s: CopilotSession, with_turns: bool = False) -> dict:
    dataset = db.get(Dataset, s.dataset_id)
    version = db.get(DatasetVersion, s.dataset_version_id) if s.dataset_version_id else None
    latest = latest_version(db, dataset) if dataset else None
    out = {"id": s.id, "title": s.title, "dataset_id": s.dataset_id,
           "dataset_filename": dataset.filename if dataset else None,
           "dataset_version": version.version_number if version else None,
           "latest_version": latest.version_number if latest else None,
           "stale": bool(latest and version and latest.id != version.id),
           "dataset_available": dataset is not None and version is not None,
           "state": s.state_json or empty_state(), "created_at": s.created_at, "updated_at": s.updated_at}
    if with_turns:
        turns = db.query(CopilotTurn).filter_by(session_id=s.id).order_by(CopilotTurn.seq).all()
        out["turns"] = [turn_out(t) for t in turns]
    return out


def handle_message(db: Session, session: CopilotSession, message: str, allow_model: bool = True) -> dict:
    """One turn: interpret, apply, recompute, record. Returns the assistant turn."""
    message = (message or "").strip()
    if not message:
        raise CopilotError("Message is empty")
    if len(message) > MAX_MESSAGE_CHARS:
        raise CopilotError(f"Message is longer than {MAX_MESSAGE_CHARS} characters")
    if db.query(CopilotTurn).filter_by(session_id=session.id).count() >= MAX_TURNS:
        raise CopilotError("This session has reached its length limit; start a new one")
    dataset = db.get(Dataset, session.dataset_id)
    version = db.get(DatasetVersion, session.dataset_version_id) if session.dataset_version_id else None
    _add_turn(db, session, "user", message)

    def reply(content: str, **extra) -> dict:
        turn = _add_turn(db, session, "assistant", content, **extra)
        db.commit()
        db.refresh(turn)
        return {**turn_out(turn), "session": session_out(db, session)}

    if dataset is None or version is None:
        return reply("The dataset this session was built on is no longer available, so nothing can be computed. "
                     "Start a new session on a current dataset.", action={"status": "dataset_unavailable"})
    latest = latest_version(db, dataset)
    state = dict(session.state_json or empty_state())
    profile = version.profile_json or {}
    try:
        df = me.load_frame(version.filepath)
    except (OSError, me.FormulaError) as e:
        return reply(f"The data for this session could not be read ({type(e).__name__}).",
                     action={"status": "dataset_unreadable"})
    resolved = semantic.resolve(db, session.team_id, dataset.id, message)
    if resolved["ambiguous"]:
        amb = resolved["ambiguous"][0]
        return reply(f"'{amb['term']}' has more than one approved definition. Which one do you mean?",
                     action={"status": "clarification",
                             "options": [{"label": f"{c['name']} = {c['formula']}", "message": c["formula"]}
                                         for c in amb["candidates"]]})
    interpreted = interpret(message, state, profile, category_index(df, profile), resolved["metrics"])
    if interpreted.get("unmapped") and allow_model:
        interpreted = interpret_with_model(message, state, profile)
    if "clarification" in interpreted:
        c = interpreted["clarification"]
        return reply(c["reason"], action={"status": "clarification", "options": c["options"]})
    if interpreted.get("unmapped"):
        suggestion = message if _NEEDS_ANALYSIS.search(message) else None
        text = ("I could not turn that into a metric, a filter or a breakdown on this dataset, so I have not "
                "changed anything.")
        if suggestion:
            text += " A question like this needs a full analysis; you can run it from the Analyses page."
        else:
            text += " Try, for example: \"total <numeric column> by <column>\", \"only <value>\", \"in 2024\", \"drill up\", \"reset\"."
        return reply(text, action={"status": "not_understood", "suggested_question": suggestion})

    operations = interpreted["operations"]
    notes: list[str] = []
    if any(o["op"] == "rebase" for o in operations):
        if latest is None or latest.id == version.id:
            notes.append("This session is already on the latest data.")
        else:
            version, profile = latest, latest.profile_json or {}
            session.dataset_version_id = latest.id
            state, pruned = prune_state(state, profile)
            notes.append(f"Moved to version {latest.version_number} of the data.")
            notes += pruned
    try:
        new_state, decisions = apply_operations(state, operations, profile)
    except CopilotError as e:
        db.rollback()
        _add_turn(db, session, "user", message)
        return reply(f"I did not change anything: {e}.", action={"status": "invalid_operation", "operations": operations})
    session.state_json = new_state
    stale = latest is not None and latest.id != version.id
    stale_note = (f" Note: a newer version of this dataset exists (version {latest.version_number}); this session "
                  f"is still on version {version.version_number}. Say \"use latest data\" to switch.") if stale else ""
    action = {"status": "applied", "operations": operations, "decisions": notes + decisions,
              "interpreted_by": interpreted.get("interpreted_by", "rules"), "state": new_state}
    if not new_state.get("formula"):
        return reply(" ".join(notes + decisions) + " No metric is selected yet, so there is nothing to compute. "
                     "Name one, for example \"total <numeric column>\"." + stale_note, action=action)
    try:
        result = compute(version.filepath, new_state)
    except me.FormulaError as e:
        return reply(" ".join(notes + decisions) + f" That could not be computed: {e}." + stale_note, action=action)
    evidence = {
        "formula": new_state["formula"], "filters": _engine_filters(new_state), "group_by": new_state["group_by"],
        "rows_used": result["rows"], "dataset_version": version.version_number,
        "dataset_fingerprint": provenance.fingerprint(db, version),
        "computed_by": "deterministic metric engine", "reference_check": result["reference_check"],
        "note": "Recomputed from the dataset on this turn; not taken from the conversation.",
    }
    return reply(" ".join(notes + decisions) + " " + describe(new_state, result) + stale_note,
                 action=action, result=result, evidence=evidence)
