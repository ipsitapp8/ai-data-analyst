"""What-if, built by hand: the person states the objective, the variables with their starting and changed values, the formulas,
assumptions and constraints. Nothing is suggested or filled in. It runs as a Custom scenario on the same deterministic engine as
every other what-if.

`builder` is the manual starting point. `idea_builder` asks for the values of a formula model that Silt AI proposed; it keeps its
own widget keys, so neither can show the other's values."""
from __future__ import annotations

import re
import uuid

import streamlit as st

from api_client import ApiError, whatif_run
from style.theme import esc, html

S = st.session_state
KEEP_PREFIX = "wxm_v_"      # the manual draft: kept while its widgets are off screen
NO_LOWER_LIMIT = -1e15      # a variable of your own may be negative
METHODS = {"deterministic": "Exact calculation", "sensitivity": "Sensitivity analysis"}


def slug(name: str) -> str:
    """The name a variable goes by inside a formula: "Ticket price" -> ticket_price."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s if s[:1].isalpha() else f"v_{s}")[:40] if s else ""


def _rows(count_key: str, start: int, label: str, key: str) -> int:
    n = S.setdefault(count_key, start)
    a, b, _ = st.columns([1.4, 1.4, 5])
    if a.button(f"Add {label}", key=f"wxm_btn_add_{key}", disabled=n >= 12):
        S[count_key] = n + 1
        st.rerun()
    if b.button(f"Remove last", key=f"wxm_btn_del_{key}", disabled=n <= 1):
        for k in [k for k in S if str(k).startswith(f"{KEEP_PREFIX}{key}_{n - 1}_")]:
            del S[k]
        S[count_key] = n - 1
        st.rerun()
    return n


def _constraints(text: str) -> list[dict]:
    parts = (line.partition("|") for line in text.splitlines() if line.strip())
    return [{"expr": a.strip(), "message": (b or a).strip(), "severity": "error"} for a, _, b in parts]


def _extras(prefix: str) -> tuple[list[str], str, str]:
    assumptions = st.text_area("Assumptions you are making (one per line)", key=f"{prefix}assume", height=80,
                               placeholder="Demand stays the same\nNo new staff are hired")
    st.caption("Recorded with the result so a reader sees them. They are not part of the arithmetic: put a number in a variable if it should be calculated.")
    cons = st.text_area("Constraints (one per line: expression | message)", key=f"{prefix}cons", height=70,
                        placeholder="cost <= 5000 | Cost must stay within 5000")
    method = st.selectbox("Method", list(METHODS), format_func=METHODS.get, key=f"{prefix}method")
    return [a.strip() for a in assumptions.splitlines() if a.strip()], cons, method


def _scenario(objective: str, definition: dict, baseline: dict, changes: dict, assumptions: list[str], method: str) -> dict:
    return {"id": uuid.uuid4().hex[:12], "name": objective[:100], "domain": "custom", "template_id": "custom_model", "question": objective, "method": method,
            "inputs": {}, "columns": {}, "uncertainty": {}, "assumptions": {}, "constraints": [], "target": None, "custom_model": definition,
            "baseline": baseline, "changes": changes, "stated_assumptions": assumptions}


def _execute(scenario: dict) -> tuple[dict | None, list[str]]:
    """(result, problems). A scenario the engine refuses comes back as its reasons, in its words."""
    try:
        res = whatif_run(scenario, None)
    except ApiError as e:
        return None, [str(e)]
    v = res["validation"]
    if v["status"] != "valid":
        return None, [*v["errors"], *(f"Still needed: {m['label']}" for m in v["missing_inputs"])]
    return res, []


def _values(rows: list[tuple[str, str, float | None, float | None]]) -> tuple[dict, dict, list[str]]:
    """(baseline, changes, problems) from (formula name, label, starting value, changed value)."""
    baseline, changes, problems = {}, {}, []
    for name, label, base, new in rows:
        if base is None:
            problems.append(f"Enter a starting value for “{label}”.")
            continue
        baseline[name] = float(base)
        if new is not None and float(new) != float(base):
            changes[name] = {"type": "set", "value": float(new)}
    if not problems and not changes:
        problems.append("Give at least one variable a changed value that differs from its starting value, so there is something to compare.")
    return baseline, changes, problems


def _collect(nv: int, no: int) -> tuple[dict | None, list[str]]:
    get = lambda key: S.get(KEEP_PREFIX + key)  # noqa: E731
    problems: list[str] = []
    objective = (get("obj") or "").strip()
    if not objective:
        problems.append("State the scenario objective.")
    rows, inputs = [], {}
    for i in range(nv):
        label, base, new = (get(f"var_{i}_name") or "").strip(), get(f"var_{i}_base"), get(f"var_{i}_new")
        if not label:
            if base is not None or new is not None:
                problems.append(f"Name variable {i + 1}.")
            continue
        name = slug(label)
        if not name:
            problems.append(f"“{label}” needs at least one letter or digit in its name.")
        elif name in inputs:
            problems.append(f"Two variables are both called “{name}” in formulas. Rename one.")
        else:
            inputs[name] = {"label": label, "unit": (get(f"var_{i}_unit") or "").strip(), "min": NO_LOWER_LIMIT}
            rows.append((name, label, base, new))
    outputs = {}
    for i in range(no):
        label, formula = (get(f"out_{i}_name") or "").strip(), (get(f"out_{i}_formula") or "").strip()
        if not label and not formula:
            continue
        if not label or not formula:
            problems.append(f"Outcome {i + 1} needs both a name and a formula.")
        elif not slug(label) or slug(label) in outputs or slug(label) in inputs:
            problems.append(f"Give outcome “{label}” a name no other variable or outcome uses.")
        else:
            outputs[slug(label)] = {"label": label, "unit": "", "formula": formula}
    if not inputs:
        problems.append("Add at least one variable.")
    if not outputs:
        problems.append("Add at least one outcome with a formula.")
    baseline, changes, more = _values(rows)
    problems += more if inputs else []
    if problems:
        return None, problems
    definition = {"inputs": inputs, "outputs": outputs, "primary_output": next(iter(outputs)), "constraints": _constraints(get("cons") or "")}
    assumptions = [a.strip() for a in (get("assume") or "").splitlines() if a.strip()]
    return _scenario(objective, definition, baseline, changes, assumptions, get("method") or "deterministic"), []


def builder(workflow: str) -> bool:
    """The manual starting point. True once a run has succeeded and its result is in session state."""
    html('<div class="wx-section">Build your own scenario</div>')
    st.caption("You define everything here: no questions are suggested and no value is filled in.")
    st.text_input("Scenario objective", key=f"{KEEP_PREFIX}obj", max_chars=200, placeholder="What do you want to find out?")

    st.markdown("**Variables**")
    st.caption("Each variable has a starting value. Give it a changed value to vary it, or leave that empty to hold it constant.")
    nv = S.get(f"{KEEP_PREFIX}nvars", 2)
    for i in range(nv):
        show = "visible" if i == 0 else "collapsed"
        a, b, c, d = st.columns([3, 1.4, 1.7, 1.7], vertical_alignment="top")
        label = a.text_input("Variable name", key=f"{KEEP_PREFIX}var_{i}_name", label_visibility=show, placeholder=f"Variable {i + 1}")
        if label.strip():
            a.caption(f"In formulas: `{slug(label)}`")
        b.text_input("Unit (optional)", key=f"{KEEP_PREFIX}var_{i}_unit", label_visibility=show, placeholder="Unit")
        c.number_input("Starting value", value=None, key=f"{KEEP_PREFIX}var_{i}_base", label_visibility=show, placeholder="Starting value", format="%g")
        d.number_input("Changed value", value=None, key=f"{KEEP_PREFIX}var_{i}_new", label_visibility=show, placeholder="Changed value", format="%g")
    _rows(f"{KEEP_PREFIX}nvars", 2, "variable", "var")

    st.markdown("**Outcomes**")
    st.caption("A formula for each result you care about, using the variable names above. Allowed: + − * / **, min, max, abs, sqrt, log, exp.")
    no = S.get(f"{KEEP_PREFIX}nouts", 1)
    for i in range(no):
        show = "visible" if i == 0 else "collapsed"
        a, b = st.columns([2, 5])
        a.text_input("Outcome name", key=f"{KEEP_PREFIX}out_{i}_name", label_visibility=show, placeholder=f"Outcome {i + 1}")
        b.text_input("Formula", key=f"{KEEP_PREFIX}out_{i}_formula", label_visibility=show, placeholder="for example: variable_a * variable_b")
    _rows(f"{KEEP_PREFIX}nouts", 1, "outcome", "out")

    st.markdown("**Assumptions and constraints**")
    _extras(KEEP_PREFIX)
    if not st.button("Run scenario", type="primary", key="wxm_btn_run"):
        return False
    scenario, problems = _collect(nv, no)
    res = None
    if scenario:
        res, problems = _execute(scenario)
    for p in problems:
        st.error(p)
    if res is None:
        return False
    S["wxu_result"], S["wxu_scn"] = res, scenario
    S["wxu_sel"] = {"id": "custom", "workflow": workflow, "question": scenario["question"], "inputs": [], "domain": "custom",
                    "template_id": "custom_model", "method": scenario["method"]}
    return True


def idea_builder(sel: dict) -> None:
    """The values for a formula model Silt AI proposed. Its formulas are shown for checking; every number is typed by the person."""
    definition, prefix = sel["custom_model"], f"wxb_{sel['id']}_"
    html('<div class="wx-section">Scenario builder</div>')
    st.markdown(f"**{sel['question']}**")
    st.caption("Silt AI proposed these variables and formulas. Check them before relying on the result; Silt validated that they run, not that they are right.")
    for o in definition["outputs"].values():
        html(f'<div class="wx-meta"><b>{esc(o["label"])}</b> = {esc(o["formula"])}</div>')
    values = []
    for i, (name, rule) in enumerate(definition["inputs"].items()):
        a, b, c = st.columns([3, 2, 2], vertical_alignment="bottom")
        a.markdown(f"**{rule['label']}**" + (f" ({rule['unit']})" if rule["unit"] else "") + f"  \n`{name}`")
        base = b.number_input("Starting value", value=None, key=f"{prefix}{i}_base", format="%g")
        new = c.number_input("Changed value", value=None, key=f"{prefix}{i}_new", format="%g")
        values.append((name, rule["label"], base, new))
    assumptions, cons, method = _extras(prefix)
    if st.button("Run scenario", type="primary", key=f"wxbtn_run_{sel['id']}"):
        baseline, changes, problems = _values(values)
        S.pop("wxu_result", None)
        if not problems:
            scenario = _scenario(sel["question"], {**definition, "constraints": _constraints(cons)}, baseline, changes, assumptions, method)
            res, problems = _execute(scenario)
            if res is not None:
                S["wxu_result"], S["wxu_scn"] = res, scenario
        for p in problems:
            st.error(p)
