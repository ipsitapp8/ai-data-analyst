"""Custom what-if models: build, validate, save and version a model from your own metrics and formulas.

Rendering only. Validation, ordering, unit checks and every calculation are done by the backend
(`/api/scenario-models`); this page assembles a definition from the form, sends it, and shows what comes back.
A saved model is immutable: editing saves a new version, and earlier scenarios keep the version they used.
"""
from __future__ import annotations

import streamlit as st

from api_client import (ApiError, archive_custom_model, create_custom_model, add_custom_model_version, get_custom_model,
                        list_metrics, restore_custom_model, validate_custom_model)
from style.theme import badge, esc, html
from what_if_decision import section

STATE_DEFAULTS = {"cm_n_in": 2, "cm_n_as": 0, "cm_n_out": 2, "cm_n_con": 0}
SOURCES = {"none": "No data source (I enter it)", "metric": "An approved metric", "formula": "My own formula over the data"}


def model_choices(catalog: list[dict], custom: list[dict]) -> dict[str, dict]:
    """Every model the page can run, keyed by the reference the API takes. Custom ones are pinned to their version."""
    out = {m["model"]: m for m in catalog}
    for m in custom:
        d = dict(m["describe"])
        d["title"] = f"{m['name']} (custom, v{m['current_version']})"
        out[m["ref"]] = d
    return out


def _bump(key: str, delta: int, low: int = 0) -> None:
    st.session_state[key] = max(low, st.session_state.get(key, STATE_DEFAULTS[key]) + delta)


def _load(model: dict) -> None:
    """Callback: copy a saved definition into the form's widgets (runs before the widgets are built)."""
    d = model["definition"]
    s = st.session_state
    s["cm_loaded_id"], s["cm_loaded_version"] = model["id"], model["current_version"]
    s["cm_name"], s["cm_desc"], s["cm_strict"] = model["name"], model["description"], bool(d.get("strict_units"))
    for prefix, count_key, rows in (("in", "cm_n_in", d["inputs"]), ("as", "cm_n_as", d["assumptions"]),
                                    ("out", "cm_n_out", d["outputs"])):
        s[count_key] = len(rows)
        for i, (name, r) in enumerate(rows.items()):
            s[f"cm_{prefix}_{i}_name"], s[f"cm_{prefix}_{i}_label"], s[f"cm_{prefix}_{i}_unit"] = name, r.get("label", name), r.get("unit", "")
            if prefix == "in":
                src = r.get("source") or {}
                s[f"cm_in_{i}_min"], s[f"cm_in_{i}_max"] = float(r.get("min", 0.0)), r.get("max")
                s[f"cm_in_{i}_good"] = r.get("good", "up")
                s[f"cm_in_{i}_src"] = src.get("type", "none") if src.get("type") in ("metric", "formula") else "none"
                s[f"cm_in_{i}_metric"] = src.get("metric_id")
                s[f"cm_in_{i}_formula"] = src.get("formula", "")
            elif prefix == "as":
                s[f"cm_as_{i}_default"], s[f"cm_as_{i}_min"], s[f"cm_as_{i}_max"] = (float(r["default"]), float(r["min"]), float(r["max"]))
            else:
                s[f"cm_out_{i}_formula"], s[f"cm_out_{i}_additive"] = r.get("formula", ""), bool(r.get("additive"))
    s["cm_primary"] = d.get("primary_output")
    cons = d.get("constraints") or []
    s["cm_n_con"] = len(cons)
    for i, c in enumerate(cons):
        s[f"cm_con_{i}_expr"], s[f"cm_con_{i}_msg"], s[f"cm_con_{i}_sev"] = c["expr"], c.get("message", ""), c.get("severity", "error")


def _new() -> None:
    for k in [k for k in st.session_state if str(k).startswith("cm_")]:
        del st.session_state[k]


def assemble() -> dict:
    """The definition described by the form, exactly as typed. The backend decides whether it is valid."""
    s = st.session_state
    inputs, assumptions, outputs = {}, {}, {}
    for i in range(s.get("cm_n_in", STATE_DEFAULTS["cm_n_in"])):
        name = (s.get(f"cm_in_{i}_name") or "").strip()
        src_kind = s.get(f"cm_in_{i}_src", "none")
        source = None
        if src_kind == "metric" and s.get(f"cm_in_{i}_metric") is not None:
            source = {"type": "metric", "metric_id": int(s[f"cm_in_{i}_metric"])}
        elif src_kind == "formula" and (s.get(f"cm_in_{i}_formula") or "").strip():
            source = {"type": "formula", "formula": s[f"cm_in_{i}_formula"].strip()}
        inputs[name] = {"label": s.get(f"cm_in_{i}_label") or name, "unit": s.get(f"cm_in_{i}_unit", ""),
                        "min": float(s.get(f"cm_in_{i}_min", 0.0)), "max": s.get(f"cm_in_{i}_max"),
                        "good": s.get(f"cm_in_{i}_good", "up"), "source": source}
    for i in range(s.get("cm_n_as", STATE_DEFAULTS["cm_n_as"])):
        name = (s.get(f"cm_as_{i}_name") or "").strip()
        assumptions[name] = {"label": s.get(f"cm_as_{i}_label") or name, "unit": s.get(f"cm_as_{i}_unit", ""),
                             "default": float(s.get(f"cm_as_{i}_default", 0.0)), "min": float(s.get(f"cm_as_{i}_min", 0.0)),
                             "max": float(s.get(f"cm_as_{i}_max", 1.0))}
    for i in range(s.get("cm_n_out", STATE_DEFAULTS["cm_n_out"])):
        name = (s.get(f"cm_out_{i}_name") or "").strip()
        outputs[name] = {"label": s.get(f"cm_out_{i}_label") or name, "unit": s.get(f"cm_out_{i}_unit", ""),
                         "formula": s.get(f"cm_out_{i}_formula", ""), "additive": bool(s.get(f"cm_out_{i}_additive", False))}
    constraints = [{"expr": s.get(f"cm_con_{i}_expr", ""), "message": s.get(f"cm_con_{i}_msg", ""),
                    "severity": s.get(f"cm_con_{i}_sev", "error")} for i in range(s.get("cm_n_con", 0)) if (s.get(f"cm_con_{i}_expr") or "").strip()]
    return {"description": s.get("cm_desc", ""), "strict_units": bool(s.get("cm_strict", False)), "inputs": inputs,
            "assumptions": assumptions, "outputs": outputs, "primary_output": s.get("cm_primary"), "constraints": constraints}


def builder_tab(custom: list[dict]) -> None:
    s = st.session_state
    for k, v in STATE_DEFAULTS.items():
        s.setdefault(k, v)
    try:
        metrics = [m for m in list_metrics() if m.get("status") == "approved"]
    except ApiError:
        metrics = []
    metric_ids = [m["id"] for m in metrics]
    metric_names = {m["id"]: f"{m['name']} — {m['formula']}" for m in metrics}

    if s.get("cm_saved_note"):
        st.success(s.pop("cm_saved_note"))
    section("Your custom models", "A model is a set of inputs and formulas you define. It runs on the same simulation, "
                                  "goal-seek and verification engines as the built-in models. Saved models are never edited in "
                                  "place: a change is a new version, and scenarios you saved keep the version they used.")
    if not custom:
        html('<div class="ds-row-meta" style="padding:6px 0;">No custom models yet. Build one below.</div>')
    for m in custom:
        c1, c2, c3 = st.columns([5, 1, 1])
        c1.markdown(f"**{m['name']}** · version {m['current_version']} · {len(m['describe']['inputs'])} inputs, "
                    f"{len(m['describe']['outputs'])} outputs  \n{m['description'] or ''}")
        if c2.button("Edit", key=f"cm_edit_{m['id']}", on_click=lambda mid=m["id"]: _load_by_id(mid)):
            pass
        if c3.button("Archive", key=f"cm_arch_{m['id']}"):
            try:
                archive_custom_model(m["id"])
                st.rerun()
            except ApiError as e:
                st.error(str(e))

    loaded = s.get("cm_loaded_id")
    section("Model builder" + (f" — editing '{s.get('cm_name')}' (version {s.get('cm_loaded_version')})" if loaded else ""))
    n1, n2 = st.columns([2, 3])
    n1.text_input("Model name", key="cm_name", max_chars=120)
    n2.text_input("Description", key="cm_desc", max_chars=1000)
    st.checkbox("Enforce units (a unit mismatch is an error, not a warning)", key="cm_strict")

    html('<div class="ds-section-title" style="margin-top:10px;">Inputs — the numbers you can change</div>')
    for i in range(s["cm_n_in"]):
        a, b, c, d = st.columns([2, 3, 2, 2])
        a.text_input("Name", key=f"cm_in_{i}_name", placeholder="e.g. customers", help="lowercase letters, digits and underscores")
        b.text_input("Label", key=f"cm_in_{i}_label")
        c.text_input("Unit", key=f"cm_in_{i}_unit", placeholder="e.g. currency/unit")
        d.selectbox("Higher is", ["up", "down"], key=f"cm_in_{i}_good", format_func=lambda v: "better" if v == "up" else "worse")
        e, f, g, h = st.columns([1, 1, 2, 4])
        e.number_input("Minimum", key=f"cm_in_{i}_min", value=0.0, format="%.4f")
        f.number_input("Maximum (blank = none)", key=f"cm_in_{i}_max", value=None, format="%.4f")
        g.selectbox("Baseline from", list(SOURCES), format_func=SOURCES.get, key=f"cm_in_{i}_src")
        if s.get(f"cm_in_{i}_src") == "metric":
            h.selectbox("Approved metric", metric_ids, format_func=lambda k: metric_names.get(k, str(k)), key=f"cm_in_{i}_metric",
                        index=None, placeholder="Choose an approved metric" if metrics else "No approved metrics yet")
        elif s.get(f"cm_in_{i}_src") == "formula":
            h.text_input("Formula over the dataset", key=f"cm_in_{i}_formula", placeholder="sum(revenue) / count()")
    b1, b2, _ = st.columns([1, 1, 6])
    b1.button("Add input", key="cm_add_in", on_click=_bump, args=("cm_n_in", 1))
    b2.button("Remove last", key="cm_rm_in", on_click=_bump, args=("cm_n_in", -1, 1))

    html('<div class="ds-section-title" style="margin-top:10px;">Assumptions — settings you choose (optional)</div>')
    for i in range(s["cm_n_as"]):
        a, b, c, d, e, f = st.columns([2, 3, 2, 1, 1, 1])
        a.text_input("Name", key=f"cm_as_{i}_name", placeholder="e.g. churn_rate")
        b.text_input("Label", key=f"cm_as_{i}_label")
        c.text_input("Unit", key=f"cm_as_{i}_unit", placeholder="fraction")
        d.number_input("Default", key=f"cm_as_{i}_default", value=0.0, format="%.4f")
        e.number_input("Min", key=f"cm_as_{i}_min", value=0.0, format="%.4f")
        f.number_input("Max", key=f"cm_as_{i}_max", value=1.0, format="%.4f")
    c1, c2, _ = st.columns([1, 1, 6])
    c1.button("Add assumption", key="cm_add_as", on_click=_bump, args=("cm_n_as", 1))
    c2.button("Remove last", key="cm_rm_as", on_click=_bump, args=("cm_n_as", -1))

    html('<div class="ds-section-title" style="margin-top:10px;">Outputs — formulas over the names above</div>')
    st.caption("Use + − × ÷ brackets, a ** 2, and abs, sqrt, log, exp, floor, ceil, min, max. An output can use other outputs.")
    for i in range(s["cm_n_out"]):
        a, b, c = st.columns([2, 3, 2])
        a.text_input("Name", key=f"cm_out_{i}_name", placeholder="e.g. mrr")
        b.text_input("Label", key=f"cm_out_{i}_label")
        c.text_input("Unit", key=f"cm_out_{i}_unit", placeholder="currency")
        d, e = st.columns([6, 2])
        d.text_input("Formula", key=f"cm_out_{i}_formula", placeholder="customers * price")
        e.checkbox("Adds up across segments", key=f"cm_out_{i}_additive")
    o1, o2, _ = st.columns([1, 1, 6])
    o1.button("Add output", key="cm_add_out", on_click=_bump, args=("cm_n_out", 1))
    o2.button("Remove last", key="cm_rm_out", on_click=_bump, args=("cm_n_out", -1, 1))
    out_names = [(s.get(f"cm_out_{i}_name") or "").strip() for i in range(s["cm_n_out"]) if (s.get(f"cm_out_{i}_name") or "").strip()]
    st.selectbox("Headline output", out_names or [None], key="cm_primary", disabled=not out_names)

    html('<div class="ds-section-title" style="margin-top:10px;">Constraints — rules every scenario must satisfy (optional)</div>')
    for i in range(s["cm_n_con"]):
        a, b, c = st.columns([3, 4, 1])
        a.text_input("Rule", key=f"cm_con_{i}_expr", placeholder="price >= unit_cost")
        b.text_input("Message", key=f"cm_con_{i}_msg", placeholder="Price must cover cost")
        c.selectbox("If broken", ["error", "warning"], key=f"cm_con_{i}_sev")
    k1, k2, _ = st.columns([1, 1, 6])
    k1.button("Add constraint", key="cm_add_con", on_click=_bump, args=("cm_n_con", 1))
    k2.button("Remove last", key="cm_rm_con", on_click=_bump, args=("cm_n_con", -1))

    definition = assemble()
    v1, v2, v3 = st.columns([1, 1, 1])
    if v1.button("Check the model", key="cm_validate", type="primary"):
        try:
            s["cm_check"] = validate_custom_model({"definition": definition})
        except ApiError as e:
            s["cm_check"] = {"valid": False, "errors": [str(e)], "warnings": [], "order": [], "units": {}}
    check = s.get("cm_check")
    if check:
        _show_check(check)

    note = st.text_input("What changed (saved with a new version)", key="cm_note", max_chars=500)
    name = (s.get("cm_name") or "").strip()
    if v2.button("Save as a new model", key="cm_save_new", disabled=not name):
        _save(lambda: create_custom_model({"name": name, "description": s.get("cm_desc", ""), "definition": definition, "note": note}))
    if loaded and v3.button(f"Save as version {s.get('cm_loaded_version', 0) + 1}", key="cm_save_version"):
        _save(lambda: add_custom_model_version(loaded, {"definition": definition, "note": note}))
    if v3.button("Start a new model", key="cm_clear", on_click=_new):
        pass


def _load_by_id(model_id: int) -> None:
    try:
        _load(get_custom_model(model_id))
    except ApiError:
        st.session_state["cm_error"] = "That model could not be opened."


def _save(action) -> None:
    try:
        saved = action()
        st.session_state["cm_saved_note"] = f"Saved '{saved['name']}' as version {saved['current_version']}."
        st.session_state["cm_loaded_id"], st.session_state["cm_loaded_version"] = saved["id"], saved["current_version"]
        st.rerun()
    except ApiError as e:
        st.error(f"Not saved: {e}")


def _show_check(check: dict) -> None:
    if check["valid"]:
        html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge("Valid", "verified")} Outputs are computed in this order: '
             f'{esc(" → ".join(check["order"]))}</div>')
    else:
        html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge("Not valid", "error")} Fix these before saving:</div>')
    for e in check["errors"]:
        html(f'<div class="ds-row-meta" style="padding:2px 0;">✗ {esc(e)}</div>')
    for w in check["warnings"]:
        html(f'<div class="ds-row-meta" style="padding:2px 0;">! {esc(w)}</div>')
    if check["valid"] and check.get("units"):
        html('<div class="ds-row-meta" style="padding:4px 0;">Units: ' + esc(" · ".join(f"{k}: {v}" for k, v in check["units"].items())) + "</div>")
