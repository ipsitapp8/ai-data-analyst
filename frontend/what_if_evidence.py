"""Explanations, "How was this calculated?" and sealed evidence for the What-If page.

Rendering only. Explanations, traces and evidence records all come from the backend, which recomputes them from the request; the
page never calculates a number and never trusts one it is not shown by the server.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import ApiError, inspect_calculation, list_evidence, reproduce_evidence
from style.theme import badge, esc, html
from what_if_decision import num, section, verification_panel

SECTION_TITLES = {"what_changed": "What changed", "why": "What accounts for it", "assumptions": "Assumptions",
                  "risks": "Risks to watch", "cannot_conclude": "What this cannot tell you"}


def explanation_panel(explanation: dict | None) -> None:
    """The plain-English explanation of a result, with its sections."""
    section("In plain English")
    if not explanation or not explanation.get("summary"):
        st.caption("No explanation is available for this result.")
        return
    html(f'<div class="ds-row-meta" style="padding:4px 0;">{esc(explanation["summary"])}</div>')
    st.caption(explanation.get("note", "") or "Written from the verified result.")
    for key, title in SECTION_TITLES.items():
        items = explanation["sections"].get(key) or []
        if items:
            with st.expander(title, expanded=key in ("what_changed", "cannot_conclude")):
                for x in items:
                    html(f'<div class="ds-row-meta" style="padding:2px 0;">• {esc(x)}</div>')


def inspector(mk: str, model: dict, payload: dict, scenario_names: list[str], dataset_id: int | None) -> None:
    """Pick a number, see exactly how it was calculated, and optionally seal the calculation as an evidence record."""
    section("How was this calculated?", "Pick any result to see its formula, your numbers put into it, each step, and where every input came "
                                        "from. Sealing stores the calculation as a record you can re-run later.")
    a, b = st.columns(2)
    scenario = a.selectbox("Which scenario", ["baseline"] + scenario_names, key=f"ins_scn_{mk}")
    outputs = list(model["outputs"])
    output = b.selectbox("Which result", outputs, index=outputs.index(model["primary_output"]), format_func=model["outputs"].get, key=f"ins_out_{mk}")
    body = {**payload, "scenario": scenario, "output": output}
    key = f"ins_result_{mk}"
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Show the calculation", key="ins_go"):
        try:
            st.session_state[key] = inspect_calculation(body)
        except ApiError as e:
            st.session_state.pop(key, None)
            st.error(f"That could not be shown: {e}")
    if c2.button("Seal it as evidence", key="ins_seal", help="Stores the request, the result, the checks and the data used, with a hash."):
        try:
            st.session_state[key] = inspect_calculation({**body, "persist": True, "dataset_id": dataset_id, "note": "sealed from the What-If page"})
        except ApiError as e:
            st.error(f"Not sealed: {e}")
    r = st.session_state.get(key)
    if r:
        _show_trace(r, model)


def _show_trace(r: dict, model: dict) -> None:
    t = r["trace"]
    html(f'<div class="ds-row-meta" style="padding:6px 0;"><b>{esc(t["label"])}</b> in <b>{esc(t["scenario"])}</b> is <b>{num(t["value"], 4)}</b> '
         f'{esc(t["unit"])} {badge("Matches the engine", "verified") if t["matches_engine"] else badge("Does not match the engine", "error")}</div>')
    html(f'<div class="ds-row-meta" style="padding:2px 0;">Formula: <code>{esc(t["output"])} = {esc(t["formula"])}</code></div>')
    rows = [{"Step": s["label"], "Formula": s["formula"], "With your numbers": s["substituted"], "Result": s["value"], "Unit": s["unit"]} for s in t["steps"]]
    st.dataframe(pd.DataFrame(rows).style.format({"Result": "{:,.4f}"}, na_rep="undefined"), use_container_width=True, hide_index=True)
    inputs = [{"Input": i["label"], "Unit": i["unit"], "Baseline": i["baseline"], "Used here": i["value"],
               "Change": (f"{i['adjustment']['type']} {i['adjustment']['value']:+g}" if i["adjustment"] else "—"),
               "Where it came from": (i["source"] or {}).get("source", "entered by you").replace("_", " ") + (f" — {i['source']['formula']}" if (i["source"] or {}).get("formula") else "")}
              for i in t["inputs"]]
    st.dataframe(pd.DataFrame(inputs).style.format({"Baseline": "{:,.4f}", "Used here": "{:,.4f}"}, na_rep="—"), use_container_width=True, hide_index=True)
    for i in t["inputs"]:
        if i.get("derived_from"):
            html(f'<div class="ds-row-meta" style="padding:2px 0;">{esc(i["label"])} was not typed: it is {esc(i["derived_from"])}.</div>')
    for a in t["assumptions"]:
        src = a["source"].get("source", "manual")
        html(f'<div class="ds-row-meta" style="padding:2px 0;">Assumption {esc(a["label"])} = {num(a["value"], 4)} ({esc(src)})'
             f'{" — applied by the engine, not in a formula" if a["applied_by_engine"] else ""}</div>')
    for c in t["constraints"]:
        ok = c.get("satisfied", True)
        html(f'<div class="ds-row-meta" style="padding:2px 0;">{badge("Rule met" if ok else "Rule broken", "verified" if ok else "error")} {esc(c["rule"])} — {esc(c["message"])}</div>')
    html(f'<div class="ds-row-meta" style="padding:2px 0;">Engine {esc(t["engine_version"])}'
         + (f' · model {esc(t["model"])}' if t["model"] not in ("unit_economics", "funnel") else f' · built-in model {esc(t["model"])}') + "</div>")
    verification_panel(r.get("verification"), "Independent check of the scenario arithmetic")
    ev = r.get("evidence")
    if ev:
        html(f'<div class="ds-row-meta" style="padding:6px 0;">{badge("Sealed", "verified")} Evidence record #{ev["id"]} · {esc(ev["verification_status"])} · '
             f'fingerprint <code>{esc(ev["content_sha256"][:16])}</code></div>')


def evidence_list(scenario_id: int, key_prefix: str) -> None:
    """The sealed records of a saved scenario, each with a button that runs it again and says whether it still gives the same result."""
    try:
        records = list_evidence(scenario_id)["records"]
    except ApiError as e:
        st.caption(f"Evidence could not be loaded: {e}")
        return
    if not records:
        st.caption("No evidence records for this scenario.")
        return
    for rec in records:
        a, b = st.columns([5, 1])
        d = rec["dataset"]
        a.markdown(f"**#{rec['id']}** · {rec['kind'].replace('_', ' ')} · {rec['verification_status']} · fingerprint `{rec['content_sha256'][:12]}`"
                   + (f" · data v{d['dataset_version_id']}" if d.get("dataset_version_id") else "") + (f" · seed {rec['seed']}" if rec.get("seed") is not None else ""))
        if b.button("Re-run", key=f"{key_prefix}_rep_{rec['id']}"):
            try:
                r = reproduce_evidence(rec["id"])
                (st.success if r["reproduced"] else st.error)(r["explanation"])
            except ApiError as e:
                st.error(str(e))
