"""The Saved scenarios tab: reopen and edit, rename, duplicate, data-status warnings, comparison and export.

Rendering only. Freshness, comparison arithmetic, recalculation and export files all come from the backend; the page never decides
whether data is stale and never recalculates a saved number. Opening a scenario loads its stored settings into the builder; it never
fetches newer data on its own.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import (ApiError, compare_scenarios, delete_scenario, duplicate_scenario, export_scenario, get_scenario, rebase_scenario,
                        rename_scenario, compare_scenarios_csv)
from style.theme import badge, esc, html
from what_if_evidence import evidence_list

STATUS = {"current": ("Data is current", "verified"), "typed_inputs": ("Typed inputs", "neutral"), "stale": ("Newer data exists", "warn"),
          "changed": ("Data changed", "error"), "unavailable": ("Data unavailable", "error")}
FORMATS = {"json": ("JSON (complete)", "application/json"), "csv": ("CSV (numbers)", "text/csv"), "md": ("Markdown (readable)", "text/markdown")}


def status_badge(fresh: dict | None) -> str:
    label, tone = STATUS.get((fresh or {}).get("status"), ("", "neutral"))
    return badge(label, tone) if label else ""


def _open(s: dict, models: dict) -> None:
    """Callback: put a saved scenario's stored settings into the builder's widgets (runs before they are drawn).

    Only what the builder can show is loaded. The page says what it could not load, and Update replaces the saved settings with what
    the builder shows, so nothing is dropped without being named."""
    st.session_state["wi_model"] = s["model"]
    for k, v in (s["baseline"] or {}).items():
        st.session_state[f"wi_in_{s['model']}_{k}"] = float(v)
    req = s["request"] or {}
    for k, v in (req.get("assumptions") or {}).items():
        st.session_state[f"wi_as_{s['model']}_{k}"] = float(v)
    mine = (req.get("scenarios") or {}).get("your scenario") or {}
    for k in models[s["model"]]["inputs"]:
        st.session_state[f"wi_adj_{s['model']}_{k}"] = float(mine[k]["value"]) if k in mine and mine[k]["type"] == "pct" else 0.0
    st.session_state["wi_editing"] = {"id": s["id"], "name": s["name"], "revision": s.get("revision", 1), "model": s["model"],
                                      "dataset_id": s["dataset_id"], "baseline_source": _source(req), "freshness": s.get("freshness"),
                                      "not_loaded": _not_loaded(req, models[s["model"]]), "spread_pct": req.get("spread_pct"),
                                      "sensitivity_pct": req.get("sensitivity_pct", 10.0)}
    st.session_state["wi_opened_notice"] = s["name"]
    st.session_state["wi_spread"] = float(req.get("spread_pct") or 0.0)
    st.session_state["wi_sens"] = float(req.get("sensitivity_pct") or 10.0)


def _open_by_id(sid: int, models: dict) -> None:
    """Callback for the Open button: widget values can only be set before the widgets are drawn, which is what a callback allows."""
    try:
        _open(get_scenario(sid), models)
    except ApiError as e:
        st.session_state["wi_open_error"] = str(e)


def _source(req: dict) -> dict | None:
    src = dict(req.get("baseline_source") or {})
    src.pop("provenance", None)
    return src or None


def _not_loaded(req: dict, model: dict) -> list[str]:
    """What the form cannot show, in words: scenarios other than 'your scenario' and any change that is not a percentage."""
    out = []
    scenarios = req.get("scenarios") or {}
    others = [n for n in scenarios if n not in ("your scenario", "optimistic", "pessimistic")]
    if others:
        out.append("other named scenarios: " + ", ".join(others))
    for k, a in (scenarios.get("your scenario") or {}).items():
        if a.get("type") != "pct":
            out.append(f"{model['inputs'].get(k, {}).get('label', k)} is set as '{a['type']}', which the form cannot show")
    if req.get("simulation"):
        out.append("the risk simulation settings (re-enter them to keep a simulation)")
    return out


def editing_banner() -> dict | None:
    """Shown at the top of the builder while a saved scenario is open. Returns the editing context, or None."""
    ctx = st.session_state.get("wi_editing")
    if not ctx:
        return None
    c1, c2 = st.columns([5, 1])
    c1.info(f"Editing saved scenario “{ctx['name']}” (revision {ctx['revision']}). Update replaces its saved settings with what you see here; "
            "evidence sealed for earlier revisions is kept.")
    if c2.button("Stop editing", key="wi_stop_edit"):
        st.session_state.pop("wi_editing", None)
        st.rerun()
    for r in (ctx.get("freshness") or {}).get("reasons", []):
        st.warning(r)
    if ctx["not_loaded"]:
        st.warning("The form could not load: " + "; ".join(ctx["not_loaded"]) + ". Updating will not keep these.")
    return ctx


def library_tab(models: dict, saved: list[dict]) -> None:
    if not saved:
        html('<div class="ds-row-meta" style="padding:16px 0;">Nothing saved yet.</div>')
        return
    if st.session_state.get("wi_open_error"):
        st.error(f"It could not be opened: {st.session_state.pop('wi_open_error')}")
    done = st.session_state.pop("wi_rebased", None)
    if done:
        _rebased(done)
    if st.session_state.get("wi_opened_notice"):
        st.success(f"“{st.session_state.pop('wi_opened_notice')}” is open in the “Build a scenario” tab.")
    chosen = []
    for s in saved:
        a, b, c = st.columns([0.5, 5, 1])
        if a.checkbox("Select", key=f"wi_pick_{s['id']}", label_visibility="collapsed"):
            chosen.append(s["id"])
        base = s.get("baseline_primary")
        b.markdown(f"**{s['name']}** · {models.get(s['model'], {}).get('title', s['model'])}"
                   + (f" · baseline {s.get('primary_output')}: {base:,.2f}" if base is not None else "") + f" · revision {s.get('revision', 1)}")
        html(f'<div class="ds-row-meta" style="padding:0 0 4px 0;">{status_badge(s.get("freshness"))}</div>')
        for r in (s.get("freshness") or {}).get("reasons", []):
            html(f'<div class="ds-row-meta" style="padding:0 0 2px 0;">{esc(r)}</div>')
        if c.button("Delete", key=f"wi_del_{s['id']}"):
            try:
                delete_scenario(s["id"])
            except ApiError as e:
                st.error(str(e))
            st.rerun()
        with b.expander("Open, rename, copy, export, evidence"):
            _actions(s, models)
    if len(chosen) >= 2:
        _comparison(chosen)
    else:
        st.caption("Tick two or more scenarios of the same model to compare them.")


def _actions(s: dict, models: dict) -> None:
    sid = s["id"]
    c1, c2, c3 = st.columns(3)
    if s["model"] in models:
        c1.button("Open in the builder", key=f"wi_open_{sid}", on_click=_open_by_id, args=(sid, models))
    else:
        c1.caption("This was saved with a model version that is no longer selectable, so it cannot be opened in the builder. It can still be copied and exported.")
    if c2.button("Duplicate", key=f"wi_dup_{sid}"):
        try:
            duplicate_scenario(sid)
            st.rerun()
        except ApiError as e:
            st.error(str(e))
    if (s.get("freshness") or {}).get("status") == "stale":
        if st.button("Recalculate on the newest data (saves a new scenario)", key=f"wi_rebase_{sid}",
                     help="Takes the baseline again from the newer version with the same mapping and period, applies the same changes, and saves the "
                          "result as a new scenario. This one is left exactly as it is."):
            try:
                st.session_state["wi_rebased"] = rebase_scenario(sid)
                st.rerun()
            except ApiError as e:
                st.error(str(e))
    name = c3.text_input("New name", value=s["name"], key=f"wi_ren_{sid}", max_chars=120, label_visibility="collapsed")
    if c3.button("Rename", key=f"wi_rename_{sid}", disabled=not name.strip() or name.strip() == s["name"]):
        try:
            rename_scenario(sid, name.strip())
            st.rerun()
        except ApiError as e:
            st.error(str(e))
    f1, f2 = st.columns([2, 1])
    fmt = f1.selectbox("Export as", list(FORMATS), format_func=lambda k: FORMATS[k][0], key=f"wi_fmt_{sid}")
    key = f"wi_file_{sid}"
    if f2.button("Prepare file", key=f"wi_prep_{sid}"):
        try:
            st.session_state[key] = (fmt, export_scenario(sid, fmt))
        except ApiError as e:
            st.error(str(e))
    ready = st.session_state.get(key)
    if ready and ready[0] == fmt:
        data, sha = ready[1]
        st.download_button(f"Download {fmt.upper()}", data, file_name=f"scenario-{sid}.{fmt}", mime=FORMATS[fmt][1], key=f"wi_dl_{sid}")
        st.caption(f"Exactly what was saved, not recalculated. Fingerprint {sha[:16]}…")
    evidence_list(sid, f"wi_ev_{sid}")


def _comparison(ids: list[int]) -> None:
    try:
        cmp = compare_scenarios(ids)
    except ApiError as e:
        st.error(str(e))
        return
    an = cmp["analysis"]
    for w in an["warnings"]:
        st.warning(w)
    rows = [{"Saved as": r["name"], "Scenario": r["scenario"], "Result": r["value"], "Change": r["change"], "Change %": r["change_pct"],
             "Data": STATUS.get(r["freshness"], (r["freshness"],))[0]} for r in an["rows"]]
    st.dataframe(pd.DataFrame(rows).style.format({"Result": "{:,.2f}", "Change": "{:+,.2f}", "Change %": "{:+,.1f}%"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    if an["highest"]:
        lo = an["lowest"]
        st.caption(f"Highest: {an['highest']['name']} / {an['highest']['scenario']} ({an['highest']['value']:,.2f})"
                   + (f" · Lowest: {lo['name']} / {lo['scenario']} ({lo['value']:,.2f})" if lo else "") + ". " + an["note"])
    st.caption(cmp["disclaimer"])
    try:
        st.download_button("Download comparison (CSV)", compare_scenarios_csv(ids), file_name="scenario-comparison.csv", mime="text/csv", key="wi_cmp_dl")
    except ApiError as e:
        st.error(str(e))


def _rebased(r: dict) -> None:
    """What a recalculation on newer data did, shown beside what it replaced (which still exists unchanged)."""
    new = r["scenario"]
    st.success(f"Saved “{new['name']}” on data version {r['to_version']}. “{r['analysis']['rows'][0]['name']}” (version {r['from_version']}) is unchanged.")
    moved = [(k, v["was"], v["now"]) for k, v in r["inputs"].items() if v["was"] != v["now"]]
    if moved:
        st.caption("Inputs that moved: " + " · ".join(f"{k} {a:,.2f} → {b:,.2f}" for k, a, b in moved))
    if r["kept_from_saved"]:
        st.caption("Kept from what you saved, because the data does not supply them: " + ", ".join(r["kept_from_saved"]))
    rows = [{"Saved as": x["name"], "Scenario": x["scenario"], "Result": x["value"], "Change": x["change"], "Change %": x["change_pct"]} for x in r["analysis"]["rows"]]
    st.dataframe(pd.DataFrame(rows).style.format({"Result": "{:,.2f}", "Change": "{:+,.2f}", "Change %": "{:+,.1f}%"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
