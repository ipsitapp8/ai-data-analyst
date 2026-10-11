"""Root cause — where a change in a metric came from: segment contributions,
price versus volume, tested hypotheses, and what could not be established."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import (ApiError, create_investigation, get_investigation, list_datasets, list_investigations,
                        list_metrics)
from style.theme import (badge, esc, figure_from_json, html, page_header, page_setup, plot, render_sidebar,
                         require_active_team)

page_setup("Root cause")
render_sidebar("investigations")
require_active_team("Root-cause investigations")
page_header("Root-cause investigations", "Locate where a metric changed. Descriptive evidence, clearly separated from cause.")

try:
    datasets = list_datasets()
    listing = list_investigations()
    metrics = [m for m in list_metrics() if m["status"] == "approved"]
except ApiError as e:
    st.error(f"Backend unreachable: {e}")
    st.stop()

with st.expander("New investigation", expanded=not listing["investigations"]):
    if not datasets:
        st.info("Upload a dataset first.")
    else:
        ds = st.selectbox("Dataset", datasets, format_func=lambda d: d["filename"], key="inv_ds")
        cols = ds["profile"].get("columns", [])
        time_cols = [c["name"] for c in cols if c.get("kind") == "datetime"]
        numeric = [c["name"] for c in cols if c.get("kind") == "numeric"]
        categorical = [c["name"] for c in cols if c.get("kind") == "categorical"]
        if not time_cols:
            st.warning("This dataset has no date column, so two periods cannot be compared.")
        else:
            usable = [m for m in metrics if m.get("dataset_id") in (None, ds["id"])]
            source = st.radio("Metric", ["A column", "An approved metric", "A formula"], horizontal=True, key="inv_src")
            formula, label = "", ""
            if source == "A column" and numeric:
                agg = st.selectbox("Aggregation", ["sum", "avg", "count", "median"], key="inv_agg")
                col = st.selectbox("Column", numeric, key="inv_col")
                formula, label = f'{agg}("{col}")', f"{agg.title()} of {col}"
            elif source == "An approved metric":
                if usable:
                    m = st.selectbox("Approved metric", usable, format_func=lambda m: f"{m['name']} = {m['formula']}", key="inv_metric")
                    formula, label = m["formula"], m["name"]
                else:
                    st.caption("No approved metric applies to this dataset. Define one on the Semantic layer page.")
            else:
                formula = st.text_input("Formula", placeholder="sum(revenue) / sum(units)", key="inv_formula")
                label = st.text_input("Label", key="inv_label")
            c1, c2 = st.columns(2)
            with c1:
                time_col = st.selectbox("Date column", time_cols, key="inv_time")
            with c2:
                dims = st.multiselect("Dimensions to examine (empty = automatic)", categorical, key="inv_dims")
            manual = st.checkbox("Choose the two periods myself (otherwise the time range is split at its midpoint)", key="inv_manual")
            pa = pb = None
            if manual:
                p1, p2, p3, p4 = st.columns(4)
                a0 = p1.date_input("Earlier period: from", key="inv_a0")
                a1 = p2.date_input("Earlier period: before", key="inv_a1")
                b0 = p3.date_input("Later period: from", key="inv_b0")
                b1 = p4.date_input("Later period: to", key="inv_b1")
                pa, pb = [a0.isoformat(), a1.isoformat()], [b0.isoformat(), b1.isoformat()]
            if st.button("Investigate  →", type="primary", key="inv_go", disabled=not formula):
                try:
                    out = create_investigation({"dataset_id": ds["id"], "formula": formula, "label": label,
                                                "time_column": time_col, "dimensions": dims or None,
                                                "period_a": pa, "period_b": pb})
                    st.session_state["active_question_id"] = out["question_id"]
                    st.session_state["investigation_id"] = out["investigation_id"]
                    st.switch_page("pages/3_Analyses.py")
                except ApiError as e:
                    st.error(str(e))
    st.caption("Asking a “why did … change?” question on the Analyses page starts an investigation automatically.")

rows = listing["investigations"]
if not rows:
    html('<div class="ds-row-meta" style="padding:16px 0;">No investigations yet.</div>')
    st.stop()

STATUS = {"complete": ("Complete", "verified"), "inconclusive": ("Inconclusive", "warn"), "failed": ("Failed", "error"),
          "queued": ("Queued", "running"), "running": ("Running", "running")}
pick = st.selectbox("Investigation", rows, key="inv_pick",
                    index=next((i for i, r in enumerate(rows) if r["id"] == st.session_state.get("investigation_id")), 0),
                    format_func=lambda r: f"#{r['id']} · {r['params'].get('label') or r['params'].get('formula', '')} · {r['status']}")
try:
    inv = get_investigation(pick["id"])
except ApiError as e:
    st.error(str(e))
    st.stop()
report = inv["report"]
label, kind = STATUS.get(inv["status"], (inv["status"], "neutral"))
html(f'<div style="display:flex;gap:10px;align-items:center;margin-top:8px;">{badge(label, kind)}'
     f'<span class="ds-row-title">{esc(report.get("summary") or "Not finished yet.")}</span></div>')
if inv["status"] in ("queued", "running"):
    st.info("This investigation is still running. Open it on the Analyses page to watch progress.")
    st.stop()
if not report or report.get("error"):
    st.error(report.get("error") or "No report was produced.")
    st.stop()

if inv.get("question_id") and st.button("Open the dashboard and evidence for this investigation  →", key="inv_dash"):
    st.session_state["active_question_id"] = inv["question_id"]
    st.switch_page("pages/4_Dashboards.py")

totals, periods = report.get("totals") or {}, report.get("periods") or {}
if totals.get("a") is not None:
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Earlier ({str(periods['a'][0])[:10]} → {str(periods['a'][1])[:10]})", f"{totals['a']:,.2f}")
    k2.metric(f"Later ({str(periods['b'][0])[:10]} → {str(periods['b'][1])[:10]})", f"{totals['b']:,.2f}")
    k3.metric("Change", f"{totals.get('change', 0):,.2f}",
              f"{totals['change_pct']:+.1f}%" if totals.get("change_pct") is not None else None)
    st.caption(f"Periods: {periods.get('source', '')}. Method: {report.get('decomposition_method', '').replace('_', ' ')}.")

for chart in report.get("charts") or []:
    plot(figure_from_json(chart["plotly_json"]), height=320)

if report.get("ranked_contributors"):
    html('<div class="ds-section-title" style="margin-top:14px;">Contributors, ranked by measured contribution and evidence quality</div>')
    st.dataframe(pd.DataFrame([{
        "Dimension": r["dimension"], "Segment": r["segment"], "Contribution": round(r["contribution"], 2),
        "Share of change %": None if r["share_of_change"] is None else round(r["share_of_change"] * 100, 1),
        "Rows earlier": r["rows_a"], "Rows later": r["rows_b"], "Same direction in both halves": r["consistent_across_halves"],
        "Evidence quality": r["evidence_quality"]} for r in report["ranked_contributors"]]),
        use_container_width=True, hide_index=True)

pv = report.get("price_volume")
if pv:
    html('<div class="ds-section-title" style="margin-top:14px;">Price and volume</div>')
    st.caption(f"{pv['basis']}. {pv['note']}")
    a, b, c = st.columns(3)
    a.metric("Price effect", f"{pv['price_effect']:,.2f}")
    b.metric("Volume effect", f"{pv['volume_effect']:,.2f}")
    c.metric("Entry / exit", f"{pv['entry_exit_effect']:,.2f}")

if report.get("hypotheses"):
    html('<div class="ds-section-title" style="margin-top:14px;">Hypotheses tested</div>')
    H = {"supported": ("Supported", "verified"), "weak": ("Weak", "warn"), "not_supported": ("Not supported", "neutral")}
    for h in report["hypotheses"]:
        hl, hk = H.get(h["status"], (h["status"], "neutral"))
        html(f'<div style="padding:8px 0;border-top:1px solid var(--border);"><div style="display:flex;gap:10px;align-items:center;">'
             f'{badge(hl, hk)}<span class="ds-row-title">{esc(h["hypothesis"])}</span></div>'
             f'<div class="ds-row-meta" style="margin-top:3px;"><b>Test:</b> {esc(h["test"])}<br><b>Result:</b> {esc(h["result"])}</div></div>')

c1, c2 = st.columns(2)
with c1:
    html('<div class="ds-section-title" style="margin-top:14px;">Not established</div>')
    for line in (report.get("unresolved") or ["Nothing was left unexamined within the budget."]) + (report.get("alternative_explanations") or []):
        html(f'<div class="ds-row-meta" style="padding:4px 0;">• {esc(line)}</div>')
with c2:
    html('<div class="ds-section-title" style="margin-top:14px;">How to read this</div>')
    for line in report.get("caveats") or []:
        html(f'<div class="ds-row-meta" style="padding:4px 0;">• {esc(line)}</div>')
b = report.get("budget") or {}
st.caption(f"Bounded run: {b.get('engine_calls', 0)} computations in {b.get('seconds', 0)}s"
           + (f"; stopped at its {b['exhausted'].replace('_', ' ')} limit" if b.get("exhausted") else "")
           + f". Dataset fingerprint {((report.get('provenance') or {}).get('dataset_fingerprint') or '')[:12]}…")
