"""Root cause — where a change in a metric came from: segment contributions,
price versus volume, tested hypotheses, and what could not be established."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import (ApiError, create_hypothesis_investigation, create_investigation, get_investigation,
                        list_datasets, list_investigations, list_metrics)
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
            html('<div class="ds-section-title">Ask why</div>')
            why = st.text_area("Question", key="inv_question", height=70,
                               placeholder="Why did revenue drop in the last two months?", label_visibility="collapsed")
            st.caption("A model proposes competing hypotheses; each one is tested by a fixed, deterministic analysis on your data. "
                       "Needs a configured model provider.")
            if st.button("Investigate with hypotheses  →", type="primary", key="inv_why_go", disabled=len((why or "").strip()) < 8):
                try:
                    out = create_hypothesis_investigation(ds["id"], why.strip())
                    st.session_state["active_question_id"] = out["question_id"]
                    st.session_state["investigation_id"] = out["investigation_id"]
                    st.switch_page("pages/3_Analyses.py")
                except ApiError as e:
                    st.error(str(e))
            html('<div class="ds-section-title" style="margin-top:14px;">Or specify the metric yourself (no model)</div>')
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
                    format_func=lambda r: f"#{r['id']} · {r['params'].get('label') or r['params'].get('formula') or r['params'].get('question', '')[:70]} · {r['status']}")
try:
    inv = get_investigation(pick["id"])
except ApiError as e:
    st.error(str(e))
    st.stop()
report = inv["report"]
label, kind = STATUS.get(inv["status"], (inv["status"], "neutral"))
html(f'<div style="display:flex;gap:10px;align-items:center;margin-top:8px;">{badge(label, kind)}'
     f'<span class="ds-row-title">{esc(report.get("summary") or "Not finished yet.")}</span></div>')
HYP_STATUS = {"supported": ("Supported", "verified"), "weakened": ("Weakened", "neutral"), "inconclusive": ("Inconclusive", "warn"),
              "not_testable": ("Not testable", "warn"), "failed": ("Failed to execute", "error"), "proposed": ("Not tested", "running")}


def _bullets(title: str, lines: list, empty: str | None = None) -> None:
    if not lines and empty is None:
        return
    html(f'<div class="ds-section-title" style="margin-top:14px;">{esc(title)}</div>')
    for line in lines or [empty]:
        html(f'<div class="ds-row-meta" style="padding:4px 0;">• {esc(line)}</div>')


def _hypothesis_rows(hypotheses: list) -> None:
    for h in hypotheses:
        hl, hk = HYP_STATUS.get(h["status"], (h["status"], "neutral"))
        tests = [e for e in h.get("evidence") or [] if e.get("role", "test") == "test"]
        inside = [e for e in h.get("evidence") or [] if e.get("role") == "refinement"]
        lines = "".join(f'<br><b>{esc(e.get("analysis_id") or "not run")}</b> · {esc(str(e.get("assessment", "")).replace("_", " "))}: '
                        f'{esc("; ".join(e.get("reasons") or []))}' for e in tests)
        lines += "".join(f'<br><b>{esc(e.get("analysis_id") or "")}</b> · inside this finding: {esc("; ".join(e.get("reasons") or []))}'
                         for e in inside)
        prediction = h.get("measurable_prediction") or (h.get("prediction") or {}).get("expect") or ""
        html(f'<div style="padding:8px 0;border-top:1px solid var(--border);"><div style="display:flex;gap:10px;align-items:center;">'
             f'{badge(hl, hk)}<span class="ds-row-title">{esc(h["id"])} · {esc(h["statement"])}</span></div>'
             f'<div class="ds-row-meta" style="margin-top:3px;"><b>Prediction:</b> {esc(prediction)}'
             f'{lines or "<br>No analysis has run for this hypothesis yet."}</div></div>')


def _analysis_table(analyses: list) -> None:
    st.dataframe(pd.DataFrame([{"ID": a["id"], "Analysis": a["op"].replace("_", " "), "Parameters": str(a.get("params") or {}),
                                "Status": a["status"], "Tests": ", ".join(a.get("hypothesis_ids") or []),
                                "Asked by": a.get("requested_by"), "ms": a.get("duration_ms")} for a in analyses]),
                 use_container_width=True, hide_index=True)


if inv.get("engine") == "hypothesis" and inv["status"] in ("queued", "running"):
    st.info("This investigation is still running. Open it on the Analyses page to watch progress, or reload this page.")
    if inv.get("hypotheses"):
        html(f'<div class="ds-section-title" style="margin-top:14px;">Hypotheses so far · {len(inv.get("analyses") or [])} analyses executed</div>')
        _hypothesis_rows(inv["hypotheses"])
    st.stop()
if inv["status"] in ("queued", "running"):
    st.info("This investigation is still running. Open it on the Analyses page to watch progress.")
    st.stop()
if not report or report.get("error"):
    st.error(report.get("error") or "No report was produced.")
    st.stop()

if inv.get("question_id") and st.button("Open the dashboard and evidence for this investigation  →", key="inv_dash"):
    st.session_state["active_question_id"] = inv["question_id"]
    st.switch_page("pages/4_Dashboards.py")

if report.get("engine") == "hypothesis":
    u, totals, periods = report.get("understanding") or {}, report.get("totals") or {}, report.get("periods") or {}
    determination = report.get("determination") or {}
    (st.success if determination.get("code") == "explanations_supported" else st.warning)(determination.get("text") or "")
    if totals.get("a") is not None and totals.get("b") is not None:
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Earlier ({str(periods['a'][0])[:10]} → {str(periods['a'][1])[:10]})", f"{totals['a']:,.2f}")
        k2.metric(f"Later ({str(periods['b'][0])[:10]} → {str(periods['b'][1])[:10]})", f"{totals['b']:,.2f}")
        k3.metric("Change", f"{totals.get('change') or 0:,.2f}",
                  f"{totals['change_pct']:+.1f}%" if totals.get("change_pct") is not None else None)
    if u:
        metric = u.get("metric") or {}
        st.caption(f"Question read as: {metric.get('label', '')} = {metric.get('formula', '')} ({metric.get('source', '')}); "
                   f"population: {u.get('population', 'all rows')}; periods: {u.get('periods_source', '')}.")
        _bullets("Assumptions made", u.get("assumptions") or [])
        _bullets("Information the question did not give", u.get("missing_information") or [])
    for chart in report.get("charts") or []:
        plot(figure_from_json(chart["plotly_json"]), height=320)

    counts = report.get("hypothesis_counts") or {}
    html('<div class="ds-section-title" style="margin-top:14px;">Hypotheses tested · '
         + esc(", ".join(f"{n} {HYP_STATUS.get(s, (s,))[0].lower()}" for s, n in ((s, counts.get(s)) for s in HYP_STATUS) if n) or "none") + '</div>')
    if report.get("hypotheses"):
        _hypothesis_rows(report["hypotheses"])
    else:
        html('<div class="ds-row-meta" style="padding:4px 0;">No hypothesis was tested.</div>')
    if report.get("rejected_hypotheses"):
        with st.expander(f"Proposed by the model but rejected before testing ({len(report['rejected_hypotheses'])})"):
            for r in report["rejected_hypotheses"]:
                html(f'<div class="ds-row-meta" style="padding:4px 0;">• <b>{esc(r.get("statement") or "")}</b> — {esc(r.get("reason") or "")}</div>')

    c1, c2 = st.columns(2)
    with c1:
        _bullets("Observed facts", report.get("observed_facts") or [], "Nothing was measured.")
        _bullets("Statistical interpretation", report.get("statistical_interpretations") or [], "No significance test applied.")
    with c2:
        _bullets("Possible explanations (not established)", report.get("possible_explanations") or [], "None offered.")
        _bullets("Suggested follow-up analyses", report.get("follow_ups") or [], "None suggested.")
    _bullets("Data quality", report.get("data_quality") or [], "No data-quality problem was detected in the two periods.")
    _bullets("Limitations", report.get("limitations") or [])

    if report.get("analyses"):
        with st.expander(f"Every analysis executed ({len(report['analyses'])})"):
            _analysis_table(report["analyses"])
            st.caption("Each row is a deterministic computation on the pinned dataset version. Hypothesis statuses are derived from "
                       "these results by fixed rules, not by the model.")
    b, repro, summary = report.get("budget") or {}, report.get("reproducibility") or {}, report.get("executive_summary") or {}
    limits = b.get("limits") or {}
    st.caption(f"Bounded run: {b.get('model_calls', 0)} of {limits.get('model_calls', '?')} model calls, "
               f"{b.get('analyses', 0)} of {limits.get('analyses', '?')} analyses, {b.get('rounds', 0)} follow-up rounds, "
               f"{b.get('seconds', 0)}s. Stopped: {str(report.get('stop_reason', '')).replace('_', ' ')}. "
               f"Summary written by {'the model, with every number checked against executed results' if summary.get('source') == 'model' else 'a fixed template (' + str(summary.get('rejected_reason') or 'no model text used') + ')'}. "
               f"Dataset fingerprint {(repro.get('dataset_fingerprint') or '')[:12]}…"
               + (" Resumed after an interruption." if report.get("resumed") else ""))
    st.stop()

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
