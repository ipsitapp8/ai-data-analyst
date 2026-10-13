"""Root cause — where a change in a metric came from: segment contributions,
price versus volume, tested hypotheses, and what could not be established."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import (ApiError, ask_investigation, create_hypothesis_investigation, create_investigation, get_investigation,
                        get_investigation_statistics, list_datasets, list_investigations, list_metrics)
from style.theme import (DASHBOARD_TOKENS, badge, esc, figure_from_json, html, page_header, page_setup, plot, render_sidebar,
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
            st.caption("A root-cause investigation compares a metric between two periods. Upload a version of this dataset "
                       "with a date column (order date, event date) to use it.")
        elif not numeric:
            st.warning("This dataset has no numeric column, so there is no metric whose change could be measured.")
        else:
            if not categorical:
                st.info("This dataset has no categorical column (region, product, channel). The change can be measured and "
                        "tested, but not located in any segment.")
            html('<div class="ds-section-title">Ask why</div>')
            why = st.text_area("Question", key="inv_question", height=70,
                               placeholder="Why did revenue drop in the last two months?", label_visibility="collapsed")
            st.caption("Name a numeric column and, if you like, a period: “last month”, “the last 14 days”, “in March”. "
                       "Periods are read against the dataset's own dates.")
            too_short = len((why or "").strip()) < 8
            if st.button("Investigate  →", type="primary", key="inv_ask_go", disabled=too_short):
                try:
                    out = ask_investigation(ds["id"], why.strip())
                    st.session_state["active_question_id"] = out["question_id"]
                    st.session_state["investigation_id"] = out["investigation_id"]
                    st.switch_page("pages/3_Analyses.py")
                except ApiError as e:
                    st.error(str(e))
            st.caption("The button above uses no AI model: every figure is computed from your data. The one below also has a "
                       "model propose competing hypotheses, each tested by a fixed analysis; it needs a configured model provider.")
            if st.button("Investigate with hypotheses  →", key="inv_why_go", disabled=too_short):
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


def _num(value, digits: int = 3) -> str:
    """A number for a table, or a dash when it could not be computed. Never 0 for 'unavailable'."""
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(_num(v, digits) for v in value)
    a = abs(value)
    return f"{value:,.0f}" if a >= 1000 else f"{value:,.2f}" if a >= 1 else f"{value:.{digits}g}"


def _p(value) -> str:
    return "—" if value is None else "< 0.001" if value < 0.001 else f"{value:.3f}"


EVIDENCE_LABEL = {"complete": ("Complete", "verified"), "in_progress": ("In progress", "running"),
                  "partial": ("Partial", "warn"), "none": ("None", "neutral")}


def _evidence_line(evidence: dict) -> None:
    """Where the statistics shown come from, and whether the run that computed them finished."""
    if not evidence:
        return
    label, kind = EVIDENCE_LABEL.get(evidence.get("status"), (str(evidence.get("status")), "neutral"))
    source = f"dataset version {evidence.get('dataset_version_id')}" if evidence.get("dataset_version_id") is not None else "this dataset"
    fingerprint = f", fingerprint {str(evidence['dataset_fingerprint'])[:12]}…" if evidence.get("dataset_fingerprint") else ""
    html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge("Evidence: " + label, kind)} '
         f'{esc(evidence.get("note") or "")} Computed on {esc(source)}{esc(fingerprint)}.</div>')


def _partial_statistics(inv: dict, indent: bool = False) -> None:
    """The statistical results a run computed before it stopped, one line each.
    `indent` sets the lines in from the left, for use under a heading line."""
    padding = "4px 0 4px 18px" if indent else "4px 0"
    try:
        stats = get_investigation_statistics(inv["id"])["statistics"]
    except ApiError:
        stats = []
    for s in stats:
        lines = [l for l in s.get("interpretation") or [] if not l.startswith("The interval is the range")]
        html(f'<div class="ds-row-meta" style="padding:{padding};">• <b>{esc(s["analysis_id"])}</b> ({esc(s["op"].replace("_", " "))}): '
             f'{esc(" ".join(lines))}</div>')
    if not stats:
        html(f'<div class="ds-row-meta" style="padding:{padding};">The run left analysis results but no statistical tests.</div>')


def _fallback_section(inv: dict) -> bool:
    """One place that says a deterministic fallback happened and, when the AI
    run got far enough to compute something, keeps three things apart: what
    was computed, what the result below is, and whether the two are connected.
    Returns True when it was shown."""
    fallback = inv.get("fallback") or {}
    if not fallback.get("used"):
        return False
    st.info(fallback.get("message") or "The AI investigation could not complete. The result shown is from the deterministic "
            "investigator, which uses fixed rules and no AI model; it is not AI-generated analysis.")
    evidence = inv.get("evidence") or {}
    if evidence.get("status") != "partial":
        return True
    framing = evidence.get("framing") or {}
    day = lambda x: str(x)[:10]  # noqa: E731
    scope = ""
    if framing.get("formula") and framing.get("period_a") and framing.get("period_b"):
        scope = (f" for {framing.get('label') or 'the metric'} = {framing['formula']}, {day(framing['period_a'][0])} to "
                 f"{day(framing['period_a'][1])} against {day(framing['period_b'][0])} to {day(framing['period_b'][1])}")
    count, version = evidence.get("analyses"), evidence.get("dataset_version_id")
    numeric = isinstance(count, int) and not isinstance(count, bool) and count > 0
    how_many = "some analyses" if not numeric else "1 analysis" if count == 1 else f"{count} analyses"
    where = f"dataset version {version}" if version is not None else "this dataset"
    html('<div class="ds-section-title" style="margin-top:10px;">What was computed and what this result is based on</div>')
    html(f'<div class="ds-row-meta" style="padding:4px 0;"><b>Computed before the AI model was lost (partial evidence):</b> '
         f'{esc(how_many)} on {esc(where)}{esc(scope)}. Statistical results among them:</div>')
    _partial_statistics(inv, indent=True)
    html('<div class="ds-row-meta" style="padding:4px 0;"><b>Deterministic fallback result:</b> everything below this section '
         '(totals, contributors, price and volume, hypotheses tested by fixed rules) was produced by the deterministic '
         'investigator after the fallback.</div>')
    same = evidence.get("matches_report_framing")
    relation = {True: " It compares the same metric over the same periods.",
                False: " It compares a different metric or different periods, so the two are not directly comparable."}.get(same, "")
    html(f'<div class="ds-row-meta" style="padding:4px 0;"><b>Did the report below use the computed evidence?</b> '
         f'{"Yes." if evidence.get("report_uses_evidence") else "No. It was produced separately."}{esc(relation)}</div>')
    return True


def _partial_evidence_section(inv: dict) -> None:
    """Statistics left by a run that did not complete and was not followed by
    a fallback report: shown apart from the report and labelled for what they are."""
    evidence = inv.get("evidence") or {}
    if evidence.get("status") != "partial":
        return
    html('<div class="ds-section-title" style="margin-top:14px;">Partial statistical evidence — not part of this report</div>')
    st.warning(evidence.get("note") or "These results come from a run that did not complete.")
    _partial_statistics(inv)


def _statistics_section(report: dict, evidence: dict | None = None) -> None:
    """Statistical evidence, in plain language first; the full output of each
    method sits in an expander for whoever wants to check it."""
    stats = [s for s in report.get("statistics") or []]
    if not stats:
        return
    html('<div class="ds-section-title" style="margin-top:14px;">Statistical evidence</div>')
    _evidence_line(evidence or {})
    for line in report.get("statistical_evidence") or []:
        html(f'<div class="ds-row-meta" style="padding:4px 0;">• {esc(line)}</div>')
    m = report.get("multiplicity")
    if m:
        st.caption(f"{m['comparisons']} tests were run for the hypotheses, so their p-values are {m['method']}-adjusted "
                   f"(controls {m['controls']}).")
    _bullets("Practical significance", report.get("practical_significance") or [],
             "No estimated difference could be expressed relative to the earlier level.")
    with st.expander(f"Technical details of the statistical methods ({len(stats)})"):
        st.dataframe(pd.DataFrame([{
            "Analysis": s["analysis_id"], "For": ", ".join(s.get("hypothesis_ids") or []) or ("headline" if s.get("requested_by") == "baseline" else ""),
            "Method": s["method_label"] if s["status"] == "ok" else "not available", "Estimand": s["estimand"],
            "n": (s.get("sample") or {}).get("n"), "Estimate": _num(s.get("estimate")),
            "Interval": "—" if not s.get("interval") else f"{_num(s['interval']['low'])} to {_num(s['interval']['high'])}",
            "Statistic": "—" if not s.get("test") or s["test"].get("statistic") is None
            else f"{s['test'].get('statistic_name') or ''} = {_num(s['test']['statistic'])}",
            "df": _num((s.get("test") or {}).get("df")), "p": _p((s.get("test") or {}).get("p_value")),
            "Adjusted p": _p((s.get("adjusted") or {}).get("p_value")),
        } for s in stats]), use_container_width=True, hide_index=True)
        for s in stats:
            head = f"{s['analysis_id']} · {s['method_label']}" if s["status"] == "ok" else f"{s['analysis_id']} · no statistical conclusion"
            html(f'<div class="ds-row-title" style="margin-top:10px;">{esc(head)}</div>')
            lines = []
            if s["status"] != "ok":
                lines.append(f"Why not: {s.get('reason')}")
            if s.get("test"):
                lines.append(f"Null hypothesis: {s['test']['null']}. Alternative: {s['test']['alternative']}.")
            lines += [f"Why this method: {r}" for r in s.get("selection") or []]
            lines += [f"Not used — {r['method'].replace('_', ' ')}: {r['reason']}" for r in s.get("rejected_methods") or []]
            sm = s.get("suggested_method")
            if sm:
                lines.append(f"Method suggested by the model: {sm['method'].replace('_', ' ')} — {sm['outcome']} ({sm['reason']}).")
            lines += [f"Assumption — {a['name']}: {a['status'].replace('_', ' ')} ({a['detail']})" for a in s.get("assumptions") or []]
            lines += [f"Effect — {e['name']}: {_num(e['value'])}" + (f" {e['units']}" if e.get("units") else "")
                      + (f" ({_num(e['interval']['low'])} to {_num(e['interval']['high'])})" if e.get("interval") else "")
                      + (f" — {e['note']}" if e.get("note") else "") for e in s.get("effects") or []]
            excluded = (s.get("sample") or {}).get("excluded")
            if excluded:
                lines.append(f"Left out: {excluded}. {(s.get('sample') or {}).get('exclusion_rule') or ''}")
            lines += [f"Warning: {w}" for w in s.get("warnings") or []]
            for line in lines:
                html(f'<div class="ds-row-meta" style="padding:2px 0;">{esc(line)}</div>')
            if s.get("comparisons"):
                st.dataframe(pd.DataFrame([{"Groups": " vs ".join(c["groups"]), "Difference": _num(c.get("estimate")),
                                            "p": _p(c.get("p_value")), "Holm-adjusted p": _p(c.get("p_adjusted"))}
                                           for c in s["comparisons"]]), use_container_width=True, hide_index=True)
    _bullets("How to read these statistics", report.get("how_to_read_statistics") or [])


def _analysis_table(analyses: list) -> None:
    st.dataframe(pd.DataFrame([{"ID": a["id"], "Analysis": a["op"].replace("_", " "), "Parameters": str(a.get("params") or {}),
                                "Status": a["status"], "Tests": ", ".join(a.get("hypothesis_ids") or []),
                                "Asked by": a.get("requested_by"), "ms": a.get("duration_ms")} for a in analyses]),
                 use_container_width=True, hide_index=True)


def _n(value, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+,.2f}" if signed else f"{value:,.2f}"


def _stat_text(stat: dict | None) -> str:
    if not stat:
        return "—"
    if stat.get("status") == "note":
        return str(stat.get("text") or "—")
    if stat.get("status") != "ok":
        return "No test: " + str(stat.get("reason") or "not run")
    p = stat.get("p_adjusted", stat.get("p_value"))
    return (f"{_n(stat.get('estimate'), True)} per {stat.get('bucket') or 'bucket'} "
            f"({(stat.get('level') or 0) * 100:.0f}% interval {_n(stat.get('low'))} to {_n(stat.get('high'))}); "
            f"p {_p(p) if p is not None and p < 0.001 else '= ' + _p(p)}{' after adjustment' if 'p_adjusted' in stat else ''}")


_T = DASHBOARD_TOKENS
_EVIDENCE_CSS = f"""<style>
.rc-panel {{ background: {_T['--bg-page']}; border-radius: 20px; padding: 14px; margin: 10px 0 6px 0; }}
.rc-panel table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: {_T['--bg-card']};
  border-radius: 14px; overflow: hidden; font-size: 0.84rem; color: {_T['--text-primary']}; }}
.rc-panel th {{ text-align: left; font-weight: 600; font-size: 0.74rem; letter-spacing: .03em; text-transform: uppercase;
  color: {_T['--text-secondary']}; background: {_T['--bg-inset']}; padding: 9px 12px; border-bottom: 1px solid {_T['--border']}; }}
.rc-panel td {{ padding: 10px 12px; border-bottom: 1px solid {_T['--border']}; vertical-align: top; }}
.rc-panel tr:last-child td {{ border-bottom: none; }}
.rc-panel tr.rc-hyp td {{ background: {_T['--bg-card-hover']}; color: {_T['--text-secondary']}; }}
.rc-bar {{ height: 6px; border-radius: 3px; background: {_T['--accent-bg']}; margin-top: 5px; min-width: 70px; }}
.rc-bar span {{ display: block; height: 6px; border-radius: 3px; background: {_T['--accent']}; }}
.rc-tag {{ display: inline-block; font-size: 0.72rem; font-weight: 600; border-radius: 999px; padding: 2px 9px; white-space: nowrap; }}
.rc-strong {{ background: {_T['--accent']}; color: #fff; }}
.rc-moderate {{ background: {_T['--accent-bg']}; color: {_T['--accent-ink']}; border: 1px solid {_T['--accent-border']}; }}
.rc-weak, .rc-none, .rc-offset, .rc-exact {{ background: {_T['--bg-inset']}; color: {_T['--text-secondary']}; border: 1px solid {_T['--border']}; }}
.rc-muted {{ color: {_T['--text-muted']}; font-size: 0.78rem; }}
</style>"""
_SUPPORT = {"strong": "Strong evidence of location", "moderate": "Moderate evidence of location", "weak": "Weak evidence",
            "offset": "Offsets the change", "exact": "Exact arithmetic"}


def _evidence_section(report: dict) -> None:
    """The evidence table: what was measured (a location of the change, never a
    cause) apart from the explanations that were not and could not be tested."""
    rows = report.get("evidence_table") or []
    if not rows:
        return
    html('<div class="ds-section-title" style="margin-top:14px;">Evidence for each candidate explanation</div>')
    body = []
    for r in rows:
        limits = "; ".join(r.get("limitations") or []) or "none noted"
        if r.get("kind") == "hypothesis":
            body.append(f'<tr class="rc-hyp"><td><b>{esc(r["candidate"])}</b><div class="rc-muted">Hypothesis, not a finding</div></td>'
                        f'<td>—</td><td>—</td><td>Not tested</td><td><span class="rc-tag rc-none">{esc(r.get("support") or "")}</span></td>'
                        f'<td>{esc(limits)}</td></tr>')
            continue
        share = r.get("share_of_change")
        width = 0 if share is None or share < 0 else min(share, 1.0) * 100
        measured = "—" if r.get("value_a") is None else f"{_n(r['value_a'])} → {_n(r.get('value_b'))}"
        rows_text = "" if r.get("rows_a") is None else f'<div class="rc-muted">{r["rows_a"]:,} → {r["rows_b"]:,} rows</div>'
        support = r.get("support") or "weak"
        body.append(
            f'<tr><td><b>{esc(r["candidate"])}</b><div class="rc-muted">Measured: {esc(r.get("claim") or "")}</div></td>'
            f'<td>{esc(measured)}{rows_text}</td>'
            f'<td>{esc(_n(r.get("contribution"), True))}'
            + ("" if share is None else f'<div class="rc-muted">{share * 100:.0f}% of the change</div>' if share >= 0
               else f'<div class="rc-muted">against the change ({share * 100:.0f}%)</div>')
            + f'<div class="rc-bar"><span style="width:{width:.0f}%"></span></div></td>'
            f'<td>{esc(_stat_text(r.get("statistic")))}</td>'
            f'<td><span class="rc-tag rc-{esc(support)}">{esc(_SUPPORT.get(support, support))}</span>'
            f'<div class="rc-muted">data quality: {esc(r.get("evidence_quality") or "")}</div></td>'
            f'<td>{esc(limits)}</td></tr>')
    st.markdown(_EVIDENCE_CSS + '<div class="rc-panel"><table><thead><tr><th>Candidate explanation</th><th>Measured change</th>'
                '<th>Contribution</th><th>Statistical evidence</th><th>Support</th><th>Limitations</th></tr></thead><tbody>'
                + "".join(body) + "</tbody></table></div>", unsafe_allow_html=True)
    family = (report.get("change_evidence") or {}).get("family") or {}
    st.caption("“Evidence of location” means the measurements show the change sits in that segment. It is never evidence of what "
               "caused it: a segment that moved is a place to look, not an explanation."
               + (f" The {family['comparisons']} segment tests are {family['method']}-adjusted together."
                  if family.get("comparisons") else ""))


if inv.get("engine") == "hypothesis" and inv["status"] in ("queued", "running"):
    st.info("This investigation is still running. Open it on the Analyses page to watch progress, or reload this page.")
    _evidence_line(inv.get("evidence") or {})
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

# The AI route was asked for but could not finish: said once, before the deterministic result.
fallback_explained = _fallback_section(inv)


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
        _bullets("Observed result", report.get("observed_facts") or [], "Nothing was measured.")
    with c2:
        _bullets("Possible explanations (not established)", report.get("possible_explanations") or [], "None offered.")
        _bullets("Suggested follow-up analyses", report.get("follow_ups") or [], "None suggested.")
    _statistics_section(report, inv.get("evidence"))
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
    if totals.get("length_adjusted"):
        lfl_pct = totals.get("like_for_like_pct")
        st.caption(f"The later period is {totals['days_b']:g} days against {totals['days_a']:g}. That difference in length alone "
                   f"accounts for {_n(totals.get('length_effect'), True)} of the change; like for like the change is "
                   f"{_n(totals.get('like_for_like_change'), True)}" + (f" ({lfl_pct:+.1f}%)." if lfl_pct is not None else "."))
    overall = (report.get("change_evidence") or {}).get("overall")
    if overall:
        st.caption("Is the change larger than the metric's ordinary variation? " + _stat_text(overall)
                   + ("" if overall.get("status") != "ok" else
                      " — yes." if overall.get("significant") else " — no, it is within ordinary variation."))
for issue in (report.get("comparability") or {}).get("issues") or []:
    st.warning(f"The two periods are not fully comparable: {issue}.")
if report.get("status") == "inconclusive" and not report.get("ranked_contributors"):
    st.info("Nothing could be located for this question. " + " ".join(report.get("unresolved") or []))

for chart in report.get("charts") or []:
    plot(figure_from_json(chart["plotly_json"]), height=320)

_evidence_section(report)

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
_bullets("Recommended follow-up analyses", report.get("follow_ups") or [])
if not fallback_explained:       # after a fallback the partial evidence is already covered in the section at the top
    _partial_evidence_section(inv)
b = report.get("budget") or {}
st.caption(f"Bounded run: {b.get('engine_calls', 0)} computations in {b.get('seconds', 0)}s"
           + (f"; stopped at its {b['exhausted'].replace('_', ' ')} limit" if b.get("exhausted") else "")
           + f". Dataset fingerprint {((report.get('provenance') or {}).get('dataset_fingerprint') or '')[:12]}…")
