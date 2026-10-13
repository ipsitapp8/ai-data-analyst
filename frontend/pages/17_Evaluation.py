"""Evaluation — results of the agent regression suite, run from the command
line (`python -m app.evals run`). This page only reads them."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import ApiError, compare_eval_runs, eval_suite, get_eval_run, list_eval_runs
from style.theme import badge, esc, html, page_header, page_setup, render_sidebar

page_setup("Evaluation")
render_sidebar("evaluation")
page_header("Agent evaluation", "Regression results for the pipeline and each of its parts, against reference answers.")

try:
    suite, runs = eval_suite(), list_eval_runs()
except ApiError as e:
    st.error(f"Backend unreachable: {e}")
    st.stop()

html(f'<div class="ds-row-meta">Suite <b>{esc(suite["suite_version"])}</b>: {suite["cases"]} cases, of which '
     f'{suite["live_cases"]} run only against real model providers. Components: '
     + esc(", ".join(f"{k} ({v})" for k, v in sorted(suite["by_component"].items()))) + ".</div>")

if not runs:
    st.info("No evaluation run has been stored yet.")
    st.code("cd backend\npython -m app.evals run                 # offline: scripted model, no API key, no cost\n"
            "python -m app.evals run --live --provider gemini   # real provider; costs money\n"
            "python -m app.evals compare <base id> <new id>     # regression report", language="bash")
    st.stop()

def _label(r: dict) -> str:
    s = r["summary"]
    return (f"#{r['id']} · {str(r['started_at'])[:16].replace('T', ' ')} · {r['mode']} · {r['provider']} · "
            f"{s.get('passed')}/{s.get('cases')} passed · git {r['git_sha'] or '—'}")


pick = st.selectbox("Run", runs, format_func=_label, key="ev_pick")
try:
    run = get_eval_run(pick["id"])
except ApiError as e:
    st.error(str(e))
    st.stop()
s, p = run["summary"], run["summary"].get("pipeline") or {}
k = st.columns(5)
k[0].metric("Cases passed", f"{s['passed']} / {s['cases']}")
k[1].metric("Model calls", p.get("llm_calls", 0))
k[2].metric("Code runs", p.get("sandbox_runs", 0))
k[3].metric("Executor retries", p.get("retries", 0))
k[4].metric("Estimated cost", f"${p.get('estimated_cost_usd', 0):.4f}")
st.caption(f"Suite {run['suite_version']} · application {run['app_version']} · {p.get('tokens_in', 0):,} tokens in / "
           f"{p.get('tokens_out', 0):,} out · mean pipeline latency {p.get('mean_latency_ms', 0)} ms · "
           f"{p.get('verification_failed_checks', 0)} failed verification checks across pipeline cases (several cases "
           "plant errors on purpose).")

st.dataframe(pd.DataFrame([{"Component": c, "Passed": v["passed"], "Cases": v["cases"]}
                           for c, v in sorted(s["by_component"].items())]), use_container_width=True, hide_index=True)

only_failed = st.checkbox("Show only failed cases", value=bool(s.get("failed")), key="ev_failed")
for r in run["results"]:
    if only_failed and r["passed"]:
        continue
    with st.expander(("✓ " if r["passed"] else "✗ ") + f"{r['case_id']} · {r['component']}"):
        if r["detail"].get("note"):
            st.caption(r["detail"]["note"])
        for c in r["detail"]["checks"]:
            html(f'<div style="display:flex;gap:8px;align-items:center;padding:3px 0;">'
                 f'{badge("Pass" if c["ok"] else "Fail", "verified" if c["ok"] else "error")}'
                 f'<span class="ds-row-title">{esc(c["check"])}</span><span class="ds-row-meta">{esc(str(c["detail"]))}</span></div>')
        m = {k2: v for k2, v in r["metrics"].items() if not isinstance(v, (dict, list))}
        if m:
            st.caption(" · ".join(f"{k2}: {v}" for k2, v in m.items()))

if len(runs) > 1:
    html('<div class="ds-section-title" style="margin-top:18px;">Compare with another run</div>')
    others = [r for r in runs if r["id"] != pick["id"]]
    base = st.selectbox("Baseline", others, format_func=_label, key="ev_base")
    try:
        cmp = compare_eval_runs(base["id"], pick["id"])
    except ApiError as e:
        cmp = None
        st.error(str(e))
    if cmp:
        if not cmp["comparable"]:
            st.warning("These runs used different suite versions or modes, so the comparison is indicative only.")
        if cmp["regressions"]:
            st.error("Regressions (passed before, fail now): " + ", ".join(cmp["regressions"]))
        else:
            st.success("No regressions.")
        if cmp["fixed"]:
            st.info("Fixed since the baseline: " + ", ".join(cmp["fixed"]))
        if cmp["metric_deltas"]:
            st.dataframe(pd.DataFrame([{"Metric": k2, "Baseline": v["base"], "This run": v["new"], "Change": v["delta"]}
                                       for k2, v in cmp["metric_deltas"].items()]), use_container_width=True, hide_index=True)
