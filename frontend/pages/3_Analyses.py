"""Analyses — ask a question, then watch the agent plan / execute / verify live."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from api_client import (
    ApiError,
    ask_question,
    get_audit_trail,
    get_status,
    list_datasets,
    list_questions,
)
from style.theme import ACCENT, ACCENT_2, badge, esc, html, page_header, page_setup, plot, render_sidebar

page_setup("Analyses")
render_sidebar("analyses")

STAGE_LABELS = {
    "queued": "Queued",
    "planning": "Planning analysis",
    "executing": "Executing code",
    "critic": "Verifying results",
    "dashboard": "Generating dashboard",
    "done": "Complete",
    "failed": "Failed",
}
STAGE_ORDER = ["planning", "executing", "critic", "dashboard"]

qid = st.session_state.get("active_question_id")

# ---------------------------------------------------------------- new run --
if not qid:
    page_header("New Analysis", "Ask a question in plain English — the agent does the rest.")

    try:
        datasets = list_datasets()
    except ApiError as e:
        datasets = []
        st.error(f"Backend unreachable: {e}")

    if not datasets:
        st.info("No dataset yet — upload a CSV first.")
        if st.button("Go to Datasets  →", type="primary", key="an_go_ds"):
            st.switch_page("pages/2_Datasets.py")
        st.stop()

    with st.container(key="card_ask"):
        labels = [f"{d['filename']}  ·  {d['row_count']:,} rows" for d in datasets]
        active = st.session_state.get("active_dataset_id")
        default = next((i for i, d in enumerate(datasets) if d["id"] == active), 0)
        idx = st.selectbox("Dataset", range(len(datasets)), index=default,
                           format_func=lambda i: labels[i])

        question = st.text_area("Question", height=110,
                                placeholder="Why did revenue drop in Q3?",
                                label_visibility="collapsed")
        if st.button("Run Analysis  →", type="primary", key="an_run",
                     disabled=not question.strip()):
            try:
                res = ask_question(datasets[idx]["id"], question.strip())
                st.session_state["active_dataset_id"] = datasets[idx]["id"]
                st.session_state["active_question_id"] = res["question_id"]
                st.session_state["question_running"] = True
                st.rerun()
            except ApiError as e:
                st.error(f"Could not start: {e}")

    # previous runs
    try:
        history = list_questions()
    except ApiError:
        history = []
    if history:
        html("<div style='height:22px'></div>")
        with st.container(key="flat_hist"):
            html('<div class="ds-card-head"><div class="ds-section-title">'
                        'Previous Analyses</div></div>')
            for h in history[:8]:
                kind = {"verified": "verified", "unverified": "warn", "rejected": "warn",
                        "failed": "error"}.get(h["status"], "running")
                label = {"verified": "Verified", "unverified": "Needs review", "rejected": "Not analyzable",
                         "failed": "Failed"}.get(h["status"], "Running")
                c1, c2 = st.columns([5, 1])
                with c1:
                    html(
                        f'<div class="ds-row" style="border-top:none;">'
                        f'<div><div class="ds-row-title">{esc(h["text"][:64])}</div>'
                        f'<div class="ds-row-meta">{h["age"]}</div></div>'
                        f'<div class="ds-row-spacer"></div>{badge(label, kind)}</div>'
                    )
                with c2:
                    if st.button("Open", key=f"an_open_{h['id']}"):
                        st.session_state["active_question_id"] = h["id"]
                        st.rerun()
    st.stop()

# ---------------------------------------------------------------- live run --
if st.button("←  All analyses", key="an_back"):
    st.session_state.pop("active_question_id", None)
    st.rerun()
html("<div style='height:6px'></div>")


@st.fragment(run_every="2s")
def _render_live_run(qid: int) -> None:
    """Everything that needs to auto-refresh while a run is in progress, scoped
    to its own fragment so only this panel re-renders every 2s -- not the
    whole page (sidebar, header, back button) flashing on every poll."""
    try:
        status = get_status(qid)
    except ApiError as e:
        st.error(f"Could not fetch status: {e}")
        return

    running = status["status"] not in ("verified", "unverified", "failed", "rejected")

    top_l, top_r = st.columns([3, 1])
    with top_l:
        html(
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<div class="ds-page-title" style="margin:0;">{esc(status.get("question_text", "Analysis"))}</div>'
            f"</div>"
        )
    with top_r:
        html("<div style='height:6px'></div>")
        kind = {"verified": "verified", "unverified": "warn", "rejected": "warn",
                "failed": "error"}.get(status["status"], "running")
        label = {"verified": "Verified", "unverified": "Needs review", "rejected": "Not analyzable",
                 "failed": "Failed"}.get(status["status"], "Running")
        html(f'<div style="text-align:right;">{badge(label, kind)}</div>')

    html("<div style='height:18px'></div>")

    # Rejected by triage: nothing ran, so show guidance instead of the plan/exec
    # panels (which would all be empty) or a red error (nothing broke).
    if status["status"] == "rejected":
        msg = status.get("error") or "This question can't be analyzed against this dataset."
        lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
        reason = lines[0] if lines else msg
        tips = [ln.lstrip("• ").strip() for ln in lines[1:] if ln.startswith("•")]

        with st.container(key="card_rejected"):
            html(
                '<div style="display:flex;align-items:center;gap:11px;margin-bottom:12px;">'
                '<div class="ds-section-title">We couldn\'t analyze that question</div></div>'
                f'<div class="ds-row-meta" style="line-height:1.7;font-size:0.93rem;">{esc(reason)}</div>'
            )
            if tips:
                html(
                    '<div class="ds-section-title" style="margin:18px 0 8px 0;font-size:0.92rem;">'
                    "Try asking instead</div>"
                )
                for t in tips:
                    html(f'<div class="ds-check"><span class="ds-check-label">{esc(t)}</span></div>')

        html("<div style='height:16px'></div>")
        if st.button("Ask a different question", type="primary", key="an_retry"):
            st.session_state.pop("active_question_id", None)
            st.rerun()
        return

    steps = status.get("steps", [])
    done_count = sum(1 for s in steps if s["status"] == "done")
    stage = status["current_stage"]

    if stage in STAGE_ORDER:
        stage_frac = STAGE_ORDER.index(stage) / len(STAGE_ORDER)
    elif stage == "done":
        stage_frac = 1.0
    else:
        stage_frac = 0.0
    step_frac = (done_count / len(steps)) if steps else 0.0
    overall = int(round(max(stage_frac, step_frac * 0.9) * 100)) if running else (
        100 if status["status"] != "failed" else int(step_frac * 100))

    col_plan, col_exec, col_prog = st.columns([1, 1.65, 1], gap="medium")

    # ---- plan ----
    with col_plan:
        with st.container(key="card_plan"):
            html(
                f'<div style="display:flex;align-items:center;justify-content:space-between;">'
                f'<div class="ds-section-title">Analysis Plan</div>'
                f'<div class="ds-row-meta">{len(steps)} Steps</div></div>'
            )
            html("<div style='height:10px'></div>")
            if not steps:
                html('<div class="ds-row-meta">Planner is drafting the steps…</div>')
            for i, s in enumerate(steps, start=1):
                state = s["status"]
                num_cls = {"done": "ds-step-done", "running": "ds-step-run"}.get(state, "")
                wrap_cls = "ds-step ds-step-active" if state == "running" else "ds-step"
                text = {"done": "Completed", "running": "Running…",
                        "failed": "Failed"}.get(state, "Pending")
                mark = "✓" if state == "done" else str(i)
                html(
                    f'<div class="{wrap_cls}">'
                    f'<div class="ds-step-num {num_cls}">{mark}</div>'
                    f'<div><div class="ds-step-title">{esc(s["description"][:60])}</div>'
                    f'<div class="ds-step-status">{text}</div></div></div>'
                )

    # ---- code execution ----
    with col_exec:
        with st.container(key="card_exec"):
            html(
                '<div style="display:flex;align-items:center;justify-content:space-between;">'
                '<div class="ds-section-title">Code Execution</div>'
                '<span class="ds-badge ds-badge-neutral">▤ Sandbox</span></div>'
            )
            html("<div style='height:12px'></div>")

            code_text = None
            try:
                trail = get_audit_trail(qid)
                for e in reversed(trail.get("entries", [])):
                    if e.get("code"):
                        code_text = e["code"]
                        break
            except ApiError:
                pass

            if code_text:
                st.code(code_text[:1400], language="python")
            else:
                html(
                    f'<div style="background:var(--bg-inset);border:1px solid var(--border);'
                    f'border-radius:10px;padding:16px;font-family:var(--font-mono);'
                    f'font-size:0.83rem;color:var(--text-secondary);line-height:1.75;">'
                    f'{esc(STAGE_LABELS.get(stage, stage))}…<br/>{esc(status.get("stage_detail","")[:120])}'
                    f"</div>"
                )

            html("<div style='height:14px'></div>")
            html(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.83rem;color:var(--text-secondary);margin-bottom:7px;">'
                f'<span>{"Execution in progress…" if running else "Execution complete"}</span>'
                f"<span>{overall}%</span></div>"
                f'<div class="ds-progress-track"><div class="ds-progress-fill" '
                f'style="width:{overall}%;"></div></div>'
            )
            if running:
                html(
                    '<div style="text-align:center;color:var(--text-secondary);'
                    'font-size:0.83rem;margin-top:14px;">AI agents are working autonomously. '
                    "You'll be notified when the analysis is complete.</div>"
                )

    # ---- progress donut ----
    with col_prog:
        with st.container(key="card_prog"):
            html('<div class="ds-section-title">Execution Progress</div>')
            html("<div style='height:6px'></div>")

            ring = ACCENT_2 if running else ACCENT
            fig = go.Figure(
                go.Pie(
                    values=[overall, max(100 - overall, 0)], hole=0.72, sort=False,
                    direction="clockwise", rotation=0,
                    marker=dict(colors=[ring, "rgba(255,255,255,0.06)"], line=dict(width=0)),
                    textinfo="none", hoverinfo="skip",
                )
            )
            fig.add_annotation(text=f"<b>{overall}%</b>", x=0.5, y=0.55, showarrow=False,
                               font=dict(size=25, color="#e6e6e6", family="system-ui, sans-serif"))
            fig.add_annotation(text="Overall Progress", x=0.5, y=0.38, showarrow=False,
                               font=dict(size=11, color="#969696", family="system-ui, sans-serif"))
            plot(fig, height=200)

            html(
                f"""
                <div style="border-top:1px solid var(--border);padding-top:14px;">
                  <div class="ds-check"><span class="ds-check-label">Stage</span>
                    <span class="ds-row-meta">{STAGE_LABELS.get(stage, stage)}</span></div>
                  <div class="ds-check"><span class="ds-check-label">Steps done</span>
                    <span class="ds-row-meta">{done_count}/{len(steps) if steps else "—"}</span></div>
                  <div class="ds-check"><span class="ds-check-label">Retries</span>
                    <span class="ds-row-meta">{status.get("retry_count", 0)}</span></div>
                </div>
                """
            )

    if status["status"] == "failed" and status.get("error"):
        html("<div style='height:16px'></div>")
        st.error(status["error"][:1500])

    if not running:
        html("<div style='height:18px'></div>")
        a, b, c = st.columns([1, 1, 3])
        with a:
            if st.button("View Dashboard  →", type="primary", key="an_dash"):
                st.switch_page("pages/4_Dashboards.py")
        with b:
            if st.button("Audit Trail", key="an_audit"):
                st.switch_page("pages/6_Audit_Trail.py")
        with c:
            if st.button("Start new analysis", key="an_new"):
                st.session_state.pop("active_question_id", None)
                st.rerun()


_render_live_run(qid)
