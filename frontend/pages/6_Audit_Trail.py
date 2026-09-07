"""Audit Trail — every dashboard number traced to the code, data, and reasoning behind it."""
from __future__ import annotations

import streamlit as st

from api_client import (
    ApiError,
    get_audit_trail,
    get_critic_reviews,
    get_status,
    list_questions,
)
from style.theme import badge, esc, html, page_header, page_setup, render_sidebar

page_setup("Audit Trail")
render_sidebar("audit")

qid = st.session_state.get("active_question_id")

if not qid:
    page_header("Audit Trail", "Full traceability for every analysis.")
    try:
        questions = list_questions()
    except ApiError as e:
        questions = []
        st.error(f"Backend unreachable: {e}")
    if not questions:
        st.info("No analyses yet.")
        if st.button("Go to Analyses  →", type="primary", key="au_go"):
            st.switch_page("pages/3_Analyses.py")
        st.stop()
    with st.container(key="flat_pick"):
        html('<div class="ds-card-head"><div class="ds-section-title">'
                    'Select an analysis</div></div>')
        for q in questions[:12]:
            c1, c2 = st.columns([5, 1])
            with c1:
                html(
                    f'<div class="ds-row" style="border-top:none;">'
                    f'<div><div class="ds-row-title">{esc(q["text"][:70])}</div>'
                    f'<div class="ds-row-meta">{esc(q["status"])} • {esc(q["age"])}</div></div></div>'
                )
            with c2:
                if st.button("Open", key=f"au_open_{q['id']}"):
                    st.session_state["active_question_id"] = q["id"]
                    st.rerun()
    st.stop()

try:
    trail = get_audit_trail(qid)
    status = get_status(qid)
except ApiError as e:
    st.error(f"Could not load audit trail: {e}")
    st.stop()

try:
    reviews = get_critic_reviews(qid)
except ApiError:
    reviews = []

if st.button("←  All analyses", key="au_back"):
    st.session_state.pop("active_question_id", None)
    st.rerun()
html("<div style='height:6px'></div>")

html(f'<div class="ds-page-title">{esc(trail["question_text"])}</div>')
kind = {"verified": "verified", "unverified": "warn", "rejected": "warn",
        "failed": "error"}.get(status["status"], "running")
label = {"verified": "Verified", "unverified": "Needs review", "rejected": "Not analyzable",
         "failed": "Failed"}.get(status["status"], "Running")
html(
    f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:18px;">'
    f'<span class="ds-row-meta">Analysis #{qid}</span>{badge(label, kind)}</div>'
)

tab_overview, tab_code, tab_lineage, tab_verif = st.tabs(
    ["Overview", "Code & Logs", "Data Lineage", "Verifications"]
)

entries = trail.get("entries", [])
plan = trail.get("plan", [])

# ---------------------------------------------------------------- Overview --
with tab_overview:
    html("<div style='height:12px'></div>")
    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        with st.container(key="card_plan_ov"):
            html('<div class="ds-section-title">Analysis Plan</div>')
            html("<div style='height:10px'></div>")
            for i, step in enumerate(plan, start=1):
                html(
                    f'<div class="ds-step"><div class="ds-step-num ds-step-done">{i}</div>'
                    f'<div><div class="ds-step-title">{esc(step["description"])}</div>'
                    f'<div class="ds-step-status">{esc(step.get("goal","")[:90])}</div></div></div>'
                )
    with right:
        with st.container(key="card_elems"):
            html('<div class="ds-section-title">Dashboard Elements</div>')
            html(
                '<div class="ds-row-meta" style="margin:6px 0 12px 0;">'
                "Every KPI, chart, and the narrative — each traced to its source.</div>"
            )
            for e in entries:
                icon = {"kpi": "◆", "chart": "▤", "narrative": "¶"}.get(e["element_type"], "•")
                html(
                    f'<div class="ds-check"><span class="ds-check-label">'
                    f'{icon}&nbsp;&nbsp;{esc(e["element_label"])}</span>'
                    f'<span class="ds-row-meta">{e["element_type"]}</span></div>'
                )

# ------------------------------------------------------------ Code & Logs --
with tab_code:
    html("<div style='height:12px'></div>")
    coded = [e for e in entries if e.get("code")]

    if not coded:
        st.info("No executed code recorded for this analysis yet.")
    else:
        labels = [e["element_label"] for e in coded]
        c_steps, c_code, c_verif = st.columns([1, 1.9, 1], gap="medium")

        with c_steps:
            with st.container(key="card_steps"):
                html('<div class="ds-section-title">Steps</div>')
                html("<div style='height:8px'></div>")
                pick = st.radio("Step", range(len(coded)), format_func=lambda i: labels[i],
                                label_visibility="collapsed", key="au_step_pick")

        sel = coded[pick]
        with c_code:
            with st.container(key="card_code"):
                html(
                    '<div style="display:flex;align-items:center;justify-content:space-between;">'
                    '<div class="ds-section-title">Executed Code</div>'
                    '<span class="ds-row-meta">Python</span></div>'
                )
                st.code(sel["code"], language="python")
                if sel.get("result"):
                    html('<div class="ds-section-title" style="margin-top:8px;">'
                                "Result Output</div>")
                    st.json(sel["result"])
                if sel.get("stdout"):
                    with st.expander("Raw stdout"):
                        st.text(sel["stdout"][:4000])

        with c_verif:
            with st.container(key="card_vcheck"):
                html('<div class="ds-section-title">Verification</div>')
                html("<div style='height:10px'></div>")
                v = sel.get("critic_verdict")
                if v:
                    ok = v == "verified"
                    html(
                        f'<div class="ds-card" style="background:var(--bg-inset);'
                        f'padding:14px 16px;">'
                        f'{badge("Critic Agent Review", "verified" if ok else "warn")}'
                        f'<div class="ds-row-meta" style="margin-top:9px;line-height:1.6;">'
                        f'{esc((sel.get("critic_summary") or "")[:300])}</div></div>'
                    )
                    html("<div style='height:12px'></div>")
                    ok_bg, ok_border, ok_color = (
                        ("var(--accent-bg)", "var(--accent-border)", "var(--accent)") if ok
                        else ("var(--warn-bg)", "var(--warn-border)", "var(--warn)")
                    )
                    html(
                        f'<div style="text-align:center;padding:16px;border-radius:var(--radius-md);'
                        f'background:{ok_bg};border:1px solid {ok_border};'
                        f'color:{ok_color};font-weight:700;letter-spacing:0.04em;">'
                        f'{"✓ VERIFIED" if ok else "! NEEDS REVIEW"}</div>'
                    )
                else:
                    html('<div class="ds-row-meta">No critic verdict recorded.</div>')

# ----------------------------------------------------------- Data Lineage --
with tab_lineage:
    html("<div style='height:12px'></div>")
    with st.container(key="card_lineage"):
        html('<div class="ds-section-title">Lineage</div>')
        html(
            '<div class="ds-row-meta" style="margin:6px 0 16px 0;">'
            "How each dashboard element traces back through code to the source dataset."
            "</div>"
        )
        for e in entries:
            src = "source CSV → sandbox → " + (
                "executed code" if e.get("code") else "synthesized from prior steps"
            )
            html(
                f'<div class="ds-row" style="border-top:1px solid var(--border);">'
                f'<div><div class="ds-row-title">{esc(e["element_label"])}</div>'
                f'<div class="ds-row-meta">{esc(src)}</div></div>'
                f'<div class="ds-row-spacer"></div>'
                f'<span class="ds-row-meta">{esc(e["element_type"])}</span></div>'
            )

# ---------------------------------------------------------- Verifications --
with tab_verif:
    html("<div style='height:12px'></div>")
    if not reviews:
        st.info("No critic reviews recorded yet.")
    for r in reviews:
        ok = r["verdict"] == "verified"
        with st.container(key=f"card_rev_{r['id']}"):
            html(
                f'<div style="display:flex;align-items:center;gap:11px;">'
                f'{badge(r["verdict"].title(), "verified" if ok else "warn")}'
                f'<span class="ds-row-meta">confidence {r["confidence"]:.2f}</span></div>'
                f'<div style="margin-top:11px;line-height:1.7;font-size:0.92rem;">'
                f'{esc(r["summary"])}</div>'
            )
            if r.get("issues"):
                html('<div class="ds-section-title" style="margin-top:14px;">'
                            "Issues flagged</div>")
                for issue in r["issues"]:
                    html(f'<div class="ds-row-meta">• {esc(issue)}</div>')
            for chk in r.get("checks", []):
                html('<div class="ds-section-title" style="margin-top:14px;">'
                            "Independent re-check</div>")
                html(
                    f'<div class="ds-row-meta" style="margin-bottom:8px;">'
                    f'{esc(chk.get("what_it_checks",""))} '
                    f'<span style="opacity:.7;">· model: '
                    f'{esc(chk.get("verifier_model","n/a"))}</span></div>'
                )
                with st.expander("Verification code & recomputed values"):
                    st.code(chk.get("code", ""), language="python")
                    html(f'<div class="ds-row-meta">Recomputed: '
                                f'{esc(chk.get("result"))}</div>')
        html("<div style='height:12px'></div>")
