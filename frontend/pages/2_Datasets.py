"""Datasets — upload CSVs, browse sources, inspect the auto-generated profile."""
from __future__ import annotations

import re

import streamlit as st

from api_client import (ApiError, ask_question, create_note, delete_note, generate_insights, get_correlations, get_dataset,
                        get_insights, list_datasets, list_notes, replace_dataset_data, upload_dataset)
from style.theme import badge, esc, html, page_header, page_setup, render_sidebar, require_active_team

page_setup("Datasets")
render_sidebar("datasets")
require_active_team("Datasets")

head_l, head_r = st.columns([2.4, 1])
with head_l:
    page_header("Datasets", "Manage and explore your data sources.")
with head_r:
    html("<div style='height:10px'></div>")
    upload_open = st.button("Upload data", type="primary", key="ds_upload_btn")

from style.theme import page_intro
page_intro("Your data lives here. Upload a CSV or Excel file once and Silt checks its quality, then you can ask questions about it on any other page.",
           ["Press Upload data and choose a file", "Press Profile this dataset and wait a moment", "Check the table below: rows, columns and data quality", "Press Analyze this to ask your first question"],
           tip="Quality is the share of cells that are filled in. A low score means many blanks, which can weaken answers.")

if upload_open:
    st.session_state["show_uploader"] = not st.session_state.get("show_uploader", False)

def _notes_editor(chosen: dict) -> None:
    NOTE_KINDS = {
        "knowledge": ("What the data means", "Column definitions, units and quirks the file does not show.",
                      "e.g. “amount” is in cents. “status = 9” means test account, exclude it."),
        "lesson": ("Mistakes to avoid", "Things Silt should not repeat. Answers the checker rejects are added here automatically.",
                   "e.g. Revenue must exclude refunds, which appear as negative amounts."),
    }
    tabs = st.tabs([v[0] for v in NOTE_KINDS.values()])
    for tab, (kind, (_label, blurb, placeholder)) in zip(tabs, NOTE_KINDS.items()):
        with tab:
            st.caption(blurb)
            try:
                notes = list_notes(chosen["id"], kind)
            except ApiError as e:
                notes = []
                st.error(f"Could not load notes: {e}")

            form_key = f"note_form_{kind}_{chosen['id']}"
            with st.form(form_key, clear_on_submit=True):
                text = st.text_area("New note", placeholder=placeholder, max_chars=1000,
                                    label_visibility="collapsed", key=f"note_text_{kind}_{chosen['id']}")
                if st.form_submit_button("Add note", type="primary"):
                    if not text.strip():
                        st.warning("Write something first.")
                    else:
                        try:
                            create_note(chosen["id"], kind, text)
                            st.rerun()
                        except ApiError as e:
                            st.error(f"Could not save note: {e}")

            if not notes:
                html('<div class="ds-row-meta" style="padding:6px 0;">Nothing here yet.</div>')
            for n in notes:
                left, right = st.columns([12, 1])
                with left:
                    tag = badge("Critic", "warn") if n["source"] == "critic" else ""
                    html(f'<div style="font-size:0.88rem;line-height:1.55;padding:6px 0;">'
                         f'{tag} {esc(n["text"])}</div>')
                with right:
                    if st.button("✕", key=f"del_note_{n['id']}", help="Delete this note"):
                        try:
                            delete_note(chosen["id"], n["id"])
                            st.rerun()
                        except ApiError as e:
                            st.error(f"Could not delete: {e}")


if st.session_state.get("show_uploader"):
    with st.container(key="card_upload"):
        html('<div class="ds-section-title">Upload a CSV</div>')
        html("<div style='height:10px'></div>")
        f = st.file_uploader("CSV", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if f is not None and st.button("Profile this dataset", type="primary", key="ds_do_upload"):
            with st.spinner("Uploading & profiling…"):
                try:
                    res = upload_dataset(f.name, f.getvalue())
                    st.session_state["active_dataset_id"] = res["id"]
                    try:  # starter questions + data-quality warnings; never blocks the upload
                        generate_insights(res["id"])
                    except ApiError:
                        pass
                    st.session_state["show_uploader"] = False
                    if res.get("already_existed"):
                        st.session_state["flash"] = (f"**{res['filename']}** is already in your datasets, so nothing "
                                                     "was added. It is selected for you.")
                        st.rerun()
                    st.success(f"Profiled **{res['filename']}** — "
                               f"{res['row_count']:,} rows, {res['col_count']} columns.")
                    st.rerun()
                except ApiError as e:
                    st.error(f"Upload failed: {e}")
    html("<div style='height:14px'></div>")

try:
    datasets = list_datasets()
except ApiError as e:
    datasets = []
    st.error(f"Backend unreachable: {e}")

FILE_ICON = ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.7"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" '
             'stroke-linejoin="round"/><path d="M14 3v5h5" stroke-linejoin="round"/></svg>')

with st.container(key="flat_table"):
    if not datasets:
        html(
            '<div style="padding:34px 22px;text-align:center;">'
            '<div class="ds-row-title" style="margin-bottom:5px;">No datasets yet</div>'
            '<div class="ds-row-meta">Nothing to analyse yet. Click <b>Upload data</b> above and choose a CSV or Excel file. Your file stays in your workspace.</div>'
            "</div>"
        )
    else:
        html(
            """
            <div style="display:flex;padding:13px 22px;border-bottom:1px solid var(--border);
                        color:var(--text-secondary);font-size:0.79rem;">
              <div style="flex:2.6;">Name</div>
              <div style="flex:0.8;">Rows</div>
              <div style="flex:0.8;">Columns</div>
              <div style="flex:0.9;">Source</div>
              <div style="flex:1.1;">Last Updated</div>
              <div style="flex:1.3;">Quality</div>
            </div>
            """
        )
        PAGE_SIZE = 8
        pages = max(1, -(-len(datasets) // PAGE_SIZE))
        page_no = min(max(int(st.session_state.get("ds_page", 1)), 1), pages)
        for d in datasets[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]:
            profile = d.get("profile", {})
            cols = profile.get("columns", [])
            total_cells = max(d["row_count"] * max(d["col_count"], 1), 1)
            missing = sum(c.get("missing_count", 0) for c in cols)
            quality = round(max(0.0, 100.0 - (missing / total_cells * 100)), 1)
            # the uploaded file's type: Excel workbooks are stored as CSV but were uploaded as Excel
            source = "Excel" if re.search(r"\.xlsx?\b", d["filename"].lower()) else "CSV"
            html(
                f"""
                <div class="ds-row" style="padding:14px 22px;">
                  <div style="flex:2.6;display:flex;align-items:center;gap:11px;">
                    <div class="ds-row-icon" style="width:29px;height:29px;">{FILE_ICON}</div>
                    <div class="ds-row-title" style="font-weight:500;">{esc(d['filename'])}</div>
                  </div>
                  <div style="flex:0.8;" class="ds-row-meta">{d['row_count']:,}</div>
                  <div style="flex:0.8;" class="ds-row-meta">{d['col_count']}</div>
                  <div style="flex:0.9;" class="ds-row-meta">{source}</div>
                  <div style="flex:1.1;" class="ds-row-meta">#{d['id']} · v{d.get('version', 1)}</div>
                  <div style="flex:1.3;">
                    <div class="ds-quality">
                      <span class="ds-row-meta" style="min-width:38px;">{quality}%</span>
                      <span class="ds-quality-track">
                        <span class="ds-quality-fill" style="width:{quality}%;"></span>
                      </span>
                    </div>
                  </div>
                </div>
                """
            )

if datasets and len(datasets) > PAGE_SIZE:
    pg_l, pg_m, pg_r = st.columns([1, 2, 1])
    with pg_l:
        if st.button("‹  Previous", key="ds_prev", disabled=page_no <= 1):
            st.session_state["ds_page"] = page_no - 1
            st.rerun()
    with pg_m:
        html(f'<div class="ds-row-meta" style="text-align:center;padding-top:8px;">Page {page_no} of {pages} · '
             f'{len(datasets)} datasets</div>')
    with pg_r:
        if st.button("Next  ›", key="ds_next", disabled=page_no >= pages):
            st.session_state["ds_page"] = page_no + 1
            st.rerun()

if datasets:
    html("<div style='height:20px'></div>")
    labels = [f"{d['filename']}  ·  #{d['id']}" for d in datasets]
    active = st.session_state.get("active_dataset_id")
    default = next((i for i, d in enumerate(datasets) if d["id"] == active), 0)

    sel_l, sel_r = st.columns([3, 1])
    with sel_l:
        idx = st.selectbox("Inspect dataset", range(len(datasets)),
                           index=default, format_func=lambda i: labels[i])
    with sel_r:
        html("<div style='height:28px'></div>")
        if st.button("Analyze this  →", type="primary", key="ds_go_analyze"):
            st.session_state["active_dataset_id"] = datasets[idx]["id"]
            st.switch_page("pages/3_Analyses.py")

    chosen = datasets[idx]
    st.session_state["active_dataset_id"] = chosen["id"]

    with st.expander(f"Replace data  ·  currently v{chosen.get('version', 1)} "
                     f"({chosen.get('version_count', 1)} version(s))"):
        st.caption("Uploads a new version into this dataset. Existing dashboards keep the "
                   "version they were computed from; new and scheduled runs use the latest.")
        new_file = st.file_uploader("New CSV", type=["csv", "xlsx", "xls"], key=f"replace_{chosen['id']}",
                                    label_visibility="collapsed")
        if new_file is not None and st.button("Upload as new version", type="primary",
                                              key=f"do_replace_{chosen['id']}"):
            with st.spinner("Uploading & profiling…"):
                try:
                    res = replace_dataset_data(chosen["id"], new_file.name, new_file.getvalue())
                    st.success(f"Now on **v{res['version']}** — {res['row_count']:,} rows, "
                               f"{res['col_count']} columns.")
                    st.rerun()
                except ApiError as e:
                    st.error(f"Upload failed: {e}")

    try:
        full = get_dataset(chosen["id"])
    except ApiError as e:
        full = None
        st.error(f"Could not load profile: {e}")

    if full:
        profile = full["profile"]
        html("<div style='height:6px'></div>")
        with st.container(key="flat_profile"):
            html(
                f'<div class="ds-card-head"><div class="ds-section-title">'
                f'Profile — {esc(full["filename"])}</div></div>'
            )
            html(
                """
                <div style="display:flex;padding:11px 22px;border-bottom:1px solid var(--border);
                            color:var(--text-secondary);font-size:0.78rem;">
                  <div style="flex:1.6;">Column</div><div style="flex:0.9;">Type</div>
                  <div style="flex:1.1;">Missing</div><div style="flex:0.8;">Unique</div>
                  <div style="flex:2.4;">Detail</div>
                </div>
                """
            )
            for c in profile["columns"]:
                detail = ""
                if c["kind"] == "numeric" and c.get("stats"):
                    s = c["stats"]
                    if s.get("mean") is not None:
                        detail = (f"min {s['min']:.2f} · max {s['max']:.2f} · "
                                  f"mean {s['mean']:.2f}")
                elif c["kind"] == "categorical" and c.get("top_values"):
                    detail = ", ".join(f"{esc(t['value'])} ({t['count']})"
                                       for t in c["top_values"][:3])
                elif c["kind"] == "datetime" and c.get("stats"):
                    detail = f"{esc(c['stats'].get('min'))} → {esc(c['stats'].get('max'))}"
                html(
                    f"""
                    <div class="ds-row" style="padding:11px 22px;">
                      <div style="flex:1.6;" class="ds-row-title" >{esc(c['name'])}</div>
                      <div style="flex:0.9;" class="ds-row-meta">{esc(c['kind'])}</div>
                      <div style="flex:1.1;" class="ds-row-meta">
                        {c['missing_count']} ({c['missing_pct']}%)</div>
                      <div style="flex:0.8;" class="ds-row-meta">{c['unique_count']}</div>
                      <div style="flex:2.4;" class="ds-row-meta">{detail}</div>
                    </div>
                    """
                )

    # ------------------------------------------------------- correlations --
    html("<div style='height:20px'></div>")
    with st.container(key="card_correlations"):
        html('<div class="ds-section-title">Strongest relationships</div>')
        try:
            pairs = get_correlations(chosen["id"])["pairs"]
        except ApiError:
            pairs = []
        if not pairs:
            html('<div class="ds-row-meta" style="padding:6px 0;">No strong numeric relationships found.</div>')
        for p in pairs:
            html(f'<div style="padding:5px 0;font-size:0.86rem;"><b>{esc(p["a"])}</b> ↔ <b>{esc(p["b"])}</b> · '
                 f'{p["strength"]} {p["direction"]} (r = {p["pearson"]}, n = {p["n"]})</div>')

    # ------------------------------------------------------- auto-insights --
    html("<div style='height:20px'></div>")
    with st.container(key="card_insights"):
        head_i, btn_i = st.columns([4, 1])
        with head_i:
            html('<div class="ds-section-title">Suggested questions &amp; data checks</div>')
        try:
            ins = get_insights(chosen["id"])
        except ApiError as e:
            ins = {"generated": False, "questions": [], "warnings": []}
            st.error(f"Could not load insights: {e}")
        with btn_i:
            if st.button("Regenerate" if ins["generated"] else "Generate", key=f"ins_gen_{chosen['id']}"):
                with st.spinner("Looking at your data…"):
                    try:
                        generate_insights(chosen["id"])
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Could not generate: {e}")

        if not ins["generated"]:
            html('<div class="ds-row-meta" style="padding:6px 0;">Generate starter questions and a data-quality '
                 'check for this dataset.</div>')
        else:
            for w in ins["warnings"]:
                kind = {"high": "error", "warn": "warn"}.get(w["severity"], "neutral")
                col = f'<b>{esc(w["column"])}</b> · ' if w.get("column") else ""
                html(f'<div style="display:flex;gap:10px;align-items:center;padding:5px 0;font-size:0.86rem;">'
                     f'{badge(w["severity"].title(), kind)}<span>{col}{esc(w["message"])}</span></div>')
            if not ins["questions"]:
                html('<div class="ds-row-meta" style="padding:6px 0;">No question suggestions this time '
                     '(the model was unavailable). Try Regenerate.</div>')
            for i, item in enumerate(ins["questions"]):
                qc, bc = st.columns([6, 1])
                with qc:
                    html(f'<div style="padding:6px 0;"><div class="ds-row-title">{esc(item["question"])}</div>'
                         f'<div class="ds-row-meta">{esc(item.get("why", ""))}</div></div>')
                with bc:
                    if st.button("Run  →", key=f"ins_run_{chosen['id']}_{i}"):
                        try:
                            res_q = ask_question(chosen["id"], item["question"])
                            st.session_state["active_dataset_id"] = chosen["id"]
                            st.session_state["active_question_id"] = res_q["question_id"]
                            st.session_state["question_running"] = True
                            st.switch_page("pages/3_Analyses.py")
                        except ApiError as e:
                            st.error(f"Could not start analysis: {e}")

    # ------------------------------------------------ knowledge & lessons --
    html("<div style='height:20px'></div>")
    with st.container(key="card_knowledge"):
        html('<div class="ds-section-title">Teach Silt about this data <span class="ds-row-meta">(optional)</span></div>')
        html('<div class="ds-page-sub" style="margin:4px 0 6px 0;">Silt reads only your file, so it can miss what a column '
             'really means. Add a short note here and it will be used in every analysis of this dataset. '
             'Skip this if the column names already say it all.</div>')
        with st.expander("Add or review notes", expanded=False):
            _notes_editor(chosen)

