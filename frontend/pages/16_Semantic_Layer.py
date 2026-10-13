"""Semantic layer — the team's approved metric definitions, the words that
refer to them, and how they connect to datasets and columns."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import (ApiError, add_metric_version, approve_metric_version, create_metric, create_term,
                        deprecate_metric, get_metric, list_datasets, list_metrics, list_terms, semantic_graph,
                        semantic_search, validate_formula)
from style.theme import badge, esc, html, page_header, page_setup, plot, render_sidebar, require_active_team

page_setup("Semantic layer")
render_sidebar("semantic")
require_active_team("Semantic layer")
page_header("Business semantic layer", "One approved definition per metric. Analyses use these instead of guessing.")

try:
    metrics, datasets, terms = list_metrics(), list_datasets(), list_terms()
except ApiError as e:
    st.error(f"Backend unreachable: {e}")
    st.stop()

STATUS = {"approved": "verified", "draft": "warn", "deprecated": "neutral"}
ds_by_id = {d["id"]: d for d in datasets}
tab_metrics, tab_new, tab_terms, tab_graph, tab_search = st.tabs(
    [f"Metrics ({len(metrics)})", "New metric", f"Dimensions and entities ({sum(1 for t in terms if t['kind'] != 'alias')})",
     "Knowledge graph", "Search"])

with tab_metrics:
    if not metrics:
        html('<div class="ds-row-meta" style="padding:16px 0;">No metrics defined yet. A definition takes effect '
             "once a team owner or admin approves it.</div>")
    for m in metrics:
        with st.expander(f"{m['name']}  ·  {m['status']}  ·  v{m['current_version']}  ·  {m['formula']}"):
            try:
                detail = get_metric(m["id"])
            except ApiError as e:
                st.error(str(e))
                continue
            html(f'<div style="display:flex;gap:8px;align-items:center;">{badge(m["status"].title(), STATUS.get(m["status"], "neutral"))}'
                 f'<span class="ds-row-meta">{esc(m["description"] or "No description")}'
                 f'{" · unit " + esc(m["unit"]) if m["unit"] else ""}'
                 f'{" · dataset " + esc(ds_by_id[m["dataset_id"]]["filename"]) if m.get("dataset_id") in ds_by_id else " · any dataset with these columns"}'
                 f'{" · also called: " + esc(", ".join(m["aliases"])) if m["aliases"] else ""}</span></div>')
            st.dataframe(pd.DataFrame([{
                "Version": v["version"], "Formula": v["formula"], "Current": v["is_current"],
                "Approved": bool(v["approved_at"]), "Approved at": (v["approved_at"] or "")[:16].replace("T", " "),
                "Proposed at": (v["created_at"] or "")[:16].replace("T", " "), "Note": v["note"]} for v in detail["versions"]]),
                use_container_width=True, hide_index=True)
            pending = [v for v in detail["versions"] if not v["approved_at"]]
            c1, c2, c3 = st.columns(3)
            with c1:
                if pending:
                    pick = st.selectbox("Version to approve", [v["version"] for v in pending], key=f"sm_ap_pick_{m['id']}")
                    if st.button("Approve", type="primary", key=f"sm_ap_{m['id']}"):
                        try:
                            approve_metric_version(m["id"], pick)
                            st.rerun()
                        except ApiError as e:
                            st.error(str(e))
            with c2:
                with st.popover("Propose a new version", use_container_width=True):
                    nf = st.text_input("Formula", value=m["formula"], key=f"sm_nv_f_{m['id']}")
                    note = st.text_input("What changed and why", key=f"sm_nv_n_{m['id']}", max_chars=500)
                    if st.button("Propose", key=f"sm_nv_go_{m['id']}"):
                        try:
                            add_metric_version(m["id"], {"formula": nf, "note": note, "dataset_id": m.get("dataset_id")})
                            st.rerun()
                        except ApiError as e:
                            st.error(str(e))
            with c3:
                if m["status"] != "deprecated" and st.button("Deprecate", key=f"sm_dep_{m['id']}"):
                    try:
                        deprecate_metric(m["id"])
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))

with tab_new:
    st.caption("Formulas use aggregations — sum, avg, median, min, max, count, count_distinct — with + − × ÷. "
               "Example: (sum(revenue) - sum(cost)) / sum(revenue) * 100. Quote column names with spaces: sum(\"Unit Price\").")
    name = st.text_input("Name", key="sm_name", max_chars=80, placeholder="Gross margin %")
    scope = st.selectbox("Applies to", [None] + datasets, format_func=lambda d: "Any dataset that has the columns" if d is None else d["filename"], key="sm_scope")
    formula = st.text_input("Formula", key="sm_formula", placeholder="(sum(revenue) - sum(cost)) / sum(revenue) * 100")
    if formula:
        try:
            check = validate_formula(formula, scope["id"] if scope else None)
            if check["valid"]:
                st.success("Valid. Reads column(s): " + ", ".join(check["columns"] or ["none"])
                           + (". Segment contributions are exact for this metric." if check["additive"] else "."))
            else:
                st.error(check["error"])
        except ApiError as e:
            st.error(str(e))
    c1, c2 = st.columns(2)
    unit = c1.text_input("Unit (optional)", key="sm_unit", max_chars=30)
    aliases = c2.text_input("Also called (comma-separated)", key="sm_aliases")
    description = st.text_area("Description", key="sm_desc", max_chars=1000, height=80)
    if st.button("Save as draft", type="primary", key="sm_create", disabled=not (name.strip() and formula.strip())):
        try:
            out = create_metric({"name": name.strip(), "formula": formula.strip(), "unit": unit.strip(),
                                 "description": description.strip(), "dataset_id": scope["id"] if scope else None,
                                 "aliases": [a.strip() for a in aliases.split(",") if a.strip()]})
            if out.get("conflicts"):
                st.warning("Saved. These words already refer to another metric, so questions using them will ask which "
                           "one is meant: " + ", ".join(f"“{c['term']}” → {c['name']}" for c in out["conflicts"]))
            else:
                st.success("Saved as a draft. An owner or admin can approve it on the Metrics tab.")
        except ApiError as e:
            st.error(str(e))

with tab_terms:
    st.caption("Give a column the name people use for it, so “by territory” resolves to the region column.")
    rows = [t for t in terms if t["kind"] != "alias"]
    if rows:
        st.dataframe(pd.DataFrame([{"Term": t["term"], "Kind": t["kind"],
                                    "Dataset": ds_by_id.get(t["dataset_id"], {}).get("filename", "?"),
                                    "Column": t["column_name"], "Description": t["description"]} for t in rows]),
                     use_container_width=True, hide_index=True)
    if datasets:
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.selectbox("Kind", ["dimension", "entity"], key="st_kind")
        tds = c2.selectbox("Dataset", datasets, format_func=lambda d: d["filename"], key="st_ds")
        col = c3.selectbox("Column", [c["name"] for c in tds["profile"].get("columns", [])], key="st_col")
        term = c4.text_input("Term", key="st_term", max_chars=80)
        if st.button("Add", key="st_add", disabled=not term.strip()):
            try:
                create_term({"kind": kind, "term": term.strip(), "dataset_id": tds["id"], "column_name": col})
                st.rerun()
            except ApiError as e:
                st.error(str(e))

with tab_graph:
    try:
        g = semantic_graph()
    except ApiError as e:
        g = {"nodes": [], "edges": []}
        st.error(str(e))
    if not g["nodes"]:
        html('<div class="ds-row-meta" style="padding:16px 0;">The graph fills in as metrics and terms are defined.</div>')
    else:
        order = ["metric", "term", "column", "dataset"]
        by_type = {t: [n for n in g["nodes"] if n["type"] == t] for t in order}
        pos = {}
        for xi, t in enumerate(order):
            for yi, n in enumerate(by_type[t]):
                pos[n["id"]] = (xi, -(yi - (len(by_type[t]) - 1) / 2))
        fig = go.Figure()
        for e in g["edges"]:
            (x0, y0), (x1, y1) = pos[e["src"]], pos[e["dst"]]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text", text=e["relation"],
                                     line={"width": 1, "dash": "dot" if e["implied"] else "solid", "color": "#9ca3af"},
                                     showlegend=False))
        colors = {"metric": "#4f46e5", "term": "#0ea5a4", "column": "#f59e0b", "dataset": "#6b7280"}
        for t in order:
            if by_type[t]:
                fig.add_trace(go.Scatter(x=[pos[n["id"]][0] for n in by_type[t]], y=[pos[n["id"]][1] for n in by_type[t]],
                                         mode="markers+text", text=[n["label"][:22] for n in by_type[t]],
                                         textposition="bottom center", name=t.title(),
                                         marker={"size": 16, "color": colors[t]}))
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, title="Metrics, terms, columns and datasets")
        plot(fig, height=max(320, 60 * max(len(v) for v in by_type.values()) + 120), showlegend=True, restyle_traces=False)
        st.caption("Dotted lines are implied by the definitions (a metric reads a column); solid lines were added by hand.")
        st.dataframe(pd.DataFrame([{"From": e["src"], "Relation": e["relation"], "To": e["dst"],
                                    "Implied": e["implied"]} for e in g["edges"]]), use_container_width=True, hide_index=True)

with tab_search:
    query = st.text_input("Search definitions, terms and knowledge notes", key="sm_search")
    if query.strip():
        try:
            results = semantic_search(query.strip())["results"]
        except ApiError as e:
            results = []
            st.error(str(e))
        if not results:
            st.caption("Nothing matches.")
        for r in results:
            html(f'<div style="padding:7px 0;border-top:1px solid var(--border);"><div style="display:flex;gap:8px;align-items:center;">'
                 f'{badge(r["type"], "neutral")}<span class="ds-row-title">{esc(r["title"])}</span>'
                 f'<span class="ds-row-meta">{esc(r.get("status") or "")}</span></div>'
                 f'<div class="ds-row-meta">{esc(r.get("detail") or "")}</div></div>')
