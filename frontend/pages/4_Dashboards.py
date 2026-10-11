"""Dashboards — verified KPI cards, agent-generated charts, and the narrative.

The Chart Studio bar above the dashboard lets the user swap any chart for one
of 40 chart types and restyle it (see chart_studio.py). Only the presentation
changes and is saved per team; the verified numbers are never recomputed.
"""
from __future__ import annotations

import streamlit as st

import chart_studio as cs
from api_client import (ApiError, ask_question, chat_dashboard, compare_questions, create_scheduled, create_share,
                        get_dashboard, get_evidence, get_manifest, get_status, list_questions, list_shares,
                        rerun_question, reset_chart_view, revoke_share, save_chart_view)
from style.theme import (
    badge,
    chart_card_css,
    require_active_team,
    esc,
    figure_from_json,
    html,
    evidence_badge,
    open_inspect,
    page_header,
    render_evidence,
    page_setup,
    plot,
    render_inspect_dialog_if_open,
    render_sidebar,
    render_verdict_banner,
    verdict_badge,
)

page_setup("Dashboards")
render_sidebar("dashboards")
require_active_team("Dashboards")

# Deep link from alert emails: /Dashboards?question=<id>
_linked = st.query_params.get("question")
if _linked and str(_linked).isdigit():
    st.session_state["active_question_id"] = int(_linked)
    st.query_params.clear()

qid = st.session_state.get("active_question_id")

# ---- picker when nothing is selected ----
try:
    questions = list_questions()
except ApiError as e:
    questions = []
    st.error(f"Backend unreachable: {e}")

ready = [q for q in questions if q.get("has_dashboard")]

if not qid:
    page_header("Dashboards", "Generated from verified analyses.")
    if not ready:
        st.info("No dashboards yet — run an analysis first.")
        if st.button("Go to Analyses  →", type="primary", key="db_go"):
            st.switch_page("pages/3_Analyses.py")
        st.stop()
    with st.container(key="flat_pick"):
        html('<div class="ds-card-head"><div class="ds-section-title">'
                    'All Dashboards</div></div>')
        for q in ready:
            c1, c2 = st.columns([5, 1])
            with c1:
                html(
                    f'<div class="ds-row" style="border-top:none;">'
                    f'<div><div class="ds-row-title">{esc(q["text"][:70])}</div>'
                    f'<div class="ds-row-meta">Updated {q["age"]}</div></div>'
                    f'<div class="ds-row-spacer"></div>{verdict_badge(q.get("verdict_state"))}</div>'
                )
            with c2:
                if st.button("Open", key=f"db_open_{q['id']}"):
                    st.session_state["active_question_id"] = q["id"]
                    st.rerun()
    st.stop()

# ---- selected dashboard ----
try:
    status = get_status(qid)
except ApiError as e:
    st.error(f"Could not fetch status: {e}")
    st.stop()

if status["status"] not in ("verified", "unverified"):
    page_header("Dashboards")
    st.warning(f"This analysis isn't finished yet (status: **{status['status']}**).")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Watch it run  →", type="primary", key="db_watch"):
            st.switch_page("pages/3_Analyses.py")
    with c2:
        if st.button("←  All dashboards", key="db_back_pending"):
            st.session_state.pop("active_question_id", None)
            st.rerun()
    st.stop()

try:
    dash = get_dashboard(qid)
except ApiError as e:
    st.error(f"Could not load dashboard: {e}")
    st.stop()

if st.button("←  All dashboards", key="db_back"):
    st.session_state.pop("active_question_id", None)
    st.rerun()
html("<div style='height:4px'></div>")

verdict_state = dash.get("verdict_state")
head_l, head_r = st.columns([3, 1])
with head_l:
    html(
        f'<div class="ds-page-title">{esc(status.get("question_text", "Analysis"))}</div>'
    )
    html(
        f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:24px;">'
        f'<span class="ds-row-meta">Analysis #{qid}</span>'
        f'{verdict_badge(verdict_state)}'
        + (f'<span class="ds-row-meta">Data v{dash["dataset_version"]}</span>' if dash.get("dataset_version") else "")
        + "</div>"
    )
with head_r:
    html("<div style='height:12px'></div>")
    if st.button("Audit Trail  →", key="db_audit"):
        st.switch_page("pages/6_Audit_Trail.py")
    with st.popover("Share", use_container_width=True):
        if verdict_state == "UNVERIFIED":
            st.caption("The Critic could not verify this analysis, so it can't be shared publicly.")
        else:
            st.caption("Anyone with the link can view a read-only copy. No login needed.")
            days = st.selectbox("Link expires after", [1, 7, 30, 90], index=1,
                                format_func=lambda d: f"{d} day{'s' if d != 1 else ''}", key="share_days")
            if st.button("Create link", type="primary", key="share_create"):
                try:
                    st.session_state["_new_share_url"] = create_share(qid, days)["url"]
                except ApiError as e:
                    st.error(f"Could not create link: {e}")
            if st.session_state.get("_new_share_url"):
                st.code(st.session_state["_new_share_url"], language=None)
            try:
                active_links = list_shares(qid)
            except ApiError:
                active_links = []
            for link in active_links:
                lc1, lc2 = st.columns([3, 1])
                with lc1:
                    st.caption(f"Expires {link['expires_at'][:10]}")
                with lc2:
                    if st.button("Revoke", key=f"share_rev_{link['id']}"):
                        try:
                            revoke_share(link["id"])
                            st.session_state.pop("_new_share_url", None)
                            st.rerun()
                        except ApiError as e:
                            st.error(str(e))
    with st.popover("Track this question", use_container_width=True):
        interval = st.radio("Re-run", ["daily", "weekly"], horizontal=True, key="track_interval")
        threshold = st.number_input("Alert when a KPI changes by more than (%)", min_value=0.0,
                                    value=10.0, step=1.0, key="track_threshold")
        if st.button("Start tracking", type="primary", key="track_go"):
            src = next((q for q in questions if q["id"] == qid), None)
            if not src:
                st.error("Could not find this question's dataset.")
            else:
                try:
                    create_scheduled(src["dataset_id"], status.get("question_text") or src["text"],
                                     interval, threshold)
                    st.success("Tracking started — see the Scheduled page.")
                except ApiError as e:
                    st.error(f"Could not schedule: {e}")

# The trust signal: first thing on the page, above every KPI.
render_verdict_banner(dash)

dashboard_id = dash["id"]

# ------------------------------------------- evidence + reproducibility --
_ev = dash.get("evidence_summary") or {}
with st.expander(
    f"Evidence and reproducibility — {_ev.get('verified', 0)} of {_ev.get('claims', 0)} claims fully verified"
    if _ev.get("claims") else "Evidence and reproducibility"
):
    ev_tab, repro_tab, compare_tab = st.tabs(["Evidence", "Run record", "Rerun and compare"])
    with ev_tab:
        try:
            evidence = get_evidence(qid)
        except ApiError as e:
            evidence = {"records": []}
            st.caption(f"Evidence unavailable: {e}")
        if not evidence["records"]:
            html('<div class="ds-row-meta">This dashboard was created before evidence records existed.</div>')
        for rec in evidence["records"]:
            with st.container(key=f"ev_{rec['claim_id']}"):
                html(f'<div style="display:flex;gap:10px;align-items:center;margin-top:10px;">'
                     f'{evidence_badge(rec["status"])}<span class="ds-row-title">{esc(rec["claim_text"][:110])}</span></div>')
                with st.popover("Checks", use_container_width=False):
                    render_evidence(rec)
    with repro_tab:
        try:
            man = get_manifest(qid)
        except ApiError as e:
            man = None
            st.caption(f"Run record unavailable: {e}")
        if man:
            ds_info, ver = man["dataset"], man["versions"]
            html('<div class="ds-row-meta" style="line-height:1.9;">'
                 f'<b>Dataset:</b> {esc(str(ds_info.get("filename")))} · version {esc(str(ds_info.get("version_number")))} '
                 f'· sha256 {esc((ds_info.get("content_sha256") or "not recorded")[:16])}…<br>'
                 f'<b>Route:</b> {esc(str(man["question"].get("route")))} &nbsp;·&nbsp; '
                 f'<b>Application:</b> {esc(str(ver.get("app")))} ({esc(ver.get("git_sha") or "no git sha")}) &nbsp;·&nbsp; '
                 f'<b>Prompts:</b> {esc(str(ver.get("prompts")))}<br>'
                 f'<b>Models:</b> {esc(", ".join(man.get("models") or {}) or "none (deterministic)")}<br>'
                 f'<b>Environment:</b> Python {esc(man["environment"].get("python", ""))}, pandas '
                 f'{esc(man["environment"].get("pandas", ""))}, sandbox {esc(man["environment"].get("sandbox_backend", ""))}<br>'
                 f'<b>Scripts run:</b> {len(man.get("executions") or [])} &nbsp;·&nbsp; '
                 f'<b>Recorded:</b> {esc(str(man.get("recorded_at") or "reconstructed from history"))}</div>')
            st.caption(man.get("reproducibility_note", ""))
            st.download_button("Download run record (JSON)", __import__("json").dumps(man, indent=2, default=str),
                               file_name=f"run_{qid}.json", mime="application/json", key="dl_manifest")
    with compare_tab:
        html('<div class="ds-row-meta">Run the same question again on exactly the same data version, or compare '
             "this run with another to see whether a difference comes from the data or from the execution.</div>")
        if st.button("Rerun on the same data version", key="rerun_same"):
            try:
                new = rerun_question(qid)
                st.session_state["active_question_id"] = new["question_id"]
                st.switch_page("pages/3_Analyses.py")
            except ApiError as e:
                st.error(str(e))
        others = [q for q in questions if q["id"] != qid and q.get("has_dashboard")]
        if others:
            pick = st.selectbox("Compare with", others, format_func=lambda q: f"#{q['id']} · {q['text'][:70]} · {q['age']}",
                                key="cmp_pick")
            if st.button("Compare", key="cmp_go"):
                try:
                    cmp = compare_questions(pick["id"], qid)
                    changed = [k for k, v in cmp["changed"].items() if v] or ["nothing"]
                    st.markdown(f"**{cmp['explanation']}**")
                    html(f'<div class="ds-row-meta">Recorded inputs that differ: {esc(", ".join(changed))}</div>')
                    if cmp["kpi_changes"]:
                        st.dataframe([{"KPI": c["label"], f"Run #{pick['id']}": c["a"], f"Run #{qid}": c["b"],
                                       "Change %": None if c["pct"] is None else round(c["pct"], 2)}
                                      for c in cmp["kpi_changes"]], use_container_width=True, hide_index=True)
                    st.caption(cmp["note"])
                except ApiError as e:
                    st.error(str(e))

# ---------------------------------------------------------- chart studio --
# Session keys, all scoped to this dashboard + chart:
#   cs_view_<d>_<chart>     {"chart_type": str | None, "style": {...}} as shown
#   cs_<d>_<chart>_<field>  the studio bar's widgets for that chart
# Widgets write through on_change callbacks (which run before the script),
# so a change is saved and visible on the same rerun.

_STYLE_FIELDS = [("palette", "palette"), ("color", "color"), ("effect", "effect"),
                 ("animation", "anim"), ("opacity", "opacity"), ("corner_radius", "radius"),
                 ("line_width", "lw"), ("marker_size", "ms"), ("labels", "labels"),
                 ("legend", "legend"), ("sort", "sort")]


def _view_key(chart_key: str) -> str:
    return f"cs_view_{dashboard_id}_{chart_key}"


def _widget_prefix(chart_key: str) -> str:
    return f"cs_{dashboard_id}_{chart_key}_"


def _commit_view(chart_key: str, view: dict) -> None:
    st.session_state[_view_key(chart_key)] = view
    try:
        if view["chart_type"] is None and view["style"] == cs.DEFAULT_STYLE:
            reset_chart_view(dashboard_id, chart_key)
        else:
            save_chart_view(dashboard_id, chart_key, view["chart_type"], view["style"])
    except ApiError as e:
        st.session_state["cs_error"] = f"Couldn't save this chart's look: {e}"


def _on_type_pick(chart_key: str, widget_key: str) -> None:
    picked = st.session_state.get(widget_key)
    if picked in cs.CHART_TYPES:  # None = the active pill was clicked again: keep it
        view = dict(st.session_state[_view_key(chart_key)])
        view["chart_type"] = picked
        _commit_view(chart_key, view)


def _on_style_change(chart_key: str, changed: str | None = None) -> None:
    w = _widget_prefix(chart_key)
    if changed == "color":  # picking a color means "use my color"
        st.session_state[w + "palette"] = cs.CUSTOM_PALETTE
    view = dict(st.session_state[_view_key(chart_key)])
    raw = dict(view["style"])
    for field, suffix in _STYLE_FIELDS:
        value = st.session_state.get(w + suffix)
        if value is not None:
            raw[field] = value
    view["style"] = cs.normalize_style(raw)
    if st.session_state.get(w + "effect") is None:  # segmented control was deselected
        st.session_state[w + "effect"] = view["style"]["effect"]
    _commit_view(chart_key, view)


def _on_reset(chart_key: str) -> None:
    try:
        reset_chart_view(dashboard_id, chart_key)
    except ApiError as e:
        st.session_state["cs_error"] = f"Couldn't reset this chart: {e}"
        return
    st.session_state.pop(_view_key(chart_key), None)
    prefix = _widget_prefix(chart_key)
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[k]


def _on_customize(chart_key: str) -> None:
    st.session_state[f"cs_target_{dashboard_id}"] = chart_key


def _studio_info(entry: dict) -> dict | None:
    """What the Inspect panel needs to explain a Chart Studio chart."""
    if not entry["build"]:
        return None
    t, style = entry["shown_type"], entry["view"]["style"]
    try:
        calc, caption = cs.calculation_table(entry["data"], t, style)
    except Exception:  # noqa: BLE001 - the explanation table is optional
        calc, caption = None, ""
    plotted = entry["data"].df.rename(columns={
        "series": "Series", "x": entry["data"].x_label or "x", "y": entry["data"].y_label or "y"})
    return {"label": cs.CHART_TYPES[t]["label"], "code": entry["build"].code,
            "formula": cs.formula_for(t), "calc": calc, "calc_caption": caption, "plotted": plotted}


charts = dash.get("charts", [])
saved_views = dash.get("view_overrides") or {}
entries: list[dict] = []
for i, chart in enumerate(charts):
    chart_key = chart.get("element_id") or f"idx{i}"
    vk = _view_key(chart_key)
    if vk not in st.session_state:
        saved = saved_views.get(chart_key) or {}
        chart_type = saved.get("chart_type")
        st.session_state[vk] = {
            "chart_type": chart_type if chart_type in cs.CHART_TYPES else None,
            "style": cs.normalize_style(saved.get("style")),
        }
    view = st.session_state[vk]
    try:
        data = cs.extract_chart_data(chart["plotly_json"])
    except Exception:  # noqa: BLE001 - an unreadable figure just can't be re-charted
        data = None
    detected = cs.detect_studio_type(data) if data is not None else None
    # Style-only changes re-draw the original type through the studio too.
    shown_type = view["chart_type"] or (detected if view["style"] != cs.DEFAULT_STYLE else None)
    build, build_error = None, None
    if shown_type and data is not None:
        try:
            build = cs.build_chart(data, shown_type, view["style"], chart["title"])
        except ValueError as e:
            build_error = str(e)
    entries.append({"key": chart_key, "chart": chart, "index": i, "view": view, "data": data,
                    "detected": detected, "shown_type": shown_type if build else None,
                    "build": build, "error": build_error})

if entries:
    target_key = f"cs_target_{dashboard_id}"
    keys = [e["key"] for e in entries]
    if st.session_state.get(target_key) not in keys:
        st.session_state[target_key] = keys[0]
    by_key = {e["key"]: e for e in entries}

    with st.container(key="card_studio"):
        head_l, head_b1, head_b2 = st.columns([4, 1.1, 1.1], vertical_alignment="center")
        with head_l:
            html('<div class="ds-section-title">Chart Studio</div>'
                 '<div class="ds-row-meta" style="margin-top:3px;">Pick a chart, swap it for any of '
                 f'{len(cs.CHART_TYPES)} chart types and restyle it. The verified numbers never change; '
                 'the look is saved for your team.</div>')
        target = by_key[st.session_state[target_key]]
        ck, w = target["key"], _widget_prefix(target["key"])
        view = target["view"]
        with head_b1:
            if st.button("🔍  Code & data", key="cs_inspect", use_container_width=True,
                         disabled=not target["chart"].get("element_id")):
                open_inspect(dashboard_id, target["chart"]["element_id"], _studio_info(target))
                st.rerun()
        with head_b2:
            st.button("↺  Reset chart", key="cs_reset", use_container_width=True,
                      on_click=_on_reset, args=(ck,),
                      disabled=view["chart_type"] is None and view["style"] == cs.DEFAULT_STYLE)

        if st.session_state.get("cs_error"):
            st.error(st.session_state.pop("cs_error"))

        html("<div style='height:8px'></div>")
        pick_l, pick_r = st.columns([1.5, 3.5], gap="medium")
        with pick_l:
            html('<div class="ds-studio-label">Chart</div>')
            st.selectbox("Chart", keys, key=target_key, label_visibility="collapsed",
                         format_func=lambda k: by_key[k]["chart"]["title"])
        current = view["chart_type"] or target["detected"]
        with pick_r:
            html(f'<div class="ds-studio-label">Chart type · {len(cs.CHART_TYPES)} options</div>')
            st.session_state.setdefault(
                w + "group", cs.CHART_TYPES[current]["group"] if current else cs.GROUPS[0])
            group = st.segmented_control("Chart family", cs.GROUPS, key=w + "group",
                                         label_visibility="collapsed") or cs.GROUPS[0]

        if target["data"] is None or target["data"].df.empty:
            html('<div class="ds-row-meta">This chart has no plotted values that can be re-charted.</div>')
        else:
            pills_key = f"{w}type_{group}"
            in_group = cs.types_in_group(group)
            # keep the pills in sync with the chart as shown (the pick itself was
            # already saved by _on_type_pick before this run started)
            st.session_state[pills_key] = current if current in in_group else None
            st.pills("Chart type", in_group, key=pills_key, label_visibility="collapsed",
                     format_func=lambda t: f":material/{cs.CHART_TYPES[t]['icon']}: {cs.CHART_TYPES[t]['label']}",
                     on_change=_on_type_pick, args=(ck, pills_key))
            original = cs.CHART_TYPES[target["detected"]]["label"] if target["detected"] else target["data"].source_type
            showing = cs.CHART_TYPES[current]["label"] if view["chart_type"] else f"{original} (as generated)"
            html(f'<div class="ds-row-meta" style="margin:2px 0 10px 0;">Showing: <b>{esc(showing)}</b>'
                 f' · originally {esc(original)}</div>')

            style = view["style"]
            for field, suffix in _STYLE_FIELDS:
                st.session_state.setdefault(w + suffix, style[field])
            s1, s2, s3, s4, s5 = st.columns([1.35, 0.75, 2.6, 1.3, 1.0], gap="small",
                                            vertical_alignment="bottom")
            with s1:
                html('<div class="ds-studio-label">Palette</div>')
                st.selectbox("Palette", cs.PALETTE_NAMES, key=w + "palette", label_visibility="collapsed",
                             on_change=_on_style_change, args=(ck,))
            with s2:
                html('<div class="ds-studio-label">Color</div>')
                st.color_picker("Color", key=w + "color", label_visibility="collapsed",
                                on_change=_on_style_change, args=(ck, "color"))
            with s3:
                html('<div class="ds-studio-label">Effect</div>')
                st.segmented_control("Effect", list(cs.EFFECTS), key=w + "effect",
                                     format_func=cs.EFFECTS.get, label_visibility="collapsed",
                                     on_change=_on_style_change, args=(ck,))
            with s4:
                html('<div class="ds-studio-label">Animation</div>')
                st.selectbox("Animation", list(cs.ANIMATIONS), key=w + "anim", format_func=cs.ANIMATIONS.get,
                             label_visibility="collapsed", on_change=_on_style_change, args=(ck,))
            with s5:
                with st.popover("Fine-tune", use_container_width=True):
                    st.slider("Opacity", 0.2, 1.0, step=0.05, key=w + "opacity",
                              on_change=_on_style_change, args=(ck,))
                    st.slider("Bar corner radius", 0, 20, key=w + "radius",
                              on_change=_on_style_change, args=(ck,))
                    st.slider("Line width", 0.5, 8.0, step=0.5, key=w + "lw",
                              on_change=_on_style_change, args=(ck,))
                    st.slider("Marker size", 2, 24, key=w + "ms",
                              on_change=_on_style_change, args=(ck,))
                    st.selectbox("Sort values", list(cs.SORTS), key=w + "sort", format_func=cs.SORTS.get,
                                 on_change=_on_style_change, args=(ck,))
                    st.toggle("Data labels", key=w + "labels", on_change=_on_style_change, args=(ck,))
                    st.toggle("Legend", key=w + "legend", on_change=_on_style_change, args=(ck,))
            if target["error"]:
                st.warning(target["error"])
            elif view["style"] != cs.DEFAULT_STYLE and not target["build"]:
                html('<div class="ds-row-meta" style="margin-top:8px;">This chart\'s original type can\'t be '
                     'restyled directly; pick a chart type above to apply the style.</div>')
    html("<div style='height:22px'></div>")

kpis = dash.get("kpis", [])
if kpis:
    cols = st.columns(min(4, len(kpis)), gap="medium")
    for col, kpi in zip(cols, kpis[:4]):
        with col:
            element_id = kpi.get("element_id")
            flag = "⚠️ " if kpi.get("flagged") else ""
            mark = {"verified": "  \n:green[✓ verified]", "verified_with_caveats": "  \n:orange[✓ with caveats]",
                    "unverified": "  \n:red[not verified]"}.get(kpi.get("evidence_status"), "")
            label = f"{flag}{kpi.get('label', '')}  \n**{kpi.get('value', '')}**{mark}"
            with st.container(key=f"kpi_{element_id or kpi.get('label', '')}"):
                if st.button(label, key=f"kpibtn_{element_id or kpi.get('label', '')}",
                             use_container_width=True, disabled=not element_id):
                    open_inspect(dashboard_id, element_id)
                    st.rerun()
    html("<div style='height:22px'></div>")

if entries:
    selected_key = st.session_state.get(f"cs_target_{dashboard_id}")
    for i in range(0, len(entries), 2):
        pair = entries[i:i + 2]
        cols = st.columns(len(pair), gap="medium")
        for col, entry in zip(cols, pair):
            chart = entry["chart"]
            with col:
                element_id = chart.get("element_id")
                container_key = f"chartcard_{element_id}" if element_id else f"card_ch{i}_{chart['step_index']}"
                css = chart_card_css(container_key, entry["view"]["style"] if entry["build"] else cs.DEFAULT_STYLE,
                                     selected=entry["key"] == selected_key and len(entries) > 1)
                if css:
                    html(css)
                with st.container(key=container_key):
                    t_l, t_r = st.columns([5, 1.3], vertical_alignment="center")
                    with t_l:
                        if chart.get("flagged") and element_id:
                            f_l, f_r = st.columns([5, 1])
                            with f_l:
                                html(f'<div class="ds-section-title">{esc(chart["title"])}</div>')
                            with f_r:
                                if st.button("⚠️", key=f"flag_{element_id}", help="Flagged by the Critic — see why"):
                                    open_inspect(dashboard_id, element_id)
                                    st.rerun()
                        else:
                            html(f'<div class="ds-section-title">{esc(chart["title"])}</div>')
                    with t_r:
                        with st.container(key=f"cs_edit_{entry['key']}"):
                            st.button("✎ Customize", key=f"cs_edit_btn_{entry['key']}",
                                      on_click=_on_customize, args=(entry["key"],))
                    html("<div style='height:4px'></div>")
                    try:
                        if entry["build"]:
                            fig = entry["build"].fig
                            event = plot(fig, height=300, showlegend=entry["view"]["style"]["legend"],
                                         on_select_key=f"chart_{element_id}" if element_id else None,
                                         restyle_traces=False)
                        else:
                            fig = figure_from_json(chart["plotly_json"])
                            event = plot(fig, height=300, showlegend=True,
                                         on_select_key=f"chart_{element_id}" if element_id else None)
                        if element_id and event and event.selection and event.selection.get("points"):
                            open_inspect(dashboard_id, element_id, _studio_info(entry))
                            st.rerun()
                    except Exception as e:  # noqa: BLE001 - render one bad chart, not the page
                        html(f'<div class="ds-row-meta">Could not render: {esc(e)}</div>')
        html("<div style='height:8px'></div>")

render_inspect_dialog_if_open()

if dash.get("narrative"):
    with st.container(key="card_narr"):
        n_col, nf_col = st.columns([6, 1])
        with n_col:
            html('<div class="ds-section-title">Narrative Summary</div>')
        with nf_col:
            if dash.get("narrative_flagged"):
                if dash.get("narrative_element_id"):
                    if st.button("⚠️", key="flag_narrative", help="Flagged by the Critic — see why"):
                        open_inspect(dashboard_id, dash["narrative_element_id"])
                        st.rerun()
                else:
                    html('<span class="ds-flag" title="Flagged by the Critic">⚠️</span>')
        html(
            f'<div style="font-size:0.97rem;line-height:1.75;color:var(--text-primary);'
            f'margin-top:10px;">{esc(dash["narrative"])}</div>'
        )

if dash.get("verification_summary"):
    html("<div style='height:14px'></div>")
    with st.container(key="card_verif"):
        html(
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'{verdict_badge(verdict_state)}'
            f'<div class="ds-section-title">Verification</div></div>'
            f'<div class="ds-row-meta" style="margin-top:10px;line-height:1.7;">'
            f'{esc(dash["verification_summary"])}</div>'
        )


# ------------------------------------------------------- ask your dashboard --
html("<div style='height:22px'></div>")
chat_key = f"chat_{qid}"
history = st.session_state.setdefault(chat_key, [])
with st.container(key="card_chat"):
    html('<div class="ds-section-title">Ask this dashboard</div>')
    html('<div class="ds-page-sub" style="margin:4px 0 12px 0;">Follow-up questions are answered only from '
         'this analysis, and every answer is fact-checked by a second model. Anything it cannot answer, '
         'you can run as a new, fully verified analysis.</div>')
    for i, turn in enumerate(history):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("verified") is True:
                    html(f'<span class="ds-badge ds-badge-verified">Checked against this analysis</span>')
                elif turn.get("verified") is False:
                    claims = "".join(f"<li>{esc(c)}</li>" for c in turn.get("unsupported_claims") or [])
                    html(f'<span class="ds-badge ds-badge-warn">Not fully supported</span>'
                         f'<div class="ds-row-meta" style="margin-top:6px;">The checker could not confirm:'
                         f'<ul style="margin:4px 0 0 18px;">{claims}</ul></div>')
                elif turn.get("needs_new_analysis") is not True:
                    html('<span class="ds-badge ds-badge-neutral">Not independently checked</span>')
                if turn.get("sources"):
                    st.caption("Based on: " + " · ".join(turn["sources"]))
                if turn.get("needs_new_analysis") and turn.get("suggested_question"):
                    st.caption("This needs a new calculation.")
                    if st.button(f"Run as new analysis: {turn['suggested_question']}", key=f"chat_run_{qid}_{i}"):
                        src = next((q for q in questions if q["id"] == qid), None)
                        if src:
                            try:
                                res = ask_question(src["dataset_id"], turn["suggested_question"])
                                st.session_state["active_question_id"] = res["question_id"]
                                st.session_state["question_running"] = True
                                st.switch_page("pages/3_Analyses.py")
                            except ApiError as e:
                                st.error(f"Could not start analysis: {e}")

prompt = st.chat_input("Ask a follow-up about this analysis…", max_chars=600, key=f"chat_input_{qid}")
if prompt:
    sent = [{"role": t["role"], "content": t["content"]} for t in history][-8:]
    history.append({"role": "user", "content": prompt})
    with st.spinner("Reading the analysis and checking the answer…"):
        try:
            reply = chat_dashboard(qid, prompt, sent)
            history.append({"role": "assistant", "content": reply["answer"], **{
                k: reply.get(k) for k in ("verified", "unsupported_claims", "sources",
                                           "needs_new_analysis", "suggested_question")}})
        except ApiError as e:
            history.append({"role": "assistant", "content": f"Sorry, I couldn't answer that: {e}",
                            "verified": None, "needs_new_analysis": True})
    st.rerun()
