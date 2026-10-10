"""Shared — anonymous, read-only view of one verified dashboard, opened from a
share link (/Shared?t=<token>). Deliberately no login, no sidebar and no
team context: it only calls the public endpoint, so a viewer can see exactly
what the link grants and nothing else."""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, get_public_dashboard
from style.theme import (
    esc, figure_from_json, html, page_setup, plot, render_verdict_banner, stat_card, verdict_badge,
)

page_setup("Shared dashboard", sidebar=False, public=True)

token = st.query_params.get("t")
if not token:
    st.info("This page opens from a share link. Ask the person who shared it for the full link.")
    st.stop()

try:
    dash = get_public_dashboard(str(token))
except ApiError as e:
    html(
        '<div style="padding:60px 0;text-align:center;">'
        '<div class="ds-page-title">This link isn\'t available</div>'
        '<div class="ds-page-sub">It may have expired or been revoked by its owner.</div></div>'
    )
    if "404" not in str(e):
        st.caption(f"Details: {e}")
    st.stop()

html(
    f'<div class="ds-page-title">{esc(dash["question_text"])}</div>'
    f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:22px;flex-wrap:wrap;">'
    f'{verdict_badge(dash.get("verdict_state"))}'
    f'<span class="ds-row-meta">Read-only snapshot · link expires {esc(dash["expires_at"][:10])}</span></div>'
)
render_verdict_banner(dash)

kpis = dash.get("kpis") or []
if kpis:
    cols = st.columns(min(4, len(kpis)), gap="medium")
    for col, kpi in zip(cols, kpis[:4]):
        with col:
            html(stat_card(f"{'⚠ ' if kpi.get('flagged') else ''}{kpi.get('label', '')}", str(kpi.get("value", ""))))
    html("<div style='height:18px'></div>")

charts = dash.get("charts") or []
for i in range(0, len(charts), 2):
    pair = charts[i:i + 2]
    cols = st.columns(len(pair), gap="medium")
    for col, chart in zip(cols, pair):
        with col:
            with st.container(key=f"card_shared_chart_{i}_{chart.get('title', '')[:20]}"):
                html(f'<div class="ds-section-title">{esc(chart.get("title", ""))}</div>')
                try:
                    plot(figure_from_json(chart["plotly_json"]), height=300, showlegend=True)
                except Exception as e:  # noqa: BLE001 - one bad chart shouldn't hide the rest
                    html(f'<div class="ds-row-meta">Could not render this chart: {esc(e)}</div>')
    html("<div style='height:8px'></div>")

if dash.get("narrative"):
    with st.container(key="card_shared_narr"):
        html('<div class="ds-section-title">Summary</div>')
        html(f'<div style="font-size:0.97rem;line-height:1.75;margin-top:10px;">{esc(dash["narrative"])}</div>')

html('<div class="h-footer"><div>Shared from <b>Silt</b></div>'
     '<div>Read-only view &middot; see the verification verdict above</div></div>')
