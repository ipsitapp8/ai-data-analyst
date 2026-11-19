"""What-if result view. Pure presentation: every number shown here was calculated by the backend (/api/whatif/run) and is only
formatted, arranged and drawn. Nothing is derived or invented in the browser.

The layout follows the shape of the result rather than a fixed template:

    header ........ domain · dataset · evidence status, with the page actions
    primary ....... one of: a single measure, a measure with its siblings, a simulated range (Monte Carlo), a solved value (goal seek)
    reading ....... the backend's own headline, then what kind of statement it is
    analysis ...... only the charts the result actually has data for (sensitivity, break-even, contributions)
    assumptions ... always visible, never behind a click
    detail ........ set-up, data, method, full result: tabs, because they are reference material
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, whatif_run, whatif_save
from style.theme import esc, html, plot

S = st.session_state

# one restrained chart palette: baseline is a quiet neutral, the simulated value carries the brand lavender
BASE_C, SIM_C, LOW_C, GRID_C, INK_C = "#6d688c", "#a78bfa", "#d49a8f", "rgba(236,233,251,.08)", "#b4b0cc"
DOMAIN_LABEL = {"business": "Business", "education": "Education", "healthcare": "Healthcare", "travel": "Travel",
                "government": "Government & public services", "custom": "Custom"}
KIND_CLASS = {"Exact calculation": "exact", "Statistical estimate": "estimate", "Forecast": "forecast", "Hypothetical assumption": "hypo",
              "Unsupported conclusion": "none"}
METHOD_LABEL = {"deterministic": "deterministic", "regression": "regression", "forecast": "trend forecast", "monte_carlo": "Monte Carlo",
                "sensitivity": "sensitivity", "break_even": "break-even", "goal_seek": "goal seek"}


def num(x, unit: str = "") -> str:
    if x is None:
        return "—"
    s = f"{x:,.0f}" if abs(x) >= 1000 or float(x).is_integer() else f"{x:,.2f}"
    return s


def signed(x) -> str:
    if x is None:
        return "—"
    return ("+" if x > 0 else "−" if x < 0 else "") + num(abs(x))


def pct(x) -> str:
    return "—" if x is None else ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x):,.1f}%"


def arrow(x) -> str:
    return "" if not x else "↑" if x > 0 else "↓"


def _unit(m: dict) -> str:
    u = (m.get("unit") or "").strip()
    return "" if u in ("", "percent") else u


def _is_rate(m: dict) -> bool:
    """A percentage or share. It must not share an axis with counts or amounts, whatever unit string the backend gave it."""
    return (m.get("unit") or "") in ("%", "percent") or str(m["key"]).endswith(("_pct", "pct")) or "%" in m["label"] or m["label"].lower().startswith("share of")


def _fig_base(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(height=height, showlegend=legend, margin=dict(l=4, r=24, t=6, b=4), bargap=.42,
                      font=dict(color=INK_C, size=12), legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12)))
    fig.update_xaxes(gridcolor=GRID_C, zeroline=False, automargin=True, tickfont=dict(size=11), ticks="outside", tickcolor="rgba(0,0,0,0)")
    fig.update_yaxes(automargin=True, showgrid=False, tickfont=dict(size=12.5), ticks="outside", tickcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------ header --

def _header(res: dict, scn: dict, sel: dict, ds_id) -> None:
    r, e = res["results"], res["explanation"]
    domain = DOMAIN_LABEL.get(scn.get("domain") or sel.get("domain"), (scn.get("domain") or "").title())
    ds = scn.get("dataset") or {}
    meta = [domain]
    if ds:
        meta.append(esc(ds.get("filename") or f"dataset {ds.get('dataset_id')}"))
        if r.get("rows_used"):
            meta.append(f"{r['rows_used']:,} records")
    else:
        meta.append("No dataset: entered values")
    kind = e["evidence_label"]
    status = (f'<span class="wr-status wr-{KIND_CLASS.get(kind, "none")}"><i></i>{esc(kind)}</span>'
              f'<span class="wr-sep">·</span><span>{esc(METHOD_LABEL.get(scn.get("method"), scn.get("method") or ""))}</span>')
    title = res.get("name") or sel.get("question") or scn.get("question") or "What-if result"
    left, right = st.columns([3, 2], vertical_alignment="bottom", gap="medium")
    with left:
        html(f'<div class="wr-eyebrow">{" <span class=wr-sep>·</span> ".join(m if i else esc(m) for i, m in enumerate(meta))}</div>'
             f'<h2 class="wr-title">{esc(title)}</h2><div class="wr-statusline">{status}</div>')
    with right:
        a, b, c, d = st.columns([1, 1, 1, 1])
        with a.popover("Save", use_container_width=True):
            name = st.text_input("Scenario name", value=res.get("name") or "", key="wxu_save_name", placeholder="Name this scenario", max_chars=120)
            if st.button("Save scenario", key="wxu_save", type="primary", disabled=not name.strip()):
                try:
                    whatif_save(S["wxu_scn"], ds_id, name.strip())
                    st.success("Saved.")
                except ApiError as ex:
                    st.error(str(ex))
        if b.button("Rerun", key="wxu_rerun", use_container_width=True):
            try:
                S["wxu_result"] = whatif_run(S["wxu_scn"], ds_id)
                st.rerun()
            except ApiError as ex:
                st.error(str(ex))
        c.download_button("Export", _export(res, scn), file_name="what-if-result.md", mime="text/markdown", key="wxu_export", use_container_width=True)
        if d.button("Discuss", key="wxu_chat", use_container_width=True):
            S["cp_handoff"] = (f"I ran the what-if \"{sel.get('question') or title}\" ({', '.join(f'{k}={v:g}' for k, v in S['wxu_scn'].get('inputs', {}).items())}). "
                               f"Result: {e['headline']} Please explain it and suggest follow-up scenarios.")
            st.switch_page("pages/12_Copilot.py")


def _export(res: dict, scn: dict) -> str:
    """The same explanation the page shows, as Markdown: what was run, what came out, and every stated assumption."""
    e, r = res["explanation"], res["results"]
    titles = {"changed": "What changed", "held_constant": "Kept constant", "data_used": "Data used", "method": "Calculation method", "assumptions": "Assumptions",
              "result": "Result in detail", "reliability": "Reliability", "cannot_conclude": "What this cannot tell you"}
    out = [f"# {res.get('name') or scn.get('question') or 'What-if result'}", "", f"**{e['evidence_label']}.** {e['headline']}", ""]
    for k, t in titles.items():
        items = e["sections"].get(k) or []
        if items:
            out += [f"## {t}", *[f"- {i}" for i in items], ""]
    if r.get("seed") is not None:
        out.append(f"Seed {r['seed']}: running again with this seed gives the same result.")
    return "\n".join(out)


# ----------------------------------------------------------------- primary --

def _primary_metric(r: dict) -> dict | None:
    return next((m for m in r["metrics"] if m["key"] == r.get("primary")), None) or (r["metrics"][0] if r["metrics"] else None)


def _hero_metric(m: dict, r: dict) -> str:
    """One measure, led by the simulated value; its baseline and difference sit beside it as plain figures."""
    u = _unit(m)
    unc = r.get("uncertainty") or {}
    rng = ""
    if unc.get("kind") in ("prediction_interval", "confidence_interval") and unc.get("low") is not None:
        rng = f'<div class="wr-range">{int(unc.get("level", .95) * 100)}% interval {num(unc["low"])} – {num(unc["high"])}</div>'
    changed = m.get("delta") not in (None, 0)
    figs = (f'<div class="wr-fig"><b>{num(m["baseline"])}</b><span>baseline</span></div>'
            f'<div class="wr-fig wr-{"up" if (m.get("delta") or 0) > 0 else "down" if changed else "flat"}"><b>{signed(m.get("delta"))}</b><span>change</span></div>'
            + (f'<div class="wr-fig wr-{"up" if (m.get("pct_change") or 0) > 0 else "down" if changed else "flat"}"><b>{arrow(m.get("pct_change"))} {pct(m.get("pct_change"))}</b><span>relative</span></div>'
               if m.get("pct_change") is not None else ""))
    return (f'<div class="wr-hero"><div class="wr-hero-main"><div class="wr-k">{esc(m["label"])}</div>'
            f'<div class="wr-big">{num(m["scenario"])}{f"<small>{esc(u)}</small>" if u else ""}</div>{rng}</div>'
            f'<div class="wr-figs">{figs}</div></div>')


def _hero_simulation(sim: dict, r: dict, m: dict | None) -> str:
    names = {"exceeds_baseline": "chance above the starting value", "reaches_target": "chance of reaching the target"}
    probs = "".join(f'<div class="wr-fig"><b>{p["probability"] * 100:.0f}%</b><span>{names[k]}</span></div>'
                    for k, p in (sim.get("probabilities") or {}).items() if k in names and isinstance(p, dict) and p.get("probability") is not None)
    return (f'<div class="wr-hero"><div class="wr-hero-main"><div class="wr-k">{esc(sim["label"])} — median of {sim["iterations"]:,} simulated runs</div>'
            f'<div class="wr-big">{num(sim["median"])}</div>'
            f'<div class="wr-range">5th–95th percentile {num(sim["p5"])} – {num(sim["p95"])}'
            + (f' · starting value {num(m["baseline"])}' if m and m["key"] == sim.get("output") and m.get("baseline") is not None else "") + '</div></div>'
            f'<div class="wr-figs"><div class="wr-fig"><b>{num(sim["p5"])}</b><span>5th pct</span></div>'
            f'<div class="wr-fig"><b>{num(sim["mean"])}</b><span>mean</span></div>'
            f'<div class="wr-fig"><b>{num(sim["p95"])}</b><span>95th pct</span></div>{probs}</div></div>')


def _hero_goal(g: dict) -> str:
    sols = g.get("solutions") or []
    solved = g.get("solution") is not None
    more = "".join(f'<div class="wr-fig"><b>{num(s["value"])}</b><span>alternative</span></div>' for s in sols[1:4] if s.get("value") is not None)
    return (f'<div class="wr-hero"><div class="wr-hero-main"><div class="wr-k">{"Value needed for " + esc(str(g.get("solve_for", ""))) if solved else "Goal seek"}</div>'
            f'<div class="wr-big">{num(g["solution"]) if solved else "No solution"}</div>'
            f'<div class="wr-range">{esc(g.get("message") or "")}</div></div><div class="wr-figs">{more}</div></div>')


def _ledger(metrics: list[dict], primary: str | None) -> str:
    if len(metrics) < 2:
        return ""
    rows = "".join(
        f'<tr class="{"wr-prim" if m["key"] == primary else ""}"><th>{esc(m["label"])}{f" <em>{esc(_unit(m))}</em>" if _unit(m) else ""}</th>'
        f'<td>{num(m["baseline"])}</td><td>{num(m["scenario"])}</td><td class="wr-{"up" if (m.get("delta") or 0) > 0 else "down" if (m.get("delta") or 0) < 0 else "flat"}">{signed(m.get("delta"))}</td>'
        f'<td>{pct(m.get("pct_change"))}</td></tr>' for m in metrics)
    return ('<table class="wr-ledger"><thead><tr><th>Measure</th><td>Baseline</td><td>Simulated</td><td>Change</td><td>Relative</td></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _compare_chart(metrics: list[dict], primary: dict) -> go.Figure | None:
    """Baseline against simulated for the measures that share the headline measure's unit (so one axis is honest)."""
    same = [m for m in metrics if _unit(m) == _unit(primary) and _is_rate(m) == _is_rate(primary) and m["baseline"] is not None and m["scenario"] is not None][:6]
    if not same:
        return None
    if len(same) == 1:
        m = same[0]
        fig = go.Figure(go.Bar(y=["Simulated", "Baseline"], x=[m["scenario"], m["baseline"]], orientation="h", marker_color=[SIM_C, BASE_C],
                               text=[num(m["scenario"]), num(m["baseline"])], textposition="outside", cliponaxis=False, textfont=dict(color="#ece9fb", size=13)))
        fig.update_xaxes(rangemode="tozero")
        return _fig_base(fig, 170)
    names = [m["label"] for m in same][::-1]
    fig = go.Figure([go.Bar(name="Baseline", y=names, x=[m["baseline"] for m in same][::-1], orientation="h", marker_color=BASE_C),
                     go.Bar(name="Simulated", y=names, x=[m["scenario"] for m in same][::-1], orientation="h", marker_color=SIM_C)])
    fig.update_layout(barmode="group", bargap=.35)
    return _fig_base(fig, max(190, 62 * len(same) + 60), legend=True)


def _histogram(sim: dict) -> go.Figure | None:
    h = sim.get("histogram") or {}
    edges, counts = h.get("edges") or h.get("bin_edges"), h.get("counts")
    if not (edges and counts):
        return None
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    width = (edges[-1] - edges[0]) / len(counts)
    inside = [sim["p5"] <= x <= sim["p95"] for x in mids]
    fig = go.Figure(go.Bar(x=mids, y=counts, width=width * .92, marker_color=[SIM_C if i else BASE_C for i in inside], hoverinfo="skip"))
    for key, lab, dash in (("p5", "5th", "dot"), ("median", "median", "solid"), ("p95", "95th", "dot")):
        fig.add_vline(x=sim[key], line=dict(color="#ece9fb", width=1, dash=dash), annotation_text=lab, annotation_position="top",
                      annotation_font=dict(size=11, color=INK_C))
    fig.update_layout(bargap=0)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig = _fig_base(fig, 250)
    fig.update_layout(margin=dict(l=4, r=12, t=40, b=4), bargap=0)
    return fig


def _primary(res: dict) -> tuple[go.Figure | None, str]:
    """Returns (chart, caption) that sits beside the reading, and renders the hero. The shape of the result decides which."""
    r = res["results"]
    d = r["details"]
    m = _primary_metric(r)
    sim, goal = d.get("simulation"), d.get("goal_seek")
    if sim:
        html(_hero_simulation(sim, r, m))
        return _histogram(sim), f"Distribution of {sim['label'].lower()} across {sim['iterations']:,} runs. Shaded bars lie between the 5th and 95th percentiles."
    if goal and not r["metrics"]:
        html(_hero_goal(goal))
        return None, ""
    if m:
        html(_hero_metric(m, r))
        html(_ledger(r["metrics"], r.get("primary")))
        if not any(x.get("delta") for x in r["metrics"]):       # nothing was changed: a baseline-against-itself chart says nothing
            return None, ""
        return _compare_chart(r["metrics"], m), "Baseline against simulated, for measures in the same unit."
    return None, ""


# ----------------------------------------------------------------- analysis --

def _tornado(sens: dict, base: float) -> go.Figure:
    rows = sens["rows"][::-1]
    fig = go.Figure()
    fig.add_bar(y=[x["label"] for x in rows], x=[x["low"] - base for x in rows], orientation="h", name=f"Input −{sens.get('pct', 10):g}%", marker_color=LOW_C)
    fig.add_bar(y=[x["label"] for x in rows], x=[x["high"] - base for x in rows], orientation="h", name=f"Input +{sens.get('pct', 10):g}%", marker_color=SIM_C)
    fig.update_layout(barmode="overlay", bargap=.4)
    fig.add_vline(x=0, line=dict(color="rgba(236,233,251,.35)", width=1))
    return _fig_base(fig, max(200, 54 * len(rows) + 70), legend=True)


def _break_even_chart(cur: dict) -> go.Figure | None:
    curve = cur.get("curve")
    if not curve:
        return None
    fig = go.Figure(go.Scatter(x=curve["x"], y=curve["profit"], mode="lines", line=dict(color=SIM_C, width=2.2), hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color="rgba(236,233,251,.35)", width=1))
    if curve.get("break_even_x") is not None:
        fig.add_vline(x=curve["break_even_x"], line=dict(color="#d4a95a", width=1.2, dash="dot"), annotation_text=f"break-even {num(curve['break_even_x'])}",
                      annotation_position="top left", annotation_font=dict(size=11, color="#d4a95a"))
    if curve.get("current_x") is not None:
        fig.add_vline(x=curve["current_x"], line=dict(color="#ece9fb", width=1), annotation_text=f"now {num(curve['current_x'])}",
                      annotation_position="top right", annotation_font=dict(size=11, color=INK_C))
    fig.update_xaxes(title=dict(text=curve.get("x_label", ""), font=dict(size=11.5)))
    fig.update_yaxes(gridcolor=GRID_C)
    fig = _fig_base(fig, 260)
    fig.update_layout(margin=dict(l=4, r=16, t=40, b=4))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_C)
    return fig


def _analysis(res: dict) -> None:
    r = res["results"]
    d = r["details"]
    m = _primary_metric(r)
    panels = []                                   # (title, caption, figure)
    sens = d.get("sensitivity")
    if sens and sens.get("rows") and m:
        panels.append(("Sensitivity", f"How far {m['label'].lower()} moves when each input alone moves ±{sens.get('pct', 10):g}% from the baseline.",
                       _tornado(sens, m["baseline"] or 0)))
    be = d.get("break_even")
    if be:
        cur = (be.get("scenarios") or {}).get("scenario") or be.get("baseline") or {}
        fig = _break_even_chart(cur)
        if fig:
            panels.append(("Break-even", cur.get("explanation") or "", fig))
    con = d.get("contributions")
    if con and len(con.get("by_input") or {}) > 1:
        by = con["by_input"]
        labels = {x["input"]: x["label"] for x in (sens or {}).get("rows", [])}
        wf = go.Figure(go.Waterfall(x=[labels.get(k, k) for k in by] + ["Total change"], y=list(by.values()) + [sum(by.values())],
                                    measure=["relative"] * len(by) + ["total"], connector=dict(line=dict(color="rgba(236,233,251,.2)")),
                                    increasing=dict(marker=dict(color=SIM_C)), decreasing=dict(marker=dict(color=LOW_C)), totals=dict(marker=dict(color=BASE_C))))
        wf = _fig_base(wf, 260)
        panels.append(("What each change contributes", f"{con['method']}; contributions sum exactly to the change.", wf))
    if not panels:
        return
    html('<div class="wr-h">Further analysis</div>')
    for i in range(0, len(panels), 2):
        row = panels[i:i + 2]
        for col, (title, cap, fig) in zip(st.columns(len(row), gap="large"), row):
            with col:
                html(f'<div class="wr-sub">{esc(title)}</div>' + (f'<div class="wr-cap">{esc(cap)}</div>' if cap else ""))
                plot(fig, height=fig.layout.height, showlegend=bool(fig.layout.showlegend), restyle_traces=False)


# --------------------------------------------------------------- reading, assumptions, detail --

def _rows(items: list[tuple[str, list[str]]]) -> None:
    out = "".join(f'<div class="wr-row"><div class="wr-rk">{esc(k)}</div><div class="wr-rv">{"".join(f"<p>{esc(i)}</p>" for i in v)}</div></div>' for k, v in items if v)
    html(f'<div class="wr-rows">{out}</div>' if out else '<div class="wr-cap">Nothing to show for this part.</div>')


def _bullets(items: list[str]) -> str:
    return '<ul class="wr-list">' + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else '<div class="wr-cap">None stated.</div>'


def render(res: dict, scn: dict, ds_id, sel: dict) -> None:
    r, e = res["results"], res["explanation"]
    sec = e["sections"]
    _header(res, scn, sel, ds_id)
    html('<div class="wr-rule"></div>')
    if r["status"] != "ok":
        html(f'<div class="wr-warn">{esc(r.get("reason") or "No result.")}</div>')
    for w in e.get("warnings") or []:
        html(f'<div class="wr-warn">{esc(w)}</div>')
    chart, caption = _primary(res)
    if r["status"] == "ok":
        left, right = st.columns([1.25, 1], gap="large") if chart is not None else (None, st.container())
        if chart is not None:
            with left:
                html('<div class="wr-sub">Baseline and simulated</div>' if not r["details"].get("simulation") else '<div class="wr-sub">Range of outcomes</div>')
                plot(chart, height=chart.layout.height, showlegend=bool(chart.layout.showlegend), restyle_traces=False)
                html(f'<div class="wr-cap">{esc(caption)}</div>')
        with right:
            unc = sec.get("reliability") or []
            html('<div class="wr-reading"><div class="wr-sub">Reading the result</div>'
                 f'<p class="wr-lead">{esc(e["headline"])}</p>'
 + "".join(f"<p>{esc(u)}</p>" for u in unc)
                 + (f'<p class="wr-cap">{esc((sec.get("changed") or [""])[0])}</p>' if sec.get("changed") else "") + '</div>')
        _analysis(res)

    a, b = st.columns(2, gap="large")
    with a:
        html('<div class="wr-h">Assumptions</div>' + _bullets(sec.get("assumptions") or []))
    with b:
        html('<div class="wr-h">What this cannot tell you</div>' + _bullets(sec.get("cannot_conclude") or []))

    html('<div class="wr-h">Scenario record</div>')
    with st.container(key="wr_tabs"):
        t1, t2, t3, t4 = st.tabs(["Scenario setup", "Data used", "Method", "Result in detail"])
        with t1:
            _rows([("What changed", sec.get("changed") or []), ("Kept constant", sec.get("held_constant") or [])])
        with t2:
            _rows([("Data", sec.get("data_used") or [])])
        with t3:
            _rows([("Calculation", sec.get("method") or []), ("Reliability", sec.get("reliability") or [])])
            v = r.get("verification")
            if v:
                checks = [f'{"✓" if c.get("status") == "pass" else "•"} {c.get("detail") or c.get("check")}' for c in v.get("checks", [])]
                _rows([("Independent check", [f"{str(v.get('status', '')).title()}. {v.get('scope', '')}"]), ("Checks run", checks)])
            if r.get("seed") is not None:
                html(f'<div class="wr-cap">Seed {esc(r["seed"])}: running again with this seed gives the same result.</div>')
        with t4:
            _rows([("Result", sec.get("result") or [])])
