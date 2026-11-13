"""What-if decision tools: break-even, risk simulation (Monte Carlo) and goal seek.

Rendering only. Every number shown here comes from the backend (`/api/scenarios/*`): this module sends the user's
settings, draws what comes back, and never calculates a result itself. Each section carries the backend's independent
verification beside its result, and says what the result does not establish.
"""
from __future__ import annotations

import itertools
import json

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, goal_seek_scenario, simulate_scenario
from style.theme import CHECK_META, badge, esc, html, plot, stat_card

DISTRIBUTIONS = {"fixed": "Fixed (no uncertainty)", "normal": "Normal", "uniform": "Uniform", "triangular": "Triangular"}
OPERATORS = [">=", "<=", ">", "<", "=="]
STATUS_BADGE = {"solved": ("verified", "Solved"), "multiple_solutions": ("warn", "Several solutions"),
                "no_solution": ("error", "No solution"), "no_feasible_solution": ("error", "No feasible solution"),
                "no_dependence": ("warn", "No effect"), "undefined": ("error", "Undefined"),
                "convergence_failure": ("error", "Solver failed")}


def num(x, digits: int = 2) -> str:
    return "—" if x is None else f"{x:,.{digits}f}"


def pct(x) -> str:
    return "—" if x is None else f"{x:.1f}%"


def section(title: str, caption: str = "") -> None:
    html(f'<div class="ds-section-title" style="margin-top:22px;">{esc(title)}</div>')
    if caption:
        st.caption(caption)


def verification_panel(ver: dict | None, title: str = "Independent check") -> None:
    """The verification block of a result: status, what was and was not checked, and each check."""
    if not ver:
        html(f'<div class="ds-row-meta">{badge("Not verified", "error")} No verification was returned for this result.</div>')
        return
    verified = ver["status"] == "verified"
    html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge("Verified" if verified else "Not verified", "verified" if verified else "error")}'
         f' {esc(title)}: {esc(ver.get("scope", ""))}</div>')
    if ver.get("error"):
        st.error(ver["error"])
    with st.expander(f"What was checked ({len(ver.get('checks', []))} checks)"):
        for c in ver.get("checks", []):
            kind, label = CHECK_META.get(c["status"], ("neutral", c["status"]))
            req = "" if c.get("required", True) else " · optional"
            html(f'<div class="ds-row-meta" style="padding:3px 0;">{badge(label, kind)} <b>{esc(c["check"].replace("_", " "))}</b>{req} — {esc(c["detail"])}</div>')
        for limit in ver.get("limitations", []):
            html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(limit)}</div>')


# ---------------------------------------------------------------- break-even --

def break_even_section(out: dict, model_key: str) -> None:
    section("Break-even", "Where profit crosses zero, from this model's own formulas.")
    be = out["break_even"]["baseline"]
    mine = out["break_even"]["scenarios"].get("your scenario")
    for label, item in [("Baseline", be)] + ([("Your scenario", mine)] if mine else []):
        html(f'<div class="ds-row-meta" style="padding:6px 0 2px;"><b>{esc(label)}</b> — {esc(item["explanation"])}</div>')
        cards = _break_even_cards(model_key, item)
        for col, card in zip(st.columns(len(cards)), cards):
            col.markdown(stat_card(*card), unsafe_allow_html=True)
    curve = be["curve"]
    fig = go.Figure()
    fig.add_scatter(x=curve["x"], y=curve["profit"], mode="lines", name="Baseline profit")
    if mine:
        fig.add_scatter(x=mine["curve"]["x"], y=mine["curve"]["profit"], mode="lines", name="Your scenario")
    fig.add_hline(y=0, line_dash="dot")
    if be.get("current_profit") is not None:
        fig.add_scatter(x=[curve["current_x"]], y=[be["current_profit"]], mode="markers", name="Where you are now",
                        marker={"size": 11})
    if curve.get("break_even_x") is not None:
        fig.add_scatter(x=[curve["break_even_x"]], y=[0], mode="markers", name="Break-even", marker={"size": 11, "symbol": "diamond"})
    fig.update_layout(title="Profit and the break-even point", xaxis_title=curve["x_label"], yaxis_title="Profit")
    plot(fig, height=320, showlegend=True)


def _break_even_cards(model_key: str, be: dict) -> list[tuple]:
    if model_key == "unit_economics":
        if be["status"] != "reachable":
            return [("Break-even", "None", ""), ("Contribution per unit", num(be["contribution_per_unit"]), "")]
        gap = be["distance_units"]
        return [("Break-even units", num(be["break_even_quantity"]), ""), ("Break-even revenue", num(be["break_even_revenue"]), ""),
                ("Units above break-even" if gap >= 0 else "Units still needed", num(abs(gap)), "", "up" if gap >= 0 else "down"),
                ("Safety margin", pct(be["safety_margin_pct"]), "", "up" if (be["safety_margin_pct"] or 0) >= 0 else "down")]
    return [("Profit per visitor", num(be["profit_per_visitor"], 4), ""),
            ("Break-even conversion", pct(None if be["break_even_conversion_rate"] is None else be["break_even_conversion_rate"] * 100), ""),
            ("Break-even order value", num(be["break_even_average_order_value"]), ""),
            ("Break-even cost per visitor", num(be["break_even_cost_per_visitor"], 4), "")]


# ---------------------------------------------------------------- simulation --

def _distribution_inputs(mk: str, name: str, label: str) -> dict | None:
    """One input's uncertainty widgets; returns its distribution spec or None when fixed."""
    cols = st.columns([2, 2, 4])
    cols[0].markdown(f"**{label}**")
    kind = cols[1].selectbox("Distribution", list(DISTRIBUTIONS), format_func=DISTRIBUTIONS.get,
                             key=f"wi_mc_dist_{mk}_{name}", label_visibility="collapsed")
    if kind == "normal":
        sd = cols[2].number_input("Standard deviation, % of the scenario value", 0.01, 500.0, 5.0, 0.5, key=f"wi_mc_sd_{mk}_{name}")
        return {"type": "normal", "sd_pct": sd}
    if kind == "uniform":
        a, b = cols[2].columns(2)
        return {"type": "uniform", "low_pct": a.number_input("Low, %", -100.0, 500.0, -10.0, 1.0, key=f"wi_mc_lo_{mk}_{name}"),
                "high_pct": b.number_input("High, %", -100.0, 500.0, 10.0, 1.0, key=f"wi_mc_hi_{mk}_{name}")}
    if kind == "triangular":
        a, b, c = cols[2].columns(3)
        return {"type": "triangular", "low_pct": a.number_input("Low, %", -100.0, 500.0, -10.0, 1.0, key=f"wi_mc_tlo_{mk}_{name}"),
                "mode_pct": b.number_input("Most likely, %", -100.0, 500.0, 0.0, 1.0, key=f"wi_mc_tmo_{mk}_{name}"),
                "high_pct": c.number_input("High, %", -100.0, 500.0, 10.0, 1.0, key=f"wi_mc_thi_{mk}_{name}")}
    return None


def simulation_section(mk: str, model: dict, baseline: dict, assumptions: dict, adjustments: dict) -> dict | None:
    """Risk simulation controls and results. Returns the options to save with the scenario, or None."""
    section("Risk simulation (Monte Carlo)",
            "How far could the result move if inputs are uncertain? Choose which inputs are uncertain and how. The ranges "
            "are your assumptions: nothing is guessed or estimated from the data, and inputs left fixed stay fixed.")
    uncertainty = {}
    for name, rule in model["inputs"].items():
        spec = _distribution_inputs(mk, name, rule["label"])
        if spec:
            uncertainty[name] = spec
    uncertain = list(uncertainty)

    correlation = None
    if len(uncertain) >= 2:
        with st.expander("Correlation between uncertain inputs (optional)"):
            st.caption("Without this the inputs are sampled independently. +1 means they move together, −1 opposite.")
            matrix = [[1.0 if i == j else 0.0 for j in range(len(uncertain))] for i in range(len(uncertain))]
            any_set = False
            for (i, a), (j, b) in itertools.combinations(enumerate(uncertain), 2):
                v = st.slider(f"{model['inputs'][a]['label']} ↔ {model['inputs'][b]['label']}", -1.0, 1.0, 0.0, 0.05, key=f"wi_mc_corr_{mk}_{a}_{b}")
                matrix[i][j] = matrix[j][i] = v
                any_set = any_set or v != 0.0
            if any_set:
                correlation = {"inputs": uncertain, "matrix": matrix}

    c1, c2, c3 = st.columns(3)
    iterations = c1.number_input("Simulated runs", 100, 200_000, 10_000, 1_000, key="wi_mc_iter")
    fix_seed = c2.checkbox("Repeatable (fixed random seed)", True, key="wi_mc_fixseed",
                           help="Same seed, same settings, same result. Untick to draw a fresh seed each run; it is always shown.")
    seed = c3.number_input("Seed", 0, 2 ** 32 - 1, 42, 1, key="wi_mc_seed", disabled=not fix_seed)
    t1, t2, t3, t4 = st.columns([2, 2, 2, 2])
    outputs = list(model["outputs"])
    target_output = t1.selectbox("Outcome to judge", outputs, index=outputs.index(model["primary_output"]),
                                 format_func=model["outputs"].get, key=f"wi_mc_out_{mk}")
    use_target = t2.checkbox("Set a target", False, key="wi_mc_target_on")
    target_value = t3.number_input("Target value", value=0.0, step=100.0, key="wi_mc_target", disabled=not use_target)
    direction = t4.selectbox("Reach it by being", ["at_least", "at_most"], format_func=lambda v: "at least" if v == "at_least" else "at most",
                             key="wi_mc_dir", disabled=not use_target)

    body = {"model": mk, "baseline": baseline, "assumptions": assumptions, "adjustments": adjustments, "uncertainty": uncertainty,
            "correlation": correlation, "iterations": int(iterations), "seed": int(seed) if fix_seed else None,
            "target_output": target_output, "target_value": float(target_value) if use_target else None,
            "target_direction": direction}
    state_key = f"wi_mc_result_{mk}"
    if st.button("Run simulation", type="primary", key="wi_mc_run", disabled=not uncertainty,
                 help=None if uncertainty else "Choose a distribution for at least one input first."):
        try:
            with st.spinner(f"Running {int(iterations):,} simulations and checking them…"):
                st.session_state[state_key] = {"body": body, "result": simulate_scenario(body)}
        except ApiError as e:
            st.session_state.pop(state_key, None)
            st.error(f"The simulation could not run: {e}")
    if not uncertainty:
        st.caption("Pick Normal, Uniform or Triangular for at least one input to enable the simulation.")
    stored = st.session_state.get(state_key)
    if not stored:
        return None
    fresh = json.dumps(stored["body"], sort_keys=True) == json.dumps(body, sort_keys=True)
    if not fresh:
        st.warning("Your inputs or settings changed since this simulation ran. Run it again to update the results.")
    _simulation_results(stored["result"])
    if fresh:
        keep = {k: v for k, v in stored["body"].items() if k not in ("model", "baseline", "assumptions")}
        keep["seed"] = stored["result"]["config"]["seed"]
        return keep if st.checkbox("Save this simulation with the scenario", True, key="wi_mc_save") else None
    return None


def _simulation_results(r: dict) -> None:
    t = r["target"]
    label = t["label"]
    cfg = r["config"]
    html(f'<div class="ds-row-meta" style="padding:6px 0;">{esc(r["disclaimer"])} {int(cfg["iterations"]):,} runs · seed {cfg["seed"]} · '
         f'{esc(", ".join(r["uncertain_inputs"]))} uncertain · engine {esc(r["engine_version"])}</div>')
    cards = [(f"Low {label.lower()} (5th percentile)", num(t["low"]), ""), (f"Expected {label.lower()}", num(t["expected"]), ""),
             (f"High {label.lower()} (95th percentile)", num(t["high"]), ""), ("Median", num(t["median"]), "")]
    for col, card in zip(st.columns(4), cards):
        col.markdown(stat_card(*card), unsafe_allow_html=True)
    p = t["probabilities"]
    prob_cards = [(f"Chance {label.lower()} is above zero", pct(_p(p["positive"])), ""),
                  (f"Chance of a negative {label.lower()}", pct(_p(p["negative"])), "")]
    if p["exceeds_baseline"]:
        prob_cards.append((f"Chance of beating today's {num(t['baseline_value'])}", pct(_p(p["exceeds_baseline"])), ""))
    if p["reaches_target"]:
        rt = p["reaches_target"]
        prob_cards.append((f"Chance {label.lower()} is {'at least' if rt['direction'] == 'at_least' else 'at most'} {num(rt['target'])}",
                           pct(_p(rt)), ""))
    for col, card in zip(st.columns(len(prob_cards)), prob_cards):
        col.markdown(stat_card(*card), unsafe_allow_html=True)
    st.caption("Low and high are the 5th and 95th percentiles of the simulated outcomes. They describe this simulation; "
               "they are not guaranteed bounds. Probabilities carry a sampling error of about "
               f"±{(p['positive']['standard_error'] or 0) * 100:.2f} percentage points at the middle of the range.")

    h = t["histogram"]
    if h["counts"]:
        centers = [(a + b) / 2 for a, b in zip(h["edges"], h["edges"][1:])]
        widths = [b - a for a, b in zip(h["edges"], h["edges"][1:])]
        fig = go.Figure(go.Bar(x=centers, y=h["counts"], width=widths, name="Simulated runs"))
        for value, name in [(t["low"], "5th percentile"), (t["expected"], "Expected"), (t["high"], "95th percentile"),
                            (t["baseline_value"], "Today"), (0, "Zero")]:
            if value is not None and h["edges"][0] <= value <= h["edges"][-1]:
                fig.add_vline(x=value, line_dash="dot", annotation_text=name, annotation_position="top")
        if p["reaches_target"] and h["edges"][0] <= p["reaches_target"]["target"] <= h["edges"][-1]:
            fig.add_vline(x=p["reaches_target"]["target"], line_dash="dash", annotation_text="Target", annotation_position="bottom")
        fig.update_layout(title=f"Distribution of simulated {label.lower()}", xaxis_title=label, yaxis_title="Runs")
        plot(fig, height=320)

    rows = {r["output_labels"][k]: v for k, v in r["outputs"].items()}
    table = {name: {"Mean": s["mean"], "Std dev": s["std"], "5th %": s["p5"], "25th %": s["p25"], "Median": s["median"],
                    "75th %": s["p75"], "95th %": s["p95"], "Min": s["min"], "Max": s["max"], "Valid runs": s["n_valid"]}
             for name, s in rows.items()}
    import pandas as pd
    st.dataframe(pd.DataFrame(table).T.style.format("{:,.2f}", na_rep="—"), use_container_width=True)
    for d in r["disclosures"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(d)}</div>')
    verification_panel(r.get("verification"), "Independent check of the simulation")


def _p(item: dict | None) -> float | None:
    return None if not item or item["probability"] is None else item["probability"] * 100


# ----------------------------------------------------------------- goal seek --

def goal_seek_section(mk: str, model: dict, baseline: dict, assumptions: dict, adjustments: dict) -> None:
    section("Goal seek", "Work backwards: which value of one input reaches the result you want? Solved on the same "
                         "formulas as the scenarios above, then checked independently.")
    outputs, inputs = list(model["outputs"]), list(model["inputs"])
    c1, c2, c3 = st.columns(3)
    target_output = c1.selectbox("I want this result", outputs, index=outputs.index(model["primary_output"]),
                                 format_func=model["outputs"].get, key=f"wi_gs_out_{mk}")
    target_value = c2.number_input("to be", value=0.0, step=100.0, key=f"wi_gs_value_{mk}", format="%.4f")
    free = [i for i in inputs if i not in adjustments]
    solve_for = c3.selectbox("by changing", free, format_func=lambda k: model["inputs"][k]["label"], key=f"wi_gs_var_{mk}")
    if len(free) < len(inputs):
        st.caption("Inputs you adjusted above stay at their adjusted values; only the input chosen here is solved for.")
    r1, r2, r3 = st.columns(3)
    limit = r1.checkbox("Limit the search range", False, key=f"wi_gs_limit_{mk}")
    lower = r2.number_input("Lowest value to consider", value=0.0, key=f"wi_gs_lo_{mk}", disabled=not limit, format="%.4f")
    upper = r3.number_input("Highest value to consider", value=float(baseline.get(solve_for, 0) * 5 or 1), key=f"wi_gs_hi_{mk}",
                            disabled=not limit, format="%.4f")
    constraints = []
    with st.expander("Constraints the answer must also satisfy (optional)"):
        metrics = {**{k: v["label"] for k, v in model["inputs"].items()}, **model["outputs"]}
        for n in (1, 2):
            a, b, c = st.columns([3, 1, 2])
            metric = a.selectbox(f"Constraint {n}", [None] + list(metrics), format_func=lambda k: "—" if k is None else metrics[k], key=f"wi_gs_cm_{mk}_{n}")
            op = b.selectbox("Operator", OPERATORS, key=f"wi_gs_co_{mk}_{n}", label_visibility="collapsed")
            value = c.number_input("Value", value=0.0, key=f"wi_gs_cv_{mk}_{n}", label_visibility="collapsed", format="%.4f")
            if metric:
                constraints.append({"metric": metric, "op": op, "value": float(value)})
    body = {"model": mk, "baseline": baseline, "assumptions": assumptions, "adjustments": adjustments,
            "target_output": target_output, "target_value": float(target_value), "solve_for": solve_for, "constraints": constraints}
    if limit:
        body.update(lower=float(lower), upper=float(upper))
    state_key = f"wi_gs_result_{mk}"
    if st.button("Solve", type="primary", key="wi_gs_run", disabled=not solve_for):
        try:
            st.session_state[state_key] = {"body": body, "result": goal_seek_scenario(body)}
        except ApiError as e:
            st.session_state.pop(state_key, None)
            st.error(f"Goal seek could not run: {e}")
    stored = st.session_state.get(state_key)
    if not stored:
        return
    if json.dumps(stored["body"], sort_keys=True) != json.dumps(body, sort_keys=True):
        st.warning("Your inputs or settings changed since this was solved. Solve again to update it.")
    _goal_seek_results(stored["result"])


def _goal_seek_results(r: dict) -> None:
    kind, label = STATUS_BADGE.get(r["status"], ("neutral", r["status"]))
    html(f'<div class="ds-row-meta" style="padding:6px 0;">{badge(label, kind)} {esc(r["message"])}</div>')
    sols = r["solutions"]
    if sols:
        import pandas as pd
        solve = r["solve_for"]
        frame = pd.DataFrame([{solve["label"]: s["value"], "Change vs today": s["change_vs_baseline"]["pct"],
                               r["target"]["label"]: s["achieved"], "Meets constraints": "yes" if s["feasible"] else "no",
                               "Recommended": "◀" if i == r["recommended_index"] else ""} for i, s in enumerate(sols)])
        st.dataframe(frame.style.format({solve["label"]: "{:,.4f}", "Change vs today": "{:+.2f}%", r["target"]["label"]: "{:,.4f}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)
    curve = r["curve"]
    top = max([s["value"] for s in sols] + [r["solve_for"]["baseline_value"]]) * 2 or r["search_range"]["upper"]
    xs = [(x, y) for x, y in zip(curve["x"], curve["y"]) if y is not None and x <= min(top, r["search_range"]["upper"])]
    if len(xs) > 2:
        fig = go.Figure()
        fig.add_scatter(x=[a for a, _ in xs], y=[b for _, b in xs], mode="lines", name=r["target"]["label"])
        fig.add_hline(y=r["target"]["value"], line_dash="dash", annotation_text="Target")
        fig.add_scatter(x=[r["solve_for"]["baseline_value"]], y=[_at(xs, r["solve_for"]["baseline_value"])], mode="markers",
                        name="Today", marker={"size": 10})
        if sols:
            fig.add_scatter(x=[s["value"] for s in sols], y=[s["achieved"] for s in sols], mode="markers", name="Solution",
                            marker={"size": 12, "symbol": "diamond"})
        fig.update_layout(title=f"{r['target']['label']} as {r['solve_for']['label'].lower()} changes",
                          xaxis_title=r["solve_for"]["label"], yaxis_title=r["target"]["label"])
        plot(fig, height=300, showlegend=True)
    if r["achievable_range"] and r["status"] in ("no_solution", "no_feasible_solution"):
        a = r["achievable_range"]
        st.caption(f"Over the search range ({r['search_range']['lower']:,.6g} to {r['search_range']['upper']:,.6g}) "
                   f"{r['target']['label'].lower()} can only reach {a['min']:,.6g} to {a['max']:,.6g}.")
    for d in r["disclosures"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(d)}</div>')
    verification_panel(r.get("verification"), "Independent check of the solution")


def _at(points: list[tuple], x: float) -> float | None:
    return min(points, key=lambda p: abs(p[0] - x))[1] if points else None
