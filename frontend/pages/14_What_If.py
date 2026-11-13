"""What-if — scenario arithmetic on explicit inputs and assumptions. Outputs are
scenarios, never forecasts."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import (ApiError, compare_scenarios, delete_scenario, evaluate_scenarios, list_custom_models,
                        list_datasets, list_scenarios, save_scenario, scenario_baseline, scenario_models)
from style.theme import esc, html, page_header, page_setup, plot, render_sidebar, require_active_team
from what_if_data import elasticity_panel
from what_if_grid import grid_section
from what_if_models import builder_tab, model_choices
from what_if_segments import segments_tab
from what_if_time import MAPPING_FIELDS, baseline_caption, compare_tab, legend, period_picker, version_picker
from what_if_decision import (break_even_section, goal_seek_section, simulation_section, verification_panel)

page_setup("What-if")
render_sidebar("what_if")
require_active_team("What-if simulator")
page_header("What-if simulator", "Change prices, costs, volume or conversion and see the arithmetic. Scenarios, not forecasts.")

try:
    catalog = scenario_models()
    datasets = list_datasets()
    saved = list_scenarios()["scenarios"]
    custom_models = list_custom_models()["models"]
except ApiError as e:
    st.error(f"Backend unreachable: {e}")
    st.stop()

st.warning(catalog["disclaimer"])
models = model_choices(catalog["models"], custom_models)
build_tab, segments_tab_, periods_tab, models_tab, saved_tab = st.tabs(["Build a scenario", "Segments", "Compare periods",
                                                                         f"Custom models ({len(custom_models)})", f"Saved scenarios ({len(saved)})"])


def render_build():
    model_key = st.selectbox("Model", list(models), format_func=lambda k: models[k]["title"], key="wi_model")
    model = models[model_key]
    with st.expander("Formulas this model uses"):
        for f in model["formulas"]:
            st.markdown(f"- `{f}`")

    seed_key = f"wi_seed_{model_key}"
    with st.expander("Take the baseline from a dataset (optional)"):
        if not datasets:
            st.caption("No dataset uploaded.")
        else:
            ds = st.selectbox("Dataset", datasets, format_func=lambda d: d["filename"], key="wi_ds")
            numeric = [None] + [c["name"] for c in ds["profile"].get("columns", []) if c.get("kind") == "numeric"]
            mapping = {}
            if model.get("custom"):
                st.caption("This model's inputs are bound to approved metrics or formulas in its definition. Inputs with no "
                           "binding stay as you enter them.")
            else:
                fields = MAPPING_FIELDS[model_key]
                cols = st.columns(len(fields))
                for c, f in zip(cols, fields):
                    mapping[f] = c.selectbox(f.replace("_", " ").title(), numeric, key=f"wi_map_{model_key}_{f}",
                                             format_func=lambda v: v or "—")
            version = version_picker("wi", ds)
            date_column, period = period_picker("wi", ds, version)
            if st.button("Compute baseline from data", key="wi_base_go"):
                try:
                    base = scenario_baseline(ds["id"], model_key, {k: v for k, v in mapping.items() if v}, date_column=date_column,
                                             period=period, dataset_version=version)
                    st.session_state[seed_key] = base
                    st.session_state["wi_dataset_id"] = ds["id"]
                    st.session_state["wi_baseline_source"] = {"dataset_version": base["provenance"]["dataset_version"], "date_column": date_column,
                                                              "period": base["period"]["spec"], "mapping": {k: v for k, v in mapping.items() if v}}
                    for name, value in base["inputs"].items():
                        if value is not None:
                            st.session_state[f"wi_in_{model_key}_{name}"] = float(value)
                    st.rerun()
                except ApiError as e:
                    st.error(str(e))
    seed = st.session_state.get(seed_key)
    if seed:
        shown = {"dataset": lambda s: f"`{s['formula']}`", "formula": lambda s: f"`{s['formula']}` (your formula)",
                 "semantic_metric": lambda s: f"approved metric “{s['metric']}” v{s['metric_version']} `{s['formula']}`"}
        lines = [f"**{models[model_key]['inputs'][k]['label']}**: " + (shown[s["source"]](s) if s["source"] in shown else "not in the data — enter a value")
                 for k, s in seed["sources"].items()]
        baseline_caption(seed)
        st.caption("Where each input came from: " + " · ".join(lines))

    html('<div class="ds-section-title" style="margin-top:8px;">Baseline</div>')
    baseline, cols = {}, st.columns(len(model["inputs"]))
    for c, (name, rule) in zip(cols, model["inputs"].items()):
        baseline[name] = c.number_input(f"{rule['label']} ({rule['unit']})", min_value=float(rule.get("min", 0.0)),
                                        max_value=float(rule["max"]) if "max" in rule else None, value=0.0,
                                        key=f"wi_in_{model_key}_{name}", format="%.4f")
    assumptions = {}
    for name, rule in model["assumptions"].items():
        assumptions[name] = st.slider(rule["label"], float(rule["min"]), float(rule["max"]), float(rule["default"]), 0.01,
                                      help=rule["help"], key=f"wi_as_{model_key}_{name}")
    assumption_sources = {}
    if "price_elasticity" in model["assumptions"]:
        assumption_sources = elasticity_panel(model_key, datasets, f"wi_as_{model_key}_price_elasticity", assumptions["price_elasticity"])

    html('<div class="ds-section-title" style="margin-top:8px;">Your scenario</div>')
    adjustments, cols = {}, st.columns(len(model["inputs"]))
    for c, (name, rule) in zip(cols, model["inputs"].items()):
        pct = c.number_input(f"{rule['label']} change %", min_value=-100.0, max_value=1000.0, value=0.0, step=1.0,
                             key=f"wi_adj_{model_key}_{name}")
        if pct:
            adjustments[name] = {"type": "pct", "value": pct}
    c1, c2 = st.columns(2)
    spread = c1.number_input("Optimistic / pessimistic spread % (0 = leave out)", 0.0, 100.0, 10.0, 1.0, key="wi_spread")
    sens = c2.number_input("Sensitivity: move each input by ± %", 1.0, 100.0, 10.0, 1.0, key="wi_sens")

    payload = {"model": model_key, "baseline": baseline, "assumptions": assumptions, "assumption_sources": assumption_sources,
               "sensitivity_pct": sens,
               "scenarios": {"your scenario": adjustments} if adjustments else {},
               "spread_pct": spread if spread else None}
    try:
        out = evaluate_scenarios(payload)
    except ApiError as e:
        st.error(f"These inputs are not valid: {e}")
        return

    primary = out["primary_output"]
    names = {"baseline": out["baseline"]["outputs"], **{k: v["outputs"] for k, v in out["scenarios"].items()}}
    table = pd.DataFrame({k: {model["outputs"][o]: v.get(o) for o in model["outputs"]} for k, v in names.items()})
    html(f'<div class="ds-section-title" style="margin-top:14px;">Scenario comparison — {esc(model["outputs"][primary])} is the headline</div>')
    st.dataframe(table.style.format("{:,.2f}", na_rep="—"), use_container_width=True)
    fig = go.Figure(go.Bar(x=list(names), y=[v.get(primary) or 0 for v in names.values()]))
    fig.update_layout(title=f"{model['outputs'][primary]} by scenario (hypothetical)")
    plot(fig, height=280)

    left, right = st.columns(2)
    with left:
        rows = out["sensitivity"]["rows"]
        base_value = out["baseline"]["outputs"].get(primary) or 0
        tornado = go.Figure()
        tornado.add_bar(y=[r["label"] for r in rows][::-1], x=[r["low"] - base_value for r in rows][::-1], orientation="h", name=f"−{sens:g}%")
        tornado.add_bar(y=[r["label"] for r in rows][::-1], x=[r["high"] - base_value for r in rows][::-1], orientation="h", name=f"+{sens:g}%")
        tornado.update_layout(barmode="overlay", title=f"Sensitivity of {model['outputs'][primary].lower()} to each input")
        plot(tornado, height=300, showlegend=True, restyle_traces=False)
    with right:
        mine = out["scenarios"].get("your scenario")
        if mine and mine["contributions"]["by_input"]:
            contrib = mine["contributions"]["by_input"]
            wf = go.Figure(go.Waterfall(x=[model["inputs"][k]["label"] for k in contrib] + ["Total change"],
                                        y=list(contrib.values()) + [sum(contrib.values())],
                                        measure=["relative"] * len(contrib) + ["total"]))
            wf.update_layout(title="What each changed input contributes")
            plot(wf, height=300, restyle_traces=False)
            st.caption(mine["contributions"]["method"] + "; contributions sum exactly to the change.")
        else:
            html('<div class="ds-row-meta" style="padding:30px 0;">Change an input above to see what each change contributes.</div>')

    html('<div class="ds-section-title" style="margin-top:8px;">Assumptions the data does not support</div>')
    for a in out["unsupported_assumptions"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(a)}</div>')

    for key, notes in (out.get("constraint_notes") or {}).items():
        for n in notes:
            st.warning(f"{'Baseline' if key == 'baseline' else key}: {n['message']}")
    if out.get("break_even"):
        break_even_section(out, model_key)
    verification_panel(out.get("verification"), "Independent check of the scenario arithmetic")
    simulation_options = simulation_section(model_key, model, baseline, assumptions, adjustments)
    goal_seek_section(model_key, model, baseline, assumptions, adjustments)
    grid_section(model_key, model, baseline, assumptions, adjustments)

    s1, s2 = st.columns([3, 1])
    name = s1.text_input("Save as", key="wi_name", max_chars=120, label_visibility="collapsed", placeholder="Name this scenario to save it")
    if s2.button("Save", type="primary", key="wi_save", disabled=not name.strip()):
        try:
            save_scenario({**payload, "name": name.strip(), "dataset_id": st.session_state.get("wi_dataset_id") if seed else None,
                           **({"baseline_source": st.session_state["wi_baseline_source"]} if seed and st.session_state.get("wi_baseline_source") else {}),
                           **({"simulation": simulation_options} if simulation_options else {})})
            st.success("Saved.")
            st.rerun()
        except ApiError as e:
            st.error(str(e))

with build_tab:
    render_build()

with segments_tab_:
    segments_tab(models, datasets)

with periods_tab:
    compare_tab(models, datasets)

with models_tab:
    builder_tab(custom_models)

with saved_tab:
    if not saved:
        html('<div class="ds-row-meta" style="padding:16px 0;">Nothing saved yet.</div>')
    chosen = []
    for s in saved:
        c1, c2, c3 = st.columns([0.5, 5, 1])
        if c1.checkbox("Select", key=f"wi_pick_{s['id']}", label_visibility="collapsed"):
            chosen.append(s["id"])
        base = s.get("baseline_primary")
        c2.markdown(f"**{s['name']}** · {models.get(s['model'], {}).get('title', s['model'])} · baseline "
                    f"{s.get('primary_output')}: {base:,.2f}" if base is not None else f"**{s['name']}**")
        if c3.button("Delete", key=f"wi_del_{s['id']}"):
            try:
                delete_scenario(s["id"])
            except ApiError as e:
                st.error(str(e))
            st.rerun()
    if len(chosen) >= 2:
        try:
            cmp = compare_scenarios(chosen)
            rows = []
            for item in cmp["items"]:
                rows.append({"Saved as": item["name"], "Scenario": "baseline", **(item["baseline_outputs"] or {})})
                for k, v in (item["scenarios"] or {}).items():
                    rows.append({"Saved as": item["name"], "Scenario": k, **(v or {})})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(cmp["disclaimer"])
        except ApiError as e:
            st.error(str(e))
    elif saved:
        st.caption("Tick two or more scenarios of the same model to compare them.")
