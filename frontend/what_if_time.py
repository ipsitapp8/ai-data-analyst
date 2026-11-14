"""What-if over time: pick a period of the data for a baseline, and compare two recorded periods.

Rendering only: periods, sums, changes, attributions and checks all come from the backend (`/api/scenarios/*`). Figures
here are historical actuals unless a section says otherwise, and the page says which kind of figure is on screen.
"""
from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, compare_periods, list_dataset_versions, scenario_periods
from style.theme import badge, esc, html, plot
from what_if_decision import num, section, verification_panel

PERIOD_TYPES = {"all": "All data", "year": "A year", "quarter": "A quarter", "month": "A month", "custom": "A custom date range"}
MAPPING_FIELDS = {"unit_economics": ["quantity_column", "amount_column", "price_column", "total_cost_column", "unit_cost_column"],
                  "funnel": ["visitors_column", "orders_column", "revenue_column", "spend_column"]}
KIND_LEGEND = [("Historical actuals", "recorded in your data for the period shown"),
               ("Scenario", "arithmetic on inputs you set; not observed"),
               ("Statistical estimate", "estimated from your data with its uncertainty"),
               ("Forecast", "not produced here: nothing is extrapolated forward in time")]


def legend() -> None:
    html('<div class="ds-row-meta" style="padding:4px 0;">What the figures are: ' + " · ".join(
        f"<b>{esc(k)}</b> — {esc(v)}" for k, v in KIND_LEGEND) + "</div>")


def version_picker(prefix: str, ds: dict) -> int | None:
    """Which version of the dataset to use: the latest, or a specific earlier one."""
    try:
        versions = list_dataset_versions(ds["id"])
    except ApiError:
        return None
    if len(versions) < 2:
        return None
    latest = max(v["version"] for v in versions)
    choice = st.selectbox("Dataset version", [None] + [v["version"] for v in sorted(versions, key=lambda v: -v["version"])],
                          format_func=lambda v: f"Latest (version {latest})" if v is None else f"Version {v}", key=f"{prefix}_version")
    return choice


def period_picker(prefix: str, ds: dict, version: int | None, fixed_date: str | None = None) -> tuple[str | None, dict]:
    """Date column and period for a baseline. Returns (date_column, period spec). `fixed_date` skips choosing the column."""
    dates = [c["name"] for c in ds["profile"].get("columns", []) if c.get("kind") == "datetime"]
    if not dates:
        st.caption("This dataset has no date column, so the baseline uses all of its rows.")
        return None, {"type": "all"}
    date_column = fixed_date or st.selectbox("Date column", [None] + dates, format_func=lambda v: "None — use all rows" if v is None else v,
                                             key=f"{prefix}_date")
    kind = st.selectbox("Period", list(PERIOD_TYPES), format_func=PERIOD_TYPES.get, key=f"{prefix}_ptype", disabled=date_column is None)
    if date_column is None or kind == "all":
        return date_column, {"type": "all"}
    try:
        info = scenario_periods(ds["id"], date_column, version)
    except ApiError as e:
        st.error(str(e))
        return date_column, {"type": "all"}
    st.caption(f"The data covers {info['min']} to {info['max']} ({info['rows_with_date']:,} rows with a date"
               + (f", {info['rows_without_date']:,} without" if info["rows_without_date"] else "") + "). Only periods with rows are offered.")
    if kind == "year":
        y = st.selectbox("Year", [x["year"] for x in info["years"]], format_func=lambda y: f"{y} ({next(x['rows'] for x in info['years'] if x['year'] == y):,} rows)",
                         key=f"{prefix}_year")
        return date_column, {"type": "year", "year": y}
    if kind == "quarter":
        qs = [(x["year"], x["quarter"]) for x in info["quarters"]]
        rows = {(x["year"], x["quarter"]): x["rows"] for x in info["quarters"]}
        y, q = st.selectbox("Quarter", qs, format_func=lambda p: f"{p[0]} Q{p[1]} ({rows[p]:,} rows)", key=f"{prefix}_quarter")
        return date_column, {"type": "quarter", "year": y, "quarter": q}
    if kind == "month":
        ms = [(x["year"], x["month"]) for x in info["months"]]
        rows = {(x["year"], x["month"]): x["rows"] for x in info["months"]}
        y, m = st.selectbox("Month", ms, format_func=lambda p: f"{dt.date(p[0], p[1], 1):%B %Y} ({rows[p]:,} rows)", key=f"{prefix}_month")
        return date_column, {"type": "month", "year": y, "month": m}
    lo, hi = dt.date.fromisoformat(info["min"]), dt.date.fromisoformat(info["max"])
    a, b = st.columns(2)
    start = a.date_input("From", lo, min_value=lo, max_value=hi, key=f"{prefix}_from")
    end = b.date_input("To", hi, min_value=lo, max_value=hi, key=f"{prefix}_to")
    return date_column, {"type": "custom", "start": str(start), "end": str(end)}


def baseline_caption(seed: dict) -> None:
    """Where a baseline came from: kind of figure, period, dataset version and fingerprint."""
    p, prov = seed.get("period") or {}, seed.get("provenance") or {}
    html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge("Historical actuals", "neutral")} {esc(p.get("label", "All data"))} · '
         f'{seed["rows_used"]:,} rows · dataset version {prov.get("dataset_version")}'
         f'{" (a newer version exists)" if prov.get("latest_version") and prov.get("latest_version") != prov.get("dataset_version") else ""}'
         f' · fingerprint {esc((prov.get("content_sha256") or "")[:12])}</div>')
    if seed.get("verification"):
        verification_panel(seed["verification"], "Independent check of the baseline")
    for r in seed.get("reconciliation") or []:
        html(f'<div class="ds-row-meta" style="padding:2px 0;">{badge("Agrees" if r["agrees"] else "Differs", "verified" if r["agrees"] else "warn")} '
             f'{esc(r["label"])} against the approved metric “{esc(r["metric"])}”: model {num(r["model_value"])}, metric {num(r["observed"])}</div>')


def compare_tab(models: dict, datasets: list[dict]) -> None:
    section("Compare two periods", "What your data recorded in each period, how it changed, and which inputs drove the change. "
                                   "Recorded figures only: nothing is adjusted and nothing is forecast.")
    legend()
    if not datasets:
        st.caption("Upload a dataset with a date column to compare periods.")
        return
    model_key = st.selectbox("Model", list(models), format_func=lambda k: models[k]["title"], key="cp_model")
    model = models[model_key]
    ds = st.selectbox("Dataset", datasets, format_func=lambda d: d["filename"], key="cp_ds")
    version = version_picker("cp", ds)
    if not any(c.get("kind") == "datetime" for c in ds["profile"].get("columns", [])):
        st.caption("This dataset has no date column, so periods cannot be compared.")
        return
    numeric = [None] + [c["name"] for c in ds["profile"].get("columns", []) if c.get("kind") == "numeric"]
    mapping = {}
    if model_key in MAPPING_FIELDS:
        fields = MAPPING_FIELDS[model_key]
        for col, f in zip(st.columns(len(fields)), fields):
            mapping[f] = col.selectbox(f.replace("_", " ").title(), numeric, key=f"cp_map_{model_key}_{f}", format_func=lambda v: v or "—")
    else:
        st.caption("This model's inputs are bound to approved metrics or formulas in its definition.")
    date_column = st.selectbox("Date column", [c["name"] for c in ds["profile"]["columns"] if c.get("kind") == "datetime"], key="cp_date")
    a, b = st.columns(2)
    with a:
        st.markdown("**Period A (earlier)**")
        _, period_a = period_picker("cp_a", ds, version, fixed_date=date_column)
    with b:
        st.markdown("**Period B (later)**")
        _, period_b = period_picker("cp_b", ds, version, fixed_date=date_column)
    overrides = {}
    with st.expander("Values your data does not provide (optional, used for both periods)"):
        st.caption("For example a fixed cost. Leave blank for an input the data supplies.")
        for name, rule in model["inputs"].items():
            v = st.number_input(f"{rule['label']} ({rule['unit']})", value=None, key=f"cp_ov_{model_key}_{name}", format="%.4f")
            if v is not None:
                overrides[name] = float(v)
    body = {"dataset_id": ds["id"], "dataset_version": version, "model": model_key, "mapping": {k: v for k, v in mapping.items() if v},
            "date_column": date_column, "period_a": period_a, "period_b": period_b, "overrides": overrides,
            "assumptions": {k: float(r["default"]) for k, r in model["assumptions"].items()}}
    if st.button("Compare the periods", type="primary", key="cp_run"):
        try:
            with st.spinner("Computing both periods and checking them…"):
                st.session_state["cp_result"] = compare_periods(body)
        except ApiError as e:
            st.session_state.pop("cp_result", None)
            st.error(f"The comparison could not run: {e}")
    r = st.session_state.get("cp_result")
    if r:
        _show(r, model)


def _show(r: dict, model: dict) -> None:
    pa, pb = r["periods"]["a"], r["periods"]["b"]
    prov = r["provenance"]
    html(f'<div class="ds-row-meta" style="padding:6px 0;">{badge("Historical actuals", "neutral")} '
         f'{esc(pa["period"]["label"])} ({pa["rows_used"]:,} rows) against {esc(pb["period"]["label"])} ({pb["rows_used"]:,} rows) · '
         f'dataset version {prov["dataset_version"]} · fingerprint {esc((prov["content_sha256"] or "")[:12])}</div>')
    import pandas as pd
    rows = []
    for k, rule in model["inputs"].items():
        c = r["input_changes"][k]
        rows.append({"": rule["label"], "Kind": "input", "Period A": r["a"]["inputs"][k], "Period B": r["b"]["inputs"][k], "Change": c["abs"], "Change %": c["pct"]})
    for k, label in model["outputs"].items():
        c = r["output_changes"].get(k) or {"abs": None, "pct": None}
        rows.append({"": label, "Kind": "output", "Period A": r["a"]["outputs"].get(k), "Period B": r["b"]["outputs"].get(k), "Change": c["abs"], "Change %": c["pct"]})
    st.dataframe(pd.DataFrame(rows).style.format({"Period A": "{:,.4f}", "Period B": "{:,.4f}", "Change": "{:+,.4f}", "Change %": "{:+.1f}%"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    contrib = r["contributions"]["by_input"]
    if contrib:
        labels = [model["inputs"][k]["label"] for k in contrib]
        fig = go.Figure(go.Waterfall(x=labels + ["Total change"], y=list(contrib.values()) + [sum(contrib.values())],
                                     measure=["relative"] * len(contrib) + ["total"]))
        fig.update_layout(title=f"What drove the change in {model['outputs'][r['primary_output']].lower()}")
        plot(fig, height=300, restyle_traces=False)
        st.caption(r["contributions"]["method"] + "; the shares sum exactly to the change.")
    else:
        st.caption("No input changed between the periods, so there is nothing to attribute.")
    for note in r["limits"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(note)}</div>')
    for label, notes in (r.get("constraint_notes") or {}).items():
        for n in notes:
            st.warning(f"Period {label}: the recorded figures break a rule of this model — {n['message']}")
    verification_panel(r.get("verification"), "Independent check of the comparison")
    for side, p in (("A", pa), ("B", pb)):
        with st.expander(f"Where period {side}'s inputs came from"):
            for name, s in p["sources"].items():
                html(f'<div class="ds-row-meta" style="padding:2px 0;"><b>{esc(model["inputs"][name]["label"])}</b>: {esc(s["source"].replace("_", " "))}'
                     f'{" — <code>" + esc(s["formula"]) + "</code>" if s.get("formula") else ""}</div>')
