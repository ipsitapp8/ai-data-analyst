"""Doing the same work once: one question embedding for the planner's two
lookups, period rows / time series / segment breakdowns computed once per
investigation, and prompts that carry a brief of each result instead of the
stored result.

Every test here pins the other half of that bargain as well: what is reused or
shortened must be exactly what would have been computed or sent anyway."""
from __future__ import annotations

import copy
import json
from collections import Counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from platform_helpers import FIXTURES, ask, run_job, workspace

from app import analysis_ops as ops
from app import config, knowledge, memory
from app import metric_engine as me
from app.agents import prompts
from app.config import EMBEDDING_DIM
from app.database import SessionLocal
from app.profiling import load_and_profile_csv, profile_dataframe

SPLIT = "2025-03-01"


# ======================================================= question embedding ==

class CountingEmbed:
    """Stands in for the embedding request and records every call."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, texts, task_type):
        self.calls.extend((t, task_type) for t in texts)
        if self.fail:
            raise RuntimeError("quota")
        # One direction for every text: whatever is stored is similar enough to be retrieved,
        # so the tests see what the lookups return rather than how similarity is scored.
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]

    def queries(self) -> list[str]:
        return [t for t, task in self.calls if task == "RETRIEVAL_QUERY"]


@pytest.fixture
def embed():
    fake = CountingEmbed()
    with patch.object(memory, "embed", fake), patch.object(knowledge, "embed", fake):
        yield fake


def test_one_vector_serves_identical_requests_and_nothing_else(embed):
    shared = memory.QueryEmbedding()
    first = shared.get("why did revenue fall", embed)
    assert shared.get("why did revenue fall", embed) is first and len(embed.calls) == 1
    assert embed.calls == [("why did revenue fall", "RETRIEVAL_QUERY")]

    shared.get("why did revenue fall ", embed)
    assert len(embed.calls) == 2, "different text (even by one space) is a different request"
    with patch.object(config, "EMBEDDING_MODEL", "another-embedding-model"):
        shared.get("why did revenue fall ", embed)
        assert len(embed.calls) == 3, "a different model never reuses a vector"
    with patch.object(config, "EMBEDDING_DIM", EMBEDDING_DIM // 2):
        shared.get("why did revenue fall ", embed)
        assert len(embed.calls) == 4, "nor does a different dimensionality"


def test_a_failed_embedding_is_never_remembered():
    failing, working = CountingEmbed(fail=True), CountingEmbed()
    shared = memory.QueryEmbedding()
    with pytest.raises(RuntimeError):
        shared.get("q", failing)
    vector = shared.get("q", working)
    assert len(working.calls) == 1 and len(vector) == EMBEDDING_DIM, "the next lookup asks again and gets a real vector"


def _lesson(ws: dict, text: str = "Check totals against the raw rows.") -> None:
    db = SessionLocal()
    try:
        knowledge.add_note(db, team_id=ws["team"], dataset_id=ws["ds"], kind="lesson", text=text, source="test")
        db.commit()
    finally:
        db.close()


def _lookups(ws: dict, text: str, shared: memory.QueryEmbedding | None):
    db = SessionLocal()
    try:
        lessons = knowledge.retrieve_lessons(db, team_id=ws["team"], dataset_id=ws["ds"], question_text=text, query_embedding=shared)
        memories = memory.retrieve(db, team_id=ws["team"], dataset_id=ws["ds"], question_text=text, query_embedding=shared)
        return [n.text for n in lessons], [m.question_id for m in memories]
    finally:
        db.close()


@pytest.fixture(scope="module")
def std_case():
    cases = json.loads((FIXTURES.parent / "cases" / "suite.json").read_text(encoding="utf-8"))["cases"]
    return next(c for c in cases if c["id"] == "pipeline-correct")


def test_planning_embeds_the_question_once_for_both_lookups(client, embed, std_case):
    ws = workspace(client)
    first = ask(client, ws, std_case["question"])
    assert run_job(first["job_id"], std_case["script"]) == "succeeded"        # leaves a remembered analysis
    _lesson(ws)

    # The two lookups return the same rows whether or not they share the vector.
    embed.calls.clear()
    apart = _lookups(ws, std_case["question"], None)
    assert len(embed.queries()) == 2, "unshared, each lookup asks for its own embedding"
    embed.calls.clear()
    together = _lookups(ws, std_case["question"], memory.QueryEmbedding())
    assert len(embed.queries()) == 1 and together == apart
    assert together[0] and together[1], "both a lesson and a past analysis were there to be found"

    embed.calls.clear()
    second = ask(client, ws, std_case["question"] + " ")
    assert run_job(second["job_id"], std_case["script"]) == "succeeded"
    assert len(embed.queries()) == 1, "the planner asked for the question's embedding exactly once"


def test_no_embedding_is_requested_when_there_is_nothing_to_search(client, embed):
    ws = workspace(client)
    embed.calls.clear()
    assert _lookups(ws, "anything", memory.QueryEmbedding()) == ([], [])
    assert embed.calls == [], "the shared vector is lazy: no stored rows, no request"


def test_a_failing_embedding_still_leaves_both_lookups_empty_not_broken(client, embed):
    ws = workspace(client)
    _lesson(ws)
    embed.fail = True
    embed.calls.clear()
    assert _lookups(ws, "anything", memory.QueryEmbedding())[0] == []


# ====================================================== reused computations ==

@pytest.fixture(scope="module")
def sales():
    return load_and_profile_csv(str(FIXTURES / "sales.csv"))


def _dirty(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values in the metric, a dimension and the date; duplicated rows."""
    out = df.copy()
    rng = np.random.default_rng(7)
    out.loc[out.index[rng.choice(len(out), 25, replace=False)], "revenue"] = np.nan
    out.loc[out.index[rng.choice(len(out), 15, replace=False)], "region"] = None
    out.loc[out.index[rng.choice(len(out), 10, replace=False)], "units"] = np.nan
    out.loc[out.index[rng.choice(len(out), 6, replace=False)], "order_date"] = pd.NaT if out["order_date"].dtype.kind == "M" else None
    return pd.concat([out, out.iloc[:12]], ignore_index=True)


def _sparse(df: pd.DataFrame) -> pd.DataFrame:
    """A handful of rows, with most days absent."""
    return df.iloc[::37].reset_index(drop=True)


def _one_sided(df: pd.DataFrame) -> pd.DataFrame:
    """Nothing at all in the later period."""
    return df[df["order_date"].astype(str) < SPLIT].reset_index(drop=True)


FRAMES = {"clean": lambda df: df, "dirty": _dirty, "sparse": _sparse, "empty_later_period": _one_sided}

OPS = [
    ("metric_change", {}),
    ("component_change", {"formula": "sum(units)"}),
    ("component_change", {"formula": "sum(cost)", "filters": [{"column": "region", "op": "=", "value": "West"}]}),
    ("share_change", {"filters": [{"column": "channel", "op": "=", "value": "Online"}]}),
    ("distribution_shift", {"column": "units"}),
    ("segment_contribution", {"dimension": "region"}),
    ("segment_contribution", {"dimension": "region", "segment": "West"}),
    ("segment_contribution", {"dimension": "product", "filters": [{"column": "region", "op": "=", "value": "West"}]}),
    ("mix_shift", {"dimension": "region"}),
    ("driver_decomposition", {}),
    ("price_volume", {}),
    ("period_coverage", {}),
    ("data_quality", {}),
    ("level_shift", {}),
    ("level_shift", {"filters": [{"column": "region", "op": "=", "value": "West"}]}),
    ("pre_trend", {}),
    ("change_point", {}),
    ("paired_change", {"unit": "region"}),
    ("composition_change", {"dimension": "region"}),
    ("group_difference", {"outcome": "units", "dimension": "region"}),
    ("correlation", {"x": "unit_price", "y": "units"}),
    ("adjusted_change", {"outcome": "units", "controls": ["region", "product"]}),
]


def _context(df: pd.DataFrame, profile: dict, base_filters: list[dict] | None = None) -> ops.AnalysisContext:
    ctx = ops.AnalysisContext(formula=me.parse_formula("sum(revenue)"), label="Revenue", base_filters=base_filters or [],
                              time_column="order_date", period_a=["2025-01-01", SPLIT], period_b=[SPLIT, "2025-04-29"],
                              profile=profile)
    try:
        base = ops.execute(df, ctx, {"op": "metric_change", "params": {}})
        ctx.total_a, ctx.total_b = base["facts"]["a"], base["facts"]["b"]
    except Exception:  # noqa: BLE001 - a frame with no usable total still has to behave the same both ways
        pass
    return ctx


def _outcome(df, ctx, name, params) -> str:
    """What running one analysis produces, as comparable text: its result, or the error it raises."""
    try:
        op = ops.normalize({"op": name, "params": copy.deepcopy(params)}, ctx)
        return json.dumps(ops.execute(df, ctx, op), sort_keys=True, default=str)
    except Exception as e:  # noqa: BLE001 - the error is part of the behaviour being compared
        return f"{type(e).__name__}: {e}"


@pytest.mark.parametrize("frame", list(FRAMES))
@pytest.mark.parametrize("base_filters", [None, [{"column": "channel", "op": "=", "value": "Online"}]], ids=["all_rows", "population_filter"])
def test_sharing_work_across_analyses_changes_no_result(sales, frame, base_filters):
    df = FRAMES[frame](sales[0])
    profile = profile_dataframe(df) if frame != "clean" else sales[1]
    # Reference: every analysis on its own, with nothing to reuse.
    alone = [_outcome(df, _context(df, profile, base_filters), name, params) for name, params in OPS]
    assert any(not o.startswith(("OpError", "ValueError", "KeyError")) for o in alone), "the frame exercises real results"
    # One context for the whole run, forwards and then again (everything cached), and in reverse order.
    shared = _context(df, profile, base_filters)
    assert [_outcome(df, shared, name, params) for name, params in OPS] == alone
    assert [_outcome(df, shared, name, params) for name, params in OPS] == alone
    backwards = _context(df, profile, base_filters)
    assert [_outcome(df, backwards, name, params) for name, params in reversed(OPS)] == alone[::-1]


def _count(monkeypatch) -> Counter:
    counts: Counter = Counter()
    real_filters, real_series = me.apply_filters, ops._build_bucket_series

    def filters(df, f):
        counts["filter_passes"] += bool(f)        # a pass with no filters does not scan anything
        return real_filters(df, f)

    def series(*a, **k):
        counts["series_builds"] += 1
        return real_series(*a, **k)

    monkeypatch.setattr(me, "apply_filters", filters)
    monkeypatch.setattr(ops, "_build_bucket_series", series)
    return counts


def test_period_rows_and_series_are_computed_once_per_run(sales, monkeypatch):
    df, profile = sales
    counts = _count(monkeypatch)
    series_ops = [(n, p) for n, p in OPS if n in ("level_shift", "pre_trend", "change_point") and not p]

    for name, params in OPS:
        _outcome(df, _context(df, profile), name, params)
    alone = dict(counts)
    counts.clear()
    shared = _context(df, profile)
    for name, params in OPS:
        _outcome(df, shared, name, params)
    together = dict(counts)
    assert together["filter_passes"] < alone["filter_passes"] / 2, (alone, together)
    assert together["series_builds"] < alone["series_builds"]

    counts.clear()
    ctx = _context(df, profile)
    for name, params in series_ops:
        _outcome(df, ctx, name, params)
    assert len(series_ops) == 3 and counts["series_builds"] == 1, "level shift, trend and change point read one series"
    start = counts["filter_passes"]
    for name, params in OPS:
        _outcome(df, ctx, name, params)
    first_pass = counts["filter_passes"] - start
    for name, params in OPS:
        _outcome(df, ctx, name, params)
    second_pass = counts["filter_passes"] - start - first_pass
    assert second_pass < first_pass, "running every analysis again filters less: the period rows are already there"


def test_reuse_never_crosses_tables_populations_or_periods(sales):
    df, profile = sales
    ctx = _context(df, profile)
    west = [{"column": "region", "op": "=", "value": "West"}]
    a, b, everything = (ops.period_frame(df, ctx, w) for w in ("a", "b", "all"))
    assert len(a) + len(b) == len(everything) == len(df) and a["order_date"].astype(str).max() < SPLIT
    assert ops.period_frame(df, ctx, "a") is a, "the same request returns the same rows"
    narrowed = ops.period_frame(df, ctx, "a", west)
    assert 0 < len(narrowed) < len(a) and set(narrowed["region"]) == {"West"}

    other = df[df["region"] != "West"].reset_index(drop=True)
    assert len(ops.period_frame(other, ctx, "a")) == len(a) - len(narrowed), "another table is never answered from this one's rows"
    online = _context(df, profile, [{"column": "channel", "op": "=", "value": "Online"}])
    assert len(ops.period_frame(df, online, "a")) < len(a), "another population has its own context and its own rows"


def test_a_caller_cannot_alter_the_series_the_next_caller_reads(sales):
    df, profile = sales
    ctx = _context(df, profile)
    def text(series: dict) -> str:
        return json.dumps(series, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    first = ops.bucket_series(df, ctx, ctx.formula)
    untouched = text(first)

    def wreck(value) -> None:
        if isinstance(value, np.ndarray):
            value.fill(0)
        elif isinstance(value, list):
            value.clear()
        elif isinstance(value, dict):
            for inner in list(value.values()):
                wreck(inner)
            value.clear()

    wreck(first)
    assert text(first) != untouched and text(ops.bucket_series(df, ctx, ctx.formula)) == untouched
    west = ops.bucket_series(df, ctx, ctx.formula, [{"column": "region", "op": "=", "value": "West"}])
    assert text(west) != untouched, "a filtered series is a different series"


def test_the_headline_total_is_still_checked_by_the_independent_implementation(client, monkeypatch):
    from test_statistics import QUESTION, stat_script

    calls = []
    real = me.reference_evaluate
    monkeypatch.setattr(me, "reference_evaluate", lambda *a, **k: calls.append(a[1:]) or real(*a, **k))
    ws = workspace(client)
    q = ask(client, ws, QUESTION)
    assert run_job(q["job_id"], stat_script()) == "succeeded"
    assert len(calls) >= 2, "both period totals are recomputed from the file by the second implementation"


# ============================================================ compact prompts ==

def _executed(df, profile, name, params) -> dict:
    ctx = _context(df, profile)
    return ops.execute(df, ctx, ops.normalize({"op": name, "params": params}, ctx))


def test_a_brief_keeps_the_estimate_its_uncertainty_and_the_test(sales):
    df, profile = sales
    with_interval = 0
    for name, params in (("level_shift", {}), ("distribution_shift", {"column": "units"}), ("paired_change", {"unit": "region"}),
                         ("correlation", {"x": "unit_price", "y": "units"}),
                         ("adjusted_change", {"outcome": "units", "controls": ["region", "product"]})):
        result = _executed(df, profile, name, params)
        stat = result["facts"]["stat"]
        assert stat["status"] == "ok", name
        brief = ops.analysis_brief(name, result)["statistics"]
        assert brief["method"] == stat["method_label"] and brief["estimand"] == stat["estimand"]
        assert brief["estimate"] == stat["estimate"] and brief["significant"] == stat["significant"]
        iv = stat.get("interval")
        assert brief["interval"] == (None if not iv else [iv["low"], iv["high"]]), "no interval is reported as none, never invented"
        assert brief["level"] == (None if not iv else iv["level"])
        with_interval += bool(iv)
        p = stat["test"]["p_value"]
        assert brief["p_value"] == (p if p >= 0.001 else "<0.001"), "a tiny p-value is never shown as 0"
        adjusted = (stat.get("adjusted") or {}).get("p_value")
        assert brief["p_adjusted"] == (adjusted if adjusted is None or adjusted >= 0.001 else "<0.001")
        assert brief["n"] == stat["sample"].get("n")
        assert brief["effects"] == {e["name"]: e["value"] for e in stat["effects"] if e.get("value") is not None}
        assert brief.get("assumptions_violated", []) == [a["name"] for a in stat["assumptions"] if a["status"] == "violated"]
    assert with_interval >= 3, "most of these analyses do report an interval, and the brief carries it"


def test_a_brief_keeps_the_leading_segments_and_says_how_many_it_left_out(sales):
    df, profile = sales
    result = _executed(df, profile, "segment_contribution", {"dimension": "region"})
    brief = ops.analysis_brief("segment_contribution", result)
    table = result["table"]
    assert [r["segment"] for r in brief["top_rows"]] == [r["segment"] for r in table[:ops.BRIEF_TOP_ROWS]]
    for shown, stored in zip(brief["top_rows"], table):
        assert shown == {k: stored[k] for k in ("segment", "contribution", "share_of_change", "base_share") if k in stored}
    assert len(table) > ops.BRIEF_TOP_ROWS and brief["rows_listed"].startswith(f"top {ops.BRIEF_TOP_ROWS} of ")
    change = _executed(df, profile, "metric_change", {})
    kept = ops.analysis_brief("metric_change", change)
    assert {k: v for k, v in change["facts"].items() if k in kept} == {k: v for k, v in kept.items() if k in change["facts"]}
    for key in ("a", "b", "change"):
        assert kept[key] == change["facts"][key], "the period totals and the change are carried as computed"


def test_a_brief_reports_an_unavailable_statistic_with_its_reason(sales):
    df, profile = sales
    tiny = _sparse(df)
    result = _executed(tiny, profile_dataframe(tiny), "level_shift", {})
    stat = result["facts"]["stat"]
    brief = ops.analysis_brief("level_shift", result)["statistics"]
    if stat["status"] != "ok":
        assert brief == {"status": "unavailable", "reason": stat["reason"], "estimand": stat["estimand"]}
    else:
        assert brief["estimate"] == stat["estimate"]
    assert ops.stat_brief(None) is None
    assert ops.stat_brief({"status": "unavailable", "reason": "too few buckets", "estimand": "x"}) == {
        "status": "unavailable", "reason": "too few buckets", "estimand": "x"}


def test_the_change_point_and_adjustment_details_survive_the_brief(sales):
    df, profile = sales
    cp = _executed(df, profile, "change_point", {})["facts"]["stat"]
    brief = ops.stat_brief(cp)
    if cp["status"] == "ok":
        expected = {k: v for k, v in (cp.get("details") or {}).items() if k in ops._BRIEF_DETAILS and v is not None}
        assert expected and brief["details"] == expected, "where the shift was found is part of the brief"
    adj = _executed(df, profile, "adjusted_change", {"outcome": "units", "controls": ["region", "product"]})["facts"]["stat"]
    shown = ops.stat_brief(adj)["details"]
    stored = {k: v for k, v in adj["details"].items() if k in ops._BRIEF_DETAILS and v is not None}
    assert set(shown) == set(stored) and shown["attenuation"] == stored["attenuation"] and shown["controls"] == stored["controls"]
    assert {k: v for k, v in shown["unadjusted"].items() if k != "p_value"} == {k: v for k, v in stored["unadjusted"].items() if k != "p_value"}


def _stat(**over) -> dict:
    """A complete, valid statistical result to vary one field at a time."""
    base = {"status": "ok", "method_label": "least-squares regression", "estimand": "difference in mean units", "estimate": -2.25,
            "interval": {"low": -2.61, "high": -1.89, "level": 0.95}, "test": {"p_value": 0.03}, "adjusted": {"p_value": 0.04},
            "significant": True, "sample": {"n": 720}, "effects": [], "assumptions": [], "warnings": [], "limitations": [], "details": {}}
    return {**base, **over}


@pytest.mark.parametrize("p, shown", [(3.2e-12, "<0.001"), (0.0, "<0.001"), (0.00099, "<0.001"), (0.001, 0.001), (0.0421, 0.0421), (1.0, 1.0)])
def test_a_p_value_inside_a_detail_is_never_shown_as_zero(p, shown):
    from app.hypothesis_engine import _rounded

    unadjusted = {"estimate": -2.25, "low": -2.8, "high": -1.7, "p_value": p}
    result = _stat(details={"unadjusted": dict(unadjusted), "attenuation": 0.0})
    brief = ops.stat_brief(result)
    assert brief["details"]["unadjusted"] == {**unadjusted, "p_value": shown}
    assert result["details"]["unadjusted"] == unadjusted, "the stored result keeps the number itself"
    in_prompt = json.dumps(_rounded(brief), separators=(",", ":"))             # exactly how the engine writes it
    assert ('"p_value":0.0,' not in in_prompt) and ('"p_value":0.0}' not in in_prompt)
    assert (f'"p_value":"{shown}"' if isinstance(shown, str) else f'"p_value":{shown}') in in_prompt.split('"unadjusted":')[1]
    assert ops.stat_brief(_stat(details={"unadjusted": {"estimate": 1.0, "p_value": None}}))["details"]["unadjusted"]["p_value"] is None


def test_the_real_adjusted_change_shows_its_tiny_unadjusted_p_value_as_a_bound(sales):
    df, profile = sales
    adj = _executed(df, profile, "adjusted_change", {"outcome": "units", "controls": ["region", "product"]})["facts"]["stat"]
    stored = adj["details"]["unadjusted"]["p_value"]
    assert 0 < stored < 0.001, "the fixture's unadjusted difference is far from chance"
    assert ops.stat_brief(adj)["details"]["unadjusted"]["p_value"] == "<0.001"
    assert adj["details"]["unadjusted"]["p_value"] == stored


def test_exclusions_are_carried_when_something_was_excluded_and_only_then(sales):
    df, profile = sales
    series = _executed(df, profile, "change_point", {})["facts"]["stat"]
    left_out = series["sample"]["excluded"]["buckets_without_rows"]
    assert left_out["earlier"] > 0 and ops.stat_brief(series)["excluded"] == {"buckets_without_rows": left_out}

    clean = _executed(df, profile, "distribution_shift", {"column": "units"})["facts"]["stat"]
    assert clean["sample"]["excluded"] == {"missing_or_non_numeric": {"earlier": 0, "later": 0}}
    assert "excluded" not in ops.stat_brief(clean), "nothing excluded, nothing to say"

    holes = _dirty(df)
    dirty = _executed(holes, profile_dataframe(holes), "distribution_shift", {"column": "units"})["facts"]["stat"]
    counts = dirty["sample"]["excluded"]["missing_or_non_numeric"]
    assert sum(counts.values()) > 0
    assert ops.stat_brief(dirty)["excluded"] == {"missing_or_non_numeric": {k: v for k, v in counts.items() if v}}

    assert ops._nonzero({"rows_with_missing_values": 0, "controls_dropped": []}) == {}
    dropped = [{"control": "customer", "reason": "400 distinct values (more than 20)"}]
    assert ops._nonzero({"rows_with_missing_values": 40, "controls_dropped": dropped}) == {"rows_with_missing_values": 40, "controls_dropped": dropped}
    assert ops._nonzero({"buckets_without_rows": {"earlier": 0, "later": 0, "handling": "left out"}}) == {}, "a label with no count left is dropped too"
    assert ops._nonzero({"buckets_without_rows": {"earlier": 0, "later": 3, "handling": "counted as 0"}}) == {
        "buckets_without_rows": {"later": 3, "handling": "counted as 0"}}


def test_the_approximate_p_value_flag_and_method_limitations_survive_the_brief(sales):
    from app.stats import result as stat_result

    df, profile = sales
    general = {stat_result.FRAME_NOTE, stat_result.OBSERVATIONAL_NOTE, stat_result.BIAS_NOTE}
    cp = _executed(df, profile, "change_point", {})["facts"]["stat"]
    assert cp["details"]["p_value_is_approximate"] is True and ops.stat_brief(cp)["details"]["p_value_is_approximate"] is True
    for name, params in (("change_point", {}), ("pre_trend", {}), ("level_shift", {}),
                         ("adjusted_change", {"outcome": "units", "controls": ["region", "product"]})):
        stat = _executed(df, profile, name, params)["facts"]["stat"]
        specific = [t for t in stat["limitations"] if t not in general]
        assert specific, name
        assert ops.stat_brief(stat)["limitations"] == specific, "word for word, in the stored order, minus the notes every result carries"
    assert any("Pettitt" in t for t in ops.stat_brief(cp)["limitations"])
    welch = _executed(df, profile, "distribution_shift", {"column": "units"})["facts"]["stat"]
    assert set(welch["limitations"]) <= general and "limitations" not in ops.stat_brief(welch), "no method-specific limitation, no field"
    assert "p_value_is_approximate" not in (ops.stat_brief(welch).get("details") or {}), "the flag is never added to a method that does not set it"


def test_the_prompt_lists_each_limitation_once_and_stays_far_smaller_than_the_results(client):
    from test_statistics import QUESTION, stat_script

    from app import jobs, runtime
    from app.evals.scripted import ScriptedProvider
    from app.stats import result as stat_result

    seen: dict[str, list[str]] = {}

    class Recording(ScriptedProvider):
        def __call__(self, provider, *, system, user_content, tool_name, **kw):
            seen.setdefault(tool_name, []).append(user_content)
            return super().__call__(provider, system=system, user_content=user_content, tool_name=tool_name, **kw)

    ws = workspace(client)
    q = ask(client, ws, QUESTION)
    with runtime.provider_override(Recording(stat_script())):
        assert jobs.run_inline(q["job_id"], None) == "succeeded"
    inv = next(i for i in client.get("/api/investigations", headers=ws["h"]).json()["investigations"] if i["question_id"] == q["question_id"])
    analyses = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()["analyses"]

    facts = seen["submit_report"][0].split("Executed analyses (the only numbers you may use):\n", 1)[1]
    listed, _, footer = facts.partition("\nLimitations of the methods used:\n")
    lines = {line.split(" ", 1)[0]: line for line in listed.splitlines() if line.startswith("A")}
    assert set(lines) == {a["id"] for a in analyses}, "every executed analysis is still in front of the model"
    assert '"limitations"' not in listed, "limitations are not repeated on each line"

    general = {stat_result.FRAME_NOTE, stat_result.OBSERVATIONAL_NOTE, stat_result.BIAS_NOTE}
    expected: dict[str, list[str]] = {}
    for a in analyses:
        stat = (a["result"].get("facts") or {}).get("stat") or {}
        for text in stat.get("limitations") or [] if stat.get("status") == "ok" else []:
            if text not in general:
                expected.setdefault(text, []).append(a["id"])
    assert expected and footer.strip().splitlines() == [f"- {', '.join(ids)}: {text}" for text, ids in expected.items()]
    assert any(len(ids) > 1 for ids in expected.values()), "a limitation shared by several analyses is written once"
    for text in expected:
        assert facts.count(text) == 1

    change_point = next(a for a in analyses if a["op"] == "change_point")
    assert '"p_value_is_approximate":true' in lines[change_point["id"]] and '"excluded":{"buckets_without_rows"' in lines[change_point["id"]]
    assert '"p_value":0.0' not in facts, "no p-value anywhere in the list is shown as zero"

    full = sum(len(json.dumps(a["result"], separators=(",", ":"), default=str)) for a in analyses)
    assert len(facts) < full / 3, (len(facts), full)
    assert len(facts) < 9000 and facts.endswith(footer)
    for a in analyses:
        stored = len(json.dumps(a["result"], separators=(",", ":"), default=str))
        if (a["result"].get("facts") or {}).get("stat"):
            assert len(lines[a["id"]]) < stored / 2, (a["id"], len(lines[a["id"]]), stored)


def test_the_short_catalogue_offers_every_analysis_and_expectation():
    lines = ops.CATALOGUE_BRIEF.splitlines()
    offered = {line.split(" ", 1)[0]: line for line in lines if " params " in line}
    assert set(offered) == set(ops.EXPECTATIONS) - {"metric_change"}, "only the headline comparison (always run) is left out"
    for name, line in offered.items():
        assert line.endswith("expect: " + " | ".join(ops.EXPECTATIONS[name]))
        assert name in ops.CATALOGUE, "the short and the full catalogue describe the same analyses"
    assert '"column","op","value"' in ops.CATALOGUE_BRIEF and "estimand" in ops.CATALOGUE_BRIEF and "method" in ops.CATALOGUE_BRIEF
    assert len(ops.CATALOGUE_BRIEF) < len(ops.CATALOGUE) / 2


def test_the_compact_profile_is_the_same_profile(sales):
    _, profile = sales
    full, compact = prompts.profile_block(profile), prompts.profile_block(profile, compact=True)

    def body(block: str) -> dict:
        head, tail = '<dataset_profile untrusted="true">\n', "\n</dataset_profile>"
        assert block.startswith(head) and block.endswith(tail)
        return json.loads(block[len(head):-len(tail)])

    assert body(full) == body(compact) and len(compact) < len(full)
    assert full == prompts.profile_block(profile, compact=False), "the standard route still gets the indented form"
    assert "\n  " in full and "\n  " not in compact


def test_follow_up_prompts_are_shorter_and_still_carry_the_evidence(client, monkeypatch):
    """The whole engine on scripted replies: every prompt that lists executed
    analyses shows each one's estimate, interval and p-value, and the follow-up
    prompt uses the short catalogue while the first prompt keeps the full one."""
    from test_statistics import QUESTION, stat_script

    from app import runtime
    from app.evals.scripted import ScriptedProvider

    seen: dict[str, list[str]] = {}

    class Recording(ScriptedProvider):
        def __call__(self, provider, *, system, user_content, tool_name, **kw):
            seen.setdefault(tool_name, []).append(user_content)
            return super().__call__(provider, system=system, user_content=user_content, tool_name=tool_name, **kw)

    ws = workspace(client)
    q = ask(client, ws, QUESTION)
    from app import jobs
    with runtime.provider_override(Recording(stat_script())):
        assert jobs.run_inline(q["job_id"], None) == "succeeded"

    with_catalogue = {tool: p for tool, ps in seen.items() for p in ps if "Available analyses:" in p}
    full = [t for t, p in with_catalogue.items() if ops.CATALOGUE in p]
    short = [t for t, p in with_catalogue.items() if ops.CATALOGUE_BRIEF in p and ops.CATALOGUE not in p]
    assert len(full) == 1 and len(short) >= 1, (full, short)

    evidence_prompts = [p for ps in seen.values() for p in ps if '"statistics":{' in p]
    assert evidence_prompts, "at least the follow-up and report prompts list executed statistics"
    for p in evidence_prompts:
        assert '"p_value":' in p and '"interval":[' in p and '"estimate":' in p and '"effects":{' in p
        assert "<dataset_profile" not in p or "\n  \"" not in p, "the engine's profile is the compact one"
