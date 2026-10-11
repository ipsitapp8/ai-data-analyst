"""Hypothesis-driven investigation: the reasoning loop.

    understand ─> baseline facts ─> generate hypotheses ─> test ─> evaluate
                                           ^                          │
                                           └── choose next analysis <─┘   (bounded)
                                                                      │
                                                                   report

Division of labour, enforced by this module:

  the model   reads the question and the schema; proposes what the metric and
              periods are, which explanations are worth testing, which analysis
              would distinguish them next, and how to word the findings
  the code    validates every proposal against the real schema, executes the
              analyses (analysis_ops), decides each hypothesis's status from the
              results by fixed rules, and checks that every number in the
              model's wording traces to an executed result

So the model can be wrong about what to look at, but it cannot make a number
up, cannot mark a hypothesis supported, and cannot claim an analysis ran.

The loop is bounded four ways (model calls, analyses, wall-clock seconds,
tokens) plus a round limit, and it stops early when nothing is left unresolved,
when the model says nothing further would distinguish the hypotheses, when its
proposals stop being new, or when the data cannot support a conclusion.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from app import analysis_ops as ops
from app import config, investigator, runtime, verification
from app import metric_engine as me
from app.analysis_ops import FAILED, INCONCLUSIVE, NOT_TESTABLE, SUPPORTED, WEAKENED

logger = logging.getLogger(__name__)

ENGINE_VERSION = "1"
PROPOSED = "proposed"
# Tool schemas stay inside what every provider accepts: one type per field, no
# untyped values, no property-less objects (Gemini rejects all three).
FILTER_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "column": {"type": "string"}, "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<="]},
    "value": {"type": "string", "description": "The value to compare with; write numbers and dates as text."}},
    "required": ["column", "op", "value"]}}
ANALYSIS_SCHEMA = {"type": "object", "properties": {
    "op": {"type": "string", "enum": [k for k in ops.EXPECTATIONS if k != "metric_change"]},
    "params": {"type": "object", "description": "Only the parameters the chosen analysis takes.", "properties": {
        "formula": {"type": "string"}, "column": {"type": "string"}, "dimension": {"type": "string"},
        "segment": {"type": "string"}, "filters": FILTER_SCHEMA}}}, "required": ["op", "params"]}
PREDICTION_SCHEMA = {"type": "object", "properties": {"expect": {"type": "string"}}, "required": ["expect"]}

UNDERSTANDING_SCHEMA = {
    "type": "object",
    "properties": {
        "metric_formula": {"type": "string", "description": "Aggregation formula for the target metric, e.g. sum(revenue). "
                                                            "Leave empty if an approved metric applies or none can be identified."},
        "metric_label": {"type": "string"},
        "time_column": {"type": "string"},
        "period_a": {"type": "array", "items": {"type": "string"},
                     "description": "Earlier period as [first day, last day], ISO dates, both included. Empty if the question fixes no periods."},
        "period_b": {"type": "array", "items": {"type": "string"},
                     "description": "Later period as [first day, last day], ISO dates, both included. Empty if the question fixes no periods."},
        "population_filters": FILTER_SCHEMA,
        "dimensions": {"type": "array", "items": {"type": "string"}},
        "expected_direction": {"type": "string", "enum": ["decrease", "increase", "unknown"]},
        "claimed_change_pct": {"type": "number", "description": "The size of change the question asserts, in percent. "
                                                                 "Omit it when the question states no size."},
        "requested_outcome": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["metric_formula", "time_column", "period_a", "period_b", "expected_direction", "assumptions",
                 "missing_information"],
}
HYPOTHESES_SCHEMA = {
    "type": "object",
    "properties": {"hypotheses": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "measurable_prediction": {"type": "string", "description": "What the data would show if this were true."},
            "analysis": ANALYSIS_SCHEMA,
            "prediction": PREDICTION_SCHEMA,
            "required_columns": {"type": "array", "items": {"type": "string"}},
            "confounders": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string"},
        },
        "required": ["statement", "measurable_prediction", "analysis", "prediction", "required_columns", "priority"],
    }}},
    "required": ["hypotheses"],
}
NEXT_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["analyze", "stop"]},
        "unresolved_question": {"type": "string"},
        "hypothesis_id": {"type": "string"},
        "analysis": ANALYSIS_SCHEMA,
        "prediction": PREDICTION_SCHEMA,
        "rationale": {"type": "string"},
    },
    "required": ["decision", "rationale"],
}
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "possible_explanations": {"type": "array", "items": {"type": "string"},
                                  "description": "Explanations consistent with the evidence, worded as possibilities."},
        "follow_up_analyses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["executive_summary", "possible_explanations", "follow_up_analyses"],
}


@dataclass
class Budget:
    max_llm_calls: int = 8
    max_analyses: int = 24
    max_seconds: float = 240.0
    max_tokens: int = 80000
    max_rounds: int = 4
    llm_calls: int = 0
    analyses: int = 0
    rounds: int = 0
    started: float = field(default_factory=time.monotonic)
    tokens_used: Callable[[], int] = lambda: 0

    @classmethod
    def from_config(cls, tokens_used: Callable[[], int] | None = None) -> "Budget":
        return cls(config.HYP_MAX_LLM_CALLS, config.HYP_MAX_ANALYSES, float(config.HYP_MAX_SECONDS),
                   config.HYP_MAX_TOKENS, config.HYP_MAX_ROUNDS, tokens_used=tokens_used or (lambda: 0))

    def seconds(self) -> float:
        return time.monotonic() - self.started

    def blocked(self, need: str | None = None) -> str | None:
        """The limit that has been reached, if any. `need` asks about one more
        model call ("llm") or one more analysis ("analysis")."""
        if self.seconds() >= self.max_seconds:
            return "time"
        if self.max_tokens and self.tokens_used() >= self.max_tokens:
            return "tokens"
        if need == "llm" and self.llm_calls >= self.max_llm_calls:
            return "model_calls"
        if need == "analysis" and self.analyses >= self.max_analyses:
            return "analyses"
        return None

    def used(self) -> dict:
        return {"model_calls": self.llm_calls, "analyses": self.analyses, "rounds": self.rounds,
                "seconds": round(self.seconds(), 2), "tokens": self.tokens_used(),
                "limits": {"model_calls": self.max_llm_calls, "analyses": self.max_analyses, "rounds": self.max_rounds,
                           "seconds": self.max_seconds, "tokens": self.max_tokens}}


class MemoryStore:
    """Where a run's state is kept. The graph node uses a database-backed
    subclass; this one is for tests and for callers that need no persistence."""

    def __init__(self):
        self.understanding: dict | None = None
        self.hypotheses: list[dict] = []
        self.analyses: list[dict] = []

    def load(self) -> dict:
        return {"understanding": self.understanding, "hypotheses": [dict(h) for h in self.hypotheses],
                "analyses": [dict(a) for a in self.analyses]}

    def save_understanding(self, understanding: dict) -> None:
        self.understanding = understanding

    def save_hypothesis(self, hypothesis: dict) -> None:
        self.hypotheses = [h for h in self.hypotheses if h["id"] != hypothesis["id"]] + [dict(hypothesis)]

    def save_analysis(self, analysis: dict) -> None:
        self.analyses = [a for a in self.analyses if a["id"] != analysis["id"]] + [dict(analysis)]


class CannotInvestigate(Exception):
    """The question cannot be turned into a metric and two periods on this data."""


def _norm_statement(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (text or "").lower())).strip()


def derived_numbers(base: list[float]) -> list[float]:
    """Forms a faithful sentence may use for a computed number: the magnitude
    of a negative ("fell by 14,490" for -14,490) and a fraction as a percentage
    ("117% of the change" for 1.17)."""
    percents = [x * 100 for x in base if x != int(x) and abs(x) <= 5]
    return [abs(x) for x in base if x < 0] + percents + [abs(x) for x in percents if x < 0]


def _rounded(value: Any) -> Any:
    """Numbers as a reader would write them, so a model does not copy 15 decimals."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 2) if abs(value) >= 1 else round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


class HypothesisEngine:
    def __init__(self, *, question: str, df: pd.DataFrame, profile: dict, call_model: Callable[..., dict],
                 approved_metrics: list[dict] | None = None, dq_messages: list[str] | None = None,
                 store: MemoryStore | None = None, budget: Budget | None = None,
                 progress: Callable[[str], None] | None = None, checkpoint: Callable[[], None] | None = None,
                 provenance: dict | None = None):
        self.question, self.df, self.profile = question, df, profile
        self.call_model = call_model
        self.approved_metrics = approved_metrics or []
        self.dq_messages = dq_messages or []
        self.store = store or MemoryStore()
        self.budget = budget or Budget.from_config()
        self.progress = progress or (lambda _msg: None)
        self.checkpoint = checkpoint or (lambda: None)
        self.provenance = provenance or {}
        self.ctx: ops.AnalysisContext | None = None
        self.understanding: dict = {}
        self.hypotheses: list[dict] = []
        self.rejected: list[dict] = []
        self.analyses: list[dict] = []
        self.by_signature: dict[str, dict] = {}
        self.limitations: list[str] = []
        self.model_notes: list[str] = []
        self.stop_reason = ""
        self.resumed = False
        self.model_unavailable = False

    # ------------------------------------------------------------ model i/o --

    def _ask(self, tool: str, schema: dict, system: str, user: str, description: str) -> dict | None:
        """One model call, inside the budget. Returns None when the budget does
        not allow it or the reply is not an object; provider errors propagate."""
        self.checkpoint()
        if self.budget.blocked("llm"):
            return None
        self.budget.llm_calls += 1
        try:
            out = self.call_model(system=system, user_content=user, tool_name=tool, tool_schema=schema,
                                  tool_description=description, max_tokens=3000)
        except runtime.ProviderUnavailable:
            if self.budget.llm_calls <= 1:
                raise  # no model at all: the caller decides what to do instead
            # The model went away mid-run. Keep the evidence gathered so far and
            # finish with what the code alone can say.
            self.model_unavailable = True
            self.model_notes.append(f"{tool}: no model provider could be reached; the run continued without it")
            return None
        if not isinstance(out, dict):
            self.model_notes.append(f"{tool}: the model's reply was not an object and was discarded")
            return None
        return out

    def _columns_text(self) -> str:
        """The schema as the model sees it. Sample values are included only for
        low-cardinality columns (the ones usable as segment dimensions); values
        of identifier-like columns -- names, emails, ids -- never reach a prompt."""
        from app.agents import prompts

        slim = {"row_count": self.profile.get("row_count"), "col_count": self.profile.get("col_count"), "columns": []}
        for col in self.profile.get("columns", []):
            entry = dict(col)
            if entry.get("kind") == "categorical" and (entry.get("unique_count") or 0) > 50:
                entry.pop("top_values", None)
                entry["note"] = "high-cardinality column; sample values withheld"
            slim["columns"].append(entry)
        return prompts.profile_block(_rounded(slim))

    # --------------------------------------------------------- A. understand --

    def understand(self) -> dict:
        from app.agents import prompts

        columns = {c["name"]: c.get("kind") for c in self.profile.get("columns", [])}
        time_cols = [n for n, k in columns.items() if k == "datetime"]
        if not time_cols:
            raise CannotInvestigate("This dataset has no date column, so there are no two periods to compare.")
        approved = "\n".join(f"- {m['name']}: {m['formula']}" for m in self.approved_metrics) or "(none)"
        user = (f"Question: {self.question}\n\nApproved metric definitions that match this question (use these, do not "
                f"redefine them):\n{approved}\n\nDataset profile:\n{self._columns_text()}\n")
        raw = self._ask("submit_understanding", UNDERSTANDING_SCHEMA, prompts.HYP_UNDERSTAND_SYSTEM, user,
                        "Submit the structured reading of the question.") or {}
        assumptions = [str(a)[:300] for a in (raw.get("assumptions") or [])][:8]
        missing = [str(a)[:300] for a in (raw.get("missing_information") or [])][:8]

        # Metric: an approved definition always wins; a model's formula is only a proposal.
        formula_source = ""
        if self.approved_metrics:
            m = self.approved_metrics[0]
            formula_text, label, base_filters = m["formula"], m["name"], list(m.get("filters") or [])
            formula_source = f"approved metric '{m['name']}' (version {m.get('current_version')})"
        else:
            formula_text, label, base_filters = str(raw.get("metric_formula") or ""), str(raw.get("metric_label") or ""), []
            formula_source = "proposed by the model from the question and validated against the schema"
        formula = None
        try:
            formula = me.parse_formula(formula_text)
            if me.validate_against_profile(formula, self.profile):
                formula = None
        except me.FormulaError:
            formula = None
        if formula is None:
            spec = investigator.infer_spec(self.profile, self.question, self.approved_metrics)
            if spec is None:
                raise CannotInvestigate("No metric could be identified from the question on this dataset. "
                                        "Name a numeric column or define the metric in the semantic layer.")
            formula, label = me.parse_formula(spec["formula"]), spec["label"]
            formula_source = "inferred from a column named in the question (the model's formula was not valid for this dataset)"
            assumptions.append(f"The metric was taken to be {formula.text}.")
        label = (label or formula.text)[:120]

        try:
            extra = me.normalize_filters(raw.get("population_filters"), list(columns))
            me.apply_filters(self.df.head(50), extra)
            base_filters = base_filters + extra
        except me.FormulaError as e:
            self.model_notes.append(f"population filters proposed by the model were not valid and were ignored ({e})")

        time_column = raw.get("time_column") if raw.get("time_column") in time_cols else time_cols[0]
        if raw.get("time_column") not in time_cols and len(time_cols) > 1:
            assumptions.append(f"The date column was taken to be '{time_column}' (the dataset has {len(time_cols)}).")

        period_a, period_b, source = self._periods(raw, time_column, base_filters)
        dims_allowed = investigator.candidate_dimensions(self.profile, {time_column, *formula.columns})
        proposed_dims = [d for d in (raw.get("dimensions") or []) if d in dims_allowed]
        understanding = {
            "question": self.question, "metric": {"label": label, "formula": formula.text, "source": formula_source,
                                                  "filters": base_filters},
            "population": "all rows" if not base_filters else "rows where " + " and ".join(
                f"{f['column']} {f['op']} {f['value']}" for f in base_filters),
            "time_column": time_column, "period_a": period_a, "period_b": period_b, "periods_source": source,
            "requested_outcome": str(raw.get("requested_outcome") or "explain the change")[:300],
            "expected_direction": raw.get("expected_direction") if raw.get("expected_direction") in ("decrease", "increase") else "unknown",
            "claimed_change_pct": self._claimed(raw.get("claimed_change_pct")),
            "available_dimensions": dims_allowed, "dimensions_of_interest": proposed_dims,
            "assumptions": assumptions, "missing_information": missing,
        }
        return understanding

    def _claimed(self, value: Any) -> float | None:
        """The size of change the question asserts -- kept only when that number
        is actually written in the question (a model fills an optional number
        field with 0 or a guess when the question states none)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value == 0:
            return None
        return value if re.search(rf"(?<![\d.]){re.escape(f'{abs(value):g}')}(?![\d.]*\d)", self.question) else None

    def _periods(self, raw: dict, time_column: str, base_filters: list[dict]) -> tuple[list[str], list[str], str]:
        ts = pd.to_datetime(me.apply_filters(self.df, base_filters)[time_column], errors="coerce").dropna()
        if ts.empty or ts.min() == ts.max():
            raise CannotInvestigate("The date column has fewer than two distinct moments, so there are no two periods to compare.")
        pa, pb = raw.get("period_a") or [], raw.get("period_b") or []

        def valid(p) -> bool:
            if not (isinstance(p, list) and len(p) == 2):
                return False
            s, e = pd.to_datetime(p[0], errors="coerce"), pd.to_datetime(p[1], errors="coerce")
            return not (pd.isna(s) or pd.isna(e)) and s < e

        if valid(pa) and valid(pb):
            a0, a1, b0, b1 = (pd.Timestamp(x) for x in (*pa, *pb))
            day = pd.Timedelta(days=1)
            # People (and models) write "1 Jan to 28 Feb" meaning both days included. A date-only end
            # is therefore read as the whole of that day, never past the start of the later period.
            if a1 < b0 and a1 == a1.normalize():
                a1 = min(a1 + day, b0)
            if b1 == b1.normalize():
                b1 = b1 + day - pd.Timedelta(seconds=1)
            inside = a0 <= ts.max() and b1 >= ts.min() and a1 <= b0 + pd.Timedelta(seconds=1)
            if inside:
                return [a0.isoformat(), a1.isoformat()], [b0.isoformat(), b1.isoformat()], "stated in the question"
            self.model_notes.append("the periods proposed by the model overlap or fall outside the data and were replaced")
        auto = investigator._auto_periods(me.apply_filters(self.df, base_filters), time_column)
        if auto is None:
            raise CannotInvestigate("Two comparison periods could not be formed from the date column.")
        return auto[0], auto[1], "automatic: the available time range split at its midpoint (the question did not fix the periods)"

    # ------------------------------------------------------------- analyses --

    def run_analysis(self, op: Any, *, requested_by: str, hypothesis_ids: list[str] | None = None) -> dict:
        """Validate and execute one analysis, or return the earlier record of an
        equivalent one. Never raises for a bad proposal or a failed computation:
        the outcome is recorded in the returned record's `status`."""
        self.checkpoint()
        hypothesis_ids = hypothesis_ids or []
        try:
            normalized = ops.normalize(op, self.ctx)
        except ops.OpError as e:
            return self._record(None, op if isinstance(op, dict) else {"op": str(op), "params": {}}, NOT_TESTABLE,
                                {}, f"{e.code}: {e}", requested_by, hypothesis_ids, 0, count=False)
        sig = ops.signature(normalized)
        if sig in self.by_signature:
            existing = self.by_signature[sig]
            for h in hypothesis_ids:
                if h not in existing["hypothesis_ids"]:
                    existing["hypothesis_ids"].append(h)
            self.store.save_analysis(existing)
            return {**existing, "duplicate_of": existing["id"]}
        if self.budget.blocked("analysis"):
            return {"id": None, "status": "skipped_budget", "op": normalized["op"], "params": normalized["params"],
                    "signature": sig, "result": {}, "error": "the analysis budget was exhausted", "hypothesis_ids": hypothesis_ids}
        started = time.monotonic()
        try:
            result = ops.execute(self.df, self.ctx, normalized)
            status, error = "ok", None
        except ops.OpError as e:
            result, status, error = {}, NOT_TESTABLE, f"{e.code}: {e}"
        except me.FormulaError as e:
            result, status, error = {}, NOT_TESTABLE, f"invalid_formula: {e}"
        except Exception as e:  # noqa: BLE001 - a crashing analysis is a failed analysis, not a crashed run
            logger.exception("Analysis %s failed", normalized["op"])
            result, status, error = {}, FAILED, f"{type(e).__name__}: the analysis raised an error"
        return self._record(sig, normalized, status, result, error, requested_by, hypothesis_ids,
                            int((time.monotonic() - started) * 1000))

    def _record(self, sig, op, status, result, error, requested_by, hypothesis_ids, duration_ms, count=True) -> dict:
        if count:
            self.budget.analyses += 1
        record = {"id": f"A{len(self.analyses) + 1}", "signature": sig or "", "op": op.get("op"), "params": op.get("params") or {},
                  "status": status, "result": result, "error": error, "requested_by": requested_by,
                  "hypothesis_ids": list(hypothesis_ids), "duration_ms": duration_ms}
        self.analyses.append(record)
        if sig:
            self.by_signature[sig] = record
        self.store.save_analysis(record)
        return record

    # ------------------------------------------------------------ B. generate --

    def _facts_for_prompt(self) -> str:
        lines = []
        for a in self.analyses:
            if a["status"] != "ok":
                lines.append(f"{a['id']} {a['op']} {a['params']}: {a['status']} ({a['error']})")
                continue
            facts = _rounded({k: v for k, v in a["result"]["facts"].items() if k not in ("identity",)})
            table = _rounded(a["result"].get("table"))
            lines.append(f"{a['id']} {a['op']} {a['params']} -> {facts}" + (f" top rows: {table[:5]}" if table else ""))
        return "\n".join(lines)[:9000]

    def generate_hypotheses(self) -> None:
        from app.agents import prompts

        u = self.understanding
        user = (f"Question: {self.question}\nMetric: {u['metric']['label']} = {u['metric']['formula']}\n"
                f"Earlier period: {u['period_a']}   Later period: {u['period_b']}   Date column: {u['time_column']}\n"
                f"Columns usable as segment dimensions: {u['available_dimensions']}\n\n"
                f"Facts already computed (real results):\n{self._facts_for_prompt()}\n\n"
                f"Dataset profile:\n{self._columns_text()}\n\nAvailable analyses:\n{ops.CATALOGUE}\n"
                f"Propose at most {config.HYP_MAX_HYPOTHESES} distinct, testable hypotheses.")
        raw = self._ask("submit_hypotheses", HYPOTHESES_SCHEMA, prompts.HYP_GENERATE_SYSTEM, user,
                        "Submit hypotheses, each with the one analysis that would test it.")
        items = (raw or {}).get("hypotheses")
        if not isinstance(items, list):
            if not self.model_unavailable:
                self.model_notes.append("the model returned no usable list of hypotheses")
            items = []
        columns = {c["name"] for c in self.profile.get("columns", [])}
        seen_statements, seen_tests, accepted = set(), set(), []
        for item in items[:20]:
            if not isinstance(item, dict):
                self.rejected.append({"statement": str(item)[:200], "reason": "malformed: not an object"})
                continue
            statement = str(item.get("statement") or "").strip()[:400]
            if not statement or not str(item.get("measurable_prediction") or "").strip():
                self.rejected.append({"statement": statement or "(empty)", "reason": "untestable: no statement or no measurable prediction"})
                continue
            missing = [c for c in (item.get("required_columns") or []) if c not in columns]
            if missing:
                self.rejected.append({"statement": statement, "reason": "schema-incompatible: column(s) not in the dataset: "
                                                                        + ", ".join(map(str, missing[:5]))})
                continue
            try:
                analysis = ops.normalize(item.get("analysis"), self.ctx)
                prediction = ops.validate_prediction(analysis["op"], item.get("prediction"))
            except ops.OpError as e:
                self.rejected.append({"statement": statement, "reason": f"untestable ({e.code}): {e}"})
                continue
            key = _norm_statement(statement)
            test_key = (ops.signature(analysis), prediction["expect"])
            if key in seen_statements or test_key in seen_tests:
                self.rejected.append({"statement": statement, "reason": "duplicate of another hypothesis (same statement or same test)"})
                continue
            seen_statements.add(key)
            seen_tests.add(test_key)
            priority = item.get("priority") if isinstance(item.get("priority"), int) and 1 <= item["priority"] <= 5 else 3
            accepted.append({
                "statement": statement, "measurable_prediction": str(item["measurable_prediction"])[:400],
                "analysis": analysis, "prediction": prediction,
                "required_columns": [str(c) for c in (item.get("required_columns") or [])][:12],
                "confounders": [str(c)[:300] for c in (item.get("confounders") or [])][:6],
                "priority": priority, "rationale": str(item.get("rationale") or "")[:400],
            })
        accepted.sort(key=lambda h: -h["priority"])
        for dropped in accepted[config.HYP_MAX_HYPOTHESES:]:
            self.rejected.append({"statement": dropped["statement"], "reason": "over the hypothesis limit (lower priority)"})
        for i, h in enumerate(accepted[:config.HYP_MAX_HYPOTHESES], start=1):
            hyp = {"id": f"H{i}", **h, "status": PROPOSED, "evidence": [], "reasons": [], "effect": {}}
            self.hypotheses.append(hyp)
            self.store.save_hypothesis(hyp)

    # ------------------------------------------------------------ D. evaluate --

    def test_hypothesis(self, hyp: dict, analysis_op: dict, prediction: dict, requested_by: str,
                        role: str = "test") -> None:
        """Run one analysis for a hypothesis and re-settle its status.

        role="refinement" marks an analysis that looks *inside* a finding (a
        drill-down into one segment). It is recorded with the hypothesis but
        does not move its status: where inside West the change sits says
        nothing about whether the change sits in West."""
        self.progress(f"Testing {hyp['id']}: {hyp['statement'][:110]}")
        record = self.run_analysis(analysis_op, requested_by=requested_by, hypothesis_ids=[hyp["id"]])
        if record["status"] == "skipped_budget":
            hyp["evidence"].append({"analysis_id": None, "assessment": "not_run", "reasons": [record["error"]]})
        elif record["status"] != "ok":
            hyp["evidence"].append({"analysis_id": record["id"], "assessment": record["status"], "reasons": [record["error"] or ""]})
        else:
            verdict = ops.evaluate({"op": record["op"], "params": record["params"]}, record["result"], prediction)
            hyp["evidence"].append({"analysis_id": record["id"], "assessment": verdict["status"], "reasons": verdict["reasons"],
                                    "effect": verdict["effect"], "prediction": prediction["expect"],
                                    "warnings": record["result"].get("warnings") or []})
        hyp["evidence"][-1]["role"] = role
        self._settle(hyp)

    def _settle(self, hyp: dict) -> None:
        """A hypothesis's status from all the evidence gathered for it."""
        tests = [e for e in hyp["evidence"] if e.get("role", "test") == "test"]
        judged = [e for e in tests if e["assessment"] in (SUPPORTED, WEAKENED, INCONCLUSIVE)]
        kinds = {e["assessment"] for e in judged}
        if not judged:
            attempts = {e["assessment"] for e in tests}
            # Nothing ran to a result: that is absence of evidence, never evidence against.
            hyp["status"] = FAILED if FAILED in attempts else NOT_TESTABLE if NOT_TESTABLE in attempts else PROPOSED
        elif SUPPORTED in kinds and WEAKENED in kinds:
            hyp["status"] = INCONCLUSIVE
        elif SUPPORTED in kinds:
            hyp["status"] = SUPPORTED
        elif WEAKENED in kinds:
            hyp["status"] = WEAKENED
        else:
            hyp["status"] = INCONCLUSIVE
        hyp["reasons"] = [r for e in tests for r in e.get("reasons", []) if r]
        hyp["refinements"] = [r for e in hyp["evidence"] if e.get("role") == "refinement" for r in e.get("reasons", []) if r]
        if SUPPORTED in kinds and WEAKENED in kinds:
            hyp["reasons"].insert(0, "the evidence points both ways")
        effects = [e["effect"] for e in judged if e.get("effect")]
        hyp["effect"] = effects[0] if effects else {}
        self.store.save_hypothesis(hyp)

    # --------------------------------------------------------------- E. adapt --

    def _state_for_prompt(self) -> str:
        lines = []
        for h in self.hypotheses:
            lines.append(f"{h['id']} [{h['status']}] {h['statement']} | prediction: {h['prediction']['expect']} | "
                         f"evidence: {'; '.join(h['reasons'][:3]) or 'none yet'}"
                         + (f" | inside it: {'; '.join(h['refinements'][:2])}" if h.get("refinements") else ""))
        return "\n".join(lines)

    def adapt(self) -> None:
        from app.agents import prompts

        stale = 0
        while True:
            unresolved = [h for h in self.hypotheses if h["status"] in (PROPOSED, INCONCLUSIVE)]
            limit = self.budget.blocked("llm") or self.budget.blocked("analysis")
            if limit:
                self.stop_reason = f"budget:{limit}"
                return
            if self.budget.rounds >= self.budget.max_rounds:
                self.stop_reason = "budget:rounds"
                return
            if not unresolved and not any(h["status"] == SUPPORTED for h in self.hypotheses):
                self.stop_reason = "nothing_left_to_test"
                return
            self.budget.rounds += 1
            self.progress(f"Round {self.budget.rounds}: choosing the next analysis")
            user = (f"Question: {self.question}\nMetric: {self.understanding['metric']['formula']}\n"
                    f"Segment dimensions available: {self.understanding['available_dimensions']}\n\n"
                    f"Hypotheses and their current status:\n{self._state_for_prompt()}\n\n"
                    f"Analyses already executed (do not repeat an equivalent one):\n{self._facts_for_prompt()}\n\n"
                    f"Available analyses:\n{ops.CATALOGUE}\n"
                    "Choose the single analysis most likely to distinguish the competing hypotheses, or stop.")
            step = self._ask("submit_next_step", NEXT_STEP_SCHEMA, prompts.HYP_NEXT_SYSTEM, user,
                             "Choose the next analysis, or stop.")
            if step is None:
                if self.model_unavailable:
                    self.stop_reason = "model_unavailable"
                    self.limitations.append("The model became unavailable during the investigation, so no further "
                                            "analyses were chosen.")
                else:
                    self.stop_reason = (f"budget:{self.budget.blocked('llm')}" if self.budget.blocked("llm")
                                        else "model_gave_no_step")
                return
            if step.get("decision") != "analyze":
                # The model's own wording is not a verified statement, so it stays out of the report body.
                self.stop_reason = "model_stopped"
                self.model_notes.append("the model's reason for stopping (unverified wording): " + str(step.get("rationale") or "")[:240])
                return
            hyp = next((h for h in self.hypotheses if h["id"] == step.get("hypothesis_id")), None)
            problem = None
            if hyp is None:
                problem = f"it named an unknown hypothesis ({step.get('hypothesis_id')})"
            else:
                try:
                    analysis = ops.normalize(step.get("analysis"), self.ctx)
                    prediction = ops.validate_prediction(analysis["op"], step.get("prediction"))
                    if ops.signature(analysis) in self.by_signature and any(
                            e.get("analysis_id") == self.by_signature[ops.signature(analysis)]["id"] for e in hyp["evidence"]):
                        problem = "it repeated an analysis already used for this hypothesis"
                except ops.OpError as e:
                    problem = f"its analysis was not valid ({e.code}: {e})"
            if problem:
                stale += 1
                self.model_notes.append(f"round {self.budget.rounds}: the model's proposal was discarded because {problem}")
                if stale >= 2:
                    self.stop_reason = "no_new_analysis: the model's proposals stopped being new or valid"
                    return
                continue
            stale = 0
            before = hyp["status"]
            drill_down = analysis["op"] == "segment_contribution" and bool(analysis["params"].get("filters"))
            self.test_hypothesis(hyp, analysis, prediction, "follow_up", role="refinement" if drill_down else "test")
            hyp.setdefault("follow_ups", []).append({"round": self.budget.rounds, "question": str(step.get("unresolved_question") or "")[:300],
                                                     "rationale": str(step.get("rationale") or "")[:300],
                                                     "status_before": before, "status_after": hyp["status"]})
            self.store.save_hypothesis(hyp)

    # ------------------------------------------------------------- F. report --

    def _determination(self, totals: dict) -> dict:
        supported = [h for h in self.hypotheses if h["status"] == SUPPORTED]
        integrity = [h for h in supported if h["analysis"]["op"] in ops.DATA_INTEGRITY_OPS]
        baseline_integrity = [a for a in self.analyses if a["requested_by"] == "baseline" and a["status"] == "ok"
                              and (a["result"]["facts"].get("incomplete") or a["result"]["facts"].get("issues_present"))]
        pct = totals.get("change_pct")
        if not any(a["op"] == "metric_change" for a in self.analyses):
            return {"code": "not_measured",
                    "text": "The investigation stopped before the metric was measured, so nothing can be said about the change."}
        if totals.get("a") is None or totals.get("b") is None:
            return {"code": "no_metric_value", "text": "The metric has no value in one of the two periods, so no change could be measured."}
        if totals["change"] == 0 or (pct is not None and abs(pct) < investigator.MATERIAL_CHANGE_PCT):
            return {"code": "no_material_change",
                    "text": f"The metric moved by less than {investigator.MATERIAL_CHANGE_PCT:g}% between the periods; there is no material change to explain."}
        if integrity or baseline_integrity:
            return {"code": "data_cannot_support_conclusion",
                    "text": "The cause could not be determined: the comparison itself is unreliable (incomplete periods or "
                            "data-quality problems), so the change may be an artefact of the data rather than of the business."}
        business = [h for h in supported if h["analysis"]["op"] not in ops.DATA_INTEGRITY_OPS]
        if business:
            return {"code": "explanations_supported",
                    "text": f"{len(business)} hypothesis(es) are supported by the data. They describe where the change is; "
                            "they do not prove what caused it."}
        return {"code": "cause_not_determined",
                "text": "The cause could not be determined: no tested hypothesis is supported by the data."}

    def _numbers_pool(self, totals: dict) -> list[float]:
        pool = verification.extract_numbers([a["result"] for a in self.analyses if a["status"] == "ok"])
        pool += verification.extract_numbers(totals)
        pool += derived_numbers(pool)

        pool += [float(len(self.hypotheses)), float(len([h for h in self.hypotheses if h["status"] == SUPPORTED]))]
        return pool

    def _template_summary(self, totals: dict, determination: dict) -> str:
        label = self.understanding["metric"]["label"]
        if totals.get("a") is None or totals.get("b") is None:
            return determination["text"]
        direction = "rose" if totals["change"] > 0 else "fell" if totals["change"] < 0 else "did not change"
        pct = f" ({totals['change_pct']:+.1f}%)" if totals.get("change_pct") is not None else ""
        text = f"{label} {direction} from {_fmt(totals['a'])} to {_fmt(totals['b'])}{pct}. {determination['text']}"
        for h in [h for h in self.hypotheses if h["status"] == SUPPORTED][:3]:
            text += f" Supported: {h['statement']} ({h['reasons'][0] if h['reasons'] else 'see evidence'})."
        return text

    def _write_summary(self, totals: dict, determination: dict) -> dict:
        from app.agents import prompts

        template = self._template_summary(totals, determination)
        fallback = {"text": template, "source": "template", "possible_explanations": [], "follow_ups": [], "rejected_reason": None}
        if self.budget.blocked("llm"):
            return {**fallback, "rejected_reason": "no model call was left in the budget"}
        user = (f"Question: {self.question}\nDetermination (fixed; do not contradict it): {determination['text']}\n"
                f"Metric totals: {_rounded(totals)}\n\nHypotheses with their final status and the evidence behind it:\n{self._state_for_prompt()}\n\n"
                f"Executed analyses (the only numbers you may use):\n{self._facts_for_prompt()}\n")
        try:
            raw = self._ask("submit_report", REPORT_SCHEMA, prompts.HYP_REPORT_SYSTEM, user, "Submit the report wording.")
        except Exception as e:  # noqa: BLE001 - the report must still be produced from the evidence
            return {**fallback, "rejected_reason": f"the model was unavailable for the summary ({type(e).__name__})"}
        text = str((raw or {}).get("executive_summary") or "").strip()
        if not text:
            return {**fallback, "rejected_reason": "the model returned no summary"}
        tokens = verification.narrative_numbers(text, self.question)
        pool = self._numbers_pool(totals)
        unsupported = [t for t in tokens if not verification.number_matches(t, pool)[0]]
        if unsupported:
            return {**fallback, "rejected_reason": "the model's summary contained numbers that no executed analysis produced: "
                                                   + ", ".join(unsupported[:6])}
        if determination["code"] != "explanations_supported" and re.search(
                r"\b(the cause (is|was)|was caused by|because of|due to)\b", text, re.IGNORECASE):
            return {**fallback, "rejected_reason": "the model's summary asserted a cause although none was determined"}

        def grounded(lines) -> list[str]:
            """Explanations and suggestions are wording, not findings: a line carrying a figure
            that no analysis produced is dropped."""
            kept = []
            for line in (lines if isinstance(lines, list) else [])[:6]:
                line = str(line)[:300]
                if all(verification.number_matches(t, pool)[0] for t in verification.narrative_numbers(line, self.question)):
                    kept.append(line)
            return kept

        return {"text": text[:1800], "source": "model", "rejected_reason": None,
                "possible_explanations": grounded(raw.get("possible_explanations")),
                "follow_ups": grounded(raw.get("follow_up_analyses"))}

    def build_report(self) -> dict:
        u = self.understanding
        base = next((a for a in self.analyses if a["op"] == "metric_change" and a["status"] == "ok"), None)
        facts = base["result"]["facts"] if base else {}
        totals = {"a": facts.get("a"), "b": facts.get("b"), "change": facts.get("change"), "change_pct": facts.get("change_pct"),
                  "rows_a": base["result"]["rows_a"] if base else 0, "rows_b": base["result"]["rows_b"] if base else 0}
        determination = self._determination(totals)
        self.progress("Writing the report")
        summary = self._write_summary(totals, determination)

        order = {SUPPORTED: 0, INCONCLUSIVE: 1, WEAKENED: 2, PROPOSED: 3, NOT_TESTABLE: 4, FAILED: 5}
        ranked = sorted(self.hypotheses, key=lambda h: (order.get(h["status"], 9), -abs((h.get("effect") or {}).get("value") or 0)))
        observed = []
        if totals["a"] is not None and totals["b"] is not None:
            observed.append(f"{u['metric']['label']} was {_fmt(totals['a'])} in the earlier period and {_fmt(totals['b'])} in the later one"
                            + (f" ({totals['change_pct']:+.1f}%)." if totals.get("change_pct") is not None else "."))
        claimed = u.get("claimed_change_pct")
        if claimed is not None and totals.get("change_pct") is not None:
            observed.append(f"The question describes a change of about {abs(claimed):g}%; the data shows {totals['change_pct']:+.1f}%.")
        for a in self.analyses:  # what the engine's own scan found, whether or not a hypothesis asked
            if a["requested_by"] != "scan" or a["status"] != "ok":
                continue
            verdict = ops.evaluate({"op": a["op"], "params": a["params"]}, a["result"], {"expect": "concentrated"})
            if verdict["status"] == SUPPORTED:
                covered = any(h["status"] == SUPPORTED and h["analysis"]["op"] == "segment_contribution"
                              and h["analysis"]["params"].get("dimension") == a["params"]["dimension"] for h in self.hypotheses)
                observed.append(f"{a['id']}: {verdict['reasons'][0]}." + ("" if covered else " No supported hypothesis covers this."))
        interpretations, quality = [], []
        for a in self.analyses:
            if a["status"] != "ok":
                continue
            f = a["result"]["facts"]
            test = f.get("test") or {}
            if test.get("p_value") is not None:
                interpretations.append(f"{a['id']} ({a['op']}): {test['method']}, p = {test['p_value']:.4f}"
                                       + (f", effect size d = {test['cohens_d']:.2f}" if test.get("cohens_d") is not None else "") + ".")
            elif test.get("skipped"):
                interpretations.append(f"{a['id']} ({a['op']}): no significance test ({test['skipped']}).")
            quality += [f"{a['id']}: {w}" for w in a["result"].get("warnings") or []]
            for issue in f.get("issues") or []:
                quality.append(f"{a['id']}: {issue}")
        limitations = list(self.limitations) + [
            "Hypotheses are judged on whether the data is consistent with them. Consistency is not proof of cause.",
            "Only explanations that can be expressed with the columns in this dataset were tested.",
        ]
        untested = [h["id"] for h in self.hypotheses if h["status"] in (PROPOSED, NOT_TESTABLE, FAILED)]
        if untested:
            limitations.append("Not tested, and therefore neither supported nor weakened: " + ", ".join(untested) + ".")
        if self.stop_reason.startswith("budget:"):
            limitations.append(f"The investigation stopped at its {self.stop_reason.split(':', 1)[1].replace('_', ' ')} limit.")

        segment = next((a for a in self.analyses if a["op"] == "segment_contribution" and a["status"] == "ok"), None)
        charts = []
        if segment and segment["result"].get("table"):
            rows = segment["result"]["table"][:10]
            charts.append({"title": f"Contribution to the change by {segment['params']['dimension']}", "plotly_json": {
                "data": [{"type": "bar", "orientation": "h", "y": [str(r["segment"]) for r in rows][::-1],
                          "x": [r["contribution"] for r in rows][::-1]}],
                "layout": {"title": {"text": f"Contribution to the change by {segment['params']['dimension']}"},
                           "xaxis": {"title": {"text": "Contribution"}}, "margin": {"l": 140}}}})
        counts = {s: sum(1 for h in self.hypotheses if h["status"] == s) for s in (SUPPORTED, WEAKENED, INCONCLUSIVE, NOT_TESTABLE, FAILED, PROPOSED)}
        conclusive = determination["code"] == "explanations_supported"
        return {
            "engine": "hypothesis", "engine_version": ENGINE_VERSION, "question": self.question,
            "status": "complete" if conclusive else "inconclusive",
            "summary": summary["text"], "executive_summary": summary, "determination": determination,
            "understanding": u,
            "metric": {"label": u["metric"]["label"], "formula": u["metric"]["formula"], "filters": u["metric"]["filters"],
                       "source": u["metric"]["source"]},
            "periods": {"a": u["period_a"], "b": u["period_b"], "time_column": u["time_column"], "source": u["periods_source"]},
            "totals": totals, "findings": [{"hypothesis_id": h["id"], "status": h["status"], "statement": h["statement"],
                                            "evidence": h["reasons"][:3], "effect": h.get("effect") or {}} for h in ranked],
            "hypotheses": ranked, "rejected_hypotheses": self.rejected, "hypothesis_counts": counts,
            "analyses": self.analyses,
            "observed_facts": observed, "statistical_interpretations": interpretations,
            "possible_explanations": summary["possible_explanations"],
            "data_quality": quality, "limitations": limitations, "model_notes": self.model_notes,
            "follow_ups": summary["follow_ups"], "charts": charts, "caveats": investigator.CAVEATS[:1],
            "stop_reason": self.stop_reason, "budget": self.budget.used(), "resumed": self.resumed,
            "reproducibility": {**self.provenance, "engine_version": ENGINE_VERSION,
                                "analysis_ids": [a["id"] for a in self.analyses],
                                "analysis_signatures": {a["id"]: a["signature"] for a in self.analyses if a["signature"]},
                                "evidence_thresholds": {"material_pct": config.HYP_MATERIAL_PCT, "strong_share": config.HYP_STRONG_SHARE,
                                                        "weak_share": config.HYP_WEAK_SHARE, "min_rows": config.HYP_MIN_ROWS,
                                                        "alpha": config.HYP_SIGNIFICANCE_ALPHA},
                                "note": "Every analysis is deterministic: re-running an analysis with the same operation on the "
                                        "same dataset version reproduces its result. The choice of hypotheses and the wording "
                                        "come from a model and can differ between runs."},
        }

    # ------------------------------------------------------------------- run --

    def _restore(self, saved: dict) -> None:
        """Pick up a run that was interrupted: reuse the understanding, the
        hypotheses and every analysis already executed."""
        self.resumed = True
        self.understanding = saved["understanding"]
        self.analyses = [dict(a) for a in saved["analyses"]]
        self.by_signature = {a["signature"]: a for a in self.analyses if a["signature"]}
        self.hypotheses = [dict(h) for h in saved["hypotheses"]]

    def _make_context(self) -> None:
        u = self.understanding
        self.ctx = ops.AnalysisContext(
            formula=me.parse_formula(u["metric"]["formula"]), label=u["metric"]["label"],
            base_filters=u["metric"]["filters"], time_column=u["time_column"], period_a=u["period_a"],
            period_b=u["period_b"], profile=self.profile, dq_messages=self.dq_messages)

    def run(self) -> dict:
        saved = self.store.load()
        if saved.get("understanding"):
            self._restore(saved)
        else:
            self.progress("Reading the question")
            self.understanding = self.understand()
            self.store.save_understanding(self.understanding)
        self._make_context()

        self.progress("Measuring the change and checking the data")
        base = self.run_analysis({"op": "metric_change", "params": {}}, requested_by="baseline")
        if base["status"] == "ok":
            self.ctx.total_a, self.ctx.total_b = base["result"]["facts"]["a"], base["result"]["facts"]["b"]
        coverage = self.run_analysis({"op": "period_coverage", "params": {}}, requested_by="baseline")
        self.run_analysis({"op": "data_quality", "params": {}}, requested_by="baseline")
        if coverage["status"] == "ok" and coverage["result"]["facts"]["incomplete"]:
            self.limitations.append("The two periods are not directly comparable: " + "; ".join(coverage["result"]["facts"]["issues"]) + ".")

        facts = base["result"]["facts"] if base["status"] == "ok" else {}
        pct = facts.get("change_pct")
        nothing_to_explain = (base["status"] != "ok" or facts.get("a") is None or facts.get("b") is None or facts.get("change") == 0
                              or (pct is not None and abs(pct) < investigator.MATERIAL_CHANGE_PCT))
        if base["status"] == "skipped_budget":
            # Nothing was measured, so nothing may be said about the metric.
            self.stop_reason = f"budget:{self.budget.blocked('analysis')}"
            return self.build_report()
        if nothing_to_explain:
            self.stop_reason = "no_material_change"
            return self.build_report()

        if not self.hypotheses:
            # Before asking for explanations, measure where the change sits. The model then
            # reasons from these facts, and a concentration it overlooks is still reported.
            self.progress("Locating the change across segments")
            u = self.understanding
            wanted = [d for d in u.get("dimensions_of_interest") or [] if d in u["available_dimensions"]]
            for dim in (wanted + [d for d in u["available_dimensions"] if d not in wanted])[:config.HYP_SCAN_DIMENSIONS]:
                self.run_analysis({"op": "segment_contribution", "params": {"dimension": dim}}, requested_by="scan")
            self.progress("Generating hypotheses")
            self.generate_hypotheses()
        if not self.hypotheses and self.model_unavailable:
            self.stop_reason = "model_unavailable"
            self.limitations.append("The model became unavailable before any hypothesis was proposed.")
            return self.build_report()
        if not self.hypotheses:
            self.stop_reason = "no_testable_hypotheses"
            self.limitations.append("The model proposed no hypothesis that could be tested on this dataset.")
            return self.build_report()

        for hyp in self.hypotheses:
            if hyp["status"] != PROPOSED or hyp["evidence"]:
                continue  # already tested before an interruption
            if self.budget.blocked("analysis"):
                self.stop_reason = f"budget:{self.budget.blocked('analysis')}"
                break
            self.test_hypothesis(hyp, hyp["analysis"], hyp["prediction"], "hypothesis")
        if not self.stop_reason:
            self.adapt()
        return self.build_report()
