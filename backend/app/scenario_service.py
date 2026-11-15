"""The one place a what-if calculation is run, verified and explained.

Both interfaces call this module: the What-If API (app/routers/scenarios.py) and the Silt Copilot (app/analyst/
scenario_bridge.py). Neither has formulas, validation rules or assumptions of its own, so the same request gives the same
numbers, the same checks and the same explanation whichever way it arrived.

    request (typed form, or a sentence parsed by scenario_nl)
        -> validated structured request
        -> deterministic engines (scenarios, scenario_sim, scenario_solve, scenario_sensitivity)
        -> independent verification (scenario_verify)
        -> plain-English explanation (scenario_explain)
        -> response

No step here asks a language model for a number.
"""
from __future__ import annotations

import logging
from typing import Any

from app import scenario_explain, scenario_sensitivity, scenario_sim, scenario_solve, scenario_verify, scenarios
from app.scenarios import ScenarioError

logger = logging.getLogger(__name__)
SERVICE_VERSION = "scenario-service.1"


def explained(result: dict) -> dict:
    """Attach the explanation. A failure to explain never hides a result that was computed and verified."""
    try:
        result["explanation"] = scenario_explain.explain(result)
    except Exception:  # noqa: BLE001
        logger.exception("scenario explanation could not be written")
        result["explanation"] = {"version": scenario_explain.EXPLAIN_VERSION, "source": "unavailable", "summary": "", "sections": {}}
    return result


def evaluate(model: "str | dict", baseline: dict, scenario_adjustments: dict | None = None, assumptions: dict | None = None, *,
             sensitivity_pct: float = 10.0, spread_pct: float | None = None, assumption_sources: dict | None = None) -> dict:
    """Baseline and named scenarios with sensitivity, attribution, break-even, verification and an explanation."""
    named = dict(scenario_adjustments or {})
    if spread_pct is not None:
        named = {**scenarios.standard_scenarios(model, spread_pct), **named}
    out = scenarios.evaluate(model, baseline, named, assumptions, sensitivity_pct)
    out["verification"] = scenario_verify.safe(scenario_verify.verify_evaluation, out, scenarios._model(model))  # noqa: SLF001
    sources = {k: v for k, v in (assumption_sources or {}).items() if k in out["assumptions"]}
    out["assumption_sources"] = {k: ({"source": "manual", **v} if "source" not in v else v) for k, v in sources.items()}
    return explained(out)


def simulate(model: "str | dict", baseline: dict, assumptions: dict | None, *, adjustments: dict | None = None,
             uncertainty: dict | None = None, correlation: dict | None = None, iterations: int = scenario_sim.DEFAULT_ITERATIONS,
             seed: int | None = None, target_output: str | None = None, target_value: float | None = None,
             target_direction: str = "at_least") -> dict:
    out = scenario_sim.simulate(model, baseline, adjustments, assumptions, uncertainty, correlation, iterations, seed, target_output,
                                target_value, target_direction)
    return explained(out)


def goal_seek(model: "str | dict", baseline: dict, target_output: str, target_value: float, solve_for: str, *,
              adjustments: dict | None = None, assumptions: dict | None = None, lower: float | None = None,
              upper: float | None = None, constraints: list | None = None) -> dict:
    out = scenario_solve.goal_seek(model, baseline, target_output, target_value, solve_for, adjustments, assumptions, lower, upper,
                                   constraints)
    return explained(out)


def grid(model: "str | dict", baseline: dict, x_input: str, y_input: str, output: str | None = None, **kw: Any) -> dict:
    return explained(scenario_sensitivity.two_way(model, baseline, x_input, y_input, output, **kw))


def run_intent(model: "str | dict", baseline: dict, assumptions: dict | None, intent: dict) -> dict:
    """Run a structured request produced by the natural-language layer (or typed by hand). The intent is validated against
    the model again here, so a request cannot reach the engines unless it names real inputs, outputs and numbers."""
    from app import scenario_nl
    spec = scenarios._model(model)  # noqa: SLF001
    clean = scenario_nl.validate_intent(intent, spec)
    action = clean["action"]
    if action == "evaluate":
        return evaluate(model, baseline, clean["scenarios"], assumptions)
    if action == "goal_seek":
        g = clean["goal"]
        return goal_seek(model, baseline, g["target_output"], g["target_value"], g["solve_for"], adjustments=clean.get("adjustments"),
                         assumptions=assumptions)
    if action == "break_even":
        return evaluate(model, baseline, {}, assumptions)
    raise ScenarioError(f"The action '{action}' does not run a calculation")


# ------------------------------------------------------------------- the question box --

def ask(model: "str | dict", baseline: dict, assumptions: dict | None, message: str, context: dict | None = None, *,
        use_model: bool = False, translator: Any = None) -> dict:
    """One sentence in, one verified and explained answer out.

    The sentence is read by scenario_nl (rules first; a language model only if asked and only to translate), the request it
    becomes is validated, and the engines compute it through this module like any other request. A question the sentence
    leaves open comes back as a clarification. A follow-up ("why did profit fall?") is answered by re-running the engine on
    the stored adjustments, so no number in the answer comes from the conversation context or from a model."""
    from app import scenario_nl as nl

    spec = scenarios._model(model)  # noqa: SLF001
    baseline = scenarios.validate_inputs(model, baseline)
    assumptions = scenarios.validate_assumptions(model, assumptions)
    parsed = nl.parse(message, spec, context, baseline=baseline, assumptions=assumptions)
    translated_by = None
    if parsed["status"] == "unsupported" and use_model:
        parsed = nl.translate_with_model(message, spec, translator)
        translated_by = parsed.get("translated_by")
    ctx, _ = nl.validate_context(context, spec, baseline, assumptions)
    base = {"engine_version": nl.NL_VERSION, "service_version": SERVICE_VERSION, "message": message, "status": parsed["status"],
            "interpretation": parsed.get("interpretation", []), "notes": parsed.get("notes", []), "translated_by": translated_by,
            "model": spec["key"], "context": context if ctx is None and not context else (
                {"model": spec["key"], "baseline_sha": nl.baseline_sha(spec, baseline, assumptions), "history": (ctx or {}).get("history", []),
                 "last": (ctx or {}).get("last")} if ctx else None)}
    if parsed["status"] == "clarify":
        return {**base, "clarification": parsed["clarification"], "answer": parsed["clarification"]["question"], "result": None}
    if parsed["status"] == "unsupported":
        return {**base, "answer": parsed["message"], "examples": parsed.get("examples", nl.EXAMPLES), "result": None}
    if parsed["status"] == "explain":
        last = (ctx or {}).get("last")
        result = evaluate(model, baseline, {last["name"]: last["adjustments"]}, assumptions)
        text = _follow_up(result, last["name"], parsed["question"])
        return {**base, "status": "answered", "answer": text, "result": result}
    intent = nl.validate_intent(parsed["intent"], spec)
    result = run_intent(model, baseline, assumptions, intent)
    action = intent["action"]
    if action == "evaluate":
        names = list(intent["scenarios"])
        last_name = names[-1]
        new_ctx = None
        for name in names:
            new_ctx = nl.next_context(spec, baseline, assumptions, new_ctx or ctx, name, intent["scenarios"][name], {})
        answer = result["explanation"]["summary"]
        return {**base, "intent": intent, "result": result, "answer": answer, "context": new_ctx, "last_scenario": last_name}
    if action == "goal_seek":
        return {**base, "intent": intent, "result": result, "answer": result["explanation"]["summary"] or result["message"]}
    be = (result.get("break_even") or {}).get("baseline")
    answer = be["explanation"] if be else ("A custom model has no built-in break-even. Ask what value of an input makes your profit "
                                           "output 0, for example: what price would make profit 0?")
    return {**base, "intent": intent, "result": result, "answer": answer}


def _follow_up(result: dict, name: str, question: str) -> str:
    """'Why did it change?' and 'which input matters most?', from the engine's own attribution for the last scenario."""
    from app import scenario_explain as ex
    p = result["primary_output"]
    label = result["output_labels"][p].lower()
    s = result["scenarios"][name]
    base, new = result["baseline"]["outputs"][p], s["outputs"][p]
    il = result.get("input_labels", {})
    contrib = s["contributions"]["by_input"]
    drivers = ex._drivers(contrib, il)  # noqa: SLF001
    if question == "drivers":
        if not drivers:
            return "Nothing has changed in the last scenario, so no input accounts for any change."
        return f"For '{name}': " + " ".join(drivers[:2])
    head = f"In '{name}', {label} {ex._move(base, new)} from {ex.n(base)} to {ex.n(new)}."  # noqa: SLF001
    if base == new:
        return head + " The inputs that changed offset each other exactly."
    return head + " " + " ".join(drivers) + " This describes the arithmetic of the numbers you set; it does not say why those inputs would move."
