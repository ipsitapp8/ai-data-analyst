"""A small, safe formula language for custom what-if models.

A custom model is a set of named inputs, assumptions and outputs, where every output is a formula over those names:

    mrr           = customers * price
    churned       = customers * churn_rate * price
    net_revenue   = mrr - churned

Formulas are parsed with Python's `ast` and *never* evaluated as Python. Only numbers, names, `+ - * /`, a power with a
constant exponent, parentheses and a short list of math functions are accepted. Attribute access, subscripts, lambdas,
comprehensions, strings and every other call are rejected when the formula is parsed, so a formula (typed by a user, or
proposed by a model) can compute but cannot do anything else.

This module owns: parsing, the dependency order of outputs (with a clear error naming any cycle), evaluation on numbers
or numpy arrays (so one definition serves a single scenario and a Monte Carlo run), and a dimensional check of units.
The independent plain-Python evaluator used for verification lives in app/scenario_verify.py and shares none of this
code, only the definition it checks.
"""
from __future__ import annotations

import ast
import difflib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

MAX_FORMULA_CHARS = 300
MAX_NODES = 120
MAX_DEPTH = 20
MAX_POWER = 10.0
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
FUNCTIONS = {"abs": (1, 1), "sqrt": (1, 1), "log": (1, 1), "exp": (1, 1), "floor": (1, 1), "ceil": (1, 1),
             "min": (2, 8), "max": (2, 8)}
COMPARATORS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
DIMENSIONLESS = {"", "1", "fraction", "ratio", "percent", "%", "rate", "share"}
_UNIT_TERM = r"[a-z][a-z0-9_]*(?:\^-?\d+)?"
_UNIT_RE = re.compile(rf"{_UNIT_TERM}(?:\s*[*/]\s*{_UNIT_TERM})*")


class FormulaError(ValueError):
    """The formula, a name or a unit is not valid; the message says why."""


@dataclass(frozen=True)
class Parsed:
    text: str
    tree: Any          # ("num", v) | ("var", name) | ("neg", x) | ("bin", op, l, r) | ("pow", base, n) | ("call", fn, [args])
    names: frozenset


# ------------------------------------------------------------------ parsing --

def _walk(n: ast.AST, names: set, count: list, depth: int = 0) -> Any:
    count[0] += 1
    if count[0] > MAX_NODES:
        raise FormulaError(f"The formula is too long (more than {MAX_NODES} parts)")
    if depth > MAX_DEPTH:
        raise FormulaError("The formula is nested too deeply")
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
            raise FormulaError("Only numbers are allowed as constants (no text or true/false)")
        if not math.isfinite(float(n.value)):
            raise FormulaError("Numbers in a formula must be finite")
        return ("num", float(n.value))
    if isinstance(n, ast.Name):
        if n.id in FUNCTIONS:
            raise FormulaError(f"'{n.id}' is a function: write {n.id}(…) with brackets")
        names.add(n.id)
        return ("var", n.id)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        inner = _walk(n.operand, names, count, depth + 1)
        return ("neg", inner) if isinstance(n.op, ast.USub) else inner
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        return ("bin", _BIN[type(n.op)], _walk(n.left, names, count, depth + 1), _walk(n.right, names, count, depth + 1))
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
        exp = n.right
        if isinstance(exp, ast.UnaryOp) and isinstance(exp.op, ast.USub) and isinstance(exp.operand, ast.Constant):
            power = -float(exp.operand.value) if isinstance(exp.operand.value, (int, float)) else None
        elif isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)) and not isinstance(exp.value, bool):
            power = float(exp.value)
        else:
            power = None
        if power is None or not math.isfinite(power) or abs(power) > MAX_POWER:
            raise FormulaError(f"A power needs a constant exponent between -{MAX_POWER:g} and {MAX_POWER:g}, for example x ** 2")
        return ("pow", _walk(n.left, names, count, depth + 1), power)
    if isinstance(n, ast.Call):
        if not isinstance(n.func, ast.Name) or n.func.id not in FUNCTIONS:
            raise FormulaError("Only these functions are allowed: " + ", ".join(sorted(FUNCTIONS)))
        if n.keywords:
            raise FormulaError(f"{n.func.id}() takes plain arguments only")
        lo, hi = FUNCTIONS[n.func.id]
        if not lo <= len(n.args) <= hi:
            raise FormulaError(f"{n.func.id}() takes {lo if lo == hi else f'{lo} to {hi}'} argument(s), not {len(n.args)}")
        return ("call", n.func.id, [_walk(a, names, count, depth + 1) for a in n.args])
    raise FormulaError("Only numbers, names, + - * /, a power with a constant exponent, brackets and "
                       + ", ".join(sorted(FUNCTIONS)) + " are allowed")


def parse(text: Any) -> Parsed:
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("The formula is empty")
    if len(text) > MAX_FORMULA_CHARS:
        raise FormulaError(f"The formula is longer than {MAX_FORMULA_CHARS} characters")
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"The formula is not valid: {e.msg}") from e
    names: set = set()
    tree = _walk(node, names, [0])
    return Parsed(text.strip(), tree, frozenset(names))


def parse_constraint(text: Any) -> tuple[str, Any, Any, frozenset]:
    """`left <op> right` with one comparison. Returns (op, left_tree, right_tree, names)."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_FORMULA_CHARS:
        raise FormulaError("A constraint needs a short comparison such as: price >= unit_cost")
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"The constraint is not valid: {e.msg}") from e
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or type(node.ops[0]) not in COMPARATORS:
        raise FormulaError("A constraint is one comparison (<, <=, >, >=, ==, !=), for example: price >= unit_cost")
    names: set = set()
    count = [0]
    left, right = _walk(node.left, names, count), _walk(node.comparators[0], names, count)
    return COMPARATORS[type(node.ops[0])], left, right, frozenset(names)


# ----------------------------------------------------------- dependency order --

def order_outputs(deps: dict[str, set], known: set[str]) -> list[str]:
    """Outputs in a computable order. `deps[output]` are the names it uses; names in `known` (inputs, assumptions) are
    available from the start. A cycle raises with the loop spelled out; an unknown name raises with a suggestion."""
    for out, used in deps.items():
        for name in sorted(used):
            if name not in known and name not in deps:
                near = difflib.get_close_matches(name, sorted(known | set(deps)), n=1)
                raise FormulaError(f"The formula for '{out}' uses '{name}', which is not an input, assumption or output"
                                   + (f" (did you mean '{near[0]}'?)" if near else ""))
    order: list[str] = []
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            loop = stack[stack.index(node):] + [node]
            raise FormulaError("These outputs depend on each other in a circle: " + " → ".join(loop))
        state[node] = 1
        stack.append(node)
        for name in sorted(deps[node]):
            if name in deps:
                visit(name)
        stack.pop()
        state[node] = 2
        order.append(node)

    for node in deps:
        visit(node)
    return order


# ----------------------------------------------------------------- evaluation --

def _undefined(x):
    """Non-finite results (division by zero, log of a non-positive number, overflow) are 'undefined': NaN."""
    return np.where(np.isfinite(x), x, np.nan)


def evaluate(tree: Any, env: dict[str, Any]) -> Any:
    """Evaluate on numbers or numpy arrays. Undefined results are NaN, never 0."""
    kind = tree[0]
    if kind == "num":
        return tree[1]
    if kind == "var":
        return env[tree[1]]
    if kind == "neg":
        return -evaluate(tree[1], env)
    if kind == "bin":
        a, b = evaluate(tree[2], env), evaluate(tree[3], env)
        with np.errstate(all="ignore"):
            op = tree[1]
            return a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else _divide(a, b)
    if kind == "pow":
        with np.errstate(all="ignore"):
            return _undefined(np.power(np.asarray(evaluate(tree[1], env), dtype=float), tree[2]))
    args = [evaluate(a, env) for a in tree[2]]
    fn = tree[1]
    with np.errstate(all="ignore"):
        if fn == "abs":
            return np.abs(args[0])
        if fn == "sqrt":
            return _undefined(np.sqrt(np.asarray(args[0], dtype=float)))
        if fn == "log":
            a = np.asarray(args[0], dtype=float)
            return _undefined(np.log(np.where(a > 0, a, np.nan)))
        if fn == "exp":
            return _undefined(np.exp(np.asarray(args[0], dtype=float)))
        if fn == "floor":
            return np.floor(args[0])
        if fn == "ceil":
            return np.ceil(args[0])
        out = args[0]
        for a in args[1:]:
            out = np.fmin(out, a) if fn == "min" else np.fmax(out, a)
        return out


def _divide(a, b):
    b = np.asarray(b, dtype=float)
    return _undefined(np.divide(a, np.where(b == 0, np.nan, b)))


def evaluate_scalar(parsed: Parsed, env: dict[str, float]) -> float | None:
    v = evaluate(parsed.tree, env)
    v = float(np.asarray(v, dtype=float))
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------- units --

def parse_unit(text: Any) -> dict[str, Fraction]:
    """Unit text such as "currency/unit" or "units*units" as exponents of base units. Words that mean 'no unit'
    (fraction, ratio, percent, rate, share, 1) are dimensionless: their scale (x100) is not tracked."""
    if text is None:
        return {}
    if not isinstance(text, str) or len(text) > 40:
        raise FormulaError("A unit is short text such as: currency/unit")
    t = text.strip().lower()
    if t in DIMENSIONLESS:
        return {}
    if not _UNIT_RE.fullmatch(t):
        raise FormulaError(f"'{text}' is not a valid unit; use words joined by * and /, for example currency/unit")
    out: dict[str, Fraction] = {}
    sign = 1
    for part in re.split(r"([*/])", t):
        part = part.strip()
        if part in ("*", ""):
            sign = 1 if part == "*" else sign
            continue
        if part == "/":
            sign = -1
            continue
        m = re.fullmatch(r"([a-z][a-z0-9_]*)(?:\^(-?\d+))?", part)
        if not m:
            raise FormulaError(f"'{text}' is not a valid unit; use words joined by * and /, for example currency/unit")
        if m.group(1) in DIMENSIONLESS:
            sign = 1
            continue
        out[m.group(1)] = out.get(m.group(1), Fraction(0)) + sign * Fraction(int(m.group(2) or 1))
        sign = 1
    return {k: v for k, v in out.items() if v != 0}


def _plus(a: dict, b: dict, sign: int = 1) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + sign * v
    return {k: v for k, v in out.items() if v != 0}


def format_unit(dim: dict) -> str:
    if not dim:
        return "no unit"
    top = [f"{k}" if v == 1 else f"{k}^{v}" for k, v in sorted(dim.items()) if v > 0]
    bottom = [f"{k}" if v == -1 else f"{k}^{-v}" for k, v in sorted(dim.items()) if v < 0]
    numerator = "*".join(top) or "1"
    return numerator if not bottom else numerator + "/" + "*".join(bottom)


def unit_of(tree: Any, units: dict[str, dict], problems: list[str]) -> dict:
    """The unit of an expression. Adding or comparing different units, and log/exp of a unit, are recorded as problems."""
    kind = tree[0]
    if kind == "num":
        return {}
    if kind == "var":
        return units[tree[1]]
    if kind == "neg":
        return unit_of(tree[1], units, problems)
    if kind == "bin":
        a, b = unit_of(tree[2], units, problems), unit_of(tree[3], units, problems)
        if tree[1] in "+-":
            zero = (tree[2] == ("num", 0.0)) or (tree[3] == ("num", 0.0))
            if a != b and not zero:
                problems.append(f"{'adds' if tree[1] == '+' else 'subtracts'} {format_unit(a)} and {format_unit(b)}")
            return a or b
        return _plus(a, b, 1 if tree[1] == "*" else -1)
    if kind == "pow":
        return {k: v * Fraction(tree[2]).limit_denominator(100) for k, v in unit_of(tree[1], units, problems).items()}
    fn, args = tree[1], [unit_of(a, units, problems) for a in tree[2]]
    if fn in ("abs", "floor", "ceil"):
        return args[0]
    if fn in ("min", "max"):
        if any(a != args[0] for a in args[1:]):
            problems.append(f"compares different units in {fn}(): " + " vs ".join(format_unit(a) for a in args))
        return args[0]
    if fn == "sqrt":
        return {k: v / 2 for k, v in args[0].items()}
    if args[0]:
        problems.append(f"takes {fn}() of {format_unit(args[0])}; it needs a number with no unit")
    return {}


def unit_problems(parsed: Parsed, units: dict[str, dict], declared: dict) -> list[str]:
    problems: list[str] = []
    got = unit_of(parsed.tree, units, problems)
    if got != declared:
        problems.append(f"works out to {format_unit(got)} but is declared as {format_unit(declared)}")
    return problems
