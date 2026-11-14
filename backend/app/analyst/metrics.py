"""Derived business metrics, resolved deterministically.

A user asks for "labor cost", "conversion rate" or "ROI"; the data has columns such as clock_in, clock_out,
hourly_rate, ad_spend, leads, conversions and revenue. This module:

  1. recognises the requested metric from the question (a catalogue of standard definitions, below);
  2. finds the columns that play each input role, from the meaning of the column names and their data types
     (never from a particular dataset): a pair of clock-time columns, an hourly rate, an amount of revenue, ...;
  3. writes a calculation plan in the calculator's own format (app/analyst/calc.py), so the arithmetic, the
     exclusions, the record bookkeeping and the presentation are the calculator's, and no language model is needed;
  4. asks a specific question when an input is missing or more than one column fits, instead of substituting a
     different quantity.

Standard definitions (what is computed is always shown to the user):
  shift duration      hours from start time to end time, per record
  labor cost          hours (a recorded hours column, else the shift duration) x hourly rate, per record
  overtime            hours beyond a threshold the user states, per record (shift)
  profit              revenue - cost, per record
  profit margin       (total revenue - total cost) / total revenue
  ROI                 (total revenue - total cost) / total cost
  ROAS                total revenue / total cost
  conversion rate     total conversions / total base (visitors, sessions, clicks or leads)
  cost per conversion total cost / total conversions        cost per lead: total cost / total base
  average order value total revenue / number of orders (an orders column, or distinct order ids)
Ratios are pooled (ratio of totals), not an average of per-record ratios; that is stated in every answer.

Records that cannot be valid for the metric (a negative rate, hours or count; an end time before the start) are
left out and listed; values that can legitimately be negative (revenue, cost: refunds, credits) and exact duplicates
(possibly a genuine repeat) are kept and flagged by the calculator's data checks, with the follow-up that excludes
them offered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from app.analyst.data import _words, find_column, kind_of


# ------------------------------------------------------------------- columns --
def _tokens(col: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", _words(str(col))) if t}


_START = {"start", "begin", "begins", "in", "login", "from", "open", "arrival", "arrive", "checkin", "clockin", "on"}
_END = {"end", "ends", "finish", "out", "logout", "to", "close", "departure", "leave", "stop", "checkout", "clockout", "off"}


def _time_like(s: pd.Series) -> bool:
    from app.analyst.calc import _looks_like_time
    if kind_of(s) == "datetime":
        d = pd.to_datetime(s, errors="coerce").dropna()
        return bool(len(d)) and bool((d.dt.hour.ne(0) | d.dt.minute.ne(0)).any())
    return kind_of(s) == "categorical" and len(s.dropna()) > 0 and bool(_looks_like_time(s.dropna().astype(str)))


def _numeric(df, col) -> bool:
    return kind_of(df[col]) == "numeric"


ROLE_WORDS: dict[str, tuple[set[str], set[str]]] = {
    # role: (any of these name words, none of these)
    "hours": ({"hours", "hour", "hrs", "hr", "duration", "worked"}, {"rate", "cost", "pay", "per", "wage", "price"}),
    "rate": ({"hourly", "wage", "wages"}, {"cost", "total"}),
    "revenue": ({"revenue", "revenues", "sales", "income", "turnover", "gmv", "takings"}, {"count", "units", "qty", "target"}),
    "cost": ({"cost", "costs", "spend", "spending", "expense", "expenses", "cogs", "budget"}, {"per", "rate"}),
    "conversions": ({"conversions", "conversion", "converted", "purchases", "signups", "signup", "acquisitions"},
                    {"rate", "pct", "percent", "value", "cost"}),
    "base": ({"visitors", "visits", "sessions", "traffic", "clicks", "leads", "users", "impressions", "views"},
             {"rate", "pct", "cost", "per"}),
    "orders": ({"orders", "transactions", "invoices", "baskets"}, {"value", "rate", "cost", "per", "avg", "average"}),
}
_BASE_PREFERENCE = ["visitors", "visits", "sessions", "traffic", "users", "clicks", "views", "impressions", "leads"]


def _role_columns(df: pd.DataFrame, role: str) -> list[str]:
    want, avoid = ROLE_WORDS[role]
    out = [c for c in map(str, df.columns) if _numeric(df, c) and _tokens(c) & want and not _tokens(c) & avoid]
    if role == "rate" and not out:
        # a plain "rate" (or "pay rate") is an hourly rate only in data that records hours or clock times
        timed = _role_columns(df, "hours") or time_pair(df)[0]
        out = [c for c in map(str, df.columns) if _numeric(df, c) and "rate" in _tokens(c)
               and not _tokens(c) & {"conversion", "churn", "attrition", "growth", "tax", "interest", "exchange",
                                     "discount", "error", "success", "response", "click"}] if timed else []
    if role == "base" and len(out) > 1:
        out.sort(key=lambda c: min([_BASE_PREFERENCE.index(t) for t in _tokens(c) if t in _BASE_PREFERENCE] or [99]))
    return out


def time_pair(df: pd.DataFrame) -> tuple[tuple[str, str] | None, str]:
    """(start, end) clock-time columns, or None with the reason."""
    times = [c for c in map(str, df.columns) if _time_like(df[c])]
    starts = [c for c in times if _tokens(c) & _START or re.search(r"(clock|check|log)_?in$", c, re.I)]
    ends = [c for c in times if _tokens(c) & _END or re.search(r"(clock|check|log)_?out$", c, re.I)]
    starts = [c for c in starts if c not in ends]
    if len(starts) == 1 and len(ends) == 1:
        return (starts[0], ends[0]), ""
    if len(times) == 2 and not starts and not ends:
        return None, f"two time columns ({times[0]}, {times[1]}) but nothing says which is the start"
    if not times:
        return None, "no column holds clock times (start and end)"
    return None, "the start and end time columns could not be told apart (" + ", ".join(times) + ")"


def _pick(df: pd.DataFrame, role: str, message: str) -> tuple[str | None, list[str]]:
    """One column for a role: the only candidate, or the candidate the question names. (column, candidates)."""
    cands = _role_columns(df, role)
    if len(cands) <= 1:
        return (cands[0] if cands else None), cands
    named = [c for c in cands if c in find_column(df, message)]
    if len(named) == 1:
        return named[0], cands
    if role == "base":
        return cands[0], cands          # a conversion base: the widest funnel stage (visitors before leads), stated
    return None, cands


# ------------------------------------------------------------------- metrics --
@dataclass
class Plan:
    name: str
    label: str
    spec: dict | None = None
    clarify: str | None = None
    percent: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    definition: str = ""


@dataclass
class Metric:
    name: str
    label: str
    pattern: str
    kind: str                              # "row": a value per record, totalled; "ratio": a ratio of totals
    column_words: tuple[str, ...] = ()     # a physical column with these words IS the metric (no derivation)


CATALOGUE = [
    Metric("overtime_hours", "overtime", r"\bover[- ]?time\b|\bhours? (?:beyond|over|above|in excess of|more than|past) "
           r"(?:a |an |the )?\d", "row", ("overtime",)),
    Metric("labor_cost", "labor cost", r"\blabou?r (?:costs?|spend|expenses?|bill)\b|\bwage (?:bill|costs?)\b|\bpayroll\b|"
           r"\bstaff(?:ing)? costs?\b|\bcosts? of (?:labou?r|staff|the shifts?)\b|\bpersonnel costs?\b|\bshift costs?\b|"
           r"\b(?:total |gross )?(?:pay|wages|earnings)\b(?!\s*(?:rate|per))", "row", ("labor", "labour", "payroll", "wages", "pay")),
    Metric("shift_hours", "shift duration", r"\bshift (?:duration|length|hours|time)s?\b|\bhours? worked\b|\bwork(?:ed|ing)? "
           r"hours\b|\btime (?:worked|on (?:the )?clock)\b|\bduration\b|\blength of (?:the |each |a )?shifts?\b|"
           r"\bhow (?:many hours|long)\b|\b(?:total|average|mean|median) hours\b|\bhours per (?:shift|day|employee|person)\b|"
           r"\b(?:time|hours?)\b(?:\s+\w+){0,4}?\s+(?:works?|worked|working|spends? working|is at work|was at work|clocks?)\b|"
           r"\bworking (?:time|duration|length)\b|\btime (?:spent )?(?:at work|working|on shift)\b",
           "row", ("hours", "duration", "hrs")),
    Metric("profit_margin", "profit margin", r"\b(?:profit|gross|net|operating)? ?margins?\b", "ratio", ("margin",)),
    Metric("roas", "ROAS", r"\broas\b|\breturn on ad(?:vertising)? spend\b", "ratio", ("roas",)),
    Metric("roi", "ROI", r"\broi\b|\breturn on (?:investment|marketing|spend|the spend|investments)\b", "ratio", ("roi",)),
    Metric("cost_per_conversion", "cost per conversion", r"\bcost per (?:conversion|acquisition|customer|sale|signup|"
           r"purchase)\b|\bcpa\b|\bcac\b|\bacquisition cost\b", "ratio", ("cpa", "cac")),
    Metric("cost_per_lead", "cost per lead", r"\bcost per (?:lead|click|visit|visitor|session)\b|\bcpl\b|\bcpc\b", "ratio",
           ("cpl", "cpc")),
    Metric("conversion_rate", "conversion rate", r"\bconversion (?:rates?|ratio|percentage|%)|\bconv\.? rate\b|"
           r"\brate of conversion\b|\bconvert(?:ed|s)? (?:rate|percentage)\b|\b(?:share|percentage|proportion) of "
           r"(?:leads|visitors|visits|sessions|clicks|users) (?:that|who) convert", "ratio", ("conversion",)),
    Metric("average_order_value", "average order value", r"\baverage order (?:value|size)\b|\baov\b|\baverage (?:basket|"
           r"ticket|transaction|sale) (?:size|value)\b|\b(?:revenue|sales) per (?:order|transaction|purchase)\b",
           "ratio", ("aov",)),
    Metric("profit", "profit", r"\b(?:net |gross |operating )?profits?\b|\b(?:revenue|sales) (?:minus|less|-|after) "
           r"(?:the )?(?:costs?|spend|expenses?|ad spend)\b", "row", ("profit",)),
]
_PERCENT = {"profit_margin", "roi", "conversion_rate"}


def recognise(message: str) -> Metric | None:
    for m in CATALOGUE:
        if re.search(m.pattern, message, re.I):
            return m
    return None


def physical(metric: Metric, df: pd.DataFrame) -> str | None:
    """A column that already IS the metric (e.g. a recorded 'profit' or 'conversion_rate'): used as it is."""
    for c in map(str, df.columns):
        t = _tokens(c)
        if metric.name in ("shift_hours",) and _numeric(df, c) and t & set(metric.column_words) and not t & {"rate", "cost", "pay"}:
            return c
        if metric.name == "labor_cost" and _numeric(df, c) and t & {"labor", "labour", "payroll", "wages"} and t & {"cost", "costs", "bill", "wages", "payroll"}:
            return c
        if metric.name not in ("shift_hours", "labor_cost") and t & set(metric.column_words) \
                and (metric.name != "conversion_rate" or t & {"rate", "pct", "percent", "ratio"}):
            return c
    return None


_MEAN = re.compile(r"\b(average|avg|mean|typical)\b", re.I)
_MEDIAN = re.compile(r"\bmedian\b", re.I)
_MAX = re.compile(r"\b(longest|maximum|max|largest single|highest single|most in (?:a|one) (?:shift|record))\b", re.I)
_MIN = re.compile(r"\b(shortest|minimum|min|smallest single|lowest single)\b", re.I)
_OVERNIGHT = re.compile(r"\bovernight\b|\bpast midnight\b|\bnight shifts?\b|\bnext day\b|\bcross(?:es|ing)? midnight\b", re.I)
_DUPES = re.compile(r"\b(exclud\w*|without|remov\w*|drop\w*|ignor\w*|no|de-?dup\w*)\b[^.?!]{0,20}\bduplicat", re.I)


def _agg_func(message: str, metric: Metric) -> str:
    text = re.sub(metric.pattern, " ", message, flags=re.I)
    if _MEDIAN.search(text):
        return "median"
    if _MEAN.search(text) or re.search(r"\bper (shift|record|entry|row|transaction)\b", text, re.I):
        return "mean"
    if _MAX.search(text):
        return "max"
    if _MIN.search(text):
        return "min"
    return "sum"


_PEOPLE = {"employee", "employees", "person", "people", "worker", "workers", "staff", "member", "members", "individual",
           "individuals", "agent", "agents", "rep", "reps", "driver", "drivers"}
_TIME_UNITS = {"day", "days", "date", "dates", "week", "weeks", "month", "months", "quarter", "quarters", "year", "years",
               "period", "periods", "daily", "weekly", "monthly"}
_GROUP = re.compile(r"\b(?:by|per|for each|for every|each|every|across(?: all)?(?: the)?|in each|broken down by|split by|"
                    r"grouped by|for all(?: the)?|of each|among(?: the)?)\s+(?:the\s+)?([a-z][a-z_]*(?:\s[a-z][a-z_]*)?)", re.I)
_WHICH = re.compile(r"\bwhich\s+([a-z][a-z_]*(?:\s[a-z][a-z_]*)?)\s+(?:has|had|have|is|was|gets|got|made|makes|spent|spends|"
                    r"earned|earns|generated|generates|delivered|delivers|shows?|did|does|costs?|worked|works)\b", re.I)


def _singular(w: str) -> str:
    return w[:-3] + "y" if w.endswith("ies") else w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _abbrev(token: str, word: str) -> bool:
    """'dept' abbreviates 'department': same first letter, at least 3 letters, letters in order."""
    if len(token) < 3 or len(token) >= len(word) or token[0] != word[0] or token in ("the", "and", "per"):
        return False
    it = iter(word)
    return all(ch in it for ch in token)


_NOUN_TOTALS = re.compile(r"\b([a-z][a-z_]*)\s+(?:totals?|averages?|sums?|breakdown|summary|split)\b", re.I)


def available(df: pd.DataFrame) -> list[str]:
    """Standard metrics whose inputs this data has (and which are not already columns), by label."""
    out = []
    for m in CATALOGUE:
        if physical(m, df) or m.name == "overtime_hours":
            continue
        if m.name == "average_order_value" and not _role_columns(df, "orders") and not [
                c for c in map(str, df.columns) if _tokens(c) & {"order", "transaction", "invoice"} and _tokens(c) & {"id", "no", "number"}]:
            continue
        try:
            p = _BUILDERS[m.name](m.label, df, Plan(m.name, m.label))
        except Exception:  # noqa: BLE001
            continue
        if p is not None and (not p.clarify or "more than one column could be" in p.clarify):
            out.append(m.label)          # inputs present (a choice between columns is asked when it is used)
    return out


def group_column(message: str, df: pd.DataFrame, skip: set[str] = frozenset()) -> tuple[str | None, str | None]:
    """The column a question breaks its result down by ("by department", "for each employee", "which channel has").
    (column, phrase) -- column None with a phrase when the phrase names nothing in the data."""
    cats = [c for c in map(str, df.columns) if kind_of(df[c]) == "categorical" and not _time_like(df[c]) and c not in skip]
    for rx in (_WHICH, _GROUP, _NOUN_TOTALS):
        for m in rx.finditer(message):
            phrase = m.group(1).lower()
            words = phrase.split()
            if words[0] in _TIME_UNITS or words[0] in ("shift", "shifts", "record", "records", "row", "rows", "entry",
                                                         "the", "a", "an", "of", "hour", "hours", "order", "orders"):
                continue
            direct = [c for c in find_column(df, phrase) if c in cats]
            if direct:
                return direct[0], phrase
            for w in (words if len(words) == 1 else [words[0], words[-1]]):
                s = _singular(w)
                hits = [c for c in cats if s in _tokens(c) or w in _tokens(c)]
                if not hits:            # abbreviated names: "dept" for department, "emp" for employee, "cat" ...
                    hits = [c for c in cats if any(_abbrev(t, s) for t in _tokens(c))]
                if not hits and w in _PEOPLE:
                    hits = [c for c in cats if _tokens(c) & {"employee", "staff", "worker", "person", "name", "agent", "driver"}]
                if hits:
                    hits.sort(key=lambda c: (0 if "name" in _tokens(c) else 1 if "id" not in _tokens(c) else 2, len(c)))
                    return hits[0], phrase
    return None, None


def _ident(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()


def plan(message: str, df: pd.DataFrame, metric: Metric | None = None) -> Plan | None:
    """A calculator plan for a recognised derived metric, a clarification, or None (not a catalogue metric, or the
    metric is a column of this data)."""
    metric = metric or recognise(message)
    if metric is None or df is None or physical(metric, df):
        return None
    if metric.name == "labor_cost" and not re.search(metric.pattern.split(r"|\b(?:total |gross )?(?:pay|wages|earnings)")[0],
                                                     message, re.I):
        # bare "pay" / "wages" / "earnings" mean labour cost only in shift data (hours or clock times) that has no
        # salary-type column; in an employee table "average pay" is the salary
        salary = [c for c in map(str, df.columns) if _numeric(df, c) and _tokens(c) & {"salary", "salaries", "pay",
                  "compensation", "income", "earnings", "wage", "wages", "remuneration"} and "rate" not in _tokens(c)]
        if salary or not (_role_columns(df, "hours") or time_pair(df)[0]):
            return None
    builder = _BUILDERS[metric.name]
    p = Plan(metric.name, metric.label)
    out = builder(message, df, p)
    if out is None or out.clarify:
        return out
    group, phrase = group_column(re.sub(metric.pattern, " ", message, flags=re.I), df)
    if group is None and phrase and _GROUP.search(message) and not re.search(r"\b(all|every|each) (records?|shifts?|rows?)\b",
                                                                            message, re.I):
        out.notes.append(f"'{phrase}' is not a column of this data, so the result is not broken down by it.")
    spec = out.spec
    spec["group_by"] = group
    if _DUPES.search(message):
        spec["exclude"].insert(0, {"reason": "exact duplicate of an earlier record", "where": "duplicate()"})
    spec.setdefault("sort", "-" + metric.name)
    spec["explain"] = out.definition
    out.spec = spec
    return out


def _clar(p: Plan, text: str) -> Plan:
    p.clarify = text
    return p


def _nonneg(cols: list[str], why: str = "not a possible value") -> list[dict]:
    return [{"reason": f"negative {c.replace('_', ' ')} ({why})", "where": f"{c} < 0"} for c in cols]


def _hours(message: str, df: pd.DataFrame, p: Plan) -> tuple[str | None, dict, list[dict], str | None]:
    """(expression for hours per record, derive, exclusions, problem)."""
    hours, hcands = _pick(df, "hours", message)
    if hours:
        return hours, {}, _nonneg([hours]), None
    if len(hcands) > 1:
        return None, {}, [], ("more than one column could be the hours worked (" + ", ".join(f"`{c}`" for c in hcands)
                              + "). Which one should I use?")
    pair, why = time_pair(df)
    if pair is None:
        return None, {}, [], f"there is no hours column and {why}"
    s, e = pair
    fn = "overnight_hours" if _OVERNIGHT.search(message) else "hours_between"
    derive = {"shift_hours": f"{fn}({s}, {e})"}
    rules = [] if fn == "overnight_hours" else [{
        "reason": f"{e.replace('_', ' ')} earlier than {s.replace('_', ' ')} — an invalid time range as recorded (say "
                  "'treat them as overnight shifts' if they ran past midnight)", "where": f"hours_between({s}, {e}) < 0"}]
    if fn == "overnight_hours":
        p.notes.append(f"An {e.replace('_', ' ')} earlier than the {s.replace('_', ' ')} is counted as the next day "
                       "(overnight shift), as you asked.")
    return "shift_hours", derive, rules, None


def _b_shift_hours(message, df, p):
    h, derive, rules, problem = _hours(message, df, p)
    if problem:
        return _clar(p, f"**Shift duration** isn't a column here and I can't derive it: {problem}.")
    if not derive:
        return None                                   # a recorded hours column: an ordinary column question
    f = _agg_func(message, CATALOGUE_BY["shift_hours"])
    p.spec = {"type": "calc", "derive": derive, "exclude": rules, "aggregate": {"shift_hours": f"{f}(shift_hours)"}}
    p.definition = f"shift duration = {_read(derive['shift_hours'])} for each record; {_fname(f)} per group"
    return p


def _b_labor_cost(message, df, p):
    h, derive, rules, problem = _hours(message, df, p)
    if problem:
        return _clar(p, f"**Labor cost** = hours × hourly rate, but {problem}.")
    rate, rc = _pick(df, "rate", message)
    if rate is None:
        return _clar(p, "**Labor cost** = hours × hourly rate, but " + (
            "more than one column could be the hourly rate (" + ", ".join(f"`{c}`" for c in rc) + "). Which one?"
            if rc else "this data has no hourly rate or wage column, so I can't compute it (I won't substitute "
                       "another number)."))
    derive = {**derive, "labor_cost": f"{h} * {rate}"}
    f = _agg_func(message, CATALOGUE_BY["labor_cost"])
    p.spec = {"type": "calc", "derive": derive, "exclude": rules + _nonneg([rate]),
              "aggregate": {"labor_cost": f"{f}(labor_cost)"} | ({"shift_hours": "sum(shift_hours)"} if f == "sum" and h == "shift_hours" else {})}
    p.definition = (f"labor cost = {_read(derive.get('shift_hours', h))} × {rate.replace('_', ' ')} for each record; "
                    f"{_fname(f)} per group")
    return p


def _b_overtime(message, df, p):
    h, derive, rules, problem = _hours(message, df, p)
    if problem:
        return _clar(p, f"**Overtime** needs hours worked, but {problem}.")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-?\s*hours?|hrs?|h)\b", message, re.I) or \
        re.search(r"\b(?:beyond|over|above|past|more than|in excess of|threshold of)\s+(\d+(?:\.\d+)?)\b", message, re.I)
    if not m:
        return _clar(p, "**Overtime** is hours beyond a threshold, and the threshold depends on your contracts. Hours "
                        "beyond how many per shift? For example: *overtime beyond 8 hours per shift*.")
    if re.search(r"\bper (week|month)|\bweekly\b|\ba week\b", message, re.I):
        return _clar(p, "I can compute overtime against a **per-shift** threshold (each record is one shift). A weekly "
                        "threshold needs hours added up per person per week first; ask for *hours per employee per "
                        "week* and I'll show those totals.")
    t = float(m.group(1))
    tt = f"{t:g}"
    derive = {**derive, "overtime_hours": f"({h} - {tt}) * ({h} > {tt})"}
    f = _agg_func(message, CATALOGUE_BY["overtime_hours"])
    p.spec = {"type": "calc", "derive": derive, "exclude": rules,
              "aggregate": {"overtime_hours": f"{f}(overtime_hours)", "shifts_with_overtime": f"sum({h} > {tt})"}}
    p.definition = (f"overtime = hours beyond {tt} in each shift ({_read(derive.get('shift_hours', h))} minus {tt}, or 0 "
                    f"when not above {tt}); {_fname(f)} per group")
    return p


def _two(df, message, p, need: list[str], what: str) -> tuple[dict | None, Plan | None]:
    found = {}
    for role in need:
        col, cands = _pick(df, role, message)
        if col is None:
            word = {"revenue": "revenue (sales)", "cost": "cost (spend)", "conversions": "conversions",
                    "base": "visitors, sessions, clicks or leads", "orders": "number of orders"}[role]
            if cands:
                return None, _clar(p, f"**{p.label}** = {what}, and more than one column could be the {word}: "
                                      + ", ".join(f"`{c}`" for c in cands) + ". Which one should I use?")
            return None, _clar(p, f"**{p.label}** = {what}, but this data has no {word} column, so I can't compute it. "
                                  "I won't substitute a different number.")
        found[role] = col
    return found, None


def _b_profit(message, df, p):
    r, bad = _two(df, message, p, ["revenue", "cost"], "revenue − cost")
    if bad:
        return bad
    f = _agg_func(message, CATALOGUE_BY["profit"])
    p.spec = {"type": "calc", "derive": {"profit": f"{r['revenue']} - {r['cost']}"}, "exclude": [],
              "aggregate": {"profit": f"{f}(profit)", r["revenue"]: f"sum({r['revenue']})", r["cost"]: f"sum({r['cost']})"}}
    p.definition = f"profit = {_lab(r['revenue'])} − {_lab(r['cost'])} for each record; {_fname(f)} per group"
    return p


def _ratio(p, num_expr, den_expr, comps: dict, definition: str, rules=None, percent=True):
    # a record missing either side cannot be counted on one side only (that would bias the ratio): it is left out
    # of both, and listed
    cols = sorted({c for e in (num_expr, den_expr) for c in re.findall(r"(?:sum|nunique)\(\s*(\w+)\s*\)", e)})
    missing = [{"reason": f"missing {c.replace('_', ' ')} (a ratio needs both values of a record)",
                "where": f"isnull({c})"} for c in cols]
    rules = missing + (rules or [])
    p.spec = {"type": "calc", "derive": {}, "exclude": rules,
              "aggregate": {p.name: f"{num_expr} / {den_expr}", **comps}}
    p.definition = definition + " — a pooled ratio of totals, not an average of per-record ratios"
    den = re.sub(r"^\(?\s*sum\(|nunique\(|\)\s*\)?$", "", den_expr).strip(") (")
    p.notes.append(f"A group whose total {den.replace('_', ' ')} is zero has no {p.label} (shown as —, not 0).")
    if percent:
        p.percent.append(p.name)
    return p


def _b_profit_margin(message, df, p):
    r, bad = _two(df, message, p, ["revenue", "cost"], "(revenue − cost) ÷ revenue")
    if bad:
        return bad
    rv, c = r["revenue"], r["cost"]
    return _ratio(p, f"(sum({rv}) - sum({c}))", f"sum({rv})", {rv: f"sum({rv})", c: f"sum({c})"},
                  f"profit margin = (total {_lab(rv)} − total {_lab(c)}) ÷ total {_lab(rv)}")


def _b_roi(message, df, p):
    r, bad = _two(df, message, p, ["revenue", "cost"], "(revenue − cost) ÷ cost")
    if bad:
        return bad
    rv, c = r["revenue"], r["cost"]
    p.notes.append(f"ROI here counts only {_lab(c)} as the investment and {_lab(rv)} as the return; other costs "
                   "are not in the data.")
    return _ratio(p, f"(sum({rv}) - sum({c}))", f"sum({c})", {rv: f"sum({rv})", c: f"sum({c})"},
                  f"ROI = (total {_lab(rv)} − total {_lab(c)}) ÷ total {_lab(c)}")


def _b_roas(message, df, p):
    r, bad = _two(df, message, p, ["revenue", "cost"], "revenue ÷ ad spend")
    if bad:
        return bad
    rv, c = r["revenue"], r["cost"]
    return _ratio(p, f"sum({rv})", f"sum({c})", {rv: f"sum({rv})", c: f"sum({c})"},
                  f"ROAS = total {_lab(rv)} ÷ total {_lab(c)} (revenue per unit of spend)", percent=False)


def _b_conversion_rate(message, df, p):
    r, bad = _two(df, message, p, ["conversions", "base"], "conversions ÷ visitors (or leads)")
    if bad:
        return bad
    cv, b = r["conversions"], r["base"]
    others = [x for x in _role_columns(df, "base") if x != b]
    if others:
        p.notes.append(f"Base used: {_lab(b)}; the data also has {', '.join(map(_lab, others))} — name it to use that "
                       "instead.")
    return _ratio(p, f"sum({cv})", f"sum({b})", {cv: f"sum({cv})", b: f"sum({b})"},
                  f"conversion rate = total {_lab(cv)} ÷ total {_lab(b)}", rules=_nonneg([cv, b], "a count"))


def _b_cost_per_conversion(message, df, p):
    r, bad = _two(df, message, p, ["cost", "conversions"], "cost ÷ conversions")
    if bad:
        return bad
    c, cv = r["cost"], r["conversions"]
    return _ratio(p, f"sum({c})", f"sum({cv})", {c: f"sum({c})", cv: f"sum({cv})"},
                  f"cost per conversion = total {_lab(c)} ÷ total {_lab(cv)}", rules=_nonneg([cv], "a count"),
                  percent=False)


def _b_cost_per_lead(message, df, p):
    r, bad = _two(df, message, p, ["cost", "base"], "cost ÷ leads (or visitors)")
    if bad:
        return bad
    c, b = r["cost"], r["base"]
    return _ratio(p, f"sum({c})", f"sum({b})", {c: f"sum({c})", b: f"sum({b})"},
                  f"{p.label} = total {_lab(c)} ÷ total {_lab(b)}", rules=_nonneg([b], "a count"), percent=False)


def _b_aov(message, df, p):
    rv, rc = _pick(df, "revenue", message)
    if rv is None:
        return _clar(p, "**Average order value** = revenue ÷ number of orders, but " + (
            "more than one column could be the revenue (" + ", ".join(f"`{c}`" for c in rc) + "). Which one?" if rc
            else "this data has no revenue or sales column."))
    orders, oc = _pick(df, "orders", message)
    if orders:
        den, what = f"sum({orders})", f"total {_lab(orders)}"
    else:
        ids = [c for c in map(str, df.columns) if kind_of(df[c]) == "categorical"
               and _tokens(c) & {"order", "transaction", "invoice", "receipt", "basket"} and _tokens(c) & {"id", "no", "number", "ref", "code"}]
        if len(ids) != 1:
            return _clar(p, "**Average order value** = revenue ÷ number of orders, but this data has no order count or "
                            "order ID column, so the number of orders is unknown. If each record is one order, ask for "
                            f"*average {_lab(rv)} per record* instead; I won't treat records as orders without that.")
        den, what = f"nunique({ids[0]})", f"number of distinct {_lab(ids[0])} values"
    return _ratio(p, f"sum({rv})", den, {rv: f"sum({rv})"}, f"average order value = total {_lab(rv)} ÷ {what}",
                  percent=False)


def _lab(c: str) -> str:
    return c.replace("_", " ")


def _fname(f: str) -> str:
    return {"sum": "totalled", "mean": "averaged per record", "median": "median per record", "max": "largest record",
            "min": "smallest record"}[f]


def _read(expr: str) -> str:
    from app.analyst.present import readable
    return readable(expr)


_BUILDERS = {"shift_hours": _b_shift_hours, "labor_cost": _b_labor_cost, "overtime_hours": _b_overtime,
             "profit": _b_profit, "profit_margin": _b_profit_margin, "roi": _b_roi, "roas": _b_roas,
             "conversion_rate": _b_conversion_rate, "cost_per_conversion": _b_cost_per_conversion,
             "cost_per_lead": _b_cost_per_lead, "average_order_value": _b_aov}
CATALOGUE_BY = {m.name: m for m in CATALOGUE}


def row_definition(phrase: str, df: pd.DataFrame) -> tuple[dict | None, str]:
    """A per-record definition of an additive metric named by `phrase` (for forecasting a derived total):
    (spec with one sum aggregate, problem). Ratios are not additive, so they are not forecast this way."""
    m = recognise(phrase)
    if m is None or physical(m, df):
        return None, ""
    if m.kind != "row":
        return None, f"{m.label} is a ratio of totals, not a quantity that adds up per period"
    p = plan(phrase, df, m)
    if p is None:
        return None, ""
    if p.clarify:
        return None, re.sub(r"\*\*", "", p.clarify)
    spec = p.spec
    name = m.name if m.name in spec["derive"] else next(iter(spec["aggregate"]))
    derive = dict(spec["derive"])
    unit = re.search(r"\bin\s+(minutes?|mins?|seconds?|secs?|hours?|days?|weeks?|thousands?|millions?|cents?|"
                     r"dollars?|euros?|pounds?|rupees?|[a-z]{3,})\b", phrase, re.I)
    if unit:
        u = unit.group(1).lower()
        factor = None
        if m.name in ("shift_hours", "overtime_hours"):
            factor = {"minute": "* 60", "min": "* 60", "second": "* 3600", "sec": "* 3600", "hour": ""}.get(u.rstrip("s"))
        elif u.rstrip("s") in ("thousand", "million"):
            factor = "/ 1000" if u.startswith("thousand") else "/ 1000000"
        if factor is None:
            return None, (f"{m.label} in {u} needs a unit conversion Silt cannot verify from the data (the data does "
                          "not say which unit or currency its values are in)")
        if factor:
            derive[name + "_in_" + u] = f"{name} {factor}"
            name = name + "_in_" + u
    return {"type": "calc", "derive": derive, "exclude": [], "group_by": None,
            "aggregate": {m.name: f"sum({name})"}}, ""


# ------------------------------------------------- conditions the plan must honour --
_CLAUSE = re.compile(r"\b(?:exclud\w*|except(?: for)?|without|ignor\w*|leav\w* out|filter\w*(?: out)?|remov\w*|"
                     r"only (?:count|include|use|records|rows|shifts|where)|where|unless|deduct\w*|subtract\w*|"
                     r"net of|(?:calculated|computed|defined|worked out) as|as (?=\w+ (?:times|multiplied|minus|plus|"
                     r"divided|x|\*|-|\+|/)))\b(?=([^.;?!]*))", re.I)    # lookahead: clauses do not swallow each other
_HANDLED_ITEM = re.compile(r"duplicat|repeat|identical|negative|invalid|impossible|zero|missing|blank|empty|null|"
                           r"(?:clock|check)[- _]?out\w* (?:is )?(?:before|earlier than)|end\w* (?:is )?(?:before|earlier)|"
                           r"(?:before|earlier than) (?:the )?(?:clock|check)[- _]?in|overnight|future|"
                           r"records? (?:outside|not in|from outside)|outside|"
                           r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
                           r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b|\b(?:19|20)\d\d\b", re.I)
_FORMULA_FILLER = {"the", "a", "an", "of", "each", "per", "record", "records", "row", "shift", "shifts", "times",
                   "multiplied", "by", "minus", "less", "divided", "x", "and", "total", "sum", "their", "his", "her",
                   "its", "as", "is", "calculated", "computed", "defined", "worked", "out", "for", "every", "all"}


def unhandled_conditions(message: str, df: pd.DataFrame, p: "Plan") -> list[str]:
    """Conditions in the question that this deterministic plan would NOT apply: an exclusion it does not
    understand ("excluding cancelled orders") or a formula that differs from the standard one ("hours x rate plus
    10% overhead"). The caller must not answer while ignoring them."""
    from app.analyst.calc import named_values
    values = {v.lower() for _, v in named_values(df, message)}
    inputs = set()
    for e in list(p.spec["derive"].values()) + list(p.spec["aggregate"].values()):
        inputs |= {t for c in re.findall(r"[A-Za-z_]\w*", e) if c in df.columns for t in _tokens(c)}
    inputs |= {t for k in p.spec["derive"] for t in _tokens(k)} | set(re.findall(r"[a-z]+", p.label.lower()))
    left = []
    for m in _CLAUSE.finditer(message):
        clause = m.group(0).strip()
        body = m.group(1)
        if re.match(r"(?:calculated|computed|defined|worked out) as|as\b", clause, re.I):
            body = re.split(r",|;|\b(?:exclud\w*|except|without|ignor\w*|by|per|for|where|in)\b", body, maxsplit=1)[0]
            clause = f"{clause} {body.strip()}".strip()
            words = {w for w in re.findall(r"[a-z]+", body.lower())} - _FORMULA_FILLER
            words -= {w2 for w in inputs for w2 in (w, w + "s", w.rstrip("s"))}
            words -= {"hour", "hours", "hourly", "rate", "rates", "cost", "costs", "spend", "revenue", "sales", "time",
                      "clock", "in", "out", "duration", "worked", "working"}
            if words or re.search(r"\d|%|\bplus\b|\+", body):
                left.append(clause)
            continue
        body = re.split(r"\b(?:by|per|for each|grouped by|broken down by|(?:calculated|computed|defined) as)\b", body,
                        maxsplit=1)[0]
        items = [x for x in re.split(r",|\band\b|\bor\b", body) if x.strip()]
        for it in items:
            it_l = it.lower().strip()
            if _HANDLED_ITEM.search(it_l) or any(v in it_l for v in values) or not re.search(r"[a-z]{3,}", it_l):
                continue
            if re.fullmatch(r"\s*(?:the |all |any )?(?:records?|rows?|entries|data|them|those|these|it)\s*", it_l):
                continue
            left.append(it.strip())
    return left


# ------------------------------------------------------------- counts by group --
_COUNT_Q = re.compile(r"\b(?:which|what)\s+\w+(?:\s\w+)?\s+(?:has|had|have|gets|got)\s+(?:the\s+)?(most|fewest|least|"
                      r"highest number of|lowest number of|largest number of|smallest number of)\s+([a-z]+)\b|"
                      r"\bhow many\s+(records?|rows?|entries|shifts?|orders?|transactions?|visits?|bookings?|tickets?|"
                      r"cases?)\s+(?:are there\s+|were there\s+|do we have\s+)?(?:per|by|for each|in each|for every)\b|"
                      r"\b(?:number|count) of (records?|rows?|entries|shifts?|orders?|transactions?|visits?|bookings?|"
                      r"tickets?|cases?) (?:per|by|for each|in each)\b", re.I)


def count_plan(message: str, df: pd.DataFrame) -> Plan | None:
    """'Which department has the most shifts?' -- records counted per group. A record is one row; the answer says
    so, since a 'shift' or an 'order' is only a row when the data has one row per shift or order."""
    m = _COUNT_Q.search(message)
    if not m or df is None:
        return None
    group, _ = group_column(message, df)
    if group is None:
        return None
    noun = next(g for g in m.groups()[1:] if g) if any(m.groups()[1:]) else "records"
    one = _singular(noun.lower())
    if [c for c in find_column(df, noun) if _numeric(df, c)] or noun.lower() in ("revenue", "money", "spend", "cost",
                                                                                "profit", "value", "time", "hours"):
        return None                     # "the most revenue" ranks by a measure; it is not a count of rows
    # "employees", "customers", "orders": distinct values of the column that identifies one, when there is one
    ids = [c for c in map(str, df.columns) if c != group and kind_of(df[c]) == "categorical"
           and (one in _tokens(c) or (one in ("employee", "people", "person") and _tokens(c) & {"employee", "staff"}))
           and _tokens(c) & {"id", "name", "code", "number", "no"}]
    ids.sort(key=lambda c: 0 if "id" in _tokens(c) else 1)
    name = "number_of_" + _ident(noun if not re.fullmatch(r"records?|rows?|entries", noun, re.I) else "records")
    expr = f"nunique({ids[0]})" if ids else "count(*)"
    low = (m.group(1) or "most").lower() in ("fewest", "least", "lowest number of", "smallest number of")
    p = Plan(name, name.replace("_", " "))
    p.spec = {"type": "calc", "derive": {}, "exclude": [], "group_by": group, "aggregate": {name: expr},
              "sort": name if low else "-" + name}
    if ids:
        p.definition = f"number of distinct {_lab(ids[0])} values for each {_lab(group)}"
    else:
        p.definition = f"number of records (rows) for each {_lab(group)}"
        if not re.fullmatch(r"records?|rows?|entries", noun, re.I):
            p.notes.append(f"Each record (row) is counted as one {one}; if a {one} can span several rows, say how to "
                           "identify one.")
    return p


# --------------------------------------------------- formulas the user states --
_OPS = {"times": "*", "multiplied by": "*", "x": "*", "*": "*", "×": "*", "divided by": "/", "over": "/", "/": "/",
        "÷": "/", "minus": "-", "less": "-", "-": "-", "plus": "+", "+": "+"}
_FORMULA = re.compile(r"([A-Za-z_][\w ]{0,40}?)\s+(times|multiplied by|divided by|over|minus|less|plus|x|\*|×|/|÷|-|\+)\s+"
                      r"([A-Za-z_][\w ]{0,40}?)(?=\s*(?:\)|,|\.|\?|$|\bby\b|\bper\b|\bfor\b|\bin\b|\bexclud))", re.I)


def _numeric_named(df: pd.DataFrame, text: str) -> str | None:
    hits = [c for c in find_column(df, text) if _numeric(df, c)]
    return hits[0] if len(hits) == 1 or (hits and len(hits[0]) > len(hits[1])) else None


def formula_plan(message: str, df: pd.DataFrame) -> Plan | None:
    """'total stock value (stock times unit price) by category': a formula the user states between two numeric
    columns, applied per record and totalled (or averaged). Only one operation between two named columns; anything
    longer goes to the model's validated planner, or is refused -- never half-applied."""
    if df is None:
        return None
    m = _FORMULA.search(message)
    if not m:
        return None
    a_txt, op, b_txt = m.group(1), m.group(2).lower(), m.group(3)
    a = _numeric_named(df, " ".join(a_txt.split()[-3:]))
    b = _numeric_named(df, " ".join(b_txt.split()[:3]))
    if not a or not b or a == b:
        return None
    rest = message[m.end():]
    if re.match(r"\s*(?:times|multiplied|divided|over|minus|less|plus|[x*×/÷+-])\s", rest, re.I):
        return None                                   # a longer formula: not attempted here
    named = re.search(r"\b(?:total|average|mean|sum of|the)\s+([a-z][a-z ]{2,30}?)\s*(?:\(|\bas\b|=|:)", message, re.I)
    label = (named.group(1).strip() if named else f"{_lab(a)} {op} {_lab(b)}")
    name = _ident(label) or "value"
    if name in df.columns:
        name = name + "_calc"
    f = "mean" if _MEAN.search(message) else "sum"
    p = Plan(name, label)
    expr = f"{a} {_OPS[op]} {b}"
    p.spec = {"type": "calc", "derive": {name: expr}, "exclude": [], "aggregate": {name: f"{f}({name})"},
              "group_by": group_column(message[:m.start()] + " " + message[m.end():], df)[0], "sort": "-" + name}
    p.definition = f"{label} = {_lab(a)} {dict(zip('*/-+', '×÷−+'))[_OPS[op]]} {_lab(b)} for each record, as you stated; {_fname(f)}"
    return p


# ------------------------------------------------------- rate of a yes/no event --
_POSITIVE = ("yes", "true", "1", "y", "t", "churned", "left", "attrited", "converted", "won", "approved", "defaulted")
_EVENT_RATE = re.compile(r"\b([a-z][a-z_]{2,30}?)\s+(?:rate|rates|ratio|percentage|share)\b|\b(?:share|percentage|"
                         r"proportion|fraction|percent) of (?:\w+ )?(?:with|who|that|where)\s+([a-z_]+)", re.I)


def event_rate_plan(message: str, df: pd.DataFrame) -> Plan | None:
    """'attrition rate by department': the share of records where a two-valued column has its positive value."""
    if df is None:
        return None
    for m in _EVENT_RATE.finditer(message):
        word = (m.group(1) or m.group(2) or "").lower()
        cols = [c for c in map(str, df.columns) if kind_of(df[c]) == "categorical" and df[c].nunique(dropna=True) == 2
                and (word in _tokens(c) or _singular(word) in _tokens(c) or c.lower() == word)]
        if len(cols) != 1:
            continue
        col = cols[0]
        values = [str(v) for v in df[col].dropna().unique()]
        pos = [v for v in values if v.lower() in _POSITIVE]
        if len(pos) != 1:
            return _clar(Plan(f"{col}_rate", f"{_lab(col)} rate"), f"**{_lab(col)} rate** = share of records with one "
                         f"value of `{col}`, but its values ({', '.join(values)}) don't say which one is the event. Which "
                         "should I count?")
        name = f"{_ident(col)}_rate"
        p = Plan(name, f"{_lab(col)} rate")
        p.spec = {"type": "calc", "derive": {}, "exclude": [{"reason": f"missing {_lab(col)}", "where": f"isnull({col})"}],
                  "aggregate": {name: f"mean({col} == '{pos[0]}')", "records_with_" + _ident(pos[0]): f"sum({col} == '{pos[0]}')"},
                  "group_by": group_column(message, df, skip={col})[0], "sort": "-" + name}
        p.percent.append(name)
        p.definition = f"{_lab(col)} rate = records with {_lab(col)} = {pos[0]} ÷ all records with a {_lab(col)} value"
        return p
    return None


# ------------------------------------------------- one record-validity policy --
_NEVER_NEGATIVE = {"rate", "hourly", "wage", "wages", "hours", "hrs", "duration", "price", "qty", "quantity", "units",
                   "count", "visitors", "visits", "sessions", "clicks", "leads", "conversions", "orders", "stock",
                   "headcount", "age", "tenure"}


def validity_rules(spec: dict, df: pd.DataFrame) -> list[dict]:
    """Records that cannot be valid for a plan, whoever wrote it: a negative value of a column that can never be
    negative (a rate, hours, a price, a count) among the columns it uses, and an end time before the start in an
    hours_between() it computes. Revenue, cost, profit and amounts can legitimately be negative (refunds, credits,
    losses) and are only flagged by the calculator's data checks. Rules already in the plan are not repeated."""
    used = " ".join(list(spec.get("derive", {}).values()) + list(spec.get("aggregate", {}).values()))
    have = {e["where"].replace(" ", "") for e in spec.get("exclude", [])}
    out = []
    for c in map(str, df.columns):
        if _numeric(df, c) and re.search(rf"\b{re.escape(c)}\b", used) and _tokens(c) & _NEVER_NEGATIVE \
                and not _tokens(c) & {"change", "diff", "difference", "delta", "growth", "variance", "net", "pct"} \
                and (pd.to_numeric(df[c], errors="coerce") < 0).any():
            rule = {"reason": f"negative {_lab(c)} (not a possible value)", "where": f"{c} < 0"}
            if rule["where"].replace(" ", "") not in have and not any(f"{c}<0" in h for h in have):
                out.append(rule)
    for a, b in re.findall(r"(?<!overnight_)hours_between\(\s*(\w+)\s*,\s*(\w+)\s*\)", used):
        rule = {"reason": f"{_lab(b)} earlier than {_lab(a)} — an invalid time range as recorded (say 'treat them as "
                          "overnight shifts' if they ran past midnight)", "where": f"hours_between({a}, {b}) < 0"}
        if rule["where"].replace(" ", "") not in have and rule not in out:
            out.append(rule)
    return out
