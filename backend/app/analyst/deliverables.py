"""One message -> the list of things it asks for (deliverables), each with its own scope.

A deliverable is one requested output: a dataset summary, a statistic of a measure, a ranking, a breakdown, a
series over time, a target check, a required pace, a forecast, or a chart of one of those. Each carries ITS OWN
measure, entity/filters, period, horizon, target and condition, read from its own clause. A person, filter or
period stated in one clause never applies to another, with one explicit exception that is recorded: a pronoun
("he", "she", "they", "it", "his") or "the target" refers back to the latest clause that named one.

`extract` is deterministic (no model). `is_compound` decides whether the message needs the compound path: it asks
for two or more deliverables of different kinds, or for a summary / target / forecast / pace together with
anything else, or a chart together with two other deliverables.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from app.analyst.data import kind_of
from app.analyst.semantics import _code_named, profile

NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "twelve": 12, "a couple of": 2, "a few": 3}
PERIOD_WORDS = {"day": "D", "daily": "D", "week": "W", "weekly": "W", "month": "M", "monthly": "M",
                "quarter": "Q", "quarterly": "Q", "year": "Y", "yearly": "Y", "annual": "Y", "annually": "Y"}
STATS = [  # (regex, aggregation) -- order matters: the longest phrases first
    (r"\bstandard deviations?\b|\bstd\b|\bstdev\b", "std"),
    (r"\bvariances?\b", "var"),
    (r"\b(?:average|mean|avg)\b", "avg"),
    (r"\bmedian\b", "median"),
    (r"\b(?:total|sum)\b", "sum"),
    (r"\bmaximum\b|\bmax\b", "max"),
    (r"\bminimum\b|\bmin\b", "min"),
]
UNIT_WORDS = {"hours", "hour", "hrs", "units", "unit", "sales", "orders", "calls", "visits", "items", "tickets", "deals",
              "shifts", "days", "km", "miles", "kg", "tonnes", "dollars", "rupees", "euros", "points", "%", "percent",
              "patients", "students", "bookings", "loans", "claims", "rides", "deliveries", "cases", "minutes"}
_STOP = set("""a an the of for in on at to by with and or is are was were be been what which who whom how many much me my
our we you your i show give tell list find get see let please also then first next now this that these those it its he
she his her they their them can could would will should do does did from per each every all overall total sum average
mean median variance standard deviation std min max maximum minimum top bottom rank ranking highest lowest most least best
worst graph chart plot draw visualise visualize trend over time monthly weekly yearly quarterly month week year day
summary overview describe dataset data table file numbers values columns everything company has have target goal quota
than more less above below under over reach hit make it if yes so not no think up upto as well along together both
compare comparison versus vs into out about only just like want need know forecast forecasts forecasting predict
prediction predictions project projection projected estimate estimates expect expected likely chart graph plot plots
visualise visualize visualisation visualization draw give share result results track get there would will next ahead
period periods quarter quarters days weeks months years please across everyone everybody across then after
start begin short""".split()) | UNIT_WORDS | {"dollar", "usd", "eur", "euro", "rupee", "pound", "yen", "money"}
_PRON = re.compile(r"\b(he|she|him|her|his|hers|they|them|their|it|its|this person|that person)\b", re.I)
_SUMMARY = re.compile(r"\b(summary|summari[sz]e|overview|describe (?:the |this |my )?(?:data(?:set)?|file|table)|"
                      r"description of (?:the |this |my )?(?:data(?:set)?|file|table)|what(?:'s| is) in (?:the |this )?"
                      r"(?:data(?:set)?|file|table))\b", re.I)
_CHART = re.compile(r"\b(graph|chart|plot|visuali[sz]e|visual|draw)\b", re.I)
_FORECAST = re.compile(r"\b(forecast\w*|predict\w*|projection|project(?:ed)?|extrapolat\w*|expect\w*|next\s+(?:\d+|"
                       + "|".join(NUMWORDS) + r")?\s*(?:days?|weeks?|months?|quarters?|years?)|"
                       r"(?:can|could|will|might)\s+(?:\w+\s+){0,3}(?:reach|get|hit|achieve|do)\b|up\s*to\b|upto\b)", re.I)
_TARGET = re.compile(r"\b(target|goal|quota|benchmark|threshold)\b", re.I)
_TARGET_Q = re.compile(r"\b(will|would|can|could|does|did|is|are|was|were)\b.*\b(make|reach|hit|meet|achieve|"
                       r"exceed|cross|miss|below|above|under|over|short|on track|get there|get to)\b|\bmake it\b|"
                       r"\bon track\b|\bget there\b", re.I)
_PACE = re.compile(r"\b(need(?:s|ed)? to (?:average|do|work|reach|get|make|hit)|required (?:pace|rate|average)|"
                   r"what (?:would|does|do) (?:he|she|they|it|\w+) need|how much (?:more )?(?:would|does|do) \w+ need)\b", re.I)
_RANK = re.compile(r"\btop\s+(\d+|" + "|".join(NUMWORDS) + r")\b|\bbottom\s+(\d+|" + "|".join(NUMWORDS) + r")\b|"
                   r"\brank(?:ing|ed)?\b|\b(highest|lowest|most|least|best|worst|leading)\b", re.I)
_TREND = re.compile(r"\b(trend|over time|daily|monthly|weekly|quarterly|yearly|by day|each day|by month|each month|"
                    r"month by month|by week|each week)\b", re.I)
_TASK = re.compile("|".join([r"\b(?:average|mean|avg|median|variance|standard deviation|std|total|sum|maximum|minimum)\b",
                             r"\btop\s+\w+|\bbottom\s+\w+|\brank", r"\b(?:graph|chart|plot|visuali[sz]e|draw)\b",
                             r"\b(?:summary|summari[sz]e|overview)\b", r"\b(?:forecast|predict)", r"\btarget\b",
                             r"\b(?:trend|monthly|weekly|over time)\b", r"\b(?:how many|how much|which|what|who)\b",
                             r"\b(?:check|compare|against|versus|project|estimate)\b"]), re.I)


def subsegments(seg: str) -> list[str]:
    """A clause listing several tasks ("the total overtime, the top 3 employees by hours and Priya's average") as
    one piece per task: a new piece starts only where a list item names a task, so "East and West" stays whole."""
    pieces = [x for x in re.split(r",\s+(?:and\s+)?|\s+and\s+|\s+&\s+", seg) if x and x.strip()]
    groups: list[str] = []
    for x in pieces:
        joins_compared = bool(groups) and re.fullmatch(r"\s*(?:compare|comparing|between)\s+\S+(?:\s+\S+)?\s*", groups[-1], re.I)
        if not groups or (_TASK.search(x) and _TASK.search(groups[-1]) and not joins_compared):
            groups.append(x.strip())
        else:
            groups[-1] += " and " + x.strip()
    return groups
_COND = re.compile(r"^\s*(?:and\s+)?if\s+(yes|so|not|no)\b\s*,?", re.I)
_SPLIT = re.compile(r"(?<=[.?!])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bvs\.)\s+(?=[A-Za-z])|;\s+|,?\s*\b(?:and\s+)?then\b\s*|\s+\balso\b\s+|,\s*(?:and\s+)?"
                    r"(?=(?:show|plot|graph|chart|draw|give|tell|what|which|how|is|are|will|can|could|forecast|predict"
                    r"|list|rank|compare|and)\b)|\s+and\s+(?=(?:show|plot|graph|chart|draw|visuali[sz]e|forecast|predict"
                    r"|give|tell|list|rank|compare|what|which|how)\b)", re.I)


@dataclass
class Deliverable:
    id: int
    kind: str                       # summary | stat | ranking | breakdown | series | target_check | pace | forecast | chart
    clause: str
    order: int                      # position in the message (the answer keeps this order)
    stat: str | None = None         # avg | var | std | median | sum | min | max | count
    measures: list = field(default_factory=list)
    entities: dict = field(default_factory=dict)        # {column: value} from THIS clause (or a recorded reference)
    entity_from_context: bool = False
    group_by: str | None = None
    groups: list = field(default_factory=list)          # compare / growth: the named groups of `group_by` (all if empty)
    top: int | None = None
    order_dir: str = "desc"
    period: str | None = None       # target / pace period: D W M Q Y
    pace_unit: str | None = None
    horizon: int | None = None
    target: float | None = None
    target_unit: str | None = None
    target_from_context: bool = False
    condition: str | None = None    # "yes" | "no" | free text: whether to show it depends on `condition_on`'s answer
    condition_on: int | None = None
    depends_on: list = field(default_factory=list)
    chart_for: int | None = None    # a chart deliverable: the deliverable it draws
    chart_type: str | None = None   # a chart deliverable: the type the user asked for (bar | line | pie | scatter)
    date_from: str | None = None    # this deliverable's own time window (ISO dates, inclusive); None = all dates
    date_to: str | None = None
    output: str = "text"            # text | table | chart
    status: str = "planned"         # planned | done | partial | failed | blocked | needs_input | condition_not_met
    validation: str | None = None
    detail: str | None = None

    def label(self) -> str:
        who = ", ".join(f"{v}" for v in self.entities.values())
        m = " / ".join(map(str, self.measures))
        if not m and self.detail:
            asked = re.search(r"“(.+?)”", self.detail)
            m = asked.group(1) if asked else ""
        stat = {"avg": "average", "var": "variance", "std": "standard deviation", "sum": "total", "min": "minimum",
                "max": "maximum", "median": "median", "count": "count"}.get(self.stat or "", self.stat or "")
        return {
            "summary": "Summary of the dataset",
            "stat": f"{stat.capitalize()} of {m or 'the measures'}" + (f" for {who}" if who else ""),
            "ranking": f"{'Top' if self.order_dir == 'desc' else 'Bottom'} {self.top or ''} {self.group_by or ''} by {m}".replace("  ", " "),
            "breakdown": f"{m} by {self.group_by}" + (f" for {who}" if who else ""),
            "series": f"{m} per {self.period_label()}" + (f" for {who}" if who else ""),
            "target_check": f"Target check: {self.target_text()}" + (f" for {who}" if who else ""),
            "pace": f"Required {'pace per ' + self.pace_unit if self.pace_unit else 'pace'} to reach {self.target_text()}" + (f" for {who}" if who else ""),
            "forecast": f"Forecast of {m}" + (f" for {who}" if who else "") + (f", next {self.horizon} period(s)" if self.horizon else ""),
            "compare": f"Comparison of {m} between " + (" and ".join(map(str, self.groups)) if self.groups else
                                                         f"{self.group_by or 'groups'}") + (f" for {who}" if who else ""),
            "growth": f"Growth of {m}" + (f" by {self.group_by}" if self.group_by else "") + (f" for {who}" if who else ""),
            "correlation": f"Correlation between {' and '.join(map(str, self.measures))}",
            "chart": "Chart",
            "unparsed": f"“{self.clause.strip()}”",
            "clarify": f"“{self.clause.strip()}”",
        }.get(self.kind, self.kind)

    def window_label(self) -> str:
        if not (self.date_from or self.date_to):
            return ""
        return f" ({self.date_from or 'start'} to {self.date_to or 'latest'})"

    def period_label(self) -> str:
        return {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}.get(self.period or "", "period")

    def target_text(self) -> str:
        if self.target is None:
            return "the target"
        num = f"{self.target:,.0f}" if float(self.target).is_integer() else f"{self.target:,}"
        return f"{num}{' ' + self.target_unit if self.target_unit else ''}" + (f" per {self.period_label()}" if self.period else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label() + (self.window_label() if self.kind != "chart" else "")
        return d


# --------------------------------------------------------------------- helpers --
_SENT = re.compile(r"(?<=[.?!])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bvs\.)(?<!\bNo\.)"
                   r"(?<!\bMrs\.)(?<!\betc\.)\s+(?=[A-Za-z])")


def segments(message: str) -> list[tuple[int, str]]:
    out = []
    for si, sent in enumerate(_SENT.split(message or "")):
        for p in _SPLIT.split(sent):
            if p and p.strip(" ,.;:"):
                out.append((si, p.strip(" ,.;:")))
    return out


def _num(tok: str) -> float | None:
    tok = tok.lower().replace(",", "").strip()
    if tok in NUMWORDS:
        return float(NUMWORDS[tok])
    try:
        return float(tok)
    except ValueError:
        return None


def entity_index(df: pd.DataFrame) -> list[tuple[str, str, re.Pattern]]:
    """(column, value, pattern) for category values that can name an entity in a question."""
    out = []
    for c in df.columns:
        s = df[c]
        if kind_of(s) != "categorical":
            continue
        vals = s.dropna().astype(str).unique()
        if len(vals) > 2000:
            continue
        for v in vals:
            v = v.strip()
            if len(v) >= 2 and not v.isdigit() and (not _code_named(c) or re.search(r"[A-Za-z]{2,}\s", v)):
                out.append((c, v, re.compile(rf"(?<![\w-]){re.escape(v)}(?![\w-])", re.I)))
    out.sort(key=lambda t: -len(t[1]))
    return out


def _entities(text: str, index) -> dict:
    found, taken = {}, []
    for col, val, rx in index:
        m = rx.search(text)
        if m and not any(a <= m.start() < b for a, b in taken):
            taken.append((m.start(), m.end()))
            if col in found:
                found[col] = (found[col] if isinstance(found[col], list) else [found[col]]) + [val]
            else:
                found[col] = val
    # keep the order of mention
    for col, v in found.items():
        if isinstance(v, list):
            v.sort(key=lambda x: min(rx.search(text).start() for c, vv, rx in index if c == col and vv == x))
    return found


def _tokens(name) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def _stem(w: str) -> str:
    """A light, consistent stem: stores/store -> store, hours/hour -> hour, worked/working -> work, categories -> category."""
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed", "er"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _named_whole(c, low: str) -> bool:
    """The clause names column c as whole words (patient_bill / "patient bill"); never as part of a longer word -- the
    column "age" is not named by "average", "rate" not by "separate"."""
    return any(re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", low)
               for f in {str(c).lower(), str(c).lower().replace("_", " ")} if f)


def plan_mismatch(d, df: pd.DataFrame, prof: dict) -> str | None:
    """Checked before a part is computed, whichever reader built it (rules or the model decomposer): the plan must
    compute what its clause asks. A measure the clause does not name, or a grouping the clause names ("compare with
    other departments", "by department") that the plan leaves out, would answer a different question -- the part is
    returned for clarification instead. None when the plan matches its clause."""
    if d.kind not in ("stat", "breakdown", "ranking") or not d.clause:
        return None
    low = d.clause.lower()
    named = measures_in(d.clause, df, prof)
    if named:
        extra = [m for m in d.measures if m not in named]
        if extra:
            return (f"“{d.clause.strip()}” names {', '.join(f'`{m}`' for m in named)}, but the plan also used "
                    f"{', '.join(f'`{m}`' for m in extra)}, which it does not ask about; nothing was substituted")
    group = _group_col(d.clause, df, prof, exclude=tuple(d.entities))
    if group and d.kind == "stat" and not d.group_by and not d.entities:
        return (f"“{d.clause.strip()}” compares across `{group}`, but the plan computed one figure over all records; "
                f"ask for it per {str(group).replace('_', ' ')} (for example: average {', '.join(d.measures[:1]) or 'value'} "
                f"by {str(group).replace('_', ' ')})")
    return None


def measures_in(text: str, df: pd.DataFrame, prof: dict) -> list:
    """Measure columns the clause names (by column name or its words); most specific first."""
    low = text.lower()
    words = {_stem(w) for w in re.findall(r"[a-z]+", low)}
    hits = []
    for c in prof["measures"]:
        toks = {_stem(t) for t in _tokens(c)} - {"of", "the", "num", "number"}
        if not toks:
            continue
        exact = _named_whole(c, low)
        overlap = len(toks & words) / len(toks)
        if exact or overlap >= 0.99 or (len(toks) >= 2 and overlap >= 0.5 and (toks & words) - {"hour", "total", "amount"}):
            hits.append((0 if exact else 1, -overlap, prof["measures"].index(c), c))
    return [c for *_, c in sorted(hits)]


def _unit_measure(unit: str | None, text: str, df, prof) -> str | None:
    """The measure a target's unit refers to ("300 working hours" -> hours_worked, not overtime_hours)."""
    if not unit:
        return None
    if unit.lower() in ("dollars", "dollar", "usd", "$", "euros", "euro", "eur", "rupees", "rupee", "inr", "pounds",
                        "gbp", "yen", "money"):
        money = [c for c in prof["measures"] if re.search(r"revenue|sales|value|amount|income|price|cost|spend|"
                                                           r"profit|turnover|gmv|earnings|usd|eur|gbp|inr|jpy|dollar|"
                                                           r"booking|billing|invoice|payment|fee|wage|salary|pay", str(c), re.I)]
        if money:
            words = {_stem(w) for w in re.findall(r"[a-z]+", text.lower())}
            return sorted(money, key=lambda c: (-len({_stem(t) for t in _tokens(c)} & words), prof["measures"].index(c)))[0]
    u = _stem(unit.lower())
    cand = [c for c in prof["measures"] if u in {_stem(t) for t in _tokens(c)}]
    if len(cand) <= 1:
        return cand[0] if cand else None
    words = {_stem(w) for w in re.findall(r"[a-z]+", text.lower())}
    scored = sorted(cand, key=lambda c: (-len({_stem(t) for t in _tokens(c)} & words),
                                         "overtime" in str(c).lower(), prof["measures"].index(c)))
    return scored[0]


def _target(text: str) -> tuple[float | None, str | None, str | None]:
    m = re.search(r"(?:target|goal|quota|benchmark|threshold)\s*(?:of|is|=|at|:)?\s*(?:about\s+)?(\d[\d,]*(?:\.\d+)?)\s*"
                  r"([a-z%]+(?:\s+(?!per\b|a\b|an\b|each\b|every\b|this\b|in\b|by\b)[a-z]+)?)?(?:\s+(?:per|a|each|every|/)\s*(day|week|month|quarter|year))?", text, re.I)
    if not m:
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*([a-z%]+(?:\s+(?!per\b|a\b|an\b|each\b|every\b|this\b|in\b|by\b)[a-z]+)?)?\s+(?:per|a|each|every|/)\s*(day|week|month|"
                      r"quarter|year)\s+(?:target|goal|quota)", text, re.I)
    if not m:
        # "check it against 320 hours per month", "compared with 56,000 dollars per week"
        m = re.search(r"\b(?:against|versus|vs\.?|compared? (?:it |this |that )?(?:with|to))\s+(?:a |the |our |an? )?(?:target |goal |"
                      r"benchmark )?(?:of )?(\d[\d,]*(?:\.\d+)?)\s*([a-z%]+(?:\s+(?!per\b|a\b|an\b|each\b|every\b)[a-z]+)?)?\s+"
                      r"(?:per|a|an|each|every)\s+(day|week|month|quarter|year)\b", text, re.I)
    if not m:
        # "we expect / need / aim for 340 hours per month", "short of 270 hours a month", "at least 50 units a week"
        m = re.search(r"\b(?:expect(?:s|ed)?|need(?:s)?|aim(?:s|ing)? for|plan(?:s|ned)? for|want(?:s)?|budget(?:ed)? for|"
                      r"short of|under|below|above|over|beyond|at least|at most)\s+(?:about\s+)?(\d[\d,]*(?:\.\d+)?)\s*"
                      r"([a-z%]+(?:\s+(?!per\b|a\b|an\b|each\b|every\b|this\b|in\b|by\b)[a-z]+)?)?\s+(?:per|a|an|each|every)\s+"
                      r"(day|week|month|quarter|year)\b", text, re.I)
    if not m:
        # "reach / hit / make / achieve 300 hours this month": a target stated as the thing to reach
        m = re.search(r"\b(?:reach|hit|make|achieve|exceed|get to|cross|do|work|sell|log)\s+(?:at least\s+)?"
                      r"(\d[\d,]*(?:\.\d+)?)\s*([a-z%]+(?:\s+(?!per\b|a\b|an\b|each\b|every\b|this\b|in\b|by\b)[a-z]+)?)?(?:\s+(?:per|a|each|every|this|in a|in the|by the "
                      r"end of the)\s*(day|week|month|quarter|year))?", text, re.I)
    if not m:
        return None, None, None
    unit = (m.group(2) or "").strip().lower() or None
    if unit:
        unit = re.sub(r"\b(per|a|each|every|target|goal|this|in|by)\b.*$", "", unit).strip() or None
        if unit:
            w = unit.split()
            unit = w[0] if w[0] in UNIT_WORDS else w[-1] if w[-1] in UNIT_WORDS else w[0]
    per = PERIOD_WORDS.get((m.group(3) or "").lower())
    return float(m.group(1).replace(",", "")), unit, per


def _horizon(text: str) -> tuple[int | None, str | None]:
    m = re.search(r"\bnext\s+(\d+|" + "|".join(NUMWORDS) + r")?\s*(days?|weeks?|months?|quarters?|years?)\b", text, re.I)
    if m:
        n = _num(m.group(1)) if m.group(1) else 1
        return int(n or 1), PERIOD_WORDS.get(m.group(2).lower().rstrip("s"))
    m = re.search(r"\b(\d+|" + "|".join(NUMWORDS) + r")\s+(days?|weeks?|months?|quarters?|years?)\s+(?:ahead|out|from now)\b",
                  text, re.I)
    if m:
        return int(_num(m.group(1)) or 1), PERIOD_WORDS.get(m.group(2).lower().rstrip("s"))
    return None, None


def _sole_measure(prof) -> list:
    """With no measure named and none in context, a table with exactly one measure means that one; otherwise none
    (never silently the first column)."""
    ms = [c for c in prof["measures"] if not _code_named(c)]
    return ms[:1] if len(ms) == 1 else []


def stat_words(text: str) -> list:
    return [agg for rx, agg in STATS if re.search(rx, text, re.I)]


def _group_col(text: str, df: pd.DataFrame, prof: dict, exclude=()) -> str | None:
    low = text.lower()
    named_cats = {c for c in df.columns if kind_of(df[c]) == "categorical" and not _code_named(c) and df[c].nunique() <= 2000}
    for c in sorted(set(prof["groups"]) | named_cats | ({prof["entity"]} if prof["entity"] else set()), key=lambda x: -len(str(x))):
        if c in exclude:
            continue
        name = str(c).lower().replace("_", " ")
        stems = {_stem(t) for t in _tokens(c)}
        if re.search(rf"\b(?:by|per|for each|each|across|among|of all|which|other|all|every|all other)\s+(?:the\s+)?"
                     rf"{re.escape(name)}s?\b", low) or \
                (stems and stems <= {_stem(w) for w in re.findall(r"[a-z]+", low)} and _RANK.search(low)):
            return c
    return None


def _people_col(df, prof) -> str | None:
    if prof.get("entity"):
        return prof["entity"]
    for c in df.columns:
        if kind_of(df[c]) == "categorical" and re.search(r"name|employee|staff|agent|rep|person|worker|member|"
                                                         r"customer|student|driver|nurse|doctor", str(c), re.I):
            return c
    return None


# ----------------------------------------------------------------------- extract --
def extract(message: str, df: pd.DataFrame) -> list[Deliverable]:
    prof = profile(df)
    index = entity_index(df)
    out: list[Deliverable] = []
    last_entities: dict = {}
    ctx_target = (None, None, None)
    ctx_measures: list = []
    nid = 0

    def _add(kind, clause, order, **kw):
        nonlocal nid
        nid += 1
        d = Deliverable(id=nid, kind=kind, clause=clause, order=order, **kw)
        out.append(d)
        return d

    def add(kind, clause, order, **kw):
        """Several values of one column in a clause ("the average of Mateo and Lena") are one deliverable each, so
        each is computed and shown on its own; a ranking or breakdown keeps the list as a filter."""
        ents = kw.get("entities") or {}
        lists = {k: v for k, v in ents.items() if isinstance(v, list)}
        if lists and kind not in ("ranking", "breakdown", "chart", "summary"):
            col, vals = next(iter(lists.items()))
            made_ = [_add(kind, clause, order, **{**kw, "entities": {**ents, col: v}}) for v in vals]
            return made_[-1]
        return _add(kind, clause, order, **kw)

    units = []          # (order, sub-clause, measures shared from a sibling)
    pending_cond = None
    sentence_of = {}
    for order, (si, seg) in enumerate(segments(message)):
        sentence_of[order] = si
        cm = _COND.match(seg)
        if cm and not _TASK.search(seg[cm.end():]) and len(seg[cm.end():].split()) <= 2:
            c0 = cm.group(1).lower()
            pending_cond = {"so": "yes"}.get(c0, c0) if c0 in ("yes", "so", "not", "no") else cm.group(0).strip(" ,")
            continue                     # "If yes," on its own: the condition of the next clause
        subs = subsegments(seg)
        lead = re.match(r"^\s*(?:then\s+|and\s+|now\s+)?(?:for|of|about)\s+([^,:;]+?)(?:\s+only)?\s*[:,]\s*(?=\S)", seg, re.I)
        lead_ents = _entities(lead.group(1), index) if lead else {}
        if lead_ents and len(subs) >= 2:
            # the leading scope applies to every listed item that names none of its own
            subs = [x if (_entities(x, index) or i == 0) else f"for {lead.group(1)}: {x}" for i, x in enumerate(subs)]
        own = [measures_in(x, df, prof) for x in subs]
        for i, x in enumerate(subs):
            share = own[i] or next((m for m in own[i + 1:] if m), None) or next((m for m in reversed(own[:i]) if m), None)
            units.append((order, x, share or [], pending_cond if i == 0 else None, seg))
        pending_cond = None
    sent_entities: dict = {}
    for order, seg, shared, cond0, whole in units:
        low = seg.lower()
        cond = cond0
        cm = _COND.match(seg)
        if cm:
            cond = cm.group(1).lower() if cm.group(1).lower() in ("yes", "so", "not", "no") else cm.group(0).strip(" ,")
            cond = {"so": "yes"}.get(cond, cond)
        ents = _entities(seg, index)
        from_ctx = False
        if not ents and _PRON.search(seg) and last_entities:
            ents, from_ctx = dict(last_entities), True
        # a clause about time with no subject of its own ("..., forecast the next 3 months and plot it") is about the
        # subject named earlier in the SAME sentence; recorded as a reference. Statistics never inherit a subject.
        same_sentence = sent_entities.get(sentence_of.get(order))
        time_task = bool(_FORECAST.search(seg) or _TARGET.search(seg) or _PACE.search(seg) or _CHART.search(seg) or
                         re.search(r"\b(hit|reach|make) (it|that)\b", low))
        overall = re.search(r"\b(overall|all|every|whole|entire|company|total|dataset|everyone|combined)\b", low)
        if not ents and time_task and not overall and not stat_words(seg):
            prev_sentence = sent_entities.get((sentence_of.get(order) or 0) - 1)
            if same_sentence:
                ents, from_ctx = dict(same_sentence), True
            elif prev_sentence and not sent_entities.get(sentence_of.get(order)):
                # "Will Zara get there? Give me a forecast for the next 3 months with a chart." -- the forecast is
                # Zara's: the subject continues from the sentence just before (recorded, shown in the answer)
                ents, from_ctx = dict(prev_sentence), True
        if ents:
            single = {k: (v[0] if isinstance(v, list) else v) for k, v in ents.items()}
            last_entities = dict(single)
            sent_entities[sentence_of.get(order)] = dict(single)
        t_val, t_unit, t_per = _target(seg)
        statement = bool(re.search(r"\b(has|have|set|is|of)\b.*\btarget\b|\btarget\b.*\b(of|is)\b\s*\d|"
                               r"^\s*target\s*:|\b(?:we|they|i)\s+(?:expect|need|want|aim for|plan for)\s+\d", low)) and \
            not _TARGET_Q.search(seg) and not re.search(r"\?\s*$", seg) and \
            not re.search(r"\b(compare|comparison|vs\.?|versus|against|check|measure|how (?:far|close))\b", low)
        if t_val is not None:
            ctx_target = (t_val, t_unit, t_per)
        # words that look like a measure no column matches ("total revenue" on data with no revenue): never replaced
        # by another column or every numeric column -- the deliverable fails and says which word was not found
        known = {w for v in index for w in re.findall(r"[a-z]+", v[1].lower())} | \
            {w for c in df.columns for w in re.findall(r"[a-z]+", str(c).lower())}
        stray = [w for w in re.findall(r"[a-z]+", low) if w not in _STOP and w not in known and len(w) > 2
                 and _stem(w) not in {_stem(k) for k in known}]
        own_meas = measures_in(seg, df, prof)
        # a column the clause names that is not a quantity ("the average customer id"): never replaced by other columns
        ent_text = " ".join(str(v).lower() for v in (ents or {}).values() for v in (v if isinstance(v, list) else [v]))
        named_other = [c for c in df.columns if c not in prof["measures"] and c not in prof["dates"] and c not in (ents or {})
                       and (str(c).lower() in low or str(c).lower().replace("_", " ") in low)
                       and str(c).lower() not in ent_text]
        meas = own_meas or ([] if (stray or named_other) else list(shared))
        if not meas and t_unit:
            um = _unit_measure(t_unit, seg, df, prof)
            meas = [um] if um else []
        if meas:
            ctx_measures = list(meas)
        made = []
        if _SUMMARY.search(seg):
            made.append(add("summary", seg, order, output="text"))
        # statistics (each word its own deliverable)
        stat_hits = []
        for rx, agg in STATS:
            for m in re.finditer(rx, seg, re.I):
                stat_hits.append((m.start(), agg))
        stat_hits.sort()
        seen_aggs = []
        is_rank = bool(_RANK.search(seg))
        group = _group_col(seg, df, prof, exclude=tuple(ents))
        for _, agg in stat_hits:
            if agg in seen_aggs or (is_rank and agg in ("max", "min")):
                continue
            if agg in ("avg", "sum") and (_PACE.search(seg) or (t_val is not None and not statement and _TARGET_Q.search(seg))):
                continue                     # "need to average 75 a week": the pace, not a statistic
            seen_aggs.append(agg)
            ms = meas or ([] if (stray or named_other) else [c for c in prof["measures"] if not _code_named(c)])
            if group and not is_rank:
                made.append(add("breakdown", seg, order, stat=agg, measures=ms[:1], entities=ents,
                                entity_from_context=from_ctx, group_by=group, output="table"))
            elif not is_rank:
                made.append(add("stat", seg, order, stat=agg, measures=ms, entities=ents, entity_from_context=from_ctx,
                                output="text", detail=(None if ms else f"`{named_other[0]}` is not a quantity, so its {agg} "
                                                       "is not computed" if named_other else
                                                       f"this data has no column for “{' '.join(stray)}”")))
        if is_rank and (group or _people_col(df, prof)) and not _TARGET_Q.search(seg):
            rm = _RANK.search(seg)
            n = _num(rm.group(1) or rm.group(2) or "") if (rm.group(1) or rm.group(2)) else None
            direction = "asc" if (rm.group(2) or re.search(r"\b(lowest|least|worst|bottom)\b", low)) else "desc"
            gcol = group or _people_col(df, prof)
            ms = meas or ([] if (stray or named_other) else (ctx_measures or _sole_measure(prof)))
            agg = next((a for _, a in stat_hits if a in ("avg", "sum", "median")), None)
            prev = next((x for x in reversed(out) if x.kind == "breakdown" and x.group_by == gcol), None)
            if agg is None and prev is not None:
                agg, ms = prev.stat, list(prev.measures)
            made.append(add("ranking", seg, order, stat=agg, measures=ms[:1], group_by=gcol, top=int(n) if n else None,
                            order_dir=direction, entities={k: v for k, v in ents.items() if k != gcol}, output="table"))
        wants_target = bool(_TARGET.search(seg) or (t_val is not None and re.search(r"\b(against|versus|vs\.?|compared? "
                                                                                   r"(?:it |this |that )?(?:with|to))\b", low))
                            or (re.search(r"\bmake it\b|\b(?:reach|hit|meet|under|below|above|over|"
                                                              r"short of|beat|miss) (?:it|that|this)\b|\bget there\b|\bon "
                                                              r"track\b", low) and ctx_target[0] is not None)
                            or (t_val is not None and re.search(r"\b(reach|hit|make|achieve|exceed|get to|cross)\s+(?:at "
                                                                r"least\s+)?\d", low))
                            or (t_val is not None and _TARGET_Q.search(seg)))
        if wants_target and not statement and not _PACE.search(seg) and (_TARGET_Q.search(seg) or re.search(r"\bvs\.?\b|\bagainst\b|\bcompare", low)):
            tv, tu, tp = (t_val, t_unit, t_per) if t_val is not None else ctx_target
            ms = meas or ([_unit_measure(tu, seg, df, prof)] if tu else []) or ctx_measures
            made.append(add("target_check", seg, order, measures=[m for m in ms if m][:1], entities=ents,
                            entity_from_context=from_ctx, target=tv, target_unit=tu, period=tp or "M",
                            target_from_context=t_val is None, condition=cond))
        if _PACE.search(seg):
            tv, tu, tp = (t_val, t_unit, t_per) if t_val is not None else ctx_target
            pm = re.search(r"\b(?:per|a|each|every)\s+(day|week|month|quarter|year)\b", low)
            pu = PERIOD_WORDS.get(pm.group(1)) if pm else (tp or "W")
            ms = meas or ([_unit_measure(tu, seg, df, prof)] if tu else []) or ctx_measures
            made.append(add("pace", seg, order, measures=[m for m in ms if m][:1], entities=ents,
                            entity_from_context=from_ctx, target=tv, target_unit=tu, period=tp or "M",
                            target_from_context=t_val is None, pace_unit={"D": "day", "W": "week", "M": "month",
                                                                           "Q": "quarter", "Y": "year"}[pu],
                            condition=cond))
        explicit_fc = re.search(r"\b(forecast|predict|project|projection|extrapolat|expect|how (?:many|much)|up\s*to|upto|next\s+"
                                r"(?:\d+|\w+)?\s*(?:days?|weeks?|months?|quarters?|years?))", low)
        if _FORECAST.search(seg) and not statement and not (_PACE.search(seg) and not re.search(r"forecast|predict", low)) \
                and (explicit_fc or not any(x.kind == "target_check" for x in made)):
            h, hp = _horizon(seg)
            ms = meas or ([] if (stray or named_other) else (ctx_measures or _sole_measure(prof)))
            made.append(add("forecast", seg, order, measures=ms[:1], entities=ents, entity_from_context=from_ctx,
                            horizon=h or 1, period=hp, target=ctx_target[0] if (ctx_target[0] is not None and
                            (_TARGET.search(seg) or any(x.kind == "target_check" for x in out))) else None,
                            target_unit=ctx_target[1], output="table", condition=cond))
        if _TREND.search(seg) and not statement and not any(x.kind in ("forecast", "ranking", "breakdown", "target_check")
                                                                for x in made) and \
                (meas or _CHART.search(seg)) and not (stat_hits and not _CHART.search(seg)):
            ms = meas or ([] if (stray or named_other) else (ctx_measures or _sole_measure(prof)))
            made.append(add("series", seg, order, measures=ms[:1], entities=ents, entity_from_context=from_ctx,
                            period=next((PERIOD_WORDS[w] for w in re.findall(r"\b(daily|weekly|monthly|quarterly|"
                                                                               r"yearly)\b", low)), None),
                            output="chart" if _CHART.search(seg) else "table"))
        if not made and not _CHART.search(seg) and not statement and not cond0 and \
                re.search(r"\b(average|mean|median|variance|standard deviation|total|sum|maximum|minimum|top \w+|bottom \w+|"
                          r"rank\w*|highest|lowest|forecast\w*|predict\w*|trend|compare|comparison|breakdown|summar\w*|"
                          r"overview|how many|how much|count)\b", low):
            made.append(add("unparsed", seg, order, detail=f"I could not read “{seg.strip()}” as an analysis of this data, "
                                                         "so nothing was computed for it"))
        if _CHART.search(seg) and not made and (own_meas or (ents and not from_ctx)):
            # "plot daily patients seen for East Clinic": the chart names its own measure / subject -- its own series,
            # never a chart of an earlier, different result
            made.append(add("series", seg, order, measures=(own_meas or meas)[:1], entities=ents,
                            entity_from_context=from_ctx, output="chart",
                            period=next((PERIOD_WORDS[w] for w in re.findall(r"\b(daily|weekly|monthly|quarterly|yearly)\b",
                                                                             low)), None)))
        if _CHART.search(seg):
            subject = next((x for x in made if x.kind == "forecast"), None) or \
                next((x for x in made if x.kind == "series"), None) or \
                next((x for x in made if x.kind in ("breakdown", "ranking")), None) or \
                next((x for x in reversed(out[:-len(made)] if made else out) if x.kind == "forecast"), None) or \
                next((x for x in made if x.kind in ("target_check", "stat")), None) or \
                next((x for x in reversed(out) if x.kind in ("forecast", "series", "breakdown")), None) or \
                next((x for x in reversed(out) if x.kind in ("ranking", "target_check")), None)
            if subject is None and meas:
                subject = add("series", seg, order, measures=meas[:1], entities=ents, entity_from_context=from_ctx,
                              output="chart")
            if subject is not None:
                subject.output = "chart"
                add("chart", seg, order, chart_for=subject.id, depends_on=[subject.id], output="chart",
                    entities=dict(subject.entities), measures=list(subject.measures))
    return finalize(out)


def finalize(out: list[Deliverable]) -> list[Deliverable]:
    """Shared by the rule reader and the model decomposer: merge duplicates, apply ordering words, wire
    dependencies (a future target check uses the forecast) and display conditions."""
    out = _merge_duplicates(out)
    # explicit ordering words: "start with / begin with / first ..." goes first, "finally / lastly / at the end" last
    for d in out:
        if re.search(r"\b(start with|begin with|starting with|first of all|to start|firstly)\b|^\s*(?:also\s+)?first\b|"
                     r"\bfirst\s*$|\b\w+ first\b(?!\s+\w+\s+(?:of|in))", d.clause, re.I):
            d.order = -1
        elif re.search(r"\b(finally|lastly|at the end|to finish|last of all)\b", d.clause, re.I):
            d.order = 10_000
    for d in out:
        if d.kind == "chart" and d.chart_for is not None:
            subj = next((x for x in out if x.id == d.chart_for), None)
            if subj is not None:
                d.order = subj.order
    # dependencies inside the message: a target check about the future uses the forecast; a pace uses the series
    by_kind = {}
    for d in out:
        by_kind.setdefault(d.kind, []).append(d)
    for d in out:
        if d.kind == "target_check":
            fc = next((f for f in by_kind.get("forecast", []) if f.entities == d.entities and f.measures == d.measures), None)
            if fc:
                d.depends_on.append(fc.id)
        if d.kind in ("forecast", "pace") and d.condition:
            tc = next((t for t in reversed(by_kind.get("target_check", [])) if t.order <= d.order), None)
            if tc:
                d.condition_on = tc.id        # a display condition, not a computation dependency (no cycle)
        if d.kind == "forecast" and d.target is None:
            tc = next((t for t in by_kind.get("target_check", []) if t.entities == d.entities), None)
            if tc:
                d.target, d.target_unit = tc.target, tc.target_unit
    return out


def _merge_duplicates(ds: list[Deliverable]) -> list[Deliverable]:
    """The same deliverable asked twice ("Will she make it next month? Forecast her next month ...") is one item:
    the first keeps its place; charts, dependencies and conditions are re-pointed to it."""
    def key(d):
        ents = tuple(sorted((k, str(v)) for k, v in d.entities.items()))
        return (d.kind, d.stat, tuple(d.measures), ents, d.group_by, tuple(map(str, d.groups)), d.top, d.target,
                d.date_from, d.date_to,
                d.horizon if d.kind == "forecast" else None, d.pace_unit)
    keep, alias, seen = [], {}, {}
    for d in ds:
        if d.kind in ("chart", "unparsed"):
            keep.append(d)
            continue
        k = key(d)
        if k in seen:
            first = seen[k]
            alias[d.id] = first.id
            first.output = "chart" if "chart" in (first.output, d.output) else first.output
            first.condition = first.condition or d.condition
            continue
        seen[k] = d
        keep.append(d)
    out, charted = [], set()
    for d in keep:
        d.depends_on = [alias.get(i, i) for i in d.depends_on]
        if d.chart_for is not None:
            d.chart_for = alias.get(d.chart_for, d.chart_for)
            if d.chart_for in charted:
                continue                      # two chart requests for the same result: one chart
            charted.add(d.chart_for)
        if d.condition_on is not None:
            d.condition_on = alias.get(d.condition_on, d.condition_on)
        out.append(d)
    return out


def spans_clauses(ds: list[Deliverable]) -> bool:
    """The deliverables come from two or more clauses of the message. A single clause ("Will we reach 900k this
    month?") is one question for Silt's specialised handlers, however many outputs it implies."""
    return len({(d.clause, d.order) for d in ds if d.kind != "chart"}) >= 2


STAT_ONLY = {"stat", "ranking", "breakdown", "unparsed"}
_HYPOTHETICAL = re.compile(r"\bwhat if\b|\bsuppose\b|\bimagine\b|\bwould (?:we|i|they|it|the)\b[^.?]*\bif\b|"
                           r"\bif (?:the |our |we )?[\w ]{0,40}\b(?:closed|close|shut|shuts|were shut|absent|removed|"
                           r"increased?|decreased?|cut|raised?|doubled?|halved?|lost)\b", re.I)


def legacy_first(ds: list[Deliverable], message: str) -> bool:
    """Silt's existing paths answer first (their answer is then checked against the deliverables) when every item is a
    statistic / ranking / breakdown, or when some part could not be read or names a measure no column holds (a
    derived quantity such as "labor cost" is built by the calculation path)."""
    if {d.kind for d in ds} <= STAT_ONLY:
        return True
    return any(d.kind == "unparsed" or (d.kind not in ("summary", "chart") and not d.measures) for d in ds)


_FOLLOWUP = re.compile(r"^\s*(now|and|also|then|what about|how about|same|instead|ok(?:ay)?,? now)\b|\bbreak it down\b|"
                       r"\b(?:do|the) same (?:for|by)\b|\bfor (?:those|them|that)\b", re.I)


def follow_up(message: str, memory: dict) -> bool:
    """A follow-up to the previous answer ("Now break it down by region: ...") belongs to the conversation handlers."""
    return bool(memory.get("_turn") or memory.get("last_result") or memory.get("history")) and bool(_FOLLOWUP.search(message or ""))


_NESTED = re.compile(r"\b(?:within|among|inside) (?:the |our )?(?:highest|lowest|top|best|worst|biggest|largest|smallest)\b|"
                     r"\bin the (?:highest|lowest|best|worst|top)[- ]\w+ (?:region|group|segment|category|team|store)\b", re.I)


def hypothetical(message: str) -> bool:
    """A what-if scenario ("how much would we lose if the South depot closed?") is a scenario for objectives.py; a
    nested scope ("within the highest-revenue region, the top five cities ...") is built by the program planner."""
    return bool(_HYPOTHETICAL.search(message or "") or _NESTED.search(message or "") or _CALC_RULES.search(message or ""))


# a calculation the user defines ("labor cost as hours times hourly rate, excluding duplicates and records outside
# September") is Silt's calculator's job (calc.py): its formulas and exclusion rules are not deliverable fields
_CALC_RULES = re.compile(r"\bexclud\w*|\bexcept\b|\bleave out\b|\bleaving out\b|\bonly (?:the )?(?:records|rows|"
                         r"entries)\b|\bwhere\b|\boutside\b|\bas (?:the )?[\w ]{1,30}\b(?:times|multiplied by|minus|plus|"
                         r"divided by|over)\b|\bduplicat\w*", re.I)


def is_compound(ds: list[Deliverable]) -> bool:
    if not spans_clauses(ds):
        return False
    kinds = [d.kind for d in ds]
    core = [k for k in kinds if k != "chart"]
    if len(core) >= 2 and len(set(kinds)) >= 2:
        return True
    if {"summary", "target_check", "forecast", "pace"} & set(kinds) and len(kinds) >= 2:
        return True
    if "chart" in kinds and len(core) >= 2:
        return True
    if len(core) >= 2 and len({json.dumps(d.entities, sort_keys=True, default=str) for d in ds}) >= 2:
        return True              # the same kind for different people ("Rahul's total and Priya's total")
    if len(core) >= 2 and len({tuple(d.measures) for d in ds if d.kind != "chart"}) >= 2:
        return True              # statistics of different measures ("... of hours and the median overtime")
    return False


# ------------------------------------------------------------------------------------- follow-ups over deliverables --
_FU_TARGET = re.compile(r"\b(?:what if|suppose|and if|with|try|instead of)\b[^?]*?\b(?:target|goal|quota|benchmark|budget)"
                        r"\b[^\d?]*?(\d[\d,]*(?:\.\d+)?)|\b(?:try|use|make it)\s+(\d[\d,]*(?:\.\d+)?)\s*(?:instead)?\b", re.I)
_FU_SAME = re.compile(r"^\s*(?:now\s+|ok,?\s+|and\s+)?(?:do\s+|show\s+)?(?:the\s+)?same\s+(?:thing\s+)?(?:for|with)\s+(.+?)"
                      r"(?:\s+please)?[?.!]*\s*$|^\s*(?:and\s+)?(?:what|how)\s+about\s+(.+?)[?.!]*\s*$|^\s*and\s+(?:for\s+)?(.+?)\?\s*$",
                      re.I)
_FU_ALL = re.compile(r"\b(everyone|everybody|all of them|overall|whole (?:dataset|table|company)|all \w+ together|"
                     r"across (?:all|every\w*)|for all)\b", re.I)
_FU_ADD = re.compile(r"\b(variance|median|standard deviation|std|maximum|minimum|total|average|mean)\b[^.?]*\b(as well|too|also)\b|"
                     r"^\s*(?:and|also|now)\s+(?:calculate\s+|give me\s+)?the\s+(variance|median|standard deviation|maximum|"
                     r"minimum|total|average|mean)\b", re.I)
_FU_PLOT = re.compile(r"\b(plot|chart|graph|visuali[sz]e|draw)\b[^.?]*\b(those|them|that|it|this|both|the two)\b|"
                      r"^\s*(?:now\s+)?(?:plot|chart|graph)\s+(?:those|them|it|that)\b", re.I)
_FU_EXCL = re.compile(r"\b(?:exclud\w*|without|except|leave out|leaving out)\s+(.+?)(?:\s+please)?[?.!]*\s*$", re.I)
FU_OTHERS = re.compile(r"\b(?:the\s+)?(?:other(?:\s+(?:two|three|four|\d+))?(?:\s+\w+)?|rest(?:\s+of\s+(?:them|the\s+\w+))?|"
                       r"remaining(?:\s+\w+)?|everyone else|everybody else|all the others)\b", re.I)
_STATWORD = {"variance": "var", "median": "median", "standard deviation": "std", "std": "std", "maximum": "max",
             "minimum": "min", "total": "sum", "average": "avg", "mean": "avg"}


def to_items(ds: list[Deliverable]) -> list[dict]:
    return [asdict(d) for d in ds]


def from_items(items: list[dict]) -> list[Deliverable]:
    keep = {f for f in Deliverable.__dataclass_fields__}
    return [Deliverable(**{k: v for k, v in it.items() if k in keep}) for it in items]


def _find_value(text: str, df: pd.DataFrame, prefer: list | None = None):
    """(column, value) for a value of a category column named in `text`; columns already in scope first."""
    hits = []
    low = text.lower().strip(" ?.!")
    for col, val, pat in entity_index(df):
        if pat.search(text) or str(val).lower() == low:
            hits.append((col, val))
    if not hits:
        return None
    hits.sort(key=lambda h: (h[0] not in (prefer or []), -len(str(h[1]))))
    return hits[0]


def followup_edit(message: str, prev: list[Deliverable], df: pd.DataFrame) -> tuple[list[Deliverable], str] | None:
    """The previous request with ONE explicit change carried out, or None. What changed and what was kept is said."""
    import copy
    base = [copy.deepcopy(d) for d in prev if d.kind not in ("unparsed", "clarify")]
    if not base:
        return None
    prev_cols = [c for d in base for c in d.entities]
    m = _FU_TARGET.search(message)
    if m and any(d.kind in ("target_check", "pace", "forecast") and d.target is not None for d in base):
        val = float((m.group(1) or m.group(2)).replace(",", ""))
        out = [d for d in base if d.kind in ("target_check", "pace", "forecast", "chart")]
        for d in out:
            if d.target is not None or d.kind in ("target_check", "pace"):
                d.target = val
        return finalize(out), f"target changed to {val:,.0f}; everything else kept"
    m = _FU_ADD.search(message)
    if m:
        st = _STATWORD[(m.group(1) or m.group(3)).lower()]
        src = [d for d in base if d.kind == "stat" and d.measures] or [d for d in base if d.measures and d.kind != "chart"][:1]
        if src:
            out = []
            for i, d in enumerate(src):
                out.append(Deliverable(id=i + 1, kind="stat", clause=message, order=i, stat=st, measures=list(d.measures[:1]),
                                       entities=dict(d.entities)))
            return finalize(out), f"added the {next(k for k, v in _STATWORD.items() if v == st)} on the same scope"
    if _FU_PLOT.search(message):
        out, n = [], 0
        for d in base:
            if d.kind == "chart":
                continue
            out.append(d)
            if d.kind in ("compare", "growth", "breakdown", "ranking", "series", "forecast", "stat"):
                subj = d
                if d.kind == "stat" and len(d.entities) and isinstance(next(iter(d.entities.values())), list):
                    continue
                if d.kind == "stat":
                    continue
                n += 1
                out.append(Deliverable(id=10_000 + n, kind="chart", clause=message, order=subj.order, chart_for=subj.id,
                                       depends_on=[subj.id], output="chart", entities=dict(subj.entities),
                                       measures=list(subj.measures)))
        stats = [d for d in base if d.kind == "stat" and d.entities and d.measures]
        if not n and len(stats) >= 2:
            # "Compare X between A and B" answered as two statistics: chart them as one comparison
            col = next(iter(stats[0].entities))
            vals = [str(v) for d in stats for v in (d.entities[col] if isinstance(d.entities.get(col), list)
                                                     else [d.entities.get(col)]) if v is not None]
            cmp_ = Deliverable(id=1, kind="compare", clause=message, order=0, stat=stats[0].stat or "avg",
                               measures=list(stats[0].measures), group_by=col, groups=vals)
            out = [cmp_, Deliverable(id=2, kind="chart", clause=message, order=0, chart_for=1, depends_on=[1],
                                     output="chart", measures=list(cmp_.measures))]
            n = 1
        if n:
            return finalize(out), "drew the previous result"
    if FU_OTHERS.search(message):
        # "and the other two regions?": every value of the previous subject's column except the one(s) asked before
        scoped = [d for d in base if d.entities]
        if scoped:
            col = next(iter(scoped[0].entities))
            before = scoped[0].entities[col]
            before = {str(v).lower() for v in (before if isinstance(before, list) else [before])}
            others = [str(v) for v in df[col].dropna().astype(str).unique() if str(v).lower() not in before]
            if others:
                out = []
                for v in others:
                    for d in base:
                        if d.kind in ("summary", "chart"):
                            continue
                        x = copy.deepcopy(d)
                        x.entities = {**{k: w for k, w in d.entities.items() if k != col}, col: v}
                        x.entity_from_context = False
                        x.id = len(out) + 1
                        x.order = len(out)
                        x.depends_on, x.condition_on = [], None
                        out.append(x)
                return finalize(out), f"{col}: the others ({', '.join(others)}) instead of {', '.join(sorted(before))}"
    m = _FU_EXCL.search(message)
    if m:
        hit = _find_value(m.group(1), df, prev_cols)
        if hit:
            col, val = hit
            others = [str(v) for v in df[col].dropna().astype(str).unique() if str(v).lower() != str(val).lower()]
            for d in base:
                if d.kind != "summary":
                    d.entities = {**{k: v for k, v in d.entities.items() if k != col}, col: others}
            return finalize(base), f"excluded {col} = {val}; everything else kept"
    if _FU_ALL.search(message):
        for d in base:
            d.entities, d.entity_from_context = {}, False
        return finalize(base), "the filter was removed (all records); everything else kept"
    m = _FU_SAME.search(message)
    if m:
        hit = _find_value(next(g for g in m.groups() if g), df, prev_cols)
        if hit:
            col, val = hit
            for d in base:
                if d.kind == "summary":
                    continue
                if col in d.entities or not d.entities or set(d.entities) & set(prev_cols):
                    d.entities = {**{k: v for k, v in d.entities.items() if k != col and k not in prev_cols}, col: val}
                    d.entity_from_context = False
            return finalize(base), f"{col} changed to {val}; everything else kept"
    return None
