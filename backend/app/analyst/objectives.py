"""Multi-step analytical objectives, built as programs (app/analyst/program.py) from what the question names.

An objective is a TYPE of question, not a question: its measure, entity, target and period come from the message,
the data and the conversation. Each builds a program, runs it, presents the result with its method, and stores the
program, its results and its parameters in conversation memory as named objects, so a later turn can change one
parameter ("change the target to 320 hours") or compare two scenarios.

  feasibility   can <entity> reach <target> <unit> by <period>?   (every entity when none is named: "who is at risk")
  absence       what if <entity> is absent for <N days>?          (on the open feasibility objective, or team workload)
  vs_rest       compare <entity> with the others
  change        change the target / days to <value>               (re-runs the last objective)
  compare       compare this with the previous scenario

What is measured is said plainly: recorded hours are hours, not output or productivity; a projection is the
historical daily pace carried forward, not a certainty; a scenario is hypothetical.
"""
from __future__ import annotations

import calendar
import json
import re

import pandas as pd

from app.analyst import metrics, semantics
from app.analyst import program as P
from app.analyst.data import find_column, kind_of
from app.analyst.explain import f as fmt

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30}
_FEAS = re.compile(r"\b(can|could|will|would|is|are|does|do)\b[^?]{0,60}\b(reach|complete|hit|achieve|make|meet|get to|"
                   r"finish|work|do|deliver)\b|\bon track (?:to|for)\b|\bat risk\b|\bmiss(?:ing)?\b[^?]{0,20}\b(?:target|goal|quota)",
                   re.I)
_ABSENT = re.compile(r"\b(absent|away|off sick|on leave|off work|unavailable|out of office|doesn'?t come|does not come|"
                     r"don'?t come|won'?t come|not come|isn'?t (?:there|here|available)|misses|skips)\b", re.I)
_VS_REST = re.compile(r"\bcompare\s+(?:him|her|them|it|this person|[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:with|to|against)\s+"
                      r"(?:the\s+)?(?:others|rest|everyone else|everybody else|team|colleagues)\b", re.I)
_CHANGE = re.compile(r"^\W*(?:ok(?:ay)?,?\s*)?(?:now\s+)?(?:change|set|make|use|try|what about|and)\s+(?:it\s+|this\s+|that\s+)?(?:the\s+)?(target|goal|"
                     r"days|absence|horizon|period)?\s*(?:to|=|of|with)?\s*(\d+(?:\.\d+)?|\w+)\s*(hours?|hrs?|days?)?\b", re.I)
_CMP_SCEN = re.compile(r"\bcompare\b[^?]{0,30}\b(previous|last|earlier|other|first|baseline|normal)\b[^?]{0,15}"
                       r"\b(scenario|one|result|case|version)\b|\bcompare (?:the )?scenarios\b|"
                       r"\bcompare\s+(?:these|those|both|the two|the 2|my two)\b[^?]{0,15}\b(?:scenarios?|analyses|"
                       r"versions|results|cases|options|targets)?\b|\b(?:difference|differences)\s+between\s+(?:these|"
                       r"those|the two|both)\b", re.I)


# "Will North clock 25000 km by month end?", "Is the team going to pass 900 hours this week?": whatever the verb, a
# yes/no question about the future + a stated target + a deadline is a target-feasibility question
_WILL = re.compile(r"(?:^\W*|[,;]\s*)(?:will|can|could|would|is|are|does|do|shall)\b|\b(?:going to|on course to|on pace to|set to)\b",
                   re.I)
_DEADLINE = re.compile(r"\bby (?:the )?(?:end of (?:the |this )?)?(?:month|week|quarter|year)(?: end)?\b|\b(?:month|week|"
                       r"quarter|year)[- ]end\b|\bbefore the end of\b|\bthis (?:month|week|quarter|year)\b|\bby (?:"
                       + "|".join(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                   "october", "november", "december"]) + r")\b", re.I)
_CHANGE_IF = re.compile(r"\b(?:what if|suppose|assume|assuming|say|imagine)\b[^?]{0,20}?\b(target|goal|days|absence)\b"
                        r"[^?\d]{0,20}?\b(?:is|were|was|changes? to|changed to|becomes?|goes (?:up|down) to|moves? to|of|"
                        r"rises? to|drops? to|=)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|days?)?", re.I)
_WHY_DIFF = re.compile(r"\bwhy\b[^?]{0,40}\b(different|differ|changed?|lower|higher|worse|better)\b", re.I)


def _num(tok: str) -> float | None:
    tok = tok.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        return float(tok)
    return float(NUM_WORDS[tok]) if tok in NUM_WORDS else None


def _days(message: str) -> int | None:
    m = re.search(r"\b(\d+|" + "|".join(NUM_WORDS) + r")\s*(working\s+)?(days?|weeks?)\b", message, re.I)
    if not m:
        return None
    n = _num(m.group(1))
    return int(n * (7 if m.group(3).lower().startswith("week") else 1)) if n is not None else None


def _target(message: str) -> tuple[float | None, str | None]:
    m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:working\s+|work\s+|billable\s+)?(hours?|hrs?|h|units?|sales|orders?|visits?|"
                  r"calls?|tickets?|points?|km|miles|%)?\b(?!\s*(?:days?|weeks?))", message, re.I)
    while m and re.match(r"\s*(days?|weeks?)", message[m.end():], re.I):
        m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:working\s+|work\s+)?(hours?|hrs?|h|units?|sales|orders?)?\b", message[m.end():], re.I)
    if not m:
        return None, None
    unit = (m.group(2) or "").lower()
    unit = "hours" if unit.startswith(("hour", "hr")) or unit == "h" else (unit or None)
    return float(m.group(1).replace(",", ".")), unit


def _period(message: str, df: pd.DataFrame, date: str) -> tuple[pd.Timestamp, pd.Timestamp, str, list[str]]:
    """(start, end, label, notes). 'This month/week/year' is the current calendar period; when the data has no
    records in it, the latest period with records is used instead -- said explicitly."""
    from app.analyst.scope import MONTHS, today
    notes = []
    t = today()
    d = pd.to_datetime(df[date], errors="coerce").dropna()
    d = d[d <= t]                       # a record dated after today is planned, not history: it never picks the period
    unit = "M"
    if re.search(r"\b(this|current)\s+week\b|\bweekly\b", message, re.I):
        unit = "W"
    elif re.search(r"\b(this|current)\s+year\b|\byearly\b|\bannual", message, re.I):
        unit = "Y"
    elif re.search(r"\b(this|current)\s+quarter\b|\bquarterly\b|\bby the end of the quarter\b", message, re.I):
        unit = "Q"
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\b(?:\s+(\d{4}))?", message, re.I)
    if m:
        mon = MONTHS.index(m.group(1).lower()) + 1
        yr = int(m.group(2)) if m.group(2) else (d.max().year if len(d) else t.year)
        per = pd.Period(year=yr, month=mon, freq="M")
    else:
        per = t.to_period(unit)
        if len(d) and not ((d >= per.start_time) & (d <= per.end_time)).any():
            latest = d.max().to_period(unit)
            notes.append(f"The data has no records in the current {('month', 'week', 'year', 'quarter')['MWYQ'.index(unit)]} "
                         f"({per}); the records end on {d.max().date()}, so the latest period with records ({latest}) is "
                         "used, as of its last record.")
            per = latest
    return per.start_time.normalize(), per.end_time.normalize(), str(per), notes


def _measure(message: str, df: pd.DataFrame, memory: dict):
    """(derive steps, column, label, unit, inputs, problem): the per-row measure the question is about."""
    m = metrics.recognise(message)
    if m is None and re.search(r"\b(hours?|hrs|time|work(?:ed|ing)?)\b", message, re.I):
        m = metrics.CATALOGUE_BY["shift_hours"]
    if m is not None and m.kind == "row" and metrics.physical(m, df):
        col = metrics.physical(m, df)           # the metric is recorded as a column: use it as it is
        if kind_of(df[col]) == "numeric":
            return [], col, col.replace("_", " "), "hours" if re.search(r"hour|hrs", col, re.I) else None, [col], None
    if m is not None and m.kind == "row" and not metrics.physical(m, df):
        spec, problem = metrics.row_definition(m.label, df)
        if spec is None:
            return None, None, m.label, None, [], problem or f"{m.label} cannot be derived from this data"
        steps = [{"op": "derive", "name": k, "formula": v} for k, v in spec["derive"].items()]
        inner = re.match(r"\s*sum\((\w+)\)", next(iter(spec["aggregate"].values()))).group(1)
        inputs = sorted({c for v in spec["derive"].values() for c in re.findall(r"[A-Za-z_]\w*", v) if c in df.columns})
        return steps, inner, m.label, "hours" if "hours_between" in json.dumps(spec) else None, inputs, None
    nums = [c for c in find_column(df, message) if kind_of(df[c]) == "numeric"]
    if nums:
        unit = "hours" if re.search(r"hour|hrs", nums[0], re.I) else None
        return [], nums[0], nums[0].replace("_", " "), unit, [nums[0]], None
    # "reach 3000 km": the target's unit names the measure when exactly one numeric column carries it (km_driven)
    tu = _target(message)[1]
    if tu and tu not in ("%", "hours"):
        hits = [c for c in df.columns if kind_of(df[c]) == "numeric" and
                {tu, tu.rstrip("s")} & set(re.split(r"[^a-z0-9]+", str(c).lower()))]
        if len(hits) == 1:
            return [], hits[0], str(hits[0]).replace("_", " "), tu, [hits[0]], None
    last = ((memory.get("objective") or {}).get("params") or {}).get("measure")
    if last:
        return last["steps"], last["column"], last["label"], last["unit"], last["inputs"], None
    return None, None, None, None, [], "the question does not say which value to measure"


def _entity(sc: dict | None, df: pd.DataFrame) -> tuple[str | None, str | None]:
    ents = (sc or {}).get("entities") or {}
    for col, vals in ents.items():
        if len(vals) == 1 and kind_of(df[col]) == "categorical":
            return col, vals[0]
    return None, None


def _people_col(df: pd.DataFrame) -> str | None:
    """The column that identifies a person (or unit); a readable name column is preferred over an id with the same
    number of values."""
    from app.analyst.followup import unit_column
    date = next((c for c in df.columns if kind_of(df[c]) == "datetime"), None)
    col = unit_column(df, date) if date else None
    if col:
        same = [c for c in df.columns if kind_of(df[c]) == "categorical" and c != col and re.search(r"name", str(c), re.I)
                and df[c].nunique() == df[col].nunique()]
        col = same[0] if same else col
    return col


def _derive_steps(steps, src):
    out, prev = [], src
    for i, s in enumerate(steps):
        out.append({**s, "id": f"m{i}", "from": prev})
        prev = f"m{i}"
    return out, prev


# --------------------------------------------------------------- feasibility --
_COLLECTIVE = re.compile(r"\b(?:we|our|us|in total|overall|combined|altogether|all together|as a whole|the (?:team|"
                         r"company|business|firm|organi[sz]ation|department|shop|store|branch)(?:'s)?)\b", re.I)


def duplicate_policy(message: str, current: str | None = None) -> str:
    """One rule for exact duplicate records, shared with the calculator: kept as recorded (and flagged) unless the
    user asks to exclude them; a later "include duplicates" restores them. `current` carries the conversation's
    earlier choice into a re-run ("what if the target is 320?")."""
    if metrics._DUPES.search(message):  # noqa: SLF001
        return "drop"
    if re.search(r"\b(?:includ\w*|keep\w*|count\w*)\b[^.?!]{0,15}\bduplicat\w*", message, re.I):
        return "keep"
    return current or "keep"


def feasibility_program(df, col, who, measure, target, unit, start, end, as_of, valid_cols, duplicates="keep"):
    steps, column = measure[0], measure[1]
    prog = [{"id": "sel", "op": "select", **({"entities": {col: [who]}} if col else {}), "valid_for": valid_cols,
             "historical": True, "period": {"end": str(as_of.date())},
             **({"duplicates": "keep"} if duplicates == "keep" else {})}]
    d, last = _derive_steps(steps, "sel")
    prog += d
    days_left = max((end - as_of).days, 0)
    prog += [{"id": "in_period", "op": "select", "from": last, "period": {"start": str(start.date()), "end": str(as_of.date())}},
             {"id": "done", "op": "aggregate", "from": "in_period", "fn": "sum", "column": column},
             {"id": "pace", "op": "project", "from": last, "column": column, "days": days_left,
              "history_end": str(as_of.date())},
             {"id": "verdict", "op": "compare", "a": "pace", "base": "done", "target": target, "days": days_left,
              **({"unit": unit} if unit else {}), **({"max_daily": 24} if unit == "hours" else {})}]
    return prog


def _feasibility(message, df, memory, sc, params=None):
    from app.analyst.scope import today
    date = next((c for c in df.columns if kind_of(df[c]) == "datetime"), None)
    if not date:
        return _clar("A target by a date needs a date column, and this data has none.")
    p = dict(params or {})
    if not p:
        target, unit = _target(message)
        if target is None:
            return None
        steps, column, label, munit, inputs, problem = _measure(message, df, memory)
        if problem:
            return _clar(f"I can check a target, but {problem}. Which value should reach {fmt(target)}?")
        if unit and munit and unit != munit:
            return _clar(f"The target is in {unit} but {label} is measured in {munit}. Which value should I compare?")
        col, who = _entity(sc, df)
        p = {"target": target, "unit": unit or munit, "measure": {"steps": steps, "column": column, "label": label,
                                                                    "unit": munit, "inputs": inputs},
             "entity_col": col, "entity": who, "message": message, "absent_days": 0,
             # "will WE reach 900k": one combined total, not every person or store against the whole target
             "combined": who is None and bool(_COLLECTIVE.search(message))}
        p["start"], p["end"], p["period"], p["period_notes"] = [str(x) if isinstance(x, pd.Timestamp) else x
                                                                for x in _period(message, df, date)]
    p["duplicates"] = duplicate_policy(message, p.get("duplicates"))
    start, end = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
    d = pd.to_datetime(df[date], errors="coerce")
    as_of = min(today(), end)
    if not ((d >= start) & (d <= end)).any() or today() > end:
        as_of = min(end, d[(d >= start) & (d <= end)].max() if ((d >= start) & (d <= end)).any() else end)
    m = p["measure"]
    valid = m["inputs"]
    if p["entity"]:
        targets = [(p["entity_col"], p["entity"])]
    elif p.get("combined"):
        targets = [(None, "All records combined")]
    else:
        col = _people_col(df)
        if not col:
            return _clar("Whose target? Name the person or group, for example *Can Rahul reach 300 hours this month?*")
        targets = [(col, v) for v in sorted(df[col].dropna().astype(str).unique())]
    rows, programs = [], {}
    for col, who in targets:
        prog = feasibility_program(df, col, who, (m["steps"], m["column"]), p["target"], p["unit"], start, end, as_of, valid,
                                   p["duplicates"])
        if p.get("absent_days"):
            # absent days are taken out of the days left: the projection and the required pace use the days available
            for st in prog:
                if st["op"] in ("project", "compare"):
                    st["days"] = max(st["days"] - p["absent_days"], 0)
        try:
            res = P.run(prog, df)
        except P.ProgramError as e:
            rows.append({"who": who, "error": str(e)})
            continue
        v = res["verdict"].value
        if v.get("expected_total") is None or v["expected_total"] != v["expected_total"]:
            v = {**v, "verdict": "cannot be assessed: no usable history"}
        rows.append({"who": who, **v, "pace": res["pace"].value, "excluded": res["sel"].provenance["excluded"]})
        programs[who] = (prog, {k: r.summary() for k, r in res.items()})
    return {"type": "feasibility", "params": p, "rows": rows, "programs": programs, "as_of": str(as_of.date()),
            "date": date}


def _feasibility_text(o):
    p, rows = o["params"], o["rows"]
    unit = p["unit"] or ""
    lab = p["measure"]["label"]
    scen = f" with {p['entity'] or 'each person'} absent for {p['absent_days']} working days" if p.get("absent_days") else ""
    lines = []
    good = [r for r in rows if "error" not in r]
    if len(rows) == 1 and good:
        r = good[0]
        lines.append(f"**{r['who']}: {r['verdict']}{scen}.** Target **{(fmt(p['target']) + ' ' + unit).strip()}** of {lab} for "
                     f"{p['period']}.")
        lines += ["", "| | value |", "|---|---:|",
                  f"| completed so far (to {o['as_of']}) | {fmt(r['completed'])} {unit} |",
                  f"| still needed | {fmt(r['remaining_needed'])} {unit} |",
                  f"| days left in the period | {r['days_left']} |",
                  f"| required pace | {fmt(r['required_daily']) + ' ' + unit + ' per day' if r['required_daily'] is not None and r['remaining_needed'] > 0 else '— (already reached)' if r['remaining_needed'] <= 0 else '—'} |",
                  f"| current pace (history average) | {fmt(r['current_daily'])} {unit} per day |",
                  f"| expected total on the current pace | {fmt(r['expected_total'])} {unit}"
                  + (f" (likely {fmt(r['expected_low'])}–{fmt(r['expected_high'])})" if r.get("expected_low") is not None else "")
                  + " |",
                  f"| expected gap to target | {fmt(r['gap'])} {unit} |"]
        pace = r["pace"]
        lines += ["", f"_Current pace is the average daily {lab} over {pace['history_days']} days of history "
                      f"({pace['history_start']} to {pace['history_end']}; {pace['active_days']} days with records, days "
                      "without records count as 0). The expected total assumes the coming days follow that pace; the range "
                      "assumes days vary as they did before, independently. With little history the range is wide._"]
    else:
        lines.append(f"**Target {fmt(p['target'])} {unit} of {lab} for {p['period']}{scen}** — per person, as of {o['as_of']}:")
        lines += ["", "| person | completed | required per day | current per day | expected total | assessment |",
                  "|---|---:|---:|---:|---:|---|"]
        for r in sorted(good, key=lambda r: r["gap"]):
            lines.append(f"| {r['who']} | {fmt(r['completed'])} | {fmt(r['required_daily']) if r['required_daily'] is not None else '—'} "
                         f"| {fmt(r['current_daily'])} | {fmt(r['expected_total'])} | {r['verdict']} |")
        lines.append("")
        lines.append("_Each person's current pace is their own daily average over their history; the expected total "
                     "carries that pace over the days left. Sorted by expected shortfall._")
    for n in p.get("period_notes") or []:
        lines.append(f"\n_{n}_")
    for r in rows:
        if "error" in r:
            lines.append(f"\n_{r['who']}: not computed — {r['error']}._")
    exc = good[0]["excluded"] if len(good) == 1 else []
    lines += _excluded_lines(exc)
    if p["unit"] == "hours":
        lines.append("\n_These are recorded working hours. They say nothing about output or productivity._")
    lines.append("\n_Type of answer: a **projection of the recorded pace** compared with your target — not a certainty "
                 "and not a machine-learning prediction._")
    return "\n".join(lines)


# ----------------------------------------------------------------- absence --
def _absence(message, df, memory, sc):
    days = _days(message)
    if days is None:
        return _clar("For how many days? For example *if Rahul is absent for ten days*.")
    col, who = _entity(sc, df)
    last = memory.get("objective")
    about_group = re.search(r"\b(team|teams|workload|everyone|others|colleagues|group|department|crew|staff)\b", message, re.I)
    if last and last["type"] == "feasibility" and not about_group and (who is None or who == last["params"].get("entity")):
        p = {**last["params"], "absent_days": days}
        o = _feasibility(message, df, memory, sc, params=p)
        o["scenario_of"] = last
        return o
    if who is None:
        return _clar("Who would be absent? Name the person, for example *if Rahul is absent for ten days*.")
    steps, column, label, munit, inputs, problem = _measure(message + " hours", df, memory)
    if problem:
        return _clar(f"I can model an absence, but {problem}.")
    team_col = col
    grp = re.search(r"\b(his|her|their)\s+(team|department|group|unit|crew|ward|site)\b", message, re.I)
    base_sel = {"id": "team", "op": "select", "valid_for": inputs, "historical": True,
                **({"duplicates": "keep"} if duplicate_policy(message) == "keep" else {})}
    group_note = "the whole dataset"
    if grp:
        cats = [c for c in df.columns if kind_of(df[c]) == "categorical" and c != team_col and 2 <= df[c].nunique() <= 30
                and df.loc[df[team_col].astype(str) == who, c].nunique() == 1]
        if cats:
            g = df.loc[df[team_col].astype(str) == who, cats[0]].iloc[0]
            base_sel["entities"] = {cats[0]: [str(g)]}
            group_note = f"{cats[0].replace('_', ' ')} {g}"
    prog = [base_sel]
    d, last_id = _derive_steps(steps, "team")
    prog += d
    prog += [{"id": "normal", "op": "project", "from": last_id, "column": column, "days": days},
             {"id": "without", "op": "scenario", "from": last_id, "name": f"{who} absent {days} days",
              "changes": [{"remove": {team_col: who}}]},
             {"id": "reduced", "op": "project", "from": "without", "column": column, "days": days},
             {"id": "cmp", "op": "compare", "a": "reduced", "b": "normal", "a_label": f"{who} absent", "b_label": "normal"}]
    try:
        res = P.run(prog, df)
    except P.ProgramError as e:
        return _clar(f"I couldn't model that absence: {e}.")
    # people who share the work: those with valid history in the group (not records that were left out)
    n_others = int(res["without"].value[team_col].nunique())
    return {"type": "absence", "params": {"entity": who, "entity_col": team_col, "days": days, "group": group_note,
                                          "measure": {"steps": steps, "column": column, "label": label, "unit": munit,
                                                      "inputs": inputs}},
            "normal": res["normal"].value, "reduced": res["reduced"].value, "cmp": res["cmp"].value, "others": n_others,
            "program": prog, "results": {k: r.summary() for k, r in res.items()}, "unit": munit}


def _absence_text(o):
    p, u = o["params"], o["unit"] or ""
    n, r = o["normal"], o["reduced"]
    lost = n["expected"] - r["expected"]
    per = lost / o["others"] if o["others"] else None
    lines = [f"**Scenario: {p['entity']} absent for {p['days']} days** — {p['group']}, {p['measure']['label']}.", "",
             "| | normal | with the absence | difference |", "|---|---:|---:|---:|",
             f"| expected {p['measure']['label']} over {p['days']} days | {fmt(n['expected'])} {u} | {fmt(r['expected'])} {u} | {fmt(-lost)} {u} |"]
    if per is not None:
        lines.append(f"| extra per remaining person, if the same work must be covered | — | {fmt(per)} {u} | +{fmt(per)} {u} |")
    lines += ["", f"_Normal = the group's average daily {p['measure']['label']} over its history ({n['history_start']} to "
                  f"{n['history_end']}), carried over {p['days']} days. With the absence, {p['entity']}'s records are "
                  f"removed and the rest carried over the same days. {o['others']} other people share the work._",
              "", "_This is a **hypothetical scenario**, not a forecast: it assumes everything else continues at its "
                  "recorded pace. It measures recorded hours of work, not workload in tasks, output or productivity, "
                  "which this data does not record._"]
    return "\n".join(lines)


def _scenario_feasibility_text(o):
    base = o["scenario_of"]
    a, b = base["rows"], o["rows"]
    text = _feasibility_text(o)
    if len(a) == 1 and len(b) == 1 and "error" not in a[0] and "error" not in b[0]:
        x, y = a[0], b[0]
        text = (f"**Scenario: {o['params']['entity']} absent for {o['params']['absent_days']} working days.**\n\n"
                f"| | as planned | with the absence |\n|---|---:|---:|\n"
                f"| expected total | {fmt(x['expected_total'])} | {fmt(y['expected_total'])} |\n"
                f"| gap to target | {fmt(x['gap'])} | {fmt(y['gap'])} |\n| assessment | {x['verdict']} | {y['verdict']} |\n\n"
                + text + "\n\n_The absent days are taken out of the days left; the target and period stay the same. "
                         "Hypothetical scenario._")
    return text


# ---------------------------------------------------------------- vs rest --
def _vs_rest(message, df, memory, sc):
    col, who = _entity(sc, df)
    if not who:
        prev = (memory.get("scope") or {}).get("entities") or {}
        col, who = next(((c, v[0]) for c, v in prev.items() if len(v) == 1), (None, None))
    if not who:
        return _clar("Compare whom with the others? Name the person or group.")
    steps, column, label, munit, inputs, problem = _measure(message, df, memory)
    if problem:
        last = memory.get("last_calc")
        if last and last["result"].get("primary"):
            m = metrics.CATALOGUE_BY.get(last["result"]["primary"])
            if m:
                steps, column, label, munit, inputs, problem = _measure(m.label, df, memory)
    if problem:                          # the value the last answer computed (a column of the data)
        ev = [e for e in ((memory.get("last_answer") or {}).get("evidence") or []) if isinstance(e, dict)]
        col_used = next((e["column"] for e in ev if e.get("column") in df.columns and kind_of(df[e["column"]]) == "numeric"), None)
        if col_used:
            steps, column, label, munit, inputs, problem = [], col_used, col_used.replace("_", " "), \
                ("hours" if re.search(r"hour|hrs", col_used, re.I) else None), [col_used], None
    if problem:
        return _clar(f"Compare {who} with the others on which value? {problem}.")
    prog = [{"id": "all", "op": "select", "valid_for": inputs, "historical": False,
             **({"duplicates": "keep"} if duplicate_policy(message) == "keep" else {})}]
    d, last_id = _derive_steps(steps, "all")
    prog += d + [{"id": "per", "op": "aggregate", "from": last_id, "fn": "sum", "column": column, "by": col}]
    try:
        res = P.run(prog, df)
    except P.ProgramError as e:
        return _clar(f"I couldn't compare: {e}.")
    series = res["per"].value
    me = series.get(who)
    others = {k: v for k, v in series.items() if k != who and v is not None}
    if me is None or not others:
        return _clar(f"{who} has no valid records of {label} to compare.")
    vals = sorted(others.values())
    rank = 1 + sum(v > me for v in others.values())
    return {"type": "vs_rest", "params": {"entity": who, "entity_col": col, "measure": {"steps": steps, "column": column,
                                                                                          "label": label, "unit": munit,
                                                                                          "inputs": inputs}},
            "me": me, "others": others, "mean": sum(vals) / len(vals), "median": vals[len(vals) // 2] if len(vals) % 2 else
            (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2, "rank": rank, "n": len(others) + 1,
            "program": prog, "results": {k: r.summary() for k, r in res.items()}, "unit": munit,
            "excluded": res["all"].provenance["excluded"]}


def _vs_rest_text(o):
    p, u = o["params"], o["unit"] or ""
    lines = [f"**{p['entity']}: {fmt(o['me'])} {u} of {p['measure']['label']}** — ranked {o['rank']} of {o['n']}. "
             f"The others average {fmt(o['mean'])} {u} (median {fmt(o['median'])}).", "",
             f"| {p['entity_col'].replace('_', ' ')} | total {p['measure']['label']} |", "|---|---:|"]
    for k, v in sorted({**o["others"], p["entity"]: o["me"]}.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {'**' + k + '**' if k == p['entity'] else k} | {fmt(v)} |")
    lines += _excluded_lines(o["excluded"])
    lines.append("\n_Totals over all valid records per person; people with more records naturally have larger totals._")
    return "\n".join(lines)


def _excluded_lines(excluded) -> list[str]:
    """Records left out, and duplicates counted as recorded -- the same wording on every objective."""
    left_out = [e for e in excluded if not e.get("kept")]
    kept = [e for e in excluded if e.get("kept")]
    out = []
    if left_out:
        out.append("\n_Not counted: " + "; ".join(f"{e['records']} record(s) — {e['reason']} (lines "
                                                 f"{', '.join(map(str, e['lines']))})" for e in left_out) + "._")
    if kept:
        out.append("\n_Counted as recorded: " + "; ".join(f"{e['records']} record(s) — {e['reason']} (lines "
                                                         f"{', '.join(map(str, e['lines']))})" for e in kept)
                   + ". Ask to exclude duplicates if they should not count._")
    return out


# ------------------------------------------------------------------ closure --
# "How much would takings drop if the Airport outlet closed?", "...over the next 7 days if the South depot were shut,
# compared with carrying on normally": a named group taken out of the data -- a hypothetical scenario, never a filter
# to that group's own records
_CLOSE = re.compile(r"\b(?:clos(?:ed|es|e|ing|ure)|shut(?:s|ting)?(?:\s+down)?|(?:were|was|is|got|get) (?:removed|dropped|"
                    r"cut|scrapped)|discontinu\w*|stopp?(?:ed|s)? (?:trading|operating)|ceased? (?:trading|operating)|"
                    r"no longer (?:operat|exist|trad)\w*)\b", re.I)
_IF = re.compile(r"\b(?:if|what if|suppose|imagine|assum\w*|were to|would|had)\b", re.I)
_HORIZON = re.compile(r"\b(?:next|coming|over|for|in)\s+(?:the\s+)?(?:next\s+|coming\s+)?(\d+|" + "|".join(NUM_WORDS)
                      + r")\s+(day|week|month)s?\b", re.I)


def _closure(message, df, memory, sc):
    from app.analyst import intent
    from app.analyst.calc import named_values
    named = [(c, v) for c, v in named_values(df, message)]
    hit = [(c, v) for c, v in named if re.search(rf"\b{re.escape(str(v))}\b[^.?!]{{0,40}}{_CLOSE.pattern}|"
                                                 rf"{_CLOSE.pattern}[^.?!]{{0,20}}\b{re.escape(str(v))}\b", message, re.I)]
    if len(hit) != 1:
        return _clar("Which one would close? Name a single value, for example *if the North depot closed*.") if hit else None
    col, who = hit[0]
    req = intent.read(message, df)
    nums = [c for c in req.measures if c in df.columns and kind_of(df[c]) == "numeric"
            and c not in (col,) and not str(c).startswith("is_")]
    if len(nums) != 1 and not nums:
        only = [c for c in df.columns if c != col and kind_of(df[c]) == "numeric" and not str(c).startswith("is_")
                and semantics.column_role(df, c)["role"] not in ("identifier", "flag", "year")]
        nums = only if len(only) == 1 else nums       # the dataset's one measure: nothing else could be meant
    if len(nums) != 1 and (memory.get("planned") or memory.get("program") or memory.get("last_calc")):
        return None                     # "...compared with the total you just worked out": an earlier result is meant
    if len(nums) != 1:
        opts = [c for c in df.columns if kind_of(df[c]) == "numeric"][:6]
        return _clar(f"If {who} closed, which value should I measure the effect on? Options: "
                     + ", ".join(f"`{c}`" for c in opts) + ".")
    m = nums[0]
    hm = _HORIZON.search(message)
    days = None
    if hm:
        n = _num(hm.group(1))
        days = int(n * {"day": 1, "week": 7, "month": 30}[hm.group(2).lower()]) if n is not None else None
    others = {c: v for c, v in ((sc or {}).get("entities") or {}).items() if c != col}
    dup = duplicate_policy(message)
    prog = [{"id": "all", "op": "select", **({"entities": others} if others else {}), "valid_for": [m, col],
             "historical": days is not None, **({"duplicates": "keep"} if dup == "keep" else {})},
            {"id": "without", "op": "scenario", "from": "all", "name": f"{who} closed",
             "changes": [{"remove": {col: who}}]}]
    if days is not None:
        prog += [{"id": "normal", "op": "project", "from": "all", "column": m, "days": days},
                 {"id": "closed", "op": "project", "from": "without", "column": m, "days": days}]
    else:
        prog += [{"id": "normal", "op": "aggregate", "from": "all", "fn": "sum", "column": m},
                 {"id": "closed", "op": "aggregate", "from": "without", "fn": "sum", "column": m}]
    prog += [{"id": "cmp", "op": "compare", "a": "closed", "b": "normal", "a_label": f"{who} closed",
              "b_label": "carrying on as recorded"}]
    try:
        res = P.run(prog, df)
    except P.ProgramError as e:
        return _clar(f"I couldn't model that closure: {e}.")
    # the closed value names what is removed, not the records the answer is limited to: the scope check and the
    # scope kept for follow-ups are the remaining entities only
    memory["_scope"] = {**sc, "entities": others} if sc and others else None
    return {"type": "closure", "params": {"entity": who, "entity_col": col, "days": days, "measure": m,
                                          "duplicates": dup},
            "normal": res["normal"].value, "closed": res["closed"].value, "cmp": res["cmp"].value,
            "hypothetical": res["cmp"].hypothetical, "program": prog,
            "results": {k: r.summary() for k, r in res.items()}, "excluded": res["all"].provenance["excluded"]}


def _closure_text(o):
    p, c = o["params"], o["cmp"]
    lab = p["measure"].replace("_", " ")
    normal, closed = c["b"], c["a"]
    lost = normal - closed
    span = f"over the next {p['days']} days" if p["days"] else "in total, over the records in this data"
    lines = [f"**Scenario (hypothetical): {p['entity']} closed** — {lab} {span}.", "",
             "| | carrying on as recorded | with " + p["entity"] + " closed | difference |", "|---|---:|---:|---:|",
             f"| {'expected ' if p['days'] else 'total '}{lab} | {fmt(normal)} | {fmt(closed)} | {fmt(-lost)} |",
             "", f"**{fmt(lost)} {lab}** would be lost ({lost / normal * 100:.1f}% of the total)." if normal else ""]
    if p["days"]:
        n = o["normal"]
        lines.append(f"\n_Expected = the average daily {lab} over the history ({n['history_start']} to "
                     f"{n['history_end']}; days without records count as 0), carried over {p['days']} days — with and "
                     f"without {p['entity']}'s records._")
    else:
        lines.append(f"\n_The recorded total with and without {p['entity']}'s records ({p['entity_col'].replace('_', ' ')})._")
    lines += _excluded_lines(o["excluded"])
    lines.append(f"\n_This is a **hypothetical scenario**, not a forecast: it assumes everything else carries on as "
                 f"recorded and none of {p['entity']}'s activity moves elsewhere. Nothing in the data was changed._")
    return "\n".join(x for x in lines if x is not None)


# ------------------------------------------------------------------ entry --
def _clar(text):
    return {"type": "clarify", "text": text}


def handle(message: str, df: pd.DataFrame, memory: dict, sc: dict | None) -> dict | None:
    """An objective's answer {answer, calls, charts, notes, plan} or None when the message is not one."""
    last = memory.get("objective")
    o = None
    if last and (_CMP_SCEN.search(message) or _WHY_DIFF.search(message)):
        hist = memory.get("objectives") or []
        if len(hist) < 2:
            return _reply_clar("There is only one scenario in this conversation so far, so there is nothing to compare "
                               "it with. Change an assumption first, for example *change the target to 320 hours*.")
        if hist[-2]["type"] == hist[-1]["type"]:
            return _compare_scenarios(hist[-2], hist[-1])
        # the last two are different kinds of analysis (a target, then a closure): they share no figures, so the two
        # most recent analyses of the same kind are compared, and the answer says so
        pair = next(((i, j) for j in range(len(hist) - 1, 0, -1) for i in range(j - 1, -1, -1)
                     if hist[i]["type"] == hist[j]["type"]), None)
        if pair is None:
            return _reply_clar("The analyses in this conversation are of different kinds ("
                               + ", ".join(dict.fromkeys(_KIND.get(o["type"], o["type"]) for o in hist))
                               + "), so they have no figures in common to compare. Change an assumption of one of "
                               "them first, for example *change the target to 45000*.")
        out = _compare_scenarios(hist[pair[0]], hist[pair[1]])
        out["answer"] = (f"_The latest analysis ({_KIND.get(hist[-1]['type'], hist[-1]['type'])}) has no earlier one of "
                         f"the same kind, so the two most recent {_KIND.get(hist[pair[1]]['type'], hist[pair[1]]['type'])}"
                         " analyses are compared._\n\n" + out["answer"].replace(
                             "Comparing the last two analyses in this conversation.",
                             f"Comparing two {_KIND.get(hist[pair[1]]['type'], hist[pair[1]]['type'])} analyses in "
                             "this conversation."))
        return out
    ch = _CHANGE.match(message)
    alt = _CHANGE_IF.search(message)
    if alt and not last and alt.group(1).lower() in ("target", "goal"):
        return _reply_clar(f"There is no target in this conversation yet, so there is nothing to change to "
                           f"{alt.group(2)}. Ask for one first — for example *Can <name> reach {alt.group(2)} "
                           "<measure> this month?*")
    if last and not (ch and ch.group(2) and _num(ch.group(2)) is not None) and alt:
        # "what if the target changes to 320?", "suppose the goal were 320 hours": the same one-parameter change
        message = f"change the {alt.group(1).lower()} to {alt.group(2)} {alt.group(3) or ''}".strip()
        ch = _CHANGE.match(message)
    if last and ch and len(message.split()) <= 9 and ch.group(2) and _num(ch.group(2)) is not None:
        what = (ch.group(1) or ch.group(3) or "").lower()
        val = _num(ch.group(2))
        p = json.loads(json.dumps(last["params"]))
        if last["type"] == "feasibility" and (what in ("target", "goal") or (ch.group(3) or "").lower().startswith(("hour", "hr"))):
            p["target"] = val
            o = _feasibility(message, df, memory, sc, params=p)
            o["changed"] = f"target changed to {fmt(val)} {p.get('unit') or ''}".strip()
        elif what.startswith(("day", "absence")) or (ch.group(3) or "").lower().startswith("day"):
            if last["type"] == "absence":
                return None if val is None else _run_and_store(_absence(f"absent for {int(val)} days", df, memory,
                                                                         {"entities": {p["entity_col"]: [p["entity"]]}}),
                                                               memory, f"absence changed to {int(val)} days", df=df)
            if last["type"] == "feasibility":
                p["absent_days"] = int(val)
                o = _feasibility(message, df, memory, sc, params=p)
                o["scenario_of"] = memory.get("objectives", [last])[0]
                o["changed"] = f"absence changed to {int(val)} days"
        if o is not None:
            return _run_and_store(o, memory, df=df)
    if _VS_REST.search(message):
        return _run_and_store(_vs_rest(message, df, memory, sc), memory, df=df)
    if _CLOSE.search(message) and _IF.search(message) and not _ABSENT.search(message):
        o = _closure(message, df, memory, sc)
        if o is not None:
            return _run_and_store(o, memory, df=df)
    if _ABSENT.search(message) and (_days(message) is not None or last):
        return _run_and_store(_absence(message, df, memory, sc), memory, df=df)
    if (_FEAS.search(message) or (_WILL.search(message) and _DEADLINE.search(message))) and _target(message)[0] is not None:
        return _run_and_store(_feasibility(message, df, memory, sc), memory, df=df)
    return None


def _reply_clar(text):
    return {"answer": text, "calls": [], "charts": [], "notes": ["Nothing was calculated."],
            "plan": {"type": "clarify", "objective": True}}


def _verification(o, df) -> list[dict]:
    from app.analyst.verify import objective_checks  # noqa: PLC0415 - one verification interface for every path
    return objective_checks(o, df)


def _run_and_store(o, memory, changed=None, df=None):
    if o is None:
        return None
    if o["type"] == "clarify":
        return _reply_clar(o["text"])
    checks = _verification(o, df)
    if not all(c["ok"] for c in checks):
        return {"answer": "**I computed this, but an independent check of the figure did not match**, so I haven't "
                          "shown it. " + "; ".join(c["detail"] for c in checks if not c["ok"]) + ".",
                "calls": [], "charts": [], "notes": ["Withheld by Silt's answer verification."],
                "plan": {"type": "error", "objective": o["type"]}, "verification": checks}
    if changed:
        o["changed"] = changed
    if o["type"] == "feasibility":
        text = _scenario_feasibility_text(o) if o.get("scenario_of") else _feasibility_text(o)
    elif o["type"] == "absence":
        text = _absence_text(o)
    elif o["type"] == "closure":
        text = _closure_text(o)
    else:
        text = _vs_rest_text(o)
    if o.get("changed"):
        text = f"_Re-ran the previous analysis with one change: {o['changed']}._\n\n" + text
    stored = {k: v for k, v in o.items() if k not in ("scenario_of",)}
    memory["objective"] = stored
    memory.setdefault("objectives", []).append(stored)
    del memory["objectives"][:-6]
    progs = o.get("programs") or ({"": (o["program"], o["results"])} if "program" in o else {})
    memory["program"] = {k: {"steps": v[0], "results": v[1]} for k, v in progs.items()}
    charts = _charts(o)
    calls = [{"tool": "analysis_program", "status": "ok",
              "result": {"objective": o["type"], "programs": {k: v[0] for k, v in progs.items()},
                         "results": {k: v[1] for k, v in progs.items()}}}]
    return {"answer": text, "calls": calls, "charts": charts,
            "notes": ["Answered by an analysis program: each step (select, derive, aggregate, project, compare, "
                      "scenario) was validated and run deterministically; the steps are under *How this was "
                      "calculated*."], "plan": {"type": "program", "objective": o["type"]},
            "details": _details(progs), "verification": checks}


def _details(progs):
    out = ["**Analysis program**"]
    for who, (prog, res) in list(progs.items())[:3]:
        if who:
            out.append(f"\n*{who}*")
        for s in prog:
            r = res.get(s["id"], {})
            v = r.get("value")
            shown = (v if isinstance(v, (int, float)) else (f"{v.get('rows')} rows" if isinstance(v, dict) and "rows" in v
                     else (f"expected {fmt(v['expected'])}" if isinstance(v, dict) and "expected" in v else
                           (v.get("verdict") if isinstance(v, dict) and "verdict" in v else ""))))
            args = {k: x for k, x in s.items() if k not in ("id", "op", "from")}
            out.append(f"- `{s['id']}` **{s['op']}** {json.dumps(args, default=str)[:160]} → {shown}")
    return "\n".join(out)


def _charts(o):
    if o["type"] == "feasibility":
        good = [r for r in o["rows"] if "error" not in r]
        if len(good) > 1:
            return [{"type": "bar", "x": "person", "y": "expected total", "additive": False,
                     "title": f"Expected total vs target {fmt(o['params']['target'])}",
                     "data": [{"person": r["who"], "expected total": r["expected_total"]} for r in good]}]
        if good:
            r = good[0]
            return [{"type": "bar", "x": "measure", "y": "value", "additive": False, "title": "Target, completed and expected",
                     "data": [{"measure": "completed", "value": r["completed"]}, {"measure": "expected total", "value": r["expected_total"]},
                              {"measure": "target", "value": r["target"]}]}]
    if o["type"] == "absence":
        return [{"type": "bar", "x": "scenario", "y": "expected", "additive": False, "title": "Expected hours: normal vs absence",
                 "data": [{"scenario": "normal", "expected": o["normal"]["expected"]},
                          {"scenario": "with absence", "expected": o["reduced"]["expected"]}]}]
    if o["type"] == "closure":
        return [{"type": "bar", "x": "scenario", "y": o["params"]["measure"], "additive": False,
                 "title": f"{o['params']['measure'].replace('_', ' ')}: as recorded vs {o['params']['entity']} closed "
                          "(hypothetical)",
                 "data": [{"scenario": "as recorded", o["params"]["measure"]: o["cmp"]["b"]},
                          {"scenario": f"{o['params']['entity']} closed", o["params"]["measure"]: o["cmp"]["a"]}]}]
    if o["type"] == "vs_rest":
        data = [{"person": k, "total": v} for k, v in sorted({**o["others"], o["params"]["entity"]: o["me"]}.items(),
                                                             key=lambda kv: -kv[1])]
        return [{"type": "bar", "x": "person", "y": "total", "additive": True, "title": "Totals per person", "data": data}]
    return []


def _key(o):
    if o["type"] == "feasibility":
        r = next((x for x in o["rows"] if "error" not in x), {})
        return {"target": o["params"]["target"], "absent days": o["params"].get("absent_days", 0),
                "expected total": r.get("expected_total"), "gap": r.get("gap"), "required per day": r.get("required_daily"),
                "assessment": r.get("verdict")}
    if o["type"] == "absence":
        return {"days": o["params"]["days"], "normal": o["normal"]["expected"], "with absence": o["reduced"]["expected"],
                "difference": o["reduced"]["expected"] - o["normal"]["expected"]}
    if o["type"] == "closure":
        return {"closed": o["params"]["entity"], "days": o["params"]["days"], "as recorded": o["cmp"]["b"],
                "with the closure": o["cmp"]["a"], "difference": o["cmp"]["difference"]}
    return {"value": o.get("me"), "rank": o.get("rank")}


_KIND = {"feasibility": "target", "absence": "absence", "closure": "closure", "vs_rest": "comparison with the rest"}


def _compare_scenarios(a, b):
    ka, kb = _key(a), _key(b)
    keys = [k for k in ka if k in kb]
    if not keys:                        # different kinds of analysis: listed, not set against each other
        lines = ["**These two analyses are of different kinds, so they have no figures in common to compare.**", ""]
        for lab, o, k in (("previous", a, ka), ("latest", b, kb)):
            lines.append(f"- **{lab}** ({_KIND.get(o['type'], o['type'])}): " + "; ".join(
                f"{n} {fmt(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}" for n, v in k.items()
                if v is not None))
        return {"answer": "\n".join(lines), "calls": [{"tool": "compare_scenarios", "status": "ok",
                                                       "result": {"previous": ka, "latest": kb}}],
                "charts": [], "notes": ["Both analyses were computed earlier in this conversation; nothing was "
                                        "recalculated."], "plan": {"type": "program", "objective": "compare"}}
    lines = ["**Comparing the last two analyses in this conversation.**", "", "| | previous | latest | change |",
             "|---|---:|---:|---:|"]
    for k in keys:
        x, y = ka[k], kb[k]
        diff = fmt(y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x is not None and y is not None else ""
        lines.append(f"| {k} | {fmt(x) if isinstance(x, (int, float)) else x} | {fmt(y) if isinstance(y, (int, float)) else y} | {diff} |")
    why = [k for k in keys if ka[k] != kb[k] and k in ("target", "absent days", "days")]
    if why:
        lines.append("\n_The results differ because " + ", ".join(f"the {k} changed ({fmt(ka[k]) if isinstance(ka[k], (int, float)) else ka[k]} → "
                                                                    f"{fmt(kb[k]) if isinstance(kb[k], (int, float)) else kb[k]})" for k in why)
                     + "; everything else is the same data and method._")
    nums = [k for k in keys if isinstance(ka[k], (int, float)) and isinstance(kb[k], (int, float))
            and ka[k] is not None and kb[k] is not None and k not in ("absent days", "days")]
    charts = [{"type": "bar", "x": "measure", "y": "value", "additive": False, "title": "Previous vs latest analysis",
               "data": [{"measure": f"{k} ({lab})", "value": v[k]} for k in nums for lab, v in (("previous", ka), ("latest", kb))]}
              ] if nums else []
    return {"answer": "\n".join(lines), "calls": [{"tool": "compare_scenarios", "status": "ok", "result": {"previous": ka, "latest": kb}}],
            "charts": charts, "notes": ["Both analyses were computed earlier in this conversation; nothing was recalculated."],
            "plan": {"type": "program", "objective": "compare"}}
