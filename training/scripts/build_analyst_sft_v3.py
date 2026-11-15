"""v3 training data: calculator plans, multi-turn follow-ups and honest clarification, on unfamiliar schemas.

Why (audit of v1, see the v3 report): every v1 example is single-turn, there are no examples of the calculator's
plan format (derive / exclude / group_by / aggregate) that the application asks the model for when a question sets
its own formula or conditions, and nothing teaches the model to ask -- rather than invent a proxy -- when a
condition is not in the data.

How:
  * Synthetic tables from a set of domains, each written in a randomly chosen naming style (snake_case, camelCase,
    Title Case, abbreviations, UPPER), with random sizes and values. 14 domains train; 4 more are held out entirely
    and appear only in the evaluation split.
  * Questions in informal, varied wording from intent templates; the gold answer is a calculator plan, or a
    clarify / decline object.
  * The prompt is captured from the application's own calculator (app.analyst.calc.model_plan) with the model
    stubbed, so training sees exactly what inference sends -- including conversation history for follow-ups.
  * Every gold plan must pass the application's validator (calc.check) and run (calc.run); its result is stored
    so the evaluation can compare executed results, not text.
  * Conversations stay whole: a follow-up and its first turn are one example group, kept in one split. Splits are
    by dataset instance; normalised questions are de-duplicated across splits.
  * A replay sample of v1's own training data keeps v1's planner, narration and chat behaviour.

    .venv313/bin/python training/scripts/build_analyst_sft_v3.py
Writes training/data/analyst_sft_v3/ (refuses to overwrite).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.analyst import calc  # noqa: E402

OUT = ROOT / "training/data/analyst_sft_v3"
SEED = 20261114
TRAIN_DOMAINS = ["retail", "hotel", "clinic", "shipping", "saas", "callcenter", "factory", "restaurant", "gym",
                 "library", "energy", "insurance", "timesheet", "marketing"]
HELDOUT_DOMAINS = ["airline", "farm", "charity", "carrental"]

# ------------------------------------------------------------------- naming --
ABBR = {"quantity": "qty", "amount": "amt", "department": "dept", "number": "num", "customer": "cust",
        "price": "prc", "category": "cat", "hours": "hrs", "revenue": "rev", "minutes": "mins", "total": "tot"}


def styler(rng: random.Random):
    style = rng.choice(["snake", "camel", "title", "abbr", "upper", "snake"])

    def name(words: str) -> str:
        w = words.split()
        if style == "snake":
            return "_".join(w)
        if style == "camel":
            return w[0] + "".join(x.capitalize() for x in w[1:])
        if style == "title":
            return " ".join(x.capitalize() for x in w)
        if style == "abbr":
            return "_".join(ABBR.get(x, x) for x in w)
        return "_".join(w).upper()
    return name, style


# ------------------------------------------------------------------ domains --
# Each domain: entity (people/things), groups, amount pairs, flag, status, times or dates. Words are canonical;
# the styler turns them into column names.
FIRST = ["Ana", "Bilal", "Chen", "Dara", "Eitan", "Femi", "Gita", "Hugo", "Ines", "Jonas", "Kofi", "Lena", "Mateo",
         "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei", "Ximena", "Yusuf", "Zoe"]
LAST = ["Alvarez", "Brandt", "Cho", "Diallo", "Eriksen", "Farouk", "Gomez", "Hale", "Ito", "Jensen", "Kowal", "Lund",
        "Mbeki", "Novak", "Okoro", "Pereira", "Quist", "Rossi", "Sato", "Tan", "Ueda", "Varga", "Weber", "Yilmaz"]

D = {
    "retail": dict(entity=("store manager", "people"), groups=[("region", ["North", "South", "East", "West"]),
                   ("product line", ["Home", "Garden", "Toys", "Tools"])], qty="units sold", price="unit price",
                   cost="cost of goods", revenue="sales amount", flag=("returned", "Yes", "No"),
                   status=("order status", ["Complete", "Cancelled", "Pending"], "Cancelled"), date="order date"),
    "hotel": dict(entity=("guest name", "people"), groups=[("room type", ["Single", "Double", "Suite"]),
                  ("booking channel", ["Direct", "Agency", "Online"])], qty="nights", price="nightly rate",
                  cost="housekeeping cost", revenue="total paid", flag=("breakfast included", "Yes", "No"),
                  status=("booking status", ["Stayed", "Cancelled", "No-show"], "Cancelled"), date="check in date"),
    "clinic": dict(entity=("doctor", "people"), groups=[("ward", ["A", "B", "C"]), ("visit type", ["Routine", "Urgent", "Follow-up"])],
                   qty="tests ordered", price="test fee", cost="supply cost", revenue="amount billed",
                   flag=("readmitted", "Yes", "No"), status=("visit status", ["Seen", "Cancelled", "Rescheduled"], "Cancelled"),
                   times=("arrival time", "discharge time"), date="visit date"),
    "shipping": dict(entity=("driver", "people"), groups=[("carrier", ["FastCo", "ShipIt", "Roadline"]),
                     ("destination region", ["Coast", "Inland", "Islands"])], qty="parcels", price="rate per parcel",
                     cost="fuel cost", revenue="invoice amount", flag=("delivered late", "Yes", "No"),
                     status=("shipment status", ["Delivered", "Lost", "Returned"], "Lost"), date="ship date"),
    "saas": dict(entity=("account manager", "people"), groups=[("plan tier", ["Basic", "Pro", "Enterprise"]),
                 ("country", ["DE", "FR", "US", "IN"])], qty="seats", price="price per seat", cost="support cost",
                 revenue="monthly bill", flag=("churned", "Yes", "No"),
                 status=("subscription status", ["Active", "Cancelled", "Trial"], "Cancelled"), date="renewal date"),
    "callcenter": dict(entity=("agent", "people"), groups=[("team", ["Alpha", "Bravo", "Delta"]),
                       ("channel", ["Phone", "Chat", "Email"])], qty="calls handled", price="cost per call",
                       cost="overtime cost", revenue="sales closed", flag=("escalated", "Yes", "No"),
                       status=("ticket status", ["Resolved", "Abandoned", "Open"], "Abandoned"),
                       times=("shift start", "shift end"), date="work date"),
    "factory": dict(entity=("line supervisor", "people"), groups=[("production line", ["L1", "L2", "L3"]),
                    ("shift", ["Day", "Night"])], qty="units produced", price="unit value", cost="scrap cost",
                    revenue="output value", flag=("failed inspection", "Yes", "No"),
                    status=("batch status", ["Released", "Scrapped", "Held"], "Scrapped"), date="batch date"),
    "restaurant": dict(entity=("server", "people"), groups=[("branch", ["Centre", "Harbour", "Uptown"]),
                       ("meal period", ["Lunch", "Dinner"])], qty="covers", price="average spend", cost="food cost",
                       revenue="bill total", flag=("tip left", "Yes", "No"),
                       status=("order state", ["Paid", "Voided", "Comped"], "Voided"), date="service date"),
    "gym": dict(entity=("trainer", "people"), groups=[("club", ["Riverside", "Hilltop", "Docks"]),
                ("membership level", ["Silver", "Gold", "Platinum"])], qty="sessions", price="session fee",
                cost="equipment cost", revenue="fees collected", flag=("renewed", "Yes", "No"),
                status=("member status", ["Active", "Frozen", "Cancelled"], "Cancelled"), date="join date"),
    "library": dict(entity=("librarian", "people"), groups=[("branch", ["Main", "East", "Campus"]),
                    ("genre", ["Fiction", "History", "Science"])], qty="loans", price="late fee rate",
                    cost="repair cost", revenue="fines collected", flag=("overdue", "Yes", "No"),
                    status=("loan status", ["Returned", "Lost", "Renewed"], "Lost"), date="loan date"),
    "energy": dict(entity=("site manager", "people"), groups=[("site", ["Plant A", "Plant B", "Plant C"]),
                   ("tariff", ["Peak", "Off-peak"])], qty="kwh used", price="price per kwh", cost="maintenance cost",
                   revenue="energy bill", flag=("outage", "Yes", "No"),
                   status=("meter status", ["Read", "Estimated", "Faulty"], "Faulty"), date="reading date"),
    "insurance": dict(entity=("adjuster", "people"), groups=[("policy type", ["Auto", "Home", "Travel"]),
                      ("state", ["NSW", "VIC", "QLD"])], qty="claims filed", price="average claim", cost="admin cost",
                      revenue="premium collected", flag=("fraud flagged", "Yes", "No"),
                      status=("claim status", ["Paid", "Denied", "Pending"], "Denied"), date="claim date"),
    "timesheet": dict(entity=("worker", "people"), groups=[("crew", ["Red", "Blue", "Green"]),
                      ("site", ["Depot", "Yard", "Office"])], qty="hours billed", price="pay rate", cost="travel cost",
                      revenue="invoice total", flag=("overtime approved", "Yes", "No"),
                      status=("entry status", ["Approved", "Rejected", "Draft"], "Rejected"),
                      times=("start time", "finish time"), date="work date"),
    "marketing": dict(entity=("campaign owner", "people"), groups=[("channel", ["Search", "Social", "Display"]),
                      ("market", ["UK", "IE", "NL"])], qty="clicks", price="cost per click", cost="ad spend",
                      revenue="attributed revenue", flag=("converted", "Yes", "No"),
                      status=("campaign status", ["Live", "Paused", "Stopped"], "Stopped"), date="report date"),
    # ---- held out: never in training
    "airline": dict(entity=("pilot", "people"), groups=[("route", ["LHR-JFK", "SIN-SYD", "CDG-DXB"]),
                    ("cabin", ["Economy", "Business"])], qty="passengers", price="average fare", cost="fuel burn cost",
                    revenue="ticket revenue", flag=("delayed", "Yes", "No"),
                    status=("flight status", ["Landed", "Cancelled", "Diverted"], "Cancelled"),
                    times=("departure time", "arrival time"), date="flight date"),
    "farm": dict(entity=("grower", "people"), groups=[("field", ["F1", "F2", "F3"]), ("crop", ["Wheat", "Barley", "Oats"])],
                 qty="tonnes harvested", price="price per tonne", cost="fertiliser cost", revenue="crop income",
                 flag=("irrigated", "Yes", "No"), status=("harvest status", ["Sold", "Spoiled", "Stored"], "Spoiled"),
                 date="harvest date"),
    "charity": dict(entity=("fundraiser", "people"), groups=[("appeal", ["Winter", "Water", "Schools"]),
                    ("donor segment", ["Regular", "One-off", "Corporate"])], qty="gifts", price="average gift",
                    cost="processing cost", revenue="donations received", flag=("gift aid", "Yes", "No"),
                    status=("pledge status", ["Received", "Lapsed", "Promised"], "Lapsed"), date="gift date"),
    "carrental": dict(entity=("branch agent", "people"), groups=[("vehicle class", ["Compact", "SUV", "Van"]),
                      ("pickup city", ["Lyon", "Porto", "Graz"])], qty="rental days", price="daily rate",
                      cost="cleaning cost", revenue="rental charge", flag=("damage reported", "Yes", "No"),
                      status=("rental status", ["Returned", "Cancelled", "Active"], "Cancelled"),
                      times=("pickup time", "return time"), date="pickup date"),
}


def make_table(domain: str, rng: random.Random):
    spec = D[domain]
    name, style = styler(rng)
    n = rng.randint(30, 80)
    npr = np.random.default_rng(rng.randint(0, 10 ** 9))
    people = list({f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(rng.randint(4, 7))})
    cols, roles = {}, {}
    roles["entity"] = name(spec["entity"][0])
    cols[roles["entity"]] = [rng.choice(people) for _ in range(n)]
    roles["groups"] = []
    for g, vals in spec["groups"]:
        c = name(g)
        cols[c] = [rng.choice(vals) for _ in range(n)]
        roles["groups"].append(c)
    for key in ("qty", "price", "cost"):
        c = name(spec[key])
        base = {"qty": (1, 40), "price": (5, 250), "cost": (10, 900)}[key]
        vals = npr.integers(*base, n) if key == "qty" else np.round(npr.uniform(*base, n), 2)
        cols[c] = vals
        roles[key] = c
    has_revenue = rng.random() < 0.6
    if has_revenue:
        c = name(spec["revenue"])
        cols[c] = np.round(np.array(cols[roles["qty"]]) * np.array(cols[roles["price"]]) * npr.uniform(0.95, 1.05, n), 2)
        roles["revenue"] = c
    if rng.random() < 0.8:
        c = name(spec["flag"][0])
        cols[c] = [rng.choice([spec["flag"][1], spec["flag"][2], spec["flag"][2]]) for _ in range(n)]
        roles["flag"] = (c, spec["flag"][1])
    if rng.random() < 0.7:
        c = name(spec["status"][0])
        cols[c] = [rng.choice(spec["status"][1]) for _ in range(n)]
        roles["status"] = (c, spec["status"][2], spec["status"][1])
    if spec.get("times") and rng.random() < 0.8:
        a, b = name(spec["times"][0]), name(spec["times"][1])
        starts = [rng.choice([6, 7, 8, 9, 10, 13]) for _ in range(n)]
        cols[a] = [f"{h:02d}:{rng.choice(['00', '30'])}" for h in starts]
        cols[b] = [f"{min(h + rng.choice([4, 5, 6, 7, 8, 9]), 23):02d}:{rng.choice(['00', '15', '45'])}" for h in starts]
        roles["times"] = (a, b)
    c = name(spec["date"])
    cols[c] = pd.to_datetime("2025-01-01") + pd.to_timedelta(npr.integers(0, 330, n), unit="D")
    roles["date"] = c
    df = pd.DataFrame(cols)
    if rng.random() < 0.4:
        df = pd.concat([df, df.iloc[[rng.randrange(n)]]], ignore_index=True)       # an exact duplicate
    roles["people"] = sorted(set(df[roles["entity"]]))
    return df, roles, style


def lab(c: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", c).replace("_", " ").lower()


def q_(c: str) -> str:
    """A column reference inside a calculator expression (backticks for names with spaces)."""
    return c if re.fullmatch(r"[A-Za-z_]\w*", c) else f"`{c}`"


def ident(s: str) -> str:
    return re.sub(r"\W+", "_", s.lower()).strip("_")


def plan(derive=None, exclude=None, group=None, aggregate=None) -> dict:
    return {"type": "calc", "derive": derive or {}, "exclude": exclude or [], "group_by": group, "aggregate": aggregate or {}}


def say(rng, *variants):
    return rng.choice(variants)


# ------------------------------------------------------------------ intents --
def intents(df, r, rng):
    """(question, gold) pairs for one table; gold is a plan dict or a clarify/decline object."""
    out = []
    g = rng.choice(r["groups"])
    qty, price, cost = r["qty"], r["price"], r["cost"]
    person = rng.choice(r["people"])
    first = person.split()[0]
    # 1. implicit product (value = qty * price) when there is no revenue column -- or a stated formula
    val = "value"
    out.append((say(rng, f"whats the total {lab(qty)} times {lab(price)} for each {lab(g)}",
                    f"total value by {lab(g)}, value being {lab(qty)} x {lab(price)}",
                    f"can u work out {lab(qty)} * {lab(price)} summed per {lab(g)}?"),
                plan({val: f"{q_(qty)} * {q_(price)}"}, [], g, {f"total_{val}": f"sum({val})"})))
    # 2. difference / margin with the revenue column when present
    if "revenue" in r:
        rev = r["revenue"]
        out.append((say(rng, f"profit by {lab(g)} pls ({lab(rev)} minus {lab(cost)})",
                        f"how much did each {lab(g)} make after {lab(cost)}?",
                        f"{lab(rev)} less {lab(cost)}, broken down by {lab(g)}"),
                    plan({"profit": f"{q_(rev)} - {q_(cost)}"}, [], g, {"profit": "sum(profit)"})))
        out.append((say(rng, f"what's the margin per {lab(g)}? i mean (rev - cost) / rev",
                        f"margin for each {lab(g)} as a share of {lab(rev)}"),
                    plan({}, [], g, {"margin": f"(sum({q_(rev)}) - sum({q_(cost)})) / sum({q_(rev)})"})))
    # 3. grounded exclusion via a status column / clarify when the data has none
    st = r.get("status")
    if st:
        col, bad, _ = st
        out.append((say(rng, f"total {lab(cost)} by {lab(g)} but leave out the {bad.lower()} ones",
                        f"{lab(cost)} per {lab(g)}, ignoring anything {bad.lower()}",
                        f"sum of {lab(cost)} for each {lab(g)} excluding {bad.lower()} records"),
                    plan({}, [{"reason": f"{bad.lower()}", "where": f"{q_(col)} == '{bad}'"}], g,
                         {f"total_{ident(cost)}": f"sum({q_(cost)})"})))
    else:
        word = say(rng, "cancelled", "refunded", "void", "test")
        out.append((f"total {lab(cost)} by {lab(g)} excluding {word} records",
                    {"type": "clarify", "question": f"Which column marks a record as {word}? This data has no "
                                                    f"{word} or status field, so I can't tell which records to leave "
                                                    "out without guessing."}))
    # 4. a person excluded / a person only
    out.append((say(rng, f"average {lab(qty)} for everyone except {person}",
                    f"mean {lab(qty)} without {person}"),
                plan({}, [{"reason": f"excluding {person}", "where": f"{q_(r['entity'])} == '{person}'"}], None,
                     {f"average_{ident(qty)}": f"mean({q_(qty)})"})))
    out.append((say(rng, f"how much {lab(cost)} did {person} rack up in total",
                    f"{person}'s total {lab(cost)}?"),
                plan({}, [{"reason": f"only {person}", "where": f"{q_(r['entity'])} != '{person}'"}], None,
                     {f"total_{ident(cost)}": f"sum({q_(cost)})"})))
    # 5. numeric threshold stated by the user
    thr = int(np.percentile(df[qty], 60))
    out.append((say(rng, f"how many records have {lab(qty)} over {thr}, per {lab(g)}",
                    f"count of rows where {lab(qty)} is above {thr} by {lab(g)}"),
                plan({}, [{"reason": f"{lab(qty)} not above {thr}", "where": f"{q_(qty)} <= {thr}"}], g,
                     {"records": "count(*)"})))
    # 6. a rate of a yes/no flag
    if r.get("flag"):
        col, pos = r["flag"]
        out.append((say(rng, f"what % of records were {lab(col)} by {lab(g)}",
                        f"{lab(col)} rate per {lab(g)}"),
                    plan({}, [], g, {f"{ident(col)}_rate": f"mean({q_(col)} == '{pos}')"})))
    # 7. duration from clock times
    if r.get("times"):
        a, b = r["times"]
        out.append((say(rng, f"average hours between {lab(a)} and {lab(b)} for each {lab(g)}",
                        f"how long on average from {lab(a)} to {lab(b)}, by {lab(g)}"),
                    plan({"duration_hours": f"hours_between({q_(a)}, {q_(b)})"}, [], g,
                         {"average_duration_hours": "mean(duration_hours)"})))
    # 8. a period named
    month = rng.choice(["March", "June", "September"])
    mnum = {"March": 3, "June": 6, "September": 9}[month]
    out.append((say(rng, f"total {lab(qty)} by {lab(g)} in {month} 2025", f"{month} 2025 {lab(qty)} per {lab(g)}"),
                plan({}, [{"reason": f"outside {month} 2025", "where": f"year({q_(r['date'])}) != 2025 or month({q_(r['date'])}) != {mnum}"}],
                     g, {f"total_{ident(qty)}": f"sum({q_(qty)})"})))
    # 9. not in the data at all: decline
    missing = say(rng, "customer satisfaction", "employee morale", "weather", "competitor prices")
    out.append((f"how does {missing} affect {lab(cost)}?",
                {"type": "decline", "reason": f"This data has no {missing} column, so its effect on {lab(cost)} can't be "
                                              "measured."}))
    return out


def followups(df, r, rng, q1, p1):
    """Second turns that change the first plan: a person, duplicates, a breakdown, the aggregation, a correction."""
    out = []
    people = r["people"]
    who = rng.choice(people)
    sel = [e for e in p1["exclude"] if r["entity"] in e["where"]]
    base_ex = [e for e in p1["exclude"] if e not in sel]
    out.append((say(rng, f"only {who.split()[0]}", f"just {who}", f"now for {who} only"),
                {**p1, "exclude": base_ex + [{"reason": f"only {who}", "where": f"{q_(r['entity'])} != '{who}'"}]}))
    out.append((say(rng, "drop duplicates", "exclude duplicate rows", "same but without duplicates"),
                {**p1, "exclude": [{"reason": "exact duplicate row", "where": "duplicate()"}] + p1["exclude"]}))
    others = [x for x in r["groups"] if x != p1["group_by"]]
    if others:
        g2 = others[0]
        out.append((say(rng, f"by {lab(g2)} instead", f"break it down by {lab(g2)}"), {**p1, "group_by": g2}))
    agg = p1["aggregate"]
    if len(agg) == 1:
        k, v = next(iter(agg.items()))
        m = re.match(r"sum\((.+)\)$", v)
        if m:
            out.append((say(rng, "average instead of total", "make it the mean", "what about the average"),
                        {**p1, "aggregate": {k.replace("total", "average") if "total" in k else "average_" + k: f"mean({m.group(1)})"}}))
    return out


# ------------------------------------------------------------------ prompts --
def captured_messages(question: str, df: pd.DataFrame, memory: dict) -> list[dict]:
    """The exact conversation the calculator sends, captured from app.analyst.calc.model_plan with the model
    stubbed (it declines immediately, so only the first prompt is needed)."""
    seen = []
    orig = calc._complete

    def fake(prompt, n):
        seen.append(prompt)
        return 'decline", "reason": "stub"}'
    calc._complete = fake
    try:
        calc.model_plan(question, df, memory)
    finally:
        calc._complete = orig
    text = seen[0]
    assert text.endswith('<|im_start|>assistant\n{"type": "')
    body = text[: -len('<|im_start|>assistant\n{"type": "')]
    msgs = []
    for part in body.split("<|im_start|>")[1:]:
        role, content = part.split("\n", 1)
        msgs.append({"role": role, "content": content[: -len("<|im_end|>\n")]})
    return msgs


def target(gold: dict) -> str:
    if gold["type"] != "calc":
        return json.dumps(gold)
    return ('{"type": "calc", "derive": ' + json.dumps(gold["derive"]) + ', "exclude": ' + json.dumps(gold["exclude"])
            + ', "group_by": ' + json.dumps(gold["group_by"]) + ', "aggregate": ' + json.dumps(gold["aggregate"]) + "}")


def valid_gold(gold, df, question, earlier=""):
    if gold["type"] != "calc":
        return None
    spec = {**gold, "sort": None, "explain": ""}
    problems = calc.check(spec, df, question, earlier=earlier)
    if problems:
        return None
    res = calc.run(spec, df, question=question)
    return {"totals": res["totals"], "groups": [{k: v for k, v in gg.items() if not k.startswith("share_of") and k != "per_row"}
                                                for gg in res["groups"]]}


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z]+", " ", q.lower()).strip()


def main() -> int:
    if OUT.exists():
        print(f"{OUT} exists; refusing to overwrite"); return 1
    rng = random.Random(SEED)
    groups = []                       # one group = one dataset instance (all its examples and conversations)
    rejected = 0
    for domain in TRAIN_DOMAINS + HELDOUT_DOMAINS:
        reps = 9 if domain in TRAIN_DOMAINS else 6
        for rep in range(reps):
            df, roles, style = make_table(domain, rng)
            rows = []
            for question, gold in intents(df, roles, rng):
                truth = valid_gold(gold, df, question)
                if gold["type"] == "calc" and truth is None:
                    rejected += 1
                    continue
                msgs = captured_messages(question, df, {})
                rows.append({"messages": msgs + [{"role": "assistant", "content": target(gold)}],
                             "meta": {"domain": domain, "style": style, "turns": 1, "gold": gold, "truth": truth,
                                      "question": question, "kind": gold["type"] if gold["type"] != "calc" else "plan"}})
                if gold["type"] == "calc" and rng.random() < 0.55:
                    for q2, g2 in followups(df, roles, rng, question, gold)[:2]:
                        spec1 = {**gold, "sort": None, "explain": ""}
                        mem = {"calc_history": [(question, {k: gold[k] for k in ("type", "derive", "exclude", "group_by", "aggregate")})],
                               "last_calc": {"question": question, "spec": spec1, "result": calc.run(spec1, df, question=question)}}
                        t2 = valid_gold(g2, df, q2, earlier=question)
                        if t2 is None:
                            rejected += 1
                            continue
                        m2 = captured_messages(q2, df, mem)
                        rows.append({"messages": m2 + [{"role": "assistant", "content": target(g2)}],
                                     "meta": {"domain": domain, "style": style, "turns": 2, "gold": g2, "truth": t2,
                                              "question": q2, "first": question, "kind": "followup"}})
            groups.append({"domain": domain, "rep": rep, "rows": rows,
                           "csv": df.to_csv(index=False), "dates": [c for c in df.columns if str(df[c].dtype).startswith("datetime")]})
    # splits: held-out domains -> test only; train domains: dataset instances split 7/1/1 (train/valid/test)
    split = {"train": [], "valid": [], "test": []}
    for gidx, gr in enumerate(groups):
        if gr["domain"] in HELDOUT_DOMAINS:
            s = "test"
        else:
            s = "train" if gr["rep"] < 7 else ("valid" if gr["rep"] == 7 else "test")
        for row in gr["rows"]:
            row["meta"]["group"] = gidx
            row["meta"]["heldout_domain"] = gr["domain"] in HELDOUT_DOMAINS
        split[s].append(gr)
    # near-duplicate guard: a normalised question + table text seen in train is dropped from valid/test
    seen = {(norm_q(r["meta"]["question"]), r["messages"][-2]["content"][-400:]) for gr in split["train"] for r in gr["rows"]}
    dropped = 0
    for s in ("valid", "test"):
        for gr in split[s]:
            keep = [r for r in gr["rows"] if (norm_q(r["meta"]["question"]), r["messages"][-2]["content"][-400:]) not in seen]
            dropped += len(gr["rows"]) - len(keep)
            gr["rows"] = keep
    # replay of v1's training data (planner prompts, narration, chat): keeps v1's skills
    v1 = [json.loads(x) for x in open(ROOT / "training/data/analyst_sft/train.jsonl")]
    v1v = [json.loads(x) for x in open(ROOT / "training/data/analyst_sft/valid.jsonl")]
    rr = random.Random(SEED + 1)
    replay_train, replay_valid = rr.sample(v1, 1400), rr.sample(v1v, 60)
    OUT.mkdir(parents=True)
    manifest = {"seed": SEED, "train_domains": TRAIN_DOMAINS, "heldout_domains": HELDOUT_DOMAINS,
                "rejected_gold_plans": rejected, "near_duplicates_dropped": dropped, "replay": {"train": 1400, "valid": 60},
                "splits": {}}
    for s in ("train", "valid", "test"):
        rows = [r for gr in split[s] for r in gr["rows"]]
        msgs = [{"messages": r["messages"]} for r in rows]
        if s == "train":
            msgs += replay_train
        if s == "valid":
            msgs += replay_valid
        random.Random(SEED + 2).shuffle(msgs)
        with open(OUT / f"{s}.jsonl", "w") as fh:
            for m in msgs:
                fh.write(json.dumps(m) + "\n")
        with open(OUT / f"{s}_meta.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps({**r["meta"], "messages": r["messages"]}, default=str) + "\n")
        with open(OUT / f"{s}_tables.json", "w") as fh:
            json.dump({str(gidx): {"csv": gr["csv"], "dates": gr["dates"]} for gidx, gr in enumerate(groups) if gr in split[s]}, fh)
        kinds = {}
        for r in rows:
            kinds[r["meta"]["kind"]] = kinds.get(r["meta"]["kind"], 0) + 1
        manifest["splits"][s] = {"rows": len(msgs), "new_examples": len(rows), "by_kind": kinds,
                                 "sha256": hashlib.sha256((OUT / f"{s}.jsonl").read_bytes()).hexdigest()}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
