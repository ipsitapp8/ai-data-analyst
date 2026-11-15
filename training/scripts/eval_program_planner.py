"""Evaluation of model-assisted program planning on requests and data Silt was not built for.

Runs every case through the real copilot (app.analyst.orchestrator.ask) and records, per case: which path answered
(the model-planned program, or a deterministic path), the planner's status and attempts, rejected plans (and whether
any named a column that does not exist), whether the program ran, whether the numbers pandas computes here appear in
the computed evidence, whether a clarification was asked when one was needed, and latency.

Datasets are generated here (a vehicle fleet, retail outlets, a support desk) with metamorphic variants (miles for km,
0/1 for Yes/No), plus the public tips data. None is one of the held-out evaluation datasets, and no case is training
material.

  python training/scripts/eval_program_planner.py --mode off|v1|base --out results.json

  off   use_model=False: deterministic paths only
  v1    the planner uses the app's model server (Qwen3-4B + v1 adapter), as configured by default
  base  the planner uses ANALYST_PLANNER_URL (e.g. the same base model without the adapter on another port)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

TODAY = pd.Timestamp("2026-11-14")


# ------------------------------------------------------------------- datasets --
def fleet(seed=21, n=420):
    rng = np.random.default_rng(seed)
    drivers = ["Kofi Mensah", "Ana Ribeiro", "Jonas Berg", "Mei Lin", "Omar Haddad", "Sara Novak", "Ravi Iyer"]
    depot_of = dict(zip(drivers, ["North", "North", "North", "South", "South", "East", "East"]))
    drv = rng.choice(drivers, n)
    km = rng.gamma(9, 14, n).round(1)
    fuel = (km * np.where(np.vectorize(depot_of.get)(drv) == "East", 0.13, 0.09) + rng.normal(0, 1.5, n)).round(2)
    late = np.where(rng.random(n) < 0.06 + 0.25 * (km > 170), "Yes", "No")
    day = TODAY - pd.to_timedelta(rng.integers(1, 75, n), "D")
    return pd.DataFrame({"trip_no": [f"T{i:05d}" for i in range(n)], "driver": drv, "depot": np.vectorize(depot_of.get)(drv),
                         "trip_date": day, "km_driven": km, "fuel_litres": fuel, "late": late})


def retail(seed=8, n=300, two_money=False):
    rng = np.random.default_rng(seed)
    outlet = rng.choice(["Harbour", "Market St", "Airport", "Old Town"], n)
    foot = rng.integers(80, 900, n)
    promo = rng.choice(["Y", "N"], n, p=[0.3, 0.7])
    staff = rng.integers(2, 9, n)
    takings = (foot * rng.normal(4.1, 0.6, n) * np.where(promo == "Y", 1.15, 1) * np.where(outlet == "Airport", 1.4, 1)).round(2)
    df = pd.DataFrame({"outlet": outlet, "sold_on": TODAY - pd.to_timedelta(rng.integers(1, 120, n), "D"),
                       "footfall": foot, "staff_on_shift": staff, "promo": promo, "takings": takings})
    if two_money:
        df["online_sales"] = (takings * rng.uniform(0.1, 0.4, n)).round(2)
    return df


def desk(seed=3, n=360):
    rng = np.random.default_rng(seed)
    agents = ["Lea Moreau", "Tariq Aziz", "Hana Sato", "Pablo Ruiz", "Ines Kova", "Femi Ade"]
    team = {a: t for a, t in zip(agents, ["Billing", "Billing", "Tech", "Tech", "Tech", "Accounts"])}
    ag = rng.choice(agents, n)
    hrs = rng.lognormal(1.6, 0.6, n).round(2)
    hrs[[4, 17, 99]] = [210.0, 260.0, 199.0]
    esc = np.where(rng.random(n) < 0.1 + 0.3 * (hrs > 9), "Yes", "No")
    return pd.DataFrame({"ticket": [f"K{i}" for i in range(n)], "agent": ag, "team": np.vectorize(team.get)(ag),
                         "opened": TODAY - pd.to_timedelta(rng.integers(1, 90, n), "D"), "resolution_hours": hrs,
                         "escalated": esc})


def tips():
    import plotly
    return pd.read_csv(Path(plotly.__file__).parent / "package_data" / "datasets" / "tips.csv.gz")


F = fleet()
F_MI = F.assign(km_driven=(F.km_driven * 0.621371).round(2)).rename(columns={"km_driven": "distance_mi"})
F_01 = F.assign(late=(F.late == "Yes").astype(int)).rename(columns={"late": "arrived_late"})
R, R2, D = retail(), retail(two_money=True), desk()
T = tips()


def month_to_date(df, date, mask=None):
    d = df if mask is None else df[mask]
    return d[(d[date] >= TODAY.replace(day=1)) & (d[date] <= TODAY)]


def iqr_count(x):
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())


# ---------------------------------------------------------------------- cases --
# expect: "answer" (truth numbers must appear in the computed evidence), "clarify", "no_answer" (nothing may be
# computed and presented as the answer: a clarification, a decline or an explanation of what is missing)
CASES = [
    dict(id="fleet-per-driver", spec=dict(measure="km_driven", fn="ratio", group="depot", parts=["associate"]), ds="F", q="Which depot gets the fewest kilometres out of each driver it employs, and is "
         "lateness linked to anything?", expect="answer",
         truth=lambda: (F.groupby("depot").km_driven.sum() / F.groupby("depot").driver.nunique()).to_dict()),
    dict(id="fleet-fuel-per-km", spec=dict(measure="fuel_litres", fn="ratio", group="depot"), ds="F", q="How does fuel burned per kilometre compare between depots, after allowing "
         "for how far each one drives?", expect="answer",
         truth=lambda: (F.groupby("depot").fuel_litres.sum() / F.groupby("depot").km_driven.sum()).to_dict()),
    dict(id="fleet-fuel-per-mile (units)", spec=dict(measure="fuel_litres", fn="ratio", group="depot"), ds="F_MI", q="How does fuel burned per mile compare between depots, after "
         "allowing for how far each one drives?", expect="answer",
         truth=lambda: (F_MI.groupby("depot").fuel_litres.sum() / F_MI.groupby("depot").distance_mi.sum()).to_dict()),
    dict(id="fleet-late-share", spec=dict(measure="late", fn="rate", group="depot"), ds="F", q="What proportion of trips run behind schedule at each depot, and which depot "
         "fares worst?", expect="answer", truth=lambda: (F.late == "Yes").groupby(F.depot).mean().to_dict()),
    dict(id="fleet-late-share (0/1)", spec=dict(measure="arrived_late", fn="rate", group="depot"), ds="F_01", q="What proportion of trips run behind schedule at each depot, and "
         "which depot fares worst?", expect="answer", truth=lambda: F_01.groupby("depot").arrived_late.mean().to_dict()),
    dict(id="fleet-depot-target", spec=dict(measure="km_driven", fn="compare"), ds="F", q="If North keeps driving at this pace, will it clock 25000 km by the end of "
         "the month?", expect="answer",
         truth=lambda: {"completed": month_to_date(F, "trip_date", F.depot == "North").km_driven.sum()}),
    dict(id="fleet-closure", spec=dict(measure="km_driven", fn="scenario"), ds="F", q="How many kilometres would we lose over the next 7 days if the South depot were "
         "shut, compared with carrying on normally?", expect="answer",
         truth=lambda: {"normal": F.groupby("trip_date").km_driven.sum().reindex(
             pd.date_range(F.trip_date.min(), F.trip_date.max()), fill_value=0).mean() * 7}),
    dict(id="fleet-unsupported-cause", ds="F", q="What is the causal effect of diesel prices on lateness?",
         expect="no_answer"),
    dict(id="fleet-impossible", ds="F", q="Forecast each driver's salary for next year and say who deserves a raise.",
         expect="no_answer"),
    dict(id="retail-conversion", spec=dict(measure="takings", fn="ratio", group="outlet", parts=["anomalies"]), ds="R", q="Which outlet turns visitors into takings least effectively, and are any "
         "takings figures out of the ordinary?", expect="answer",
         truth=lambda: (R.groupby("outlet").takings.sum() / R.groupby("outlet").footfall.sum()).to_dict()),
    dict(id="retail-promo", spec=dict(measure="takings", fn="mean", group="promo"), ds="R", q="On promotion days versus ordinary days, how do average takings stack up?",
         expect="answer", truth=lambda: R.groupby("promo").takings.mean().to_dict()),
    dict(id="retail-uplift", spec=dict(measure="takings", fn="scenario"), ds="R", q="Suppose takings rose 8% everywhere — what would total takings come to, and how "
         "much more is that?", expect="answer", truth=lambda: {"total": R.takings.sum() * 1.08}),
    dict(id="retail-per-staff", spec=dict(measure="takings", fn="ratio", group="outlet"), ds="R", q="Are takings per person on shift any different across outlets once you account "
         "for staffing levels?", expect="answer",
         truth=lambda: (R.groupby("outlet").takings.sum() / R.groupby("outlet").staff_on_shift.sum()).to_dict()),
    dict(id="retail-ambiguous", ds="R2", q="Which outlet brings in the most revenue for each person on shift?",
         expect="clarify"),
    dict(id="desk-slowest", spec=dict(measure="resolution_hours", fn="mean", group="team", parts=["associate"]), ds="D", q="Which team takes longest to close tickets on average, and does escalation go "
         "with anything?", expect="answer", truth=lambda: D.groupby("team").resolution_hours.mean().to_dict()),
    dict(id="desk-escalated-median", spec=dict(measure="resolution_hours", fn="median", group="team"), ds="D", q="Among escalated tickets only, what is the typical (median) time to "
         "resolve for each team?", expect="answer",
         truth=lambda: D[D.escalated == "Yes"].groupby("team").resolution_hours.median().to_dict()),
    dict(id="desk-odd-values", spec=dict(measure="resolution_hours", fn="anomalies"), ds="D", q="Do any resolution times look implausible enough to be data errors, and how "
         "many are there?", expect="answer", truth=lambda: {"count": iqr_count(D.resolution_hours)}),
    dict(id="tips-share", spec=dict(measure="tip", fn="ratio", group="smoker"), ds="T", q="Do smokers leave a bigger slice of the bill as a tip than non-smokers, and is "
         "that difference meaningful?", expect="answer",
         truth=lambda: (T.groupby("smoker").tip.sum() / T.groupby("smoker").total_bill.sum()).to_dict()),
    dict(id="tips-per-diner", spec=dict(measure="total_bill", fn="ratio", group="day"), ds="T", q="On which day is the spend per diner highest once party size is taken into "
         "account?", expect="answer", truth=lambda: (T.groupby("day").total_bill.sum() / T.groupby("day")["size"].sum()).to_dict()),
]


NEW_CASES = [
    dict(id="n-fleet-fuel-rank", ds="F", spec=dict(measure="fuel_litres", fn="sum", group="depot", parts=["anomalies"]),
         q="Rank the depots by how much fuel they burn in total, and flag any trips with suspicious fuel figures.",
         expect="answer", truth=lambda: F.groupby("depot").fuel_litres.sum().to_dict()),
    dict(id="n-fleet-typical-trip", ds="F", spec=dict(measure="km_driven", fn="median", group="driver"),
         q="For each driver, what is the typical trip length?", expect="answer",
         truth=lambda: F.groupby("driver").km_driven.median().to_dict()),
    dict(id="n-fleet-entity-rate", ds="F", spec=dict(measure="late", fn="rate"),
         q="What fraction of Mei Lin's trips were late?", expect="answer",
         truth=lambda: {"rate": (F[F.driver == "Mei Lin"].late == "Yes").mean()}),
    dict(id="n-fleet-late-why", ds="F", spec=dict(measure="late", fn="rate", group="depot", parts=["associate"]),
         q="Which depot has the most late trips relative to how many trips it runs, and what is lateness associated with?",
         expect="answer", truth=lambda: (F.late == "Yes").groupby(F.depot).mean().to_dict()),
    dict(id="n-fleet-avg-trip", ds="F", spec=dict(measure="km_driven", fn="mean", group="depot"),
         q="How far does an average trip go at each depot, and which depot's trips are longest?", expect="answer",
         truth=lambda: F.groupby("depot").km_driven.mean().to_dict()),
    dict(id="n-fleet-why-late", ds="F", spec=dict(measure="late", fn="associate", parts=["associate"]),
         q="Do longer trips run late more often? Show what lateness goes with.", expect="answer", truth=lambda: {}),
    dict(id="n-retail-total-min", ds="R", spec=dict(measure="takings", fn="sum", group="outlet"),
         q="What are total takings for each outlet, and which outlet takes the least?", expect="answer",
         truth=lambda: R.groupby("outlet").takings.sum().to_dict()),
    dict(id="n-retail-avg-visitors", ds="R", spec=dict(measure="footfall", fn="mean", group="outlet"),
         q="On average how many visitors does each outlet get?", expect="answer",
         truth=lambda: R.groupby("outlet").footfall.mean().to_dict()),
    dict(id="n-retail-per-visitor", ds="R", spec=dict(measure="takings", fn="ratio", group="outlet"),
         q="Takings per visitor at each outlet — which is weakest?", expect="answer",
         truth=lambda: (R.groupby("outlet").takings.sum() / R.groupby("outlet").footfall.sum()).to_dict()),
    dict(id="n-retail-closure", ds="R", spec=dict(measure="takings", fn="scenario"),
         q="How much would total takings drop if the Airport outlet closed?", expect="answer",
         truth=lambda: {"without": R[R.outlet != "Airport"].takings.sum()}),
    dict(id="n-retail-promo-share", ds="R", spec=dict(measure="promo", fn="rate", group="outlet"),
         q="What proportion of trading days ran a promotion at each outlet?", expect="answer",
         truth=lambda: (R.promo == "Y").groupby(R.outlet).mean().to_dict()),
    dict(id="n-retail-odd", ds="R", spec=dict(measure="takings", fn="anomalies"),
         q="Do any takings figures look odd, and how many such values are there?", expect="answer",
         truth=lambda: {"count": iqr_count(R.takings)}),
    dict(id="n-desk-esc-why", ds="D", spec=dict(measure="escalated", fn="rate", group="team", parts=["associate"]),
         q="What's the escalation rate for each team, and why are some teams escalating more?", expect="answer",
         truth=lambda: (D.escalated == "Yes").groupby(D.team).mean().to_dict()),
    dict(id="n-desk-entity-vs", ds="D", spec=dict(measure="resolution_hours", fn="mean"),
         q="On average, how long does Lea Moreau take to resolve a ticket compared with everyone else?", expect="answer",
         truth=lambda: {"lea": D[D.agent == "Lea Moreau"].resolution_hours.mean()}),
    dict(id="n-desk-count", ds="D", spec=dict(measure=None, fn="count", group="agent"),
         q="How many tickets does each agent handle, and who handles the most?", expect="answer",
         truth=lambda: D.groupby("agent").size().astype(float).to_dict()),
    dict(id="n-desk-per-agent", ds="D", spec=dict(measure="resolution_hours", fn="ratio", group="team"),
         q="Per agent, how many hours of resolution work does each team carry?", expect="answer",
         truth=lambda: (D.groupby("team").resolution_hours.sum() / D.groupby("team").agent.nunique()).to_dict()),
    dict(id="n-tips-day", ds="T", spec=dict(measure="tip", fn="mean", group="day"),
         q="What is the average tip on each day, and which day tips best?", expect="answer",
         truth=lambda: T.groupby("day").tip.mean().to_dict()),
    dict(id="n-tips-smoker-share", ds="T", spec=dict(measure="smoker", fn="rate", group="day"),
         q="What share of parties were smokers on each day?", expect="answer",
         truth=lambda: (T.smoker == "Yes").groupby(T.day).mean().to_dict()),
    dict(id="n-tips-why", ds="T", spec=dict(measure="tip", fn="associate", parts=["associate"]),
         q="Is tip size linked to anything else in the data?", expect="answer", truth=lambda: {}),
    dict(id="n-tips-median-time", ds="T", spec=dict(measure="total_bill", fn="median", group="time"),
         q="What is the typical bill at dinner versus lunch?", expect="answer",
         truth=lambda: T.groupby("time").total_bill.median().to_dict()),
    dict(id="n-retail-ambiguous", ds="R2", q="How much revenue does each outlet make on average?", expect="clarify"),
    dict(id="n-fleet-missing-col", ds="F", q="Which depot has the best customer satisfaction, and why?", expect="no_answer"),
    dict(id="n-tips-missing-col", ds="T", q="Compare the tips each waiter earns and say who is best.", expect="no_answer"),
]
CASES = CASES + NEW_CASES

# one-day reliability sprint: paraphrases that exercise each fix (different wording from the unit tests)
SPRINT_CASES = [
    dict(id="s-target-any-verb", ds="F", spec=dict(measure="km_driven", fn="compare"),
         q="Is East going to log 12000 km before the end of the month?", expect="answer",
         truth=lambda: {"completed": month_to_date(F, "trip_date", F.depot == "East").km_driven.sum()}),
    dict(id="s-entity-rate", ds="F", spec=dict(measure="late", fn="rate"),
         q="What share of Ravi Iyer's trips ran late?", expect="answer",
         truth=lambda: {"rate": (F[F.driver == "Ravi Iyer"].late == "Yes").mean()}),
    dict(id="s-entity-rate-2", ds="F", spec=dict(measure="late", fn="rate"),
         q="What fraction of trips by Omar Haddad were late?", expect="answer",
         truth=lambda: {"rate": (F[F.driver == "Omar Haddad"].late == "Yes").mean()}),
    dict(id="s-synonym-length", ds="F", spec=dict(measure="km_driven", fn="mean", group="depot"),
         q="Average trip length by depot?", expect="answer", truth=lambda: F.groupby("depot").km_driven.mean().to_dict()),
    dict(id="s-synonym-visitors", ds="R", spec=dict(measure="footfall", fn="mean"),
         q="How many visitors does Harbour get on average?", expect="answer",
         truth=lambda: {"mean": R[R.outlet == "Harbour"].footfall.mean()}),
    dict(id="s-synonym-spend", ds="T", spec=dict(measure="total_bill", fn="sum", group="day"),
         q="What is the total spend on each day?", expect="answer", truth=lambda: T.groupby("day").total_bill.sum().to_dict()),
    dict(id="s-per-visitor-unnamed", ds="R", q="What do outlets take per visitor, and which is lowest?",
         expect="clarify"),                     # "take" names no column: Silt must ask what to divide, not guess
    dict(id="s-per-visitor", ds="R", spec=dict(measure="takings", fn="ratio", group="outlet"),
         q="What are the takings per visitor at each outlet, and which is lowest?", expect="answer",
         truth=lambda: (R.groupby("outlet").takings.sum() / R.groupby("outlet").footfall.sum()).to_dict()),
    dict(id="s-last-30-days", ds="D", spec=dict(measure="resolution_hours", fn="sum"),
         q="Over the last 30 days, what were the total resolution hours?", expect="answer",
         truth=lambda: {"sum": D[(D.opened >= TODAY - pd.Timedelta(days=29)) & (D.opened <= TODAY)].resolution_hours.sum()}),
    dict(id="s-last-2-weeks", ds="F", spec=dict(measure="km_driven", fn="sum"),
         q="Total km driven in the last 2 weeks", expect="answer",
         truth=lambda: {"sum": F[(F.trip_date >= TODAY - pd.Timedelta(days=13)) & (F.trip_date <= TODAY)].km_driven.sum()}),
    dict(id="s-only-escalated", ds="D", spec=dict(measure="resolution_hours", fn="median", group="team"),
         q="Median resolution hours for escalated tickets only, by team", expect="answer",
         truth=lambda: D[D.escalated == "Yes"].groupby("team").resolution_hours.median().to_dict()),
    dict(id="s-excluding", ds="R", spec=dict(measure="takings", fn="sum"),
         q="Total takings excluding Airport", expect="answer", truth=lambda: {"sum": R[R.outlet != "Airport"].takings.sum()}),
    dict(id="s-performs-worst", ds="F", q="Which depot performs worst and why?", expect="clarify"),
    dict(id="s-rate-why", ds="D", spec=dict(measure="escalated", fn="rate", group="team", parts=["associate"]),
         q="Which team has the highest escalation rate and what explains it?", expect="answer",
         truth=lambda: (D.escalated == "Yes").groupby(D.team).mean().to_dict()),
]
SPRINT_CONVERSATIONS = [
    dict(id="s-conv-compare-these", ds="F", turns=[
        dict(q="Can Kofi Mensah reach 3000 km this month?", expect="answer",
             truth=lambda: {"completed": month_to_date(F, "trip_date", F.driver == "Kofi Mensah").km_driven.sum()}),
        dict(q="What if the goal is 3600 km?", expect="answer", truth=lambda: {"target": 3600.0}),
        dict(q="Compare these scenarios and explain the difference.", expect="answer",
             truth=lambda: {"a": 3000.0, "b": 3600.0}),
    ]),
]

# reliability audit: new paraphrases of every fixed failure (wording differs from the unit tests and the cases above)
def timesheet(seed=31, n=90):
    rng = np.random.default_rng(seed)
    who = rng.choice(["Ana", "Ben", "Chen"], n)
    df = pd.DataFrame({"worker": who, "worked_on": TODAY - pd.to_timedelta(rng.integers(0, 40, n), "D"),
                       "hours": rng.choice([6.0, 7.5, 8.0, 9.0], n)})
    return pd.concat([df, df[df.worker == "Ana"].head(2)], ignore_index=True)   # two exact duplicates of Ana's records


TS = timesheet()


def _daily(df, date, col, days, window_of=None):
    w = df if window_of is None else window_of
    return df.groupby(date)[col].sum().reindex(pd.date_range(w[date].min(), w[date].max()), fill_value=0).mean() * days


AUDIT_CASES = [
    dict(id="a-late-behind", ds="F", spec=dict(measure="late", fn="rate", group="depot"),
         q="Which depot has the highest share of trips that ran behind schedule?", expect="answer",
         truth=lambda: (F.late == "Yes").groupby(F.depot).mean().to_dict()),
    dict(id="a-closure-total", ds="F", spec=dict(measure="km_driven", fn="scenario"),
         q="If the North depot shut down, how many kilometres would the rest of the fleet still have driven?",
         expect="answer", truth=lambda: {"without": F[F.depot != "North"].km_driven.sum()}),
    # the history window is the whole dataset's date range, with and without Harbour's records
    dict(id="a-closure-horizon", ds="R", spec=dict(measure="takings", fn="scenario"),
         q="Over the next 14 days, how much takings would we lose if Harbour closed?", expect="answer",
         truth=lambda: {"normal": _daily(R, "sold_on", "takings", 14),
                        "closed": _daily(R[R.outlet != "Harbour"], "sold_on", "takings", 14, window_of=R)}),
    dict(id="a-convert", ds="R", spec=dict(measure="takings", fn="ratio", group="outlet"),
         q="How well does each outlet convert footfall into takings?", expect="answer",
         truth=lambda: (R.groupby("outlet").takings.sum() / R.groupby("outlet").footfall.sum()).to_dict()),
    dict(id="a-share-of", ds="T", spec=dict(measure="tip", fn="ratio", group="day"),
         q="What is the tip as a share of the total bill on each day?", expect="answer",
         truth=lambda: (T.groupby("day").tip.sum() / T.groupby("day").total_bill.sum()).to_dict()),
    dict(id="a-for-every", ds="D", spec=dict(measure="resolution_hours", fn="ratio", group="team"),
         q="How many resolution hours does each team log for every agent?", expect="answer",
         truth=lambda: (D.groupby("team").resolution_hours.sum() / D.groupby("team").agent.nunique()).to_dict()),
    dict(id="a-versus-values", ds="T", spec=dict(measure="tip", fn="median", group="day"),
         q="Compare the median tip on Sat versus Sun.", expect="answer",
         truth=lambda: T[T.day.isin(["Sat", "Sun"])].groupby("day").tip.median().to_dict()),
    dict(id="a-versus-flag", ds="R", spec=dict(measure="takings", fn="median", group="promo"),
         q="How do median takings compare on promo days versus normal days?", expect="answer",
         truth=lambda: R.groupby("promo").takings.median().to_dict()),
    dict(id="a-bare-days", ds="R", spec=dict(measure="takings", fn="mean", group="outlet"),
         q="What are average takings per outlet across all trading days?", expect="answer",
         truth=lambda: R.groupby("outlet").takings.mean().to_dict()),
]
AUDIT_CONVERSATIONS = [
    dict(id="a-conv-duplicates", ds="TS", turns=[
        dict(q="Total hours for Ana", expect="answer", truth=lambda: {"total": TS[TS.worker == "Ana"].hours.sum()}),
        dict(q="Can Ana reach 200 hours this month?", expect="answer",
             truth=lambda: {"completed": month_to_date(TS, "worked_on", TS.worker == "Ana").hours.sum()}),
        dict(q="Can Ana reach 200 hours this month, excluding duplicates?", expect="answer",
             truth=lambda: {"completed": month_to_date(TS.drop_duplicates(), "worked_on",
                                                       TS.drop_duplicates().worker == "Ana").hours.sum()}),
    ]),
    dict(id="a-conv-ordinal", ds="D", turns=[
        dict(q="Go through this file and point out anything important.", expect="answer", truth=lambda: {}),
        dict(q="Explain the third thing you found.", expect="detail"),
    ]),
    dict(id="a-conv-register", ds="R", turns=[
        dict(q="Total takings for Harbour", expect="answer", truth=lambda: {"h": R[R.outlet == "Harbour"].takings.sum()}),
        dict(q="Total takings for Airport", expect="answer", truth=lambda: {"a": R[R.outlet == "Airport"].takings.sum()}),
        dict(q="Compare that with the previous result.", expect="answer",
             truth=lambda: {"h": R[R.outlet == "Harbour"].takings.sum(), "a": R[R.outlet == "Airport"].takings.sum()}),
    ]),
]

CONVERSATIONS = [
    dict(id="conv-target-rerun-compare", ds="F", turns=[
        dict(q="Can Kofi Mensah reach 3000 km this month?", expect="answer",
             truth=lambda: {"completed": month_to_date(F, "trip_date", F.driver == "Kofi Mensah").km_driven.sum()}),
        dict(q="Suppose the goal were 3500 instead — does that change the picture?", expect="answer",
             truth=lambda: {"target": 3500.0}),
        dict(q="Set that next to the original ask and tell me what moved.", expect="answer",
             truth=lambda: {"a": 3000.0, "b": 3500.0}),
        dict(q="Show the difference as a chart.", expect="chart"),
    ]),
    dict(id="conv-findings", ds="D", turns=[
        dict(q="Look through this whole file and tell me what stands out.", expect="answer", truth=lambda: {}),
        dict(q="Go back to the second thing you found and explain exactly how it was measured.", expect="detail"),
    ]),
    dict(id="conv-fresh-isolation", ds="F", fresh=True, turns=[
        dict(q="Set that next to the original ask and tell me what moved.", expect="no_answer"),
    ]),
]
DATA = {"F": F, "F_MI": F_MI, "F_01": F_01, "R": R, "R2": R2, "D": D, "T": T, "TS": TS}


# ---------------------------------------------------------------------- check --
def numbers(x, out):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)) and np.isfinite(x):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            numbers(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            numbers(v, out)


def found(truth: dict, evidence: list[float]) -> bool:
    return all(any(abs(e - v) <= 1e-6 * max(1.0, abs(v)) for e in evidence) for v in truth.values())


def grade_program(out: dict, spec: dict | None) -> dict:
    """Intent, metric, column and completeness of the program that answered (model-planned or built by Silt), judged
    against the case's specification -- written from the question, independently of the planner's own checks."""
    prog = next((c["result"]["programs"][""] for c in out.get("calls", []) if c.get("tool") == "analysis_program"
                 and c.get("status") == "ok" and "" in (c["result"].get("programs") or {})), None)
    if prog is None or not spec:
        return {}
    from app.analyst import intent as I
    by = {s["id"]: s for s in prog}
    ops = [s["op"] for s in prog]
    aggs = [s for s in prog if s["op"] == "aggregate"]
    cols = set()
    for s in prog:
        if s["op"] in ("aggregate", "project", "anomalies"):
            cols |= I._measure_cols(s, by)                                          # noqa: SLF001
        if s["op"] == "associate":
            cols.add(s["target"])
        if s["op"] == "derive":
            cols |= set(__import__("re").findall(r"[A-Za-z_]\w*", s["formula"]))
    fn = spec.get("fn")
    fns = {s["fn"] for s in aggs}
    flagmean = any(s["fn"] == "mean" and any(str(c).startswith("is_") or c == spec.get("measure")
                                             for c in I._measure_cols(s, by)) for s in aggs)  # noqa: SLF001
    metric = {"ratio": "ratio" in ops, "rate": flagmean or "ratio" in ops, "mean": "mean" in fns,
              "median": "median" in fns, "sum": "sum" in fns, "count": bool({"count", "nunique"} & fns),
              "anomalies": "anomalies" in ops, "associate": "associate" in ops, "scenario": "scenario" in ops,
              "compare": "compare" in ops}.get(fn, False)
    column = spec.get("measure") is None or spec["measure"] in cols
    group = spec.get("group") is None or any(s.get("by") == spec["group"] for s in aggs)
    complete = all(op in ops for op in spec.get("parts", []))
    return {"metric_ok": bool(metric and group), "column_ok": bool(column), "complete": complete}


def evaluate(out: dict, turn: dict, mem: dict) -> dict:
    ev = []
    for c in out.get("calls", []):
        if c.get("status") == "ok":
            numbers(c.get("result"), ev)
    plan = out.get("plan") or {}
    planner = mem.get("_nl_last") or {}      # the planner's last attempt this turn, whichever path answered
    clar = plan.get("type") == "clarify"
    e = turn["expect"]
    if e == "answer":
        ok = bool(ev) and found(turn["truth"](), ev)
    elif e == "clarify":
        ok = clar and not any(c.get("status") == "ok" for c in out.get("calls", []))
    elif e == "no_answer":
        ok = not any(c.get("status") == "ok" and c.get("tool") == "analysis_program" for c in out.get("calls", [])) and \
            (clar or plan.get("type") in ("decline", "chat") or not any(c.get("status") == "ok" for c in out.get("calls", [])))
    elif e == "chart":
        ok = bool(out.get("charts"))
    elif e == "detail":
        ok = "**Calculation:**" in out.get("answer", "")
    else:
        ok = False
    g = grade_program(out, turn.get("spec"))
    text = out.get("answer", "")
    parts = (turn.get("spec") or {}).get("parts", [])
    text_complete = all({"associate": "associated with" in text, "anomalies": "unusual" in text.lower()}[x] for x in parts)
    intent_ok = {"answer": any(c.get("status") == "ok" for c in out.get("calls", [])) and not clar,
                 "clarify": clar, "no_answer": ok}.get(e, ok)
    return {"ok": ok, "intent_ok": intent_ok, **g, "text_complete": text_complete if parts else None,
            "unneeded_clarify": e == "answer" and clar,
            "path": out.get("plan_source"), "plan_type": plan.get("type"),
            "model_planned": out.get("plan_source") == "model_program",
            "planner_status": planner.get("status"), "planner_attempts": planner.get("attempts"),
            "planner_ms": planner.get("model_ms"), "rejected": planner.get("rejected") or [],
            "total_ms": out.get("elapsed_ms"), "answer": out.get("answer", "")[:600]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "v1", "base"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--set", choices=["all", "sprint", "audit"], default="all")
    a = ap.parse_args()
    if a.mode == "base" and not os.getenv("ANALYST_PLANNER_URL"):
        sys.exit("--mode base needs ANALYST_PLANNER_URL (and ANALYST_PLANNER_MODEL)")
    if a.mode == "v1":
        os.environ.pop("ANALYST_PLANNER_URL", None)
        os.environ.pop("ANALYST_PLANNER_MODEL", None)
    from app.analyst import orchestrator, scope
    scope.today = lambda: TODAY
    use_model = a.mode != "off"
    rows = []
    cases, convs = {"sprint": (SPRINT_CASES, SPRINT_CONVERSATIONS), "audit": (AUDIT_CASES, AUDIT_CONVERSATIONS)}.get(
        a.set, (CASES + SPRINT_CASES, CONVERSATIONS + SPRINT_CONVERSATIONS))
    for c in cases:
        if a.only and a.only not in c["id"]:
            continue
        t0 = time.monotonic()
        mem = {}
        out = orchestrator.ask(c["q"], DATA[c["ds"]].copy(), mem, use_model=use_model, filename=f"{c['ds']}.csv")
        r = {"id": c["id"], "q": c["q"], "expect": c["expect"], **evaluate(out, c, mem),
             "wall_s": round(time.monotonic() - t0, 2)}
        rows.append(r)
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['wall_s']:6.1f}s {r['path']:<15} {str(r['planner_status']):<12} {c['id']}",
              flush=True)
    for conv in convs:
        if a.only and a.only not in conv["id"]:
            continue
        mem = {}
        for i, t in enumerate(conv["turns"], 1):
            t0 = time.monotonic()
            mem.pop("_nl_last", None)
            out = orchestrator.ask(t["q"], DATA[conv["ds"]].copy(), mem, use_model=use_model, filename=f"{conv['ds']}.csv")
            r = {"id": f"{conv['id']}#{i}", "q": t["q"], "expect": t["expect"], **evaluate(out, t, mem),
                 "wall_s": round(time.monotonic() - t0, 2)}
            rows.append(r)
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['wall_s']:6.1f}s {r['path']:<15} {str(r['planner_status']):<12} "
                  f"{r['id']}", flush=True)
    planned = [r for r in rows if r["planner_status"]]
    s = {"mode": a.mode, "cases": len(rows), "passed": sum(r["ok"] for r in rows),
         "passed_by_model_program": sum(r["ok"] and r["model_planned"] for r in rows),
         "passed_by_deterministic": sum(r["ok"] and not r["model_planned"] for r in rows),
         "planner_invoked": len(planned),
         "valid_plan": sum(r["planner_status"] in ("ok", "reference", "clarify") for r in planned),
         "planned_and_answered_correctly": sum(r["ok"] and r["model_planned"] for r in planned),
         "planned_but_wrong_numbers": sum((not r["ok"]) and r["model_planned"] and r["expect"] == "answer" for r in planned),
         "invalid_column_attempts": sum(any("no column" in x for x in r["rejected"]) for r in planned),
         "clarifications": sum(r["plan_type"] == "clarify" for r in rows),
         "planner_failed_fell_back": sum(r["planner_status"] in ("failed", "unsupported") for r in planned),
         "median_planner_ms": float(np.median([r["planner_ms"] for r in planned if r["planner_ms"]])) if planned else None,
         "median_total_s": float(np.median([r["wall_s"] for r in rows])),
         "max_total_s": float(max(r["wall_s"] for r in rows))}
    def rate(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return f"{sum(v)}/{len(v)}" if v else "0/0"
    mp = [r for r in rows if r["model_planned"]]
    progs = [r for r in rows if r.get("metric_ok") is not None]
    s.update({
        "intent_ok": rate(rows, "intent_ok"),
        "numerically_correct_answers": f"{sum(r['ok'] for r in rows if r['expect'] == 'answer')}/"
                                       f"{sum(r['expect'] == 'answer' for r in rows)}",
        "model_planned: metric_ok": rate(mp, "metric_ok"), "model_planned: column_ok": rate(mp, "column_ok"),
        "model_planned: complete": rate([r for r in mp if r.get("text_complete") is not None], "complete"),
        "model_planned: numerically_correct": f"{sum(r['ok'] for r in mp)}/{len(mp)}",
        "all programs (model or rule): metric_ok": rate(progs, "metric_ok"),
        "composite parts present in answer": rate(rows, "text_complete"),
        "clarification accuracy (asked when needed)": f"{sum(r['ok'] for r in rows if r['expect'] == 'clarify')}/"
                                                      f"{sum(r['expect'] == 'clarify' for r in rows)}",
        "unneeded clarifications": sum(r["unneeded_clarify"] for r in rows),
        "execution_success (valid model programs that ran)": f"{sum(r['model_planned'] for r in planned if r['planner_status'] == 'ok')}/"
                                                            f"{sum(r['planner_status'] in ('ok', 'run_failed') for r in planned)}",
        "p50_total_s": float(np.percentile([r["wall_s"] for r in rows], 50)),
        "p90_total_s": float(np.percentile([r["wall_s"] for r in rows], 90)),
        "rule_program_answers": sum(r["path"] == "rule_program" for r in rows)})
    Path(a.out).write_text(json.dumps({"summary": s, "rows": rows}, indent=1, default=str))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
