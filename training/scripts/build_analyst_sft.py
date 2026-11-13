"""Supervised fine-tuning data for the Analyst Copilot's planner and narrator.

    .venv313/bin/python training/scripts/build_analyst_sft.py

Writes training/data/analyst_sft/{train,valid,test}.jsonl (chat "messages"
format, as mlx_lm.lora expects) and MANIFEST.json.

What the model learns
    plan     question + schema -> one JSON plan (chat / clarify / decline /
             explain_previous / tool calls). The prompt is exactly the one the
             application sends (app.analyst.planner.build_prompt).
    narrate  question + real tool results -> a short summary that uses only
             numbers present in the results.

Targets are written by construction from templates whose intended plan is known;
they are never produced by the rule planner, and no target contains a number
that is not in the prompt (plans) or the tool results (summaries).

Sources and protected data
    * Spider TRAINING split, "tune" databases only (training/data/eval/spider_train_dev,
      split == "tune"). The "check" split and Spider's validation split (which
      holds the 138 held-out questions) are never read.
    * Invented schemas and data (seeded), across several business domains.

Splits: the test split holds two whole domains never seen in training
(education, real_estate) plus a share of the question phrasings; valid is a
random 6% of the rest.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.analyst import explain, orchestrator, planner, tools  # noqa: E402
from app.analyst.data import kind_of  # noqa: E402

OUT = ROOT / "training/data/analyst_sft"
SPIDER_DEV = ROOT / "training/data/eval/spider_train_dev/cases.jsonl"
TEST_DOMAINS = {"education", "real_estate"}
SEED = 20261111
R = random.Random(SEED)


# ----------------------------------------------------------------- schemas --

def pick(*options):
    return R.choice(options)


def nat(col: str) -> str:
    return col.replace("_", " ").lower()


def domain_frame(domain: str, rng: np.random.Generator, n: int = 60) -> tuple[pd.DataFrame, dict]:
    """A small invented table and what its columns mean."""
    meta: dict = {"domain": domain, "entity": "rows"}
    if domain == "hr":
        idc, dep, lvl = pick("employee_id", "emp_no", "staff_id"), pick("department", "team", "division"), pick("level", "grade", "seniority")
        sal, yrs = pick("salary", "annual_salary", "pay", "base_salary"), pick("years_experience", "tenure_years", "years_at_company")
        tgt = pick("left_company", "attrition", "resigned")
        df = pd.DataFrame({idc: range(1, n + 1), dep: rng.choice(["Engineering", "Sales", "Finance", "Support"], n),
                           lvl: rng.choice(["Junior", "Mid", "Senior"], n), yrs: rng.integers(0, 30, n),
                           "age": rng.integers(21, 65, n), pick("bonus", "annual_bonus"): rng.integers(0, 9000, n),
                           sal: rng.integers(40000, 150000, n), tgt: rng.choice(["Yes", "No"], n)})
        meta.update(entity=pick("employees", "staff"), measure=[sal, yrs, "age"], group=[dep, lvl], cls_target=tgt,
                    reg_target=sal, class_word={"left_company": "leave the company", "attrition": "leave",
                                                "resigned": "resign"}[tgt])
    elif domain == "churn":
        cid, plan, reg = pick("customer_id", "account_id", "client_id"), pick("contract", "plan", "subscription_type"), pick("region", "state", "market")
        tgt = pick("churn", "churned", "cancelled")
        ten, chg = pick("tenure_months", "months_active", "tenure"), pick("monthly_charges", "monthly_fee", "monthly_spend")
        df = pd.DataFrame({cid: [f"C{i}" for i in range(n)], plan: rng.choice(["Monthly", "Annual", "Two-year"], n),
                           reg: rng.choice(["North", "South", "East", "West"], n), ten: rng.integers(1, 72, n),
                           chg: rng.uniform(15, 120, n).round(2), pick("support_tickets", "complaints"): rng.poisson(2, n),
                           tgt: rng.choice(["Yes", "No"], n)})
        meta.update(entity=pick("customers", "clients", "subscribers"), measure=[chg, ten], group=[plan, reg],
                    cls_target=tgt, reg_target=chg, class_word={"churn": "churn", "churned": "churn", "cancelled": "cancel"}[tgt])
    elif domain == "sales":
        store, cat = pick("store", "branch", "outlet"), pick("product_category", "category", "department")
        rev, units = pick("revenue", "sales", "total_sales"), pick("units_sold", "quantity", "units")
        df = pd.DataFrame({"order_id": range(n), store: rng.choice(["Downtown", "Airport", "Mall", "Harbor"], n),
                           cat: rng.choice(["Toys", "Books", "Garden", "Kitchen"], n), units: rng.integers(1, 50, n),
                           pick("unit_price", "price"): rng.uniform(2, 80, n).round(2), pick("discount", "discount_pct"): rng.uniform(0, 0.3, n).round(2),
                           rev: rng.uniform(10, 4000, n).round(2), pick("returned", "is_returned"): rng.choice(["Yes", "No"], n)})
        meta.update(entity=pick("orders", "transactions"), measure=[rev, units], group=[store, cat], reg_target=rev,
                    cls_target=[c for c in df.columns if c in ("returned", "is_returned")][0], class_word="be returned")
    elif domain in ("marketing", "marketing_alt"):
        area = pick("area", "region", "market", "city", "territory")
        spend = pick("ad_spend", "marketing_spend", "spend", "campaign_cost", "budget")
        leads = pick("leads", "clicks", "visits")
        conv = pick("conversions", "new_customers", "customers_acquired", "acquisitions")
        rev = pick("revenue", "sales")
        places = R.sample(["Area A", "Area B", "Area C", "Area D", "North", "South", "East", "West", "Lagos", "Leeds",
                           "Austin", "Denver", "Zone 1", "Zone 2", "Zone 3"], 4)
        df = pd.DataFrame({pick("month", "week", "date"): pd.date_range("2025-01-01", periods=n, freq="W"),
                           area: rng.choice(places, n), pick("channel", "campaign"): rng.choice(["Search", "Social", "Email"], n),
                           spend: rng.uniform(500, 9000, n).round(2), leads: rng.integers(100, 2000, n),
                           conv: rng.integers(5, 120, n), rev: rng.uniform(500, 20000, n).round(2)})
        meta.update(entity="campaigns", measure=[spend, rev, conv], group=[area], marketing=True, area=area, places=places,
                    reg_target=rev, cls_target=None)
    elif domain == "inventory":
        df = pd.DataFrame({pick("sku", "item_code"): [f"S{i}" for i in range(n)], pick("category", "product_type"): rng.choice(["Tools", "Garden", "Paint"], n),
                           pick("stock", "on_hand", "quantity_in_stock"): rng.integers(0, 500, n),
                           pick("unit_price", "cost_price"): rng.uniform(1, 90, n).round(2), pick("supplier", "vendor"): rng.choice(["Acme", "Globex", "Initech"], n)})
        cols = list(df.columns)
        meta.update(entity="items", measure=[cols[2], cols[3]], group=[cols[1], cols[4]], reg_target=None, cls_target=None)
    elif domain == "education":
        df = pd.DataFrame({"student_id": range(n), pick("school", "campus"): rng.choice(["Northside", "Riverside", "Hillcrest"], n),
                           pick("hours_studied", "study_hours"): rng.uniform(0, 30, n).round(1), pick("attendance_rate", "attendance"): rng.uniform(0.5, 1, n).round(2),
                           pick("exam_score", "final_score", "grade_points"): rng.uniform(30, 100, n).round(1),
                           pick("passed", "graduated"): rng.choice(["Yes", "No"], n), pick("gender", "cohort"): rng.choice(["A", "B"], n)})
        cols = list(df.columns)
        meta.update(entity="students", measure=[cols[4], cols[2]], group=[cols[1], cols[6]], cls_target=cols[5],
                    reg_target=cols[4], class_word="pass" if cols[5] == "passed" else "graduate")
    elif domain == "real_estate":
        df = pd.DataFrame({"listing_id": range(n), pick("city", "neighbourhood", "district"): rng.choice(["Oakwood", "Bayview", "Elm Park"], n),
                           pick("bedrooms", "rooms"): rng.integers(1, 6, n), pick("floor_area_sqm", "size_sqm", "area_sqft"): rng.uniform(30, 300, n).round(0),
                           pick("sale_price", "price", "listing_price"): rng.uniform(80000, 900000, n).round(0),
                           pick("sold_within_90_days", "sold_quickly"): rng.choice(["Yes", "No"], n), pick("property_type", "home_type"): rng.choice(["Flat", "House"], n)})
        cols = list(df.columns)
        meta.update(entity=pick("listings", "properties", "homes"), measure=[cols[4], cols[3]], group=[cols[1], cols[6]],
                    cls_target=cols[5], reg_target=cols[4], class_word="sell within 90 days")
    else:
        raise ValueError(domain)
    meta.setdefault("marketing", False)
    return df, meta


DOMAINS = ["hr", "churn", "sales", "marketing", "marketing_alt", "inventory", "education", "real_estate"]


# ---------------------------------------------------------------- examples --

def plan_tools(intent, *calls):
    return {"type": "tools", "intent": intent, "calls": [{"tool": t, "args": a} for t, a in calls]}


AGG_WORDS = {"avg": ["average", "mean"], "sum": ["total", "sum of"], "max": ["maximum", "highest", "largest"],
             "min": ["minimum", "lowest", "smallest"], "median": ["median"]}
CHAT = [
    (["Hello", "Hi!", "hey there", "Good morning", "hello, who are you?"],
     "Hello! I'm Silt's analyst. I can compute figures from your dataset, run statistical tests, build Random Forest "
     "predictions and give evidence-based recommendations. What would you like to know?"),
    (["Thanks!", "thank you", "Great, thanks a lot"], "You're welcome! Ask me anything else about your data."),
    (["What can you do?", "How can you help me?", "what kind of questions can I ask"],
     "I can answer questions about your dataset (averages, totals, comparisons between groups), run statistical tests "
     "with effect sizes, build Random Forest models to predict a column, analyse campaign performance and recommend "
     "where to advertise, and explain the evidence behind every answer. Every number I give is computed from your data."),
    (["What is Random Forest?", "Explain random forests", "how does a random forest work?"], planner.CONCEPTS[r"random forest"]),
    (["What is a p-value?", "what does p value mean", "Explain p-values simply"], planner.CONCEPTS[r"p[- ]?value"]),
    (["What is ROI?", "What does return on investment mean?"], planner.CONCEPTS[r"\broi\b|return on investment"]),
    (["What is CAC?", "what is customer acquisition cost"], planner.CONCEPTS[r"\bcac\b|customer acquisition cost"]),
    (["What is a confidence interval?", "explain confidence intervals"], planner.CONCEPTS[r"confidence interval"]),
    (["What is overfitting?", "Explain overfitting"],
     "Overfitting is when a model learns noise in its training data instead of the general pattern, so it performs "
     "well on the rows it was trained on and worse on new rows. Evaluating on held-out rows, as I do, reveals it."),
    (["Does correlation mean causation?", "If two things are correlated does one cause the other?"],
     "No. Correlation shows two variables move together; it does not show that one causes the other. A third factor, "
     "reverse causation or chance can produce a correlation. A controlled experiment is the usual way to test cause."),
    (["Why do you split data into training and test sets?", "what is a train test split"],
     "A model is fitted on the training rows and evaluated on test rows it never saw, so the reported accuracy "
     "reflects how it would do on new data rather than how well it memorised the training data."),
]


def phrasing(variants: list[str], key: str) -> tuple[str, bool]:
    """A phrasing and whether it is reserved for the test split (every 5th variant, by a stable hash)."""
    i = R.randrange(len(variants))
    held = int(hashlib.sha256(f"{key}:{i}".encode()).hexdigest(), 16) % 5 == 0 and len(variants) > 2
    return variants[i], held


def examples_for(df: pd.DataFrame, meta: dict) -> list[tuple[str, dict, str, bool]]:
    """(question, plan, context, held_phrasing) for one schema."""
    out = []
    num = [c for c in df.columns if kind_of(df[c]) == "numeric" and not c.endswith("_id") and c != "order_id"]
    cat = [c for c in df.columns if kind_of(df[c]) == "categorical" and df[c].nunique() <= 10]
    m, g = R.choice(meta["measure"]), R.choice(meta["group"])
    ent = meta["entity"]

    agg = R.choice(list(AGG_WORDS))
    w = R.choice(AGG_WORDS[agg])
    q, h = phrasing([f"What is the {w} {nat(m)}?", f"{w} {nat(m)}", f"Can you tell me the {w} {nat(m)} across all {ent}?",
                     f"what's the {w} {nat(m)}", f"Give me the {w} of {nat(m)}"], "metric")
    out.append((q, plan_tools("metric", ("compute_metric", {"aggregation": agg, "column": m})), "", h))

    q, h = phrasing([f"Show the {w} {nat(m)} by {nat(g)}", f"{w} {nat(m)} for each {nat(g)}",
                     f"Break down {w} {nat(m)} per {nat(g)}", f"What is the {w} {nat(m)} in each {nat(g)}?"], "metric_by")
    out.append((q, plan_tools("metric", ("compute_metric", {"aggregation": agg, "column": m, "group_by": g})), "", h))

    val = str(df[g].dropna().iloc[0])
    q, h = phrasing([f"What is the average {nat(m)} for {val}?", f"average {nat(m)} where {nat(g)} is {val}",
                     f"How much {nat(m)} on average in {val}?"], "metric_filter")
    out.append((q, plan_tools("metric", ("compute_metric", {"aggregation": "avg", "column": m,
                                                              "filters": [{"column": g, "op": "==", "value": val}]})), "", h))

    thr = int(np.percentile(df[m].dropna(), 60))
    q, h = phrasing([f"How many {ent} have {nat(m)} above {thr}?", f"count of {ent} with {nat(m)} greater than {thr}",
                     f"How many rows have {nat(m)} over {thr}?"], "count_filter")
    out.append((q, plan_tools("metric", ("compute_metric", {"aggregation": "count",
                                                              "filters": [{"column": m, "op": ">", "value": thr}]})), "", h))

    q, h = phrasing(["How many rows are there?", f"How many {ent} are in the data?", "count the records"], "count")
    out.append((q, plan_tools("metric", ("compute_metric", {"aggregation": "count"})), "", h))

    hi = R.random() < 0.7
    q, h = phrasing([f"Which {nat(g)} has the {'highest' if hi else 'lowest'} average {nat(m)}?",
                     f"Which {nat(g)} {'is best' if hi else 'is worst'} on {nat(m)}?",
                     f"which {nat(g)} has the {'most' if hi else 'least'} {nat(m)} on average"], "which_most")
    out.append((q, plan_tools("group_comparison",
                              ("compute_metric", {"aggregation": "avg", "column": m, "group_by": g, "order": "desc" if hi else "asc"}),
                              ("compare_groups", {"outcome": m, "group": g})), "", h))

    q, h = phrasing([f"What is the standard deviation of {nat(m)}?", f"Describe the distribution of {nat(m)}",
                     f"Give me summary statistics for {nat(m)}", f"variance of {nat(m)}",
                     f"What is the 95% confidence interval for the mean {nat(m)}?"], "describe")
    out.append((q, plan_tools("statistics", ("describe_column", {"column": m})), "", h))

    q, h = phrasing([f"Is there a significant difference in {nat(m)} between {nat(g)} groups?",
                     f"Does {nat(m)} differ by {nat(g)}?", f"Compare {nat(m)} across {nat(g)} with a statistical test",
                     f"test whether {nat(m)} varies between {nat(g)}s"], "compare")
    out.append((q, plan_tools("comparison", ("compare_groups", {"outcome": m, "group": g})), "", h))

    if len(num) >= 2:
        x, y = R.sample(num, 2)
        q, h = phrasing([f"Is {nat(x)} correlated with {nat(y)}?", f"What is the relationship between {nat(x)} and {nat(y)}?",
                         f"correlation between {nat(x)} and {nat(y)}"], "corr")
        out.append((q, plan_tools("correlation", ("correlation", {"x": x, "y": y})), "", h))
        q, h = phrasing([f"Run a linear regression of {nat(y)} on {nat(x)}", f"Regress {nat(y)} on {nat(x)}",
                         f"How does {nat(x)} relate to {nat(y)} in a regression?"], "regression")
        out.append((q, plan_tools("regression", ("linear_regression", {"target": y, "features": [x]})), "", h))

    if meta.get("cls_target"):
        t = meta["cls_target"]
        q, h = phrasing([f"Predict {nat(t)}", f"Can you predict which {ent} will {meta['class_word']}?",
                         f"Build a model to predict {nat(t)}", f"Which {ent} are likely to {meta['class_word']}? Use a model.",
                         f"train a classifier for {nat(t)}"], "predict_cls")
        out.append((q, plan_tools("prediction", ("random_forest", {"target": t, "task": "classification"})), "", h))
    if meta.get("reg_target"):
        t = meta["reg_target"]
        q, h = phrasing([f"Predict {nat(t)}", f"Build a model that predicts {nat(t)}", f"Can you estimate {nat(t)} with a model?",
                         f"forecast {nat(t)} from the other columns"], "predict_reg")
        out.append((q, plan_tools("prediction", ("random_forest", {"target": t, "task": "regression"})), "", h))
    candidates = planner._target_candidates(df)
    if candidates:
        q, h = phrasing(["Build a prediction model", "Make a prediction for me", "Can you predict something useful?",
                         "Train a machine learning model"], "clarify_target")
        out.append((q, {"type": "clarify", "question": "Which column should I predict?", "options": candidates[:8]}, "", h))
    if not any("churn" in c for c in df.columns) and candidates:
        q, h = phrasing(["Predict churn.", "Which customers will churn?", "predict customer churn"], "clarify_churn")
        out.append((q, {"type": "clarify", "question": "There is no churn column. Which column should I predict?",
                        "options": candidates[:8]}, "", h))
    if not candidates:
        q, h = phrasing(["Predict sales", "Build a model to predict churn"], "decline_predict")
        out.append((q, {"type": "decline", "reason": "No column in this dataset is a suitable prediction target."}, "", h))

    if meta["marketing"]:
        area, place = meta["area"], R.choice(meta["places"])
        q, h = phrasing([f"Which {nat(area)} should receive our next advertising campaign?", "Where should we advertise next?",
                         f"Which {nat(area)} gives the best return on ad spend?", "Where should our next campaign go?",
                         f"Recommend a {nat(area)} for the next marketing push"], "rec_where")
        out.append((q, plan_tools("recommendation", ("recommend", {"kind": "where"})), "", h))
        q, h = phrasing([f"Should we increase the advertising budget in {place}?", f"Is it worth spending more on ads in {place}?",
                         f"Should we put more marketing money into {place}?", f"Do you think we should advertise more in {place}?"], "rec_increase")
        out.append((q, plan_tools("recommendation", ("recommend", {"kind": "increase", "focus": place})), "", h))
        q, h = phrasing([f"Show campaign performance by {nat(area)}", f"Compare ROI and CAC across {nat(area)}s",
                         f"How did each {nat(area)} perform on ad spend?"], "performance")
        out.append((q, plan_tools("performance", ("business_performance", {"group": area})), "", h))
        pct = R.choice([5, 10, 15, 20, 25, 30, 50])
        up = R.random() < 0.7
        scoped = R.random() < 0.4
        args = {"change_pct": float(pct if up else -pct)}
        if scoped:
            args["focus"] = place
        where = f" in {place}" if scoped else ""
        q, h = phrasing([f"What if we {'increase' if up else 'cut'} the budget{where} by {pct}%?",
                         f"What happens if ad spend{where} goes {'up' if up else 'down'} {pct}%?",
                         f"Simulate a {pct}% budget {'increase' if up else 'reduction'}{where}"], "scenario")
        out.append((q, plan_tools("scenario", ("budget_scenario", args)), "", h))
        ctx = "A recommendation was made earlier in this conversation."
        q, h = phrasing([f"Why do you recommend {place}?", "Why did you suggest that?", "What is the evidence for your recommendation?",
                         "why that area?"], "why")
        out.append((q, {"type": "explain_previous"}, ctx, h))
    else:
        q, h = phrasing(["Which area should receive our next advertising campaign?", "Should we increase our ad budget?",
                         "Where should we advertise next?"], "decline_ads")
        out.append((q, {"type": "decline", "reason": "This dataset has no advertising history (spend with revenue or "
                                                      "customers acquired per area or campaign), so I cannot make an "
                                                      "evidence-based advertising recommendation from it."}, "", h))
    if meta.get("cls_target") or meta.get("reg_target"):
        t = meta.get("cls_target") or meta["reg_target"]
        ctx = f"A Random Forest for '{t}' was trained earlier in this conversation."
        q, h = phrasing(["Which factors influenced your prediction?", "What drove the model's predictions?",
                         "Which features matter most?", "show feature importance"], "importance")
        out.append((q, plan_tools("explain_prediction", ("feature_importance", {})), ctx, h))
    return out


def messages(df: pd.DataFrame, question: str, context: str, assistant: str) -> list[dict]:
    mem = {}
    if "Random Forest" in context:
        mem["model"] = {"target": context.split("'")[1]}
    if "recommendation" in context:
        mem["last_recommendation"] = True
    msgs = planner.build_prompt(question, df, mem)
    return msgs + [{"role": "assistant", "content": assistant}]


def narration(df: pd.DataFrame, meta: dict, rng) -> tuple[list[dict], str] | None:
    """A real tool result on invented data and a summary written only from it."""
    m, g = R.choice(meta["measure"]), R.choice(meta["group"])
    kind = R.choice(["metric", "which", "compare"] + (["rec"] * 2 if meta["marketing"] else []))
    if kind == "metric":
        call = {"tool": "compute_metric", "args": {"aggregation": "avg", "column": m}}
        q = f"What is the average {nat(m)}?"
    elif kind == "which":
        call = {"tool": "compute_metric", "args": {"aggregation": "avg", "column": m, "group_by": g}}
        q = f"Which {nat(g)} has the highest average {nat(m)}?"
    elif kind == "compare":
        call = {"tool": "compare_groups", "args": {"outcome": m, "group": g}}
        q = f"Does {nat(m)} differ by {nat(g)}?"
    else:
        call = {"tool": "recommend", "args": {"kind": "where"}}
        q = "Which area should receive our next advertising campaign?"
    try:
        r = orchestrator._jsonable(tools.run(tools.validate_call(call, df), df, {}))
    except Exception:  # noqa: BLE001
        return None
    f, pct = explain.f, lambda x: explain.f(x, pct=True)
    if kind == "metric":
        text = f"The average {nat(m)} is {f(r['value'])}, computed over {r['rows']} rows."
    elif kind == "which":
        top, bottom = r["groups"][0], r["groups"][-1]
        text = (f"{top['group']} has the highest average {nat(m)} at {f(top['value'])}, and {bottom['group']} the lowest "
                f"at {f(bottom['value'])}. This compares averages only; see the statistical test before treating the gap as real.")
    elif kind == "compare":
        t = r.get("test")
        if not t:
            return None
        sig = (t.get("p_value") or 1) < 0.05
        text = (f"The {t['name']} gives {explain.p(t['p_value'])}. " +
                (f"{nat(m).capitalize()} differs between {nat(g)} groups at the 5% level." if sig else
                 f"The data do not show a reliable difference in {nat(m)} between {nat(g)} groups."))
    else:
        rec, perf = r["recommendation"], r["performance"]
        row = next(x for x in perf["groups"] if x["group"] == rec["focus"])
        text = (f"{rec['focus']} had the highest return: ROI {pct(row['roi'])} on spend of {f(row['spend'])}. "
                f"The evidence is {rec['evidence_level']}, so the recommendation is: {rec['recommendation']} "
                "This is based on historical, observational data, which does not prove that spending caused the results.")
    if explain.ungrounded(text, r, q):
        return None
    msgs = [{"role": "system", "content": orchestrator.NARRATE_SYSTEM},
            {"role": "user", "content": f"Question: {q}\nResults: {json.dumps([r], default=str)[:6000]}"},
            {"role": "assistant", "content": text}]
    return msgs, kind


def spider_examples() -> list[dict]:
    """Spider training questions from the tune databases, with their verified reference plans."""
    if not SPIDER_DEV.exists():
        return []
    out = []
    for c in map(json.loads, SPIDER_DEV.open(encoding="utf-8")):
        if c["split"] != "tune":
            continue
        calls, ok = [], True
        for formula in c["spec"]["formulas"]:
            if formula == "count()":
                args = {"aggregation": "count"}
            else:
                fn, col = formula.split("(", 1)
                args = {"aggregation": fn, "column": col.strip(')"')}
            if c["spec"]["filters"]:
                fs = []
                for flt in c["spec"]["filters"]:
                    if flt["op"] not in tools.OPS:
                        ok = False
                    v = flt["value"]
                    fs.append({"column": flt["column"], "op": flt["op"],
                               "value": int(v) if isinstance(v, float) and v.is_integer() else v})
                args["filters"] = fs
            calls.append(("compute_metric", args))
        if not ok:
            continue
        df = pd.read_csv(ROOT / c["csv"])
        plan = plan_tools("metric", *calls)
        try:
            for call in plan["calls"]:
                tools.validate_call(call, df)
        except tools.InvalidCall:
            continue
        out.append({"messages": messages(df, c["question"], "", json.dumps(plan)), "source": "spider_tune",
                    "db": c["db_id"]})
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    train, test = [], []
    for domain in DOMAINS:
        reps = 26 if domain in TEST_DOMAINS else 40
        for i in range(reps):
            df, meta = domain_frame(domain, rng)
            for q, plan, ctx, held in examples_for(df, meta):
                ex = {"messages": messages(df, q, ctx, json.dumps(plan)), "source": f"template:{domain}",
                      "intent": plan.get("intent", plan["type"])}
                (test if domain in TEST_DOMAINS or held else train).append(ex)
            for _ in range(2):
                n = narration(df, meta, rng)
                if n:
                    ex = {"messages": n[0], "source": f"narrate:{domain}", "intent": f"narrate_{n[1]}"}
                    (test if domain in TEST_DOMAINS else train).append(ex)
        for variants, reply in CHAT:
            df, _ = domain_frame(domain, rng)
            for v in variants:
                ex = {"messages": messages(df, v, "", json.dumps({"type": "chat", "reply": reply})), "source": "chat",
                      "intent": "chat"}
                held = int(hashlib.sha256(f"chat:{v}:{domain}".encode()).hexdigest(), 16) % 7 == 0
                (test if held or domain in TEST_DOMAINS else train).append(ex)
    spider = spider_examples()
    R.shuffle(spider)
    train += spider[:450]
    R.shuffle(train)
    n_valid = int(len(train) * 0.06)
    valid, train = train[:n_valid], train[n_valid:]
    R.shuffle(test)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "base_prompt": "app.analyst.planner.build_prompt (identical at inference)",
                "protected_data": "Spider validation split (138 held-out questions) and the spider_train_dev 'check' "
                                  "split are not read; Spider examples come from 'tune' databases only.",
                "test_domains": sorted(TEST_DOMAINS), "splits": {}}
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
        with (OUT / f"{name}_meta.jsonl").open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps({k: v for k, v in ex.items() if k != "messages"}) + "\n")
        from collections import Counter
        manifest["splits"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "by_source": dict(Counter(e["source"].split(":")[0] for e in rows)),
                                    "by_intent": dict(Counter(e.get("intent", "?") for e in rows))}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["rows"] for k, v in manifest["splits"].items()}), json.dumps(manifest["splits"]["train"]["by_source"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
