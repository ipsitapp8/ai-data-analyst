"""Turning a message into a plan.

A plan is one of
    {"type": "chat", "reply": str}                              conversation, concepts
    {"type": "clarify", "question": str, "options": [str]}     the request is ambiguous
    {"type": "decline", "reason": str}                          cannot be answered from this data
    {"type": "tools", "intent": str, "calls": [{"tool", "args"}]}
    {"type": "explain_previous"}                                "why did you recommend that?"

`model_plan` asks the analyst model (a local OpenAI-compatible server, e.g.
`mlx_lm.server` serving the fine-tuned adapter) for a plan; `rule_plan` is the
deterministic planner used when no model is served or the model's plan is
invalid. Either way the plan's tool calls are validated before anything runs.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from app.analyst import tools
from app.analyst.data import business_roles, find_column, kind_of, schema, schema_text

MODEL_URL = os.getenv("ANALYST_MODEL_URL", "http://127.0.0.1:8080/v1")
# mlx_lm.server applies its --model and --adapter-path to requests for "default_model".
MODEL_NAME = os.getenv("ANALYST_MODEL_NAME", "default_model")
MODEL_LABEL = os.getenv("ANALYST_MODEL_LABEL", "Qwen3-4B-Instruct + Silt analyst LoRA")
MODEL_TIMEOUT = float(os.getenv("ANALYST_MODEL_TIMEOUT", "60"))

SYSTEM = """You are Silt's analyst copilot. You plan; the backend computes. Never write a number that a tool did not return.
Reply with ONE JSON object and nothing else, one of:
{"type":"chat","reply":"..."}  for greetings and general questions (no dataset numbers)
{"type":"clarify","question":"...","options":["..."]}  when the request is ambiguous (e.g. which target column)
{"type":"decline","reason":"..."}  when this dataset cannot answer it
{"type":"explain_previous"}  when the user asks why you recommended or concluded something earlier
{"type":"tools","intent":"...","calls":[{"tool":"<name>","args":{...}}]}  to compute
Use only these tools and only column names from the schema:
"""

CONCEPTS = {
    r"random forest": "A Random Forest is an ensemble of decision trees. Each tree is trained on a bootstrap sample of "
                      "the rows and considers a random subset of features at each split; the forest averages the trees "
                      "(regression) or takes a vote (classification). It handles non-linear effects and mixed feature "
                      "types well. Its feature importance shows what the model relied on, not what causes the outcome.",
    r"p[- ]?value": "A p-value is the probability of seeing a difference at least as large as the one observed if "
                    "there were truly no difference. A small p-value is evidence against 'no difference'; it does "
                    "not measure how large or important the difference is.",
    r"confidence interval": "A 95% confidence interval is a range computed so that, over repeated samples, 95% of such "
                            "ranges would contain the true value. Wide intervals mean the data pin the value down poorly.",
    r"\broi\b|return on investment": "ROI (return on investment) = (revenue - spend) / spend. An ROI of 0.5 means each "
                                     "unit spent returned 1.5 units of revenue.",
    r"\bcac\b|customer acquisition cost": "Customer acquisition cost (CAC) = marketing spend / customers acquired.",
    r"conversion rate": "Conversion rate = conversions / leads (or visitors): the share of prospects who became customers.",
    r"correlation": "Correlation measures how strongly two variables move together (-1 to 1). It does not establish "
                    "that one causes the other.",
}


# ------------------------------------------------------------------- model --

def model_available() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(MODEL_URL.rstrip("/") + "/models", timeout=1.5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def build_prompt(message: str, df: pd.DataFrame, memory: dict) -> list[dict]:
    context = []
    if memory.get("model"):
        context.append(f"A Random Forest for '{memory['model']['target']}' was trained earlier in this conversation.")
    if memory.get("last_recommendation"):
        context.append("A recommendation was made earlier in this conversation.")
    user = (f"Dataset schema:\n{schema_text(schema(df))}\n"
            + (f"Conversation context: {' '.join(context)}\n" if context else "")
            + f"Question: {message}")
    return [{"role": "system", "content": SYSTEM + tools.catalogue()}, {"role": "user", "content": user}]


def model_chat(messages: list[dict], max_tokens: int = 700) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=MODEL_URL, api_key=os.getenv("ANALYST_MODEL_KEY", "local"), timeout=MODEL_TIMEOUT)
    r = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def parse_plan(text: str) -> dict | None:
    """The first JSON object in the model's reply, if it is a well-formed plan."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("type") not in ("chat", "clarify", "decline", "tools", "explain_previous"):
        return None
    if obj["type"] == "tools" and not (isinstance(obj.get("calls"), list) and obj["calls"]):
        return None
    if obj["type"] == "chat" and not isinstance(obj.get("reply"), str):
        return None
    return obj


def model_plan(message: str, df: pd.DataFrame, memory: dict) -> dict | None:
    return parse_plan(model_chat(build_prompt(message, df, memory)))


# ------------------------------------------------------------------- rules --

_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)")


def _focus_value(df: pd.DataFrame, message: str, column: str | None) -> str | None:
    if column is None:
        return None
    low = message.lower()
    hits = [v for v in df[column].dropna().astype(str).unique() if re.search(rf"(?<![a-z0-9]){re.escape(v.lower())}(?![a-z0-9])", low)]
    return max(hits, key=len) if hits else None


def _target_candidates(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        k, u = kind_of(df[c]), df[c].nunique(dropna=True)
        if (k == "categorical" and 2 <= u <= 10) or (k == "numeric" and u > 10 and not re.search(r"(^|_)id$", c.lower())):
            out.append(c)
    return out


def rule_plan(message: str, df: pd.DataFrame, memory: dict) -> dict:
    m = message.strip()
    low = m.lower()
    if re.fullmatch(r"(hi|hello|hey|hiya|good (morning|afternoon|evening)|thanks|thank you)[!. ]*", low):
        return {"type": "chat", "reply": "Hello! I'm Silt's analyst. Ask me about this dataset: averages and comparisons, "
                                         "statistical tests, Random Forest predictions, or advertising recommendations."}
    if re.search(r"\bwhy\b.*\b(recommend|suggest|conclude|choose|chose)", low):
        return {"type": "explain_previous"}
    if re.search(r"\bwhich factors|feature importance|what (influenced|drove|drives)|most important (features|factors)", low):
        return {"type": "tools", "intent": "explain_prediction", "calls": [{"tool": "feature_importance", "args": {}}]}
    if re.match(r"(what|explain|define)\b.*\b(is|are|does|mean)\b", low) or re.match(r"what is (a |an )?", low):
        for pat, text in CONCEPTS.items():
            if re.search(pat, low) and not find_column(df, low):
                return {"type": "chat", "reply": text}

    roles = business_roles(df)["roles"]
    area_col = roles.get("area") or roles.get("campaign")
    focus = _focus_value(df, m, area_col)

    pct = _PCT.search(low)
    if pct and re.search(r"what if|scenario|increase|decrease|raise|cut|reduce", low) and re.search(r"budget|spend", low):
        change = float(pct.group(1)) * (-1 if re.search(r"decrease|cut|reduce|lower", low) else 1)
        args = {"change_pct": change}
        if focus:
            args["focus"] = focus
        return {"type": "tools", "intent": "scenario", "calls": [{"tool": "budget_scenario", "args": args}]}

    if re.search(r"advertis|campaign|marketing|ad spend|budget", low) and \
            re.search(r"should|which|where|recommend|worth|best|next", low):
        if "spend" not in roles or not ({"revenue", "conversions"} & roles.keys()) or area_col is None:
            return {"type": "decline", "reason": "This dataset has no advertising history (spend with revenue or customers "
                                                 "acquired per area or campaign), so I cannot make an evidence-based "
                                                 "advertising recommendation from it."}
        kind = "increase" if re.search(r"increase|more|raise|expand|grow", low) else "where"
        args = {"kind": kind}
        if focus:
            args["focus"] = focus
        return {"type": "tools", "intent": "recommendation", "calls": [{"tool": "recommend", "args": args}]}

    pm = re.search(r"\b(predict|forecast|estimate)\b(.*)", low)
    if pm:
        named = [c for c in find_column(df, pm.group(2))]
        if not named and "churn" in pm.group(2):
            named = [c for c in df.columns if "churn" in c.lower()]
        if len(named) == 1:
            return {"type": "tools", "intent": "prediction", "calls": [{"tool": "random_forest", "args": {"target": named[0]}}]}
        options = named or _target_candidates(df)
        if not options:
            return {"type": "decline", "reason": "No column in this dataset is a suitable prediction target."}
        return {"type": "clarify", "question": "Which column should I predict?", "options": options[:8]}

    cols = find_column(df, low)
    num_cols = [c for c in cols if kind_of(df[c]) == "numeric"]
    cat_cols = [c for c in cols if kind_of(df[c]) == "categorical"]
    if re.search(r"\bpay|paid|earn|wage", low) and not num_cols:
        num_cols = [c for c in df.columns if re.search(r"salary|pay|wage|income|compensation", c.lower())
                    and kind_of(df[c]) == "numeric"][:1]
    if re.search(r"correlat|relationship between|associated with", low) and len(num_cols) >= 2:
        return {"type": "tools", "intent": "correlation", "calls": [{"tool": "correlation", "args": {"x": num_cols[0], "y": num_cols[1]}}]}
    if re.search(r"significant|difference|differ|compare|test", low) and cat_cols and (num_cols or len(cat_cols) >= 2):
        outcome = num_cols[0] if num_cols else cat_cols[1]
        return {"type": "tools", "intent": "comparison", "calls": [{"tool": "compare_groups", "args": {"outcome": outcome, "group": cat_cols[0]}}]}
    if re.search(r"standard deviation|variance|\bstd\b|confidence interval|describe|summary|distribution", low) and num_cols:
        return {"type": "tools", "intent": "statistics", "calls": [{"tool": "describe_column", "args": {"column": num_cols[0]}}]}
    if re.search(r"\bwhich\b.*\b(most|highest|least|lowest|best|worst|largest|smallest)\b", low) and cat_cols and num_cols:
        order = "asc" if re.search(r"least|lowest|worst|smallest", low) else "desc"
        agg = "sum" if re.search(r"total|in all|overall", low) else "avg"
        return {"type": "tools", "intent": "group_comparison", "calls": [
            {"tool": "compute_metric", "args": {"aggregation": agg, "column": num_cols[0], "group_by": cat_cols[0], "order": order}},
            {"tool": "compare_groups", "args": {"outcome": num_cols[0], "group": cat_cols[0]}}]}
    agg = next((a for pat, a in ((r"average|mean", "avg"), (r"median", "median"), (r"total|sum", "sum"),
                                  (r"maximum|max|highest|largest", "max"), (r"minimum|min|lowest|smallest", "min"))
                if re.search(rf"\b(?:{pat})\b", low)), None)
    if agg and num_cols:
        args = {"aggregation": agg, "column": num_cols[0]}
        by = re.search(r"\b(by|per|for each|across)\b", low)
        if by and cat_cols:
            args["group_by"] = cat_cols[0]
        return {"type": "tools", "intent": "metric", "calls": [{"tool": "compute_metric", "args": args}]}
    if re.search(r"how many (rows|records)|number of (rows|records)", low):
        return {"type": "tools", "intent": "metric", "calls": [{"tool": "compute_metric", "args": {"aggregation": "count"}}]}
    if re.search(r"\b(average|mean|median|total|sum|maximum|minimum)\b", low) and not num_cols:
        return {"type": "clarify", "question": "Which numeric column do you mean?",
                "options": [c for c in df.columns if kind_of(df[c]) == "numeric"][:8]}
    return {"type": "chat", "reply": "I'm not sure what to compute. You can ask, for example: 'What is the average "
                                     "<column>?', 'Which <group> has the highest <column>?', 'Predict <column>', or "
                                     "'Which area should receive our next advertising campaign?'"}
