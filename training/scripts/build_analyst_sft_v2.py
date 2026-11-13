"""Fine-tuning data v2 for the Analyst Copilot: v1 plus corrected, contrastive examples.

    .venv313/bin/python training/scripts/build_analyst_sft_v2.py

Writes training/data/analyst_sft_v2/. v1 (training/data/analyst_sft and its
builder) is read, never changed.

What changed and why (failures found in the v1 evaluation):
  * churn        v1 taught "predict customer churn" only where no churn column
                 existed (-> clarify). v2 adds churn phrasings on churn
                 datasets (-> random_forest) paired with the same phrasings on
                 datasets without one (-> clarify).
  * advertising  v1 sometimes declined advertising questions on advertising
                 data. v2 adds identical questions on marketing schemas
                 (-> recommend / business_performance) and on schemas with no
                 advertising history (-> decline).
  * budget       "budget" questions where spend is named ad_spend, marketing_spend, ...
  * ROI / CAC    ROI and CAC are computed by business_performance, never columns.

Each pair has the same question with different correct outcomes depending on
the columns available: the plan must be read from the schema, not the words.

Protected evaluation data: the test split is v1's test split, copied byte for
byte (the frozen held-out set). No v2 training or validation question equals
(case and punctuation folded) any held-out test question, any of the 25 frozen
execution questions or any of the 14 acceptance questions; the build fails if
one does. MANIFEST.json records the check, and also the overlaps v1 already had.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/evals"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_analyst_sft as v1  # noqa: E402

V1 = ROOT / "training/data/analyst_sft"
OUT = ROOT / "training/data/analyst_sft_v2"
SEED = 20261112
R = random.Random(SEED)
DRAWS = 9
REPEAT_NEW = 3   # each new training example appears this many times (validation: once)
NON_MARKETING = ["hr", "churn", "sales", "inventory"]
NON_CHURN = ["hr", "sales", "inventory"]
DECLINE_ADS = ("This dataset has no advertising history (spend with revenue or customers acquired per area or "
               "campaign), so I cannot make an evidence-based advertising recommendation from it.")

CHURN_Q = ["Can you model customer churn?", "Build a churn prediction model", "predict which customers churn",
           "Which customers are going to churn? Build a model.", "Forecast churn for our customers",
           "Train a model for customer churn", "I want to predict churn for our clients"]
WHERE_Q = ["Where should our next ad campaign run?", "Which area deserves the next marketing campaign?",
           "Pick the best region for our next campaign", "Which market should we target with the next campaign?",
           "Where would advertising money work best?", "Recommend where to run our next campaign",
           "which territory should get the next round of ads"]
INCREASE_Q = ["Should we raise our advertising spend?", "Is it a good idea to spend more on marketing?",
              "Should we grow the ad budget?", "Do you think we should invest more in ads?",
              "Should the marketing budget go up?", "Would more ad spend pay off for us?"]
ROI_Q = ["What are ROI, CAC and conversion rates per {a}?", "Give me return on ad spend and acquisition cost for every {a}",
         "How do the {a}s compare on ROI and customer acquisition cost?", "Campaign ROI and CAC broken down by {a}",
         "Which {a} has the best ROI and the lowest CAC?", "compare return on investment across {a}s"]


def fold(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def frozen_questions() -> dict[str, set[str]]:
    import analyst_acceptance
    import analyst_model_execution
    test_q = {fold(json.loads(l)["messages"][1]["content"].split("Question: ")[-1])
              for l in (V1 / "test.jsonl").open(encoding="utf-8")}
    exec_q = {fold(q[4]) for q in analyst_model_execution.questions()}
    acc_q = {fold(s[2]) for s in analyst_acceptance.SCENARIOS}
    return {"heldout_test": test_q, "execution_25": exec_q, "acceptance_14": acc_q}


def question_of(ex: dict) -> str:
    return ex["messages"][1]["content"].split("Question: ")[-1]


def contrastive(rng: np.random.Generator) -> list[dict]:
    out = []

    def add(df, q, plan, source, intent, ctx=""):
        out.append({"messages": v1.messages(df, q, ctx, json.dumps(plan)), "source": source, "intent": intent})

    for _ in range(DRAWS):
        # churn: column present -> random_forest; absent -> clarify
        df, meta = v1.domain_frame("churn", rng)
        q = R.choice(CHURN_Q)
        add(df, q, v1.plan_tools("prediction", ("random_forest", {"target": meta["cls_target"], "task": "classification"})),
            "v2:churn_present", "prediction")
        df, meta = v1.domain_frame(R.choice(NON_CHURN), rng)
        cands = v1.planner._target_candidates(df)
        if cands:
            add(df, q, {"type": "clarify", "question": "There is no churn column. Which column should I predict?",
                        "options": cands[:8]}, "v2:churn_absent", "clarify")
        # advertising: marketing data -> recommend; otherwise -> decline
        for questions, kind in ((WHERE_Q, "where"), (INCREASE_Q, "increase")):
            q = R.choice(questions)
            df, meta = v1.domain_frame(R.choice(["marketing", "marketing_alt"]), rng)
            add(df, q, v1.plan_tools("recommendation", ("recommend", {"kind": kind})), f"v2:ads_{kind}_present", "recommendation")
            df, meta = v1.domain_frame(R.choice(NON_MARKETING), rng)
            add(df, q, {"type": "decline", "reason": DECLINE_ADS}, f"v2:ads_{kind}_absent", "decline")
        # ROI / CAC: computed by business_performance, never a column
        df, meta = v1.domain_frame(R.choice(["marketing", "marketing_alt"]), rng)
        area = meta["area"]
        q = R.choice(ROI_Q).format(a=v1.nat(area))
        add(df, q, v1.plan_tools("performance", ("business_performance", {"group": area})), "v2:roi_present", "performance")
        df2, meta2 = v1.domain_frame(R.choice(NON_MARKETING), rng)
        add(df2, R.choice(ROI_Q).format(a=v1.nat(meta2["group"][0])), {"type": "decline", "reason": DECLINE_ADS},
            "v2:roi_absent", "decline")
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    v1_train = [json.loads(l) for l in (V1 / "train.jsonl").open(encoding="utf-8")]
    v1_valid = [json.loads(l) for l in (V1 / "valid.jsonl").open(encoding="utf-8")]
    frozen = frozen_questions()
    extra = []
    for _ in range(4):
        extra += contrastive(rng)
    # drop any generated example whose question equals a frozen evaluation question
    blocked = set().union(*frozen.values())
    removed = [question_of(e) for e in extra if fold(question_of(e)) in blocked]
    extra = [e for e in extra if fold(question_of(e)) not in blocked]
    R.shuffle(extra)
    n_valid = max(20, len(extra) // 16)
    new_valid, new_train = extra[:n_valid], extra[n_valid:]

    train = [{"messages": e["messages"]} for e in v1_train] + [{"messages": e["messages"]} for e in new_train] * REPEAT_NEW
    valid = [{"messages": e["messages"]} for e in v1_valid] + [{"messages": e["messages"]} for e in new_valid]
    R.shuffle(train)

    v2_new_questions = {fold(question_of(e)) for e in new_train + new_valid}
    v1_train_questions = {fold(question_of(e)) for e in v1_train}
    overlap = {"v2_new_examples": {k: sorted(v2_new_questions & v) for k, v in frozen.items()},
               "inherited_from_v1_train": {k: sorted(v1_train_questions & v) for k, v in frozen.items() if k != "heldout_test"}}
    if any(overlap["v2_new_examples"].values()):
        raise SystemExit(f"contamination: {overlap['v2_new_examples']}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in ("test.jsonl", "test_meta.jsonl"):
        shutil.copyfile(V1 / f, OUT / f)
    with (OUT / "new_examples_meta.jsonl").open("w", encoding="utf-8") as fh:
        for e in new_train + new_valid:
            fh.write(json.dumps({"source": e["source"], "intent": e["intent"], "question": question_of(e),
                                 "split": "valid" if e in new_valid else "train"}) + "\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "seed": SEED, "built_from": {"v1_dir": str(V1.relative_to(ROOT)), "v1_manifest": json.loads((V1 / "MANIFEST.json").read_text())["splits"]},
        "splits": {s: {"rows": sum(1 for _ in (OUT / f"{s}.jsonl").open()), "sha256": sha(OUT / f"{s}.jsonl")} for s in ("train", "valid", "test")},
        "test_identical_to_v1": sha(OUT / "test.jsonl") == sha(V1 / "test.jsonl"),
        "new_examples": {"train": len(new_train), "train_repeats": REPEAT_NEW, "valid": len(new_valid),
                         "by_source": dict(Counter(e["source"] for e in new_train + new_valid))},
        "generated_then_removed_as_frozen_questions": sorted(set(removed)),
        "overlap_with_frozen_questions": overlap,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("splits", "test_identical_to_v1", "new_examples")}, indent=1))
    print("inherited v1 overlap:", json.dumps(overlap["inherited_from_v1_train"]))
    print("removed:", manifest["generated_then_removed_as_frozen_questions"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
