"""Regressions for two defects found while preparing the multi-domain training data.

A. A Random Forest classifier for a target stored as numbers (0/1, 1/2) crashed with
   "The classes, [0, 1], are not in class_weight" (scikit-learn 1.9 rejects
   class_weight="balanced" when a label looks like a number).
B. Ingestion read a column of diagnosis codes (250.83, V57, E909) as numbers because
   95% of the values parsed, silently blanking every V/E code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import ingest
from app.analyst import ml, tools
from app.analyst.data import load_table


def _frame(labels, n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"signal": x, "noise": rng.normal(size=n), "kind": rng.choice(["a", "b", "c"], n),
                         "target": np.where(x > 0.8, labels[1], labels[0])})


@pytest.mark.parametrize("labels", [(0, 1), (1, 2), ("0", "1"), ("no", "yes")])
def test_random_forest_classifies_numeric_and_text_labels(labels):
    df = _frame(labels)
    call = tools.validate_call({"tool": "random_forest", "args": {"target": "target", "task": "classification"}}, df)
    memory: dict = {}
    report = tools.run(call, df, memory)
    assert report["task"] == "classification"
    assert report["metrics"]["balanced_accuracy"] > report["metrics"]["baseline_balanced_accuracy"]
    assert {c["class"] for c in report["metrics"]["per_class"]} == {str(v) for v in labels}
    # the trained model is kept and predicts one of the original labels
    out = ml.predict(memory["model"], [{"signal": 2.0, "noise": 0.0, "kind": "a"}])
    assert str(out[0]["prediction"]) in {str(v) for v in labels}


def test_balanced_weights_match_class_weight_balanced():
    """The fix weights rows exactly as class_weight="balanced" did, so text-label models are unchanged."""
    from sklearn.ensemble import RandomForestClassifier
    df = _frame(("no", "yes"))
    X, y = df[["signal", "noise"]], df["target"]
    old = RandomForestClassifier(n_estimators=50, random_state=0, class_weight="balanced").fit(X, y)
    from sklearn.pipeline import Pipeline
    new = ml._fit(Pipeline([("forest", RandomForestClassifier(n_estimators=50, random_state=0))]), X, y, "classification")
    assert (old.predict_proba(X) == new.predict_proba(X)).all()


def _write(tmp_path, text):
    p = tmp_path / "codes.csv"
    p.write_text(text)
    return p


def test_diagnosis_codes_with_letters_survive_ingestion(tmp_path):
    codes = ["250.83", "401", "276", "428"] * 30 + ["V57", "E909"]      # 98% numeric-looking
    rows = "\n".join(f"{i},{c},{i % 7}" for i, c in enumerate(codes))
    df = load_table(str(_write(tmp_path, "encounter,diag_1,stay\n" + rows + "\n")))
    assert df["diag_1"].dtype == object
    assert {"V57", "E909", "250.83"} <= set(df["diag_1"].astype(str))
    assert df["diag_1"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(df["stay"])


def test_missing_markers_still_read_as_numbers(tmp_path):
    """Markers without digits ("?", "-") are missing values, as before: the column is still numeric."""
    vals = [str(v) for v in range(1, 60)] + ["?", "-"]
    df, _ = ingest.read_table(str(_write(tmp_path, "amount\n" + "\n".join(vals) + "\n")))
    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert df["amount"].isna().sum() == 2
