"""Random Forest classification and regression with the checks that decide
whether a model should be trained at all.

Order: suitability (target, rows, classes) -> feature screening (identifiers,
dates, free text, leakage) -> held-out split -> fit -> metrics against a naive
baseline -> permutation importance on held-out rows. A model is refused, with
the reason, when the target or the data cannot support it.

Feature importance says which inputs the model relied on for its predictions.
It is not evidence that an input causes the outcome; every result says so.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.analyst.data import kind_of

MIN_ROWS = 40
MIN_PER_CLASS = 8
MAX_CLASSES = 20
MAX_TEXT_CARDINALITY = 50
LEAKAGE_CORR = 0.98
TEST_SIZE = 0.25
SEED = 42
TEST_SIZES = (0.2, 0.25, 0.3)        # the user may choose 80/20, 75/25 or 70/30
IMBALANCE_RATIO = 3.0                # majority / minority class count above which imbalance is reported


class Unsuitable(ValueError):
    """The data cannot responsibly support this model; the message says why."""


def infer_task(target: pd.Series) -> str:
    if kind_of(target) == "numeric" and target.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def _is_identifier(name: str, s: pd.Series) -> bool:
    n = s.notna().sum()
    if n == 0:
        return False
    if re.search(r"(?:^|_)(?:id|uuid|key|code)$|^id", name.lower()) and s.nunique() > 0.9 * n:
        return True
    return kind_of(s) == "categorical" and s.nunique() == n and n > 20


def screen_features(df: pd.DataFrame, target: str, task: str, features: list[str] | None = None) -> tuple[list[str], list[dict]]:
    """Features the model may use, and every column left out with the reason."""
    dropped: list[dict] = []
    candidates = [c for c in (features or df.columns) if c != target]
    keep = []
    y = df[target]
    tname = re.sub(r"[^a-z0-9]", "", target.lower())
    for c in candidates:
        s = df[c]
        kind = kind_of(s)
        cname = re.sub(r"[^a-z0-9]", "", c.lower())
        if s.notna().sum() == 0:
            dropped.append({"column": c, "reason": "empty column"})
        elif _is_identifier(c, s):
            dropped.append({"column": c, "reason": "identifier: unique per row, it cannot generalise"})
        elif kind == "datetime":
            dropped.append({"column": c, "reason": "date column: left out to avoid learning the time of collection"})
        elif kind == "categorical" and s.nunique() > MAX_TEXT_CARDINALITY:
            dropped.append({"column": c, "reason": f"free text or too many categories ({s.nunique()})"})
        elif len(tname) >= 4 and tname in cname:
            dropped.append({"column": c, "reason": f"possible leakage: its name contains the target '{target}'"})
        elif kind == "numeric" and task == "regression" and s.std() > 0 and \
                abs(pd.concat([s, y], axis=1).dropna().corr().iloc[0, 1]) > LEAKAGE_CORR:
            dropped.append({"column": c, "reason": "possible leakage: almost perfectly correlated with the target"})
        elif task == "classification" and s.nunique() >= 2 and \
                pd.concat([s, y], axis=1).dropna().groupby(c)[target].nunique().max() == 1 and s.nunique() < len(s) * 0.5 \
                and pd.concat([s, y], axis=1).dropna()[target].nunique() > 1:
            dropped.append({"column": c, "reason": "possible leakage: each of its values determines the target exactly"})
        else:
            keep.append(c)
    return keep, dropped


def check_target(df: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    if target not in df.columns:
        raise Unsuitable(f"There is no column named '{target}'.")
    data = df[df[target].notna()]
    if len(data) < MIN_ROWS:
        raise Unsuitable(f"Only {len(data)} rows have a value for '{target}'; at least {MIN_ROWS} are needed to train "
                         "and test a model honestly.")
    if task == "classification":
        counts = data[target].astype(str).value_counts()
        if len(counts) < 2:
            raise Unsuitable(f"'{target}' has a single value, so there is nothing to predict.")
        if len(counts) > MAX_CLASSES:
            raise Unsuitable(f"'{target}' has {len(counts)} categories; that is too many for a reliable classifier "
                             "on this data.")
        if counts.min() < MIN_PER_CLASS:
            raise Unsuitable(f"The class '{counts.idxmin()}' of '{target}' has only {int(counts.min())} rows; each class "
                             f"needs at least {MIN_PER_CLASS} to be learned and tested.")
    else:
        if kind_of(data[target]) != "numeric":
            raise Unsuitable(f"'{target}' is not numeric, so regression does not apply.")
        if data[target].std() == 0:
            raise Unsuitable(f"'{target}' is constant, so there is nothing to predict.")
    return data


def train(df: pd.DataFrame, target: str, task: str | None = None, features: list[str] | None = None,
          test_size: float = TEST_SIZE, seed: int = SEED) -> dict:
    """Train and evaluate. Returns the fitted pipeline under "_model" (kept by
    the caller for predictions) and a JSON-safe report. Preprocessing is part
    of the pipeline, so it is fitted on the training rows only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 mean_absolute_error, mean_squared_error, precision_recall_fscore_support,
                                 r2_score, roc_auc_score, root_mean_squared_error)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    if test_size not in TEST_SIZES:
        raise Unsuitable(f"The test share must be one of {', '.join(f'{int(t * 100)}%' for t in TEST_SIZES)}.")
    seed = int(seed)
    if target not in df.columns:
        raise Unsuitable(f"There is no column named '{target}'.")
    task = task or infer_task(df[target])
    data = check_target(df, target, task)
    feats, dropped = screen_features(data, target, task, features)
    if not feats:
        raise Unsuitable("No usable input columns remain after removing identifiers, dates, free text and "
                         "possible leakage: " + "; ".join(f"{d['column']} ({d['reason']})" for d in dropped))
    numeric = [c for c in feats if kind_of(data[c]) == "numeric"]
    categorical = [c for c in feats if c not in numeric]
    X = data[feats].copy()
    for c in categorical:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan).astype(str).replace("nan", np.nan)
    y = data[target].astype(str) if task == "classification" else data[target].astype(float)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if task == "classification":
        forest = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1,
                                        min_samples_leaf=2)
    else:
        forest = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    model = Pipeline([("prepare", pre), ("forest", forest)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if task == "classification" else None)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if task == "classification":
        majority = y_train.value_counts().idxmax()
        metrics = {"accuracy": float(accuracy_score(y_test, pred)),
                   "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                   "macro_f1": float(f1_score(y_test, pred, average="macro")),
                   "baseline_accuracy": float((y_test == majority).mean()),
                   # a constant prediction recalls one class fully and the others not at all
                   "baseline_balanced_accuracy": 1.0 / y_train.nunique(),
                   "baseline": f"always predicting the most common class '{majority}'"}
        classes = list(model.classes_)
        prec, rec, f1s, sup = precision_recall_fscore_support(y_test, pred, labels=classes, zero_division=0)
        metrics["per_class"] = [{"class": str(c), "precision": float(a), "recall": float(b), "f1": float(d),
                                 "support": int(n)} for c, a, b, d, n in zip(classes, prec, rec, f1s, sup)]
        metrics["macro_precision"] = float(np.mean(prec))
        metrics["macro_recall"] = float(np.mean(rec))
        metrics["confusion_matrix"] = {"labels": [str(c) for c in classes],
                                       "matrix": confusion_matrix(y_test, pred, labels=classes).tolist()}
        counts = y.value_counts()
        balance = {"counts": {str(k): int(v) for k, v in counts.items()},
                   "ratio": float(counts.max() / counts.min())}
        balance["imbalanced"] = balance["ratio"] >= IMBALANCE_RATIO
        if len(classes) == 2:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score((y_test == classes[1]).astype(int), proba))
        scoring = "balanced_accuracy"
        primary = ("balanced_accuracy", metrics["balanced_accuracy"], "baseline_accuracy")
        beats = metrics["balanced_accuracy"] > 0.5 + 0.05 if len(classes) == 2 else metrics["accuracy"] > metrics["baseline_accuracy"]
    else:
        baseline_mae = float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))
        metrics = {"r2": float(r2_score(y_test, pred)), "mae": float(mean_absolute_error(y_test, pred)),
                   "rmse": float(root_mean_squared_error(y_test, pred)), "mse": float(mean_squared_error(y_test, pred)),
                   "baseline_mae": baseline_mae,
                   "baseline": "always predicting the training mean"}
        scoring = "neg_mean_absolute_error"
        primary = ("mae", metrics["mae"], "baseline_mae")
        beats = metrics["mae"] < baseline_mae * 0.95
        classes = None
        balance = None

    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, scoring=scoring)
    importance = sorted(({"feature": f, "importance": float(m), "sd": float(s)}
                         for f, m, s in zip(feats, imp.importances_mean, imp.importances_std)),
                        key=lambda d: -d["importance"])
    limitations = [
        "Feature importance shows what the model relied on to predict, not what causes the outcome.",
        f"Evaluated on {len(y_test)} held-out rows ({int(round(test_size * 100))}% of the data); results on new data may differ.",
    ]
    if balance and balance["imbalanced"]:
        limitations.append(f"The classes are imbalanced (largest/smallest = {balance['ratio']:.1f}): accuracy can look "
                           "good by favouring the common class, so read balanced accuracy, recall and F1 per class. "
                           "Classes were weighted during training to compensate.")
    if not beats:
        limitations.insert(0, "The model does not clearly beat the naive baseline, so its predictions add little.")
    if task == "regression" and metrics["r2"] >= 0.995:
        limitations.insert(0, f"Near-perfect fit (R² {metrics['r2']:.3f}): '{target}' is probably (almost) a formula of "
                              "the inputs (for example units × price). That is a calculation, not a prediction problem; "
                              "consider leaving those inputs out to learn something you could not compute directly.")
    if len(data) < 200:
        limitations.append(f"Small dataset ({len(data)} rows): metrics and importances are uncertain.")
    return {
        "_model": model,
        "task": task, "target": target, "features": feats, "dropped": dropped,
        "rows_used": int(len(data)), "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
        "classes": classes, "metrics": metrics, "beats_baseline": bool(beats), "class_balance": balance,
        "split": {"test_size": test_size, "train_share": round(1 - test_size, 2), "seed": seed,
                  "stratified": task == "classification"},
        "importance": importance,
        # held-out rows, for an actual-versus-predicted chart (regression) or a check of the errors
        "test_predictions": ([{"actual": float(a), "predicted": float(b)} for a, b in zip(y_test, pred)][:500]
                             if task == "regression" else None),
        "method": (f"Random Forest {'classifier' if task == 'classification' else 'regressor'}, 300 trees, "
                   f"{int(round((1 - test_size) * 100))}/{int(round(test_size * 100))} split (seed {seed})"
                   f"{' stratified by class' if task == 'classification' else ''}, median/most-frequent imputation, "
                   "one-hot encoding of categories; importance by permutation on held-out rows"),
        "limitations": limitations,
    }


def predict(trained: dict, rows: list[dict]) -> list[dict]:
    """Predictions for new rows. Inputs the row does not give are imputed and
    listed, so the user sees what the prediction assumed."""
    model, feats = trained["_model"], trained["features"]
    out = []
    for row in rows:
        unknown = [k for k in row if k not in feats]
        missing = [f for f in feats if f not in row or row[f] in (None, "")]
        if len(missing) > len(feats) / 2:
            raise Unsuitable(f"Too few inputs: give at least half of {', '.join(feats)}.")
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
        item = {"input": {f: row.get(f) for f in feats if f in row}, "imputed": missing, "ignored": unknown}
        if trained["task"] == "classification":
            proba = model.predict_proba(X)[0]
            item["prediction"] = str(model.predict(X)[0])
            item["probabilities"] = {str(c): float(p) for c, p in zip(model.classes_, proba)}
        else:
            item["prediction"] = float(model.predict(X)[0])
        out.append(item)
    return out


def public(report: dict) -> dict:
    return {k: v for k, v in report.items() if not k.startswith("_")}
