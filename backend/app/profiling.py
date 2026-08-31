"""Profile a freshly uploaded CSV: column types, row count, missing values, basic stats."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        missing_pct = round(missing / max(len(df), 1) * 100, 2)

        col_info: dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "missing_count": missing,
            "missing_pct": missing_pct,
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            col_info["kind"] = "numeric"
            if len(clean):
                col_info["stats"] = {
                    "min": _safe_float(clean.min()),
                    "max": _safe_float(clean.max()),
                    "mean": _safe_float(clean.mean()),
                    "median": _safe_float(clean.median()),
                    "std": _safe_float(clean.std()),
                }
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["kind"] = "datetime"
            clean = series.dropna()
            if len(clean):
                col_info["stats"] = {"min": str(clean.min()), "max": str(clean.max())}
        else:
            col_info["kind"] = "categorical"
            top = series.value_counts(dropna=True).head(5)
            col_info["top_values"] = [
                {"value": str(k), "count": int(v)} for k, v in top.items()
            ]

        columns.append(col_info)

    return {
        "row_count": int(len(df)),
        "col_count": int(len(df.columns)),
        "columns": columns,
    }


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load_and_profile_csv(path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    # Best-effort datetime inference on text columns that look like dates.
    # (pandas >= 2.x may report these as dtype "object" or a dedicated "str" dtype
    # depending on the `future.infer_string` setting, so check both.)
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(20)
            if len(sample) and _looks_like_date(sample):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
    return df, profile_dataframe(df)


def _looks_like_date(sample: pd.Series) -> bool:
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.8
