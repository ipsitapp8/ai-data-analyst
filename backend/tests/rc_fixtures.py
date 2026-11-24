"""Datasets for the root-cause tests, each built with a known, deliberately controlled structure.

Every scenario has two periods of about 90 days (A: Jan-Mar, B: Apr-Jun 2025) and a stated ground truth.
The tests recompute that truth from the raw frame with plain pandas, independent of the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

A = ["2025-01-01", "2025-04-01"]
B = ["2025-04-01", "2025-06-30"]
SPEC_PERIODS = {"period_a": A, "period_b": B}


def _dates(rng, n):
    days = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    return pd.Series(rng.choice(days, n)).sort_values().reset_index(drop=True)


def in_b(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["order_date"]) >= pd.Timestamp(B[0])


def sales_decline_segment(seed=1, n=6000) -> pd.DataFrame:
    """Revenue falls because region North collapsed (-45%) in period B. Nothing else moves."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "region": rng.choice(["North", "South", "East", "West"], n),
                       "channel": rng.choice(["Web", "Store", "Partner"], n), "product": rng.choice(["Alpha", "Beta", "Gamma"], n)})
    df["revenue"] = rng.gamma(9, 12, n).round(2)
    drop = in_b(df) & (df["region"] == "North")
    keep = ~(drop & (rng.random(n) < 0.45))          # 45% of North's later orders never happened
    return df[keep].reset_index(drop=True)


def cost_competing_drivers(seed=2, n=7000) -> pd.DataFrame:
    """Cost rises for two independent reasons: vendor Acme's unit cost +40%, and Freight volume doubling."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "vendor": rng.choice(["Acme", "Birch", "Cobalt", "Delta"], n),
                       "category": rng.choice(["Freight", "Parts", "Labour", "Software"], n),
                       "site": rng.choice(["Leeds", "Pune", "Oslo"], n)})
    df["cost"] = rng.gamma(8, 10, n).round(2)
    b = in_b(df)
    df.loc[b & (df["vendor"] == "Acme"), "cost"] *= 1.4
    extra = df[b & (df["category"] == "Freight")].sample(frac=1.0, random_state=seed).copy()   # Freight volume doubles
    extra["cost"] = rng.gamma(8, 10, len(extra)).round(2)             # new orders, not copies
    return pd.concat([df, extra], ignore_index=True).sort_values("order_date").reset_index(drop=True)


def productivity_decline(seed=3, n=6000) -> pd.DataFrame:
    """Output per hour falls on the night shift in B, where overtime also rose; day shifts are flat."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "shift": rng.choice(["Day", "Evening", "Night"], n),
                       "team": rng.choice(["T1", "T2", "T3", "T4"], n)})
    df["hours"] = rng.uniform(6, 9, n).round(2)
    df["overtime_hours"] = rng.gamma(2, 0.5, n).round(2)
    night_b = in_b(df) & (df["shift"] == "Night")
    df.loc[night_b, "overtime_hours"] += rng.uniform(2, 4, night_b.sum())
    rate = rng.normal(11, 1.2, n)
    rate[night_b.to_numpy()] *= 0.62
    df["output"] = (rate * df["hours"]).round(2)
    return df


def conversion_decline(seed=4, n=9000) -> pd.DataFrame:
    """Conversion falls from 12% to ~8% overall because Mobile fell from 12% to 3%; Mobile load time also rose."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "device": rng.choice(["Mobile", "Desktop", "Tablet"], n, p=[.45, .4, .15]),
                       "source": rng.choice(["Organic", "Paid", "Email"], n)})
    df["page_load_ms"] = rng.normal(1200, 150, n).round(0)
    mob_b = in_b(df) & (df["device"] == "Mobile")
    df.loc[mob_b, "page_load_ms"] += 1500
    p = np.full(n, 0.12)
    p[mob_b.to_numpy()] = 0.03
    df["converted"] = (rng.random(n) < p).astype(int)
    return df


def confounded_proxy(seed=5, n=12000) -> pd.DataFrame:
    """West looks like the culprit, but the real driver is the Paid channel (-40% everywhere).
    West is 70% Paid; elsewhere 20% Paid. Within each channel, West and the rest move identically."""
    rng = np.random.default_rng(seed)
    region = rng.choice(["West", "East", "North", "South"], n, p=[.3, .25, .25, .2])
    paid = np.where(region == "West", rng.random(n) < 0.7, rng.random(n) < 0.2)
    df = pd.DataFrame({"order_date": _dates(rng, n), "region": region, "channel": np.where(paid, "Paid", "Organic"),
                       "segment": rng.choice(["SMB", "Enterprise"], n)})
    df["revenue"] = rng.gamma(9, 12, n).round(2)
    drop = in_b(df) & (df["channel"] == "Paid")
    keep = ~(drop & (rng.random(n) < 0.40))
    return df[keep].reset_index(drop=True)


def wrong_factor(seed=6, n=6000) -> pd.DataFrame:
    """Revenue falls because of Region North; `discount` drifts upward everywhere but has nothing to do with revenue."""
    df = sales_decline_segment(seed, n)
    rng = np.random.default_rng(seed + 100)
    df["discount"] = rng.normal(10, 3, len(df)).round(2) + in_b(df).astype(int) * 4.0
    return df


def noisy_small(seed=7, n=70) -> pd.DataFrame:
    """Too little data and no structure: a small, noisy change."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "region": rng.choice(["N", "S", "E"], n), "channel": rng.choice(["A", "B"], n)})
    df["revenue"] = rng.gamma(4, 20, n).round(2)
    return df


def diffuse_decline(seed=8, n=8000) -> pd.DataFrame:
    """Every region and channel loses about 12% of its orders: no single factor."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "region": rng.choice(["N", "S", "E", "W"], n),
                       "channel": rng.choice(["Web", "Store", "Partner"], n)})
    df["revenue"] = rng.gamma(9, 12, n).round(2)
    keep = ~(in_b(df) & (rng.random(n) < 0.12))
    return df[keep].reset_index(drop=True)


def missing_inputs(seed=9, n=6000) -> pd.DataFrame:
    """The 'decline' is an artefact: revenue stopped being recorded for half of Partner rows in B."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "region": rng.choice(["N", "S", "E", "W"], n),
                       "channel": rng.choice(["Web", "Store", "Partner"], n)})
    df["revenue"] = rng.gamma(9, 12, n).round(2)
    hit = in_b(df) & (df["channel"] == "Partner") & (rng.random(n) < 0.5)
    df.loc[hit, "revenue"] = np.nan
    return df


def outlier_artefact(seed=10, n=5000) -> pd.DataFrame:
    """A handful of enormous orders in period A make period B look like a decline; typical orders did not change."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "region": rng.choice(["N", "S", "E", "W"], n),
                       "channel": rng.choice(["Web", "Store"], n)})
    df["revenue"] = rng.gamma(9, 12, n).round(2)
    a_idx = df.index[~in_b(df)][:6]
    df.loc[a_idx, "revenue"] = 60000.0
    return df


def new_vendor_slows_resolution(seed=12, n=5000) -> pd.DataFrame:
    """Resolution hours rise because a vendor that did not exist in period A ('Brightline') takes 35% of period-B tickets and is slow."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": _dates(rng, n), "channel": rng.choice(["email", "chat", "phone"], n),
                       "severity": rng.integers(1, 5, n)})
    df["vendor"] = np.where(in_b(df) & (rng.random(n) < 0.35), "Brightline", rng.choice(["InHouse", "Teleserve", "NorthDesk"], n))
    df["hours"] = 6 + df["severity"] * 1.5 + rng.normal(0, 2, n) + np.where(df["vendor"] == "Brightline", 9, 0)
    return df
