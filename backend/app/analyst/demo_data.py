"""Invented demonstration datasets (seeded, reproducible). They contain no real
people or companies; they exist to exercise the analyst end to end.

    python -m app.analyst.demo_data <output dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def employees(seed: int = 1, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Engineering", "Sales", "Marketing", "Finance", "Support"], n, p=[.3, .25, .15, .15, .15])
    base = {"Engineering": 92000, "Sales": 70000, "Marketing": 66000, "Finance": 78000, "Support": 52000}
    years = rng.integers(0, 25, n)
    level = rng.choice(["Junior", "Mid", "Senior"], n, p=[.35, .4, .25])
    bump = {"Junior": 0, "Mid": 9000, "Senior": 22000}
    salary = np.array([base[d] + bump[l] + 1300 * y for d, l, y in zip(dept, level, years)]) + rng.normal(0, 6000, n)
    return pd.DataFrame({"employee_id": np.arange(1, n + 1), "department": dept, "level": level,
                         "years_experience": years, "age": np.clip(22 + years + rng.integers(0, 12, n), 21, 66),
                         "remote": rng.choice(["Yes", "No"], n), "salary": salary.round(0)})


def customers(seed: int = 2, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    contract = rng.choice(["Monthly", "Annual", "Two-year"], n, p=[.5, .3, .2])
    support = rng.poisson(1.5, n)
    charges = rng.normal(70, 20, n).clip(15, 150).round(2)
    logit = (-0.6 + 1.6 * (contract == "Monthly") - 0.9 * (contract == "Two-year") - 0.035 * tenure
             + 0.35 * support + 0.012 * (charges - 70))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "tenure_months": tenure, "contract": contract,
                         "monthly_charges": charges, "support_tickets": support,
                         "region": rng.choice(["North", "South", "East", "West"], n),
                         "churn": np.where(churn, "Yes", "No")})


def campaigns(seed: int = 3) -> pd.DataFrame:
    """Monthly campaigns per area. Area A converts best and returns the most;
    Area D loses money; Area E has only two campaigns."""
    rng = np.random.default_rng(seed)
    profile = {"Area A": (0.065, 165, 14), "Area B": (0.045, 150, 14), "Area C": (0.040, 140, 14),
               "Area D": (0.030, 95, 14), "Area E": (0.050, 150, 2)}
    rows = []
    for area, (rate, value, k) in profile.items():
        for m in range(k):
            spend = float(rng.uniform(3000, 9000))
            leads = int(spend / rng.uniform(4.5, 6.5))
            conv = int(rng.binomial(leads, rate))
            revenue = conv * value * float(rng.uniform(0.85, 1.15))
            rows.append({"month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m % 12 + 12 * (m // 12)),
                         "area": area, "channel": rng.choice(["Search", "Social", "Display"]),
                         "ad_spend": round(spend, 2), "leads": leads, "conversions": conv,
                         "revenue": round(revenue, 2), "repeat_customers": int(conv * rng.uniform(0.1, 0.4))})
    return pd.DataFrame(rows)


def no_marketing(seed: int = 4) -> pd.DataFrame:
    """Inventory data: nothing about advertising."""
    rng = np.random.default_rng(seed)
    n = 80
    return pd.DataFrame({"sku": [f"S{i:04d}" for i in range(n)], "category": rng.choice(["Tools", "Garden", "Kitchen"], n),
                         "stock": rng.integers(0, 500, n), "unit_price": rng.uniform(2, 90, n).round(2)})


def tiny(seed: int = 5) -> pd.DataFrame:
    """Too small for a responsible model."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"store": [f"Store {i}" for i in range(12)], "size_sqm": rng.integers(80, 400, 12),
                         "churn": rng.choice(["Yes", "No"], 12), "sales": rng.uniform(1e4, 9e4, 12).round(0)})


ALL = {"employees.csv": employees, "customers.csv": customers, "campaigns.csv": campaigns,
       "inventory.csv": no_marketing, "stores_small.csv": tiny}


def write_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in ALL.items():
        p = out / name
        fn().to_csv(p, index=False)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_all(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data")):
        print(p)
