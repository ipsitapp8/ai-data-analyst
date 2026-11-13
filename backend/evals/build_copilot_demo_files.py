"""Synthetic demo files for the Silt Copilot rehearsal. Every value is invented
(seeded random numbers); nothing comes from real people or companies.

    ../.venv313/bin/python evals/build_copilot_demo_files.py [OUTPUT_DIR]   (default ~/silt-demo/upload)

Writes, refusing to overwrite:
  sales.csv            regional store sales by month (with a few missing values, outliers and duplicates)
  hr_attrition.xlsx    two sheets: "Employees" (attrition target, imbalanced) and "Read me"
  sales_report.pdf     3 text pages and 1 scanned (image-only) page
  broken.csv           not a table: for invalid-file handling
  not_really.pdf       text renamed to .pdf: for invalid-PDF handling
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sales(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = {"North": (1.00, 9000), "South": (0.92, 8000), "East": (1.05, 9500), "West": (1.22, 12000)}
    rows = []
    for region, (lift, mkt) in regions.items():
        for store in range(1, 6):
            store_id = f"{region[0]}{store:02d}"
            for m in pd.date_range("2025-01-01", periods=12, freq="MS"):
                season = 1 + 0.12 * np.sin((m.month - 3) / 12 * 2 * np.pi)
                spend = mkt * rng.uniform(0.8, 1.2)
                leads = int(spend / rng.uniform(9, 12))
                conv_rate = 0.03 * lift * rng.uniform(0.85, 1.15)
                conversions = int(leads * conv_rate)
                units = int(rng.normal(900, 90) * lift * season)
                price = round(rng.normal(42, 2) * (1.06 if region == "West" else 1.0), 2)
                rows.append({"store_id": store_id, "region": region, "month": m.strftime("%Y-%m-%d"),
                             "product_line": rng.choice(["Home", "Outdoor", "Kitchen"]),
                             "units_sold": units, "avg_price": price, "sales": round(units * price, 2),
                             "marketing_spend": round(spend, 2), "leads": leads, "conversions": conversions,
                             "discount_pct": round(float(rng.choice([0, 5, 10, 15], p=[.4, .3, .2, .1])), 1),
                             "staff": int(rng.integers(6, 14))})
    df = pd.DataFrame(rows)
    df.loc[rng.choice(len(df), 6, replace=False), "discount_pct"] = np.nan
    df.loc[rng.choice(len(df), 3, replace=False), "staff"] = np.nan
    df.loc[[17, 101], "sales"] = df.loc[[17, 101], "sales"] * 3.2          # two extreme values to find
    return pd.concat([df, df.iloc[[5, 60, 200]]], ignore_index=True)       # three duplicated rows


def attrition(seed: int = 12, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Sales", "Engineering", "Support", "Finance", "Operations"], n, p=[.25, .3, .2, .1, .15])
    level = rng.choice(["Junior", "Mid", "Senior"], n, p=[.45, .35, .2])
    age = rng.integers(21, 61, n)
    tenure = np.clip((age - 21) * rng.uniform(0.1, 0.5, n), 0, 35).round(1)
    base = {"Junior": 3200, "Mid": 4800, "Senior": 7200}
    income = np.array([base[lv] for lv in level]) * rng.uniform(0.85, 1.25, n)
    overtime = rng.choice(["Yes", "No"], n, p=[.3, .7])
    satisfaction = rng.integers(1, 6, n)
    raise_pct = rng.choice([0, 2, 3, 5, 8], n, p=[.15, .25, .3, .2, .1]).astype(float)
    logit = (-2.3 + 2.0 * (overtime == "Yes") - 0.9 * (satisfaction - 3) - 0.12 * tenure - 0.0004 * (income - 4500)
             - 0.2 * raise_pct + 0.8 * (dept == "Support"))
    left = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"employee_id": [f"E{1000 + i}" for i in range(n)], "department": dept, "job_level": level,
                         "age": age, "tenure_years": tenure, "monthly_income": income.round(0), "overtime": overtime,
                         "satisfaction": satisfaction, "last_raise_pct": raise_pct,
                         "attrition": np.where(left, "Yes", "No")})


REPORT = [
    ("Northwind Home Goods - Quarterly Sales Review (synthetic example)",
     ["Prepared for the regional leadership meeting. All figures in this document are invented for a software demo.",
      "",
      "Executive summary",
      "Total sales for the year reached 9.4 million, up 6.1% on the previous year.",
      "The West region led with 3.1 million in sales and the highest conversion rate at 3.7%.",
      "The South region fell short of its target by 8% and is under review.",
      "Marketing spend rose to 460 thousand; return on marketing spend was strongest in the West."]),
    ("Regional performance",
     ["North: sales 2.2 million, 20 stores, steady growth through the summer.",
      "South: sales 1.9 million; two stores closed for refurbishment in March and April.",
      "East: sales 2.2 million, strongest Kitchen product line sales.",
      "West: sales 3.1 million; premium pricing about 6% above other regions.",
      "",
      "Risks",
      "Staff turnover in Support teams increased to 21% and may affect service levels.",
      "Supplier lead times for the Outdoor line lengthened from 3 to 5 weeks."]),
    ("Plan for next quarter",
     ["Pilot a 10% marketing increase in the South region with a matched holdout of stores.",
      "Hire two additional staff for each West store before the holiday season.",
      "Review discount policy: discounts above 10% did not raise units sold enough to cover the margin.",
      "Next review date: 15 January."]),
]


def report_pdf(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType: text stays extractable
    with PdfPages(path) as pdf:
        for title, lines in REPORT:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.93, title, fontsize=15, weight="bold")
            for i, line in enumerate(lines):
                fig.text(0.08, 0.88 - i * 0.035, line, fontsize=10.5)
            pdf.savefig(fig)
            plt.close(fig)
        # A "scanned" page: the text exists only as pixels of an image.
        img_fig = plt.figure(figsize=(6, 2), dpi=100)
        img_fig.text(0.05, 0.6, "Appendix: signed approval (scanned copy)", fontsize=14)
        img_fig.text(0.05, 0.3, "Approved budget: 512,000", fontsize=14)
        img_fig.canvas.draw()
        pixels = np.asarray(img_fig.canvas.buffer_rgba())
        plt.close(img_fig)
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.05, 0.55, 0.9, 0.35])
        ax.imshow(pixels)
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    out = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path("~/silt-demo/upload").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    targets = ["sales.csv", "hr_attrition.xlsx", "sales_report.pdf", "broken.csv", "not_really.pdf"]
    existing = [t for t in targets if (out / t).exists()]
    if existing:
        print("refusing to overwrite:", ", ".join(existing))
        return 1
    sales().to_csv(out / "sales.csv", index=False)
    with pd.ExcelWriter(out / "hr_attrition.xlsx") as xw:
        pd.DataFrame({"About this workbook": [
            "Synthetic HR data for a software demo; no real employees.",
            "Sheet 'Employees' holds one row per employee; 'attrition' says whether they left."]}).to_excel(
            xw, sheet_name="Read me", index=False)
        attrition().to_excel(xw, sheet_name="Employees", index=False)
    report_pdf(out / "sales_report.pdf")
    (out / "broken.csv").write_bytes(b"\x00\x01\x02PK\x03\x04 this is not a table \xff\xfe")
    (out / "not_really.pdf").write_text("This is plain text with a .pdf name.")
    for t in targets:
        print(f"{t:20s} {(out / t).stat().st_size:>8,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
