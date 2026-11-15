"""Generalization: datasets Silt was never built or trained on, in formats it must detect, asked in plain language.

Data comes from public datasets bundled with installed libraries (plotly: gapminder, tips, stocks, election,
carshare; scikit-learn: diabetes, breast cancer, wine), written here in different encodings and delimiters. Every
expected value is computed in this file with pandas from the original data -- never by Silt. Nothing in production
code names these datasets or their columns. The model is offline throughout (Render runs without it).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app import ingest
from app.analyst import data, orchestrator, planner

plotly = pytest.importorskip("plotly")
sk = pytest.importorskip("sklearn.datasets")
P = os.path.join(os.path.dirname(plotly.__file__), "package_data", "datasets")


def pl(name):
    return pd.read_csv(os.path.join(P, name + ".csv.gz"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("unseen")
    out = {}
    g = pl("gapminder"); g.to_csv(d / "gapminder.csv", index=False); out["gapminder"] = (d / "gapminder.csv", g)
    t = pl("tips"); t.to_csv(d / "tips.csv", sep=";", decimal=",", index=False); out["tips"] = (d / "tips.csv", t)
    s = pl("stocks"); s.to_csv(d / "stocks.csv", sep="\t", index=False); out["stocks"] = (d / "stocks.csv", s)
    e = pl("election"); (d / "election.csv").write_text(e.to_csv(index=False), encoding="utf-8-sig")
    out["election"] = (d / "election.csv", e)
    c = pl("carshare"); c["area"] = "Zone-" + (c["peak_hour"] % 3).astype(str) + "-é"
    (d / "carshare.csv").write_text(c.to_csv(index=False), encoding="cp1252"); out["carshare"] = (d / "carshare.csv", c)
    db = sk.load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": "progression"})
    db.to_csv(d / "diabetes.csv", index=False); out["diabetes"] = (d / "diabetes.csv", db)
    bc = sk.load_breast_cancer(as_frame=True).frame
    bc["diagnosis"] = np.where(bc.pop("target") == 1, "benign", "malignant")
    with pd.ExcelWriter(d / "clinic.xlsx") as xw:
        pd.DataFrame([["Measurements"], [""]]).to_excel(xw, sheet_name="data", index=False, header=False)
        bc.to_excel(xw, sheet_name="data", index=False, startrow=3)
    out["clinic"] = (d / "clinic.xlsx", bc)
    w = sk.load_wine(as_frame=True).frame
    w.to_csv(d / "wine.csv", index=False, header=False); out["wine"] = (d / "wine.csv", w)
    return out


def load(files, k):
    return data.load_table(str(files[k][0]))


def ask(q, df, mem=None):
    return orchestrator.ask(q, df, mem if mem is not None else {}, use_model=False, filename="data.csv")


def res(out, tool="calculator"):
    return next(c["result"] for c in out["calls"] if c["tool"] == tool)


GENERIC = "I couldn't turn **“"


# --------------------------------------------------------------------- ingestion --
@pytest.mark.parametrize("key,delim", [("gapminder", "comma"), ("tips", "semicolon"), ("stocks", "tab"),
                                       ("election", "comma"), ("carshare", "comma"), ("diabetes", "comma")])
def test_every_file_is_read_with_its_own_delimiter_encoding_and_decimals(files, key, delim):
    path, truth = files[key]
    df, rep = ingest.read_table(str(path))
    assert rep["delimiter"] == delim
    assert list(df.columns) == [str(c) for c in truth.columns] and len(df) == len(truth)
    for c in truth.columns:
        if pd.api.types.is_numeric_dtype(truth[c]):
            assert np.allclose(df[c].to_numpy(float), truth[c].to_numpy(float)), c
        else:
            assert df[c].astype(str).tolist() == truth[c].astype(str).tolist() or pd.api.types.is_datetime64_any_dtype(df[c]), c


def test_excel_header_below_titles_and_headerless_csv(files):
    df, rep = ingest.read_table(str(files["clinic"][0]))
    assert list(df.columns) == list(files["clinic"][1].columns) and "row 4" in " ".join(rep["notes"])
    w, rep = ingest.read_table(str(files["wine"][0]))
    assert w.shape == files["wine"][1].shape and "column_1" in w.columns and "not column names" in " ".join(rep["notes"])


def test_text_numbers_dayfirst_dates_and_malformed_rows_are_reported(tmp_path):
    p = tmp_path / "fees.csv"
    p.write_text("Ref,Fee,Rate,Paid On\nA1,\"$1,200.50\",12%,13/02/2025\nA2,\"$950.00\",8%,01/03/2025\n"
                 "A3,\"$2,000.00\",15%,28/02/2025\nA4,$10\n")
    df, rep = ingest.read_table(str(p))
    assert df["Fee"].tolist()[:3] == [1200.5, 950.0, 2000.0] and df["Rate"].tolist()[:3] == [12, 8, 15]
    assert df["Paid On"].iloc[1] == pd.Timestamp("2025-03-01"), "day-first, decided from 13/02 and 28/02"
    assert any("fewer fields" in n and "line 5" in n for n in rep["notes"])


@pytest.mark.parametrize("text", ["a;b;c\n1;2\n3;4;5;6\n7\n", "name,score\n"])
def test_unreadable_tables_are_refused_with_a_reason(text, tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(ingest.IngestError):
        ingest.read_table(str(p))


# ------------------------------------------------------------- overview intents --
@pytest.mark.parametrize("q", ["Explain this dataset in simple terms.", "What is this data about?",
                               "Give me an overview of this file.", "Describe the columns.",
                               "What does each row represent?", "What are the important patterns?",
                               "What should we look into next?", "Find unusual records."])
@pytest.mark.parametrize("key", ["gapminder", "tips", "stocks", "wine"])
def test_overview_questions_are_understood_on_any_dataset(files, key, q):
    out = ask(q, load(files, key))
    assert GENERIC not in out["answer"] and out["plan"].get("type") not in ("clarify", "decline")
    assert out["calls"] and out["calls"][0]["status"] == "ok"


# ------------------------------------------------- calculations on unfamiliar names --
def test_abbreviated_column_names_are_understood(files):
    g = load(files, "gapminder")
    out = ask("What is the average life expectancy by continent?", g)
    truth = files["gapminder"][1].groupby("continent")["lifeExp"].mean()
    rows = res(out, "compute_metric")["groups"]
    assert {r["group"]: r["value"] for r in rows} == pytest.approx(truth.to_dict())


def test_a_year_in_the_question_filters_a_numeric_year_column(files):
    g = load(files, "gapminder")
    out = ask("Which continent has the highest total population in 2007?", g)
    t = files["gapminder"][1]
    truth = t[t.year == 2007].groupby("continent")["pop"].sum()
    rows = {r["group"]: r["value"] for r in res(out, "compute_metric")["groups"]}
    assert rows == pytest.approx(truth.to_dict()) and "year = 2007" in out["answer"]


def test_a_trend_over_a_year_column_averages_per_capita_values(files):
    out = ask("How has GDP per capita changed over time?", load(files, "gapminder"))
    t = files["gapminder"][1]
    pts = {p["period"]: p["value"] for p in res(out, "trend")["points"]}
    truth = t.groupby("year")["gdpPercap"].mean()
    assert pts == pytest.approx({str(k): v for k, v in truth.items()})


def test_numeric_breakdown_and_share_and_threshold(files):
    c = load(files, "carshare")
    rows = res(ask("What is the average car hours by peak hour?", c), "compute_metric")["groups"]
    truth = files["carshare"][1].groupby("peak_hour")["car_hours"].mean()
    assert {int(r["group"]): r["value"] for r in rows} == pytest.approx(truth.to_dict())
    t = load(files, "tips")
    share = res(ask("What percentage of bills were paid by smokers?", t))
    assert next(iter(share["totals"].values())) == pytest.approx((files["tips"][1].smoker == "Yes").mean())
    big = res(ask("How many bills have a total bill above 30?", t))
    assert big["totals"]["records_meeting"] == int((files["tips"][1].total_bill > 30).sum())


def test_a_stated_ratio_formula_is_averaged_per_record(files):
    out = ask("What is the tip rate (tip divided by total bill) by day?", load(files, "tips"))
    t = files["tips"][1]
    truth = (t.tip / t.total_bill).groupby(t.day).mean()
    assert {g["group"]: g["tip_rate"] for g in res(out)["groups"]} == pytest.approx(truth.to_dict())


def test_compare_named_categories_and_associations(files):
    out = ask("Compare life expectancy between Europe and Africa.", load(files, "gapminder"))
    r = res(out, "compare_groups")
    t = files["gapminder"][1]
    means = {g["group"]: g["mean"] for g in r["descriptives"]} if "descriptives" in r else None
    assert "Europe" in out["answer"] and "Africa" in out["answer"] and "Asia" not in out["answer"].split("**Welch")[0]
    if means:
        assert means["Europe"] == pytest.approx(t[t.continent == "Europe"].lifeExp.mean())
    a = res(ask("Which factors are associated with progression?", load(files, "diabetes")), "associations")
    d = files["diabetes"][1]
    bmi = next(x for x in a["factors"] if x["factor"] == "bmi")
    assert bmi["value"] == pytest.approx(stats.spearmanr(d.bmi, d.progression)[0])


def test_wide_data_compares_columns(files):
    out = ask("Which stock had the highest average price?", load(files, "stocks"))
    s = files["stocks"][1].drop(columns="date")
    assert res(out, "compare_columns")["values"] == pytest.approx(s.mean().to_dict())
    assert f"**{s.mean().idxmax()}**" in out["answer"]


def test_a_category_share_is_counted_not_used_as_a_filter(files):
    out = ask("What share of tumours are malignant?", load(files, "clinic"))
    assert next(iter(res(out)["totals"].values())) == pytest.approx((files["clinic"][1].diagnosis == "malignant").mean())


# ------------------------------------------------- conversation on unseen data --
def test_a_multi_turn_conversation_on_unseen_data(files):
    t, mem = load(files, "tips"), {}
    truth = files["tips"][1]
    first = ask("Show day totals", t, mem)
    assert first["plan"]["type"] == "clarify" and "tip" in first["plan"]["options"]
    done = ask("tip", t, mem)
    assert {g["group"]: g["value"] for g in res(done, "compute_metric")["groups"]} == pytest.approx(
        truth.groupby("day")["tip"].sum().to_dict())
    only = ask("What is the average tip for smokers?", t, mem)
    assert "Yes" in only["answer"] or "smoker" in only["answer"]
    ask("What percentage of bills were paid by smokers?", t, mem)
    by = ask("By day.", t, mem)
    assert {g["group"]: g[next(k for k in g if k.startswith("share_"))] for g in res(by)["groups"]} == pytest.approx(
        (truth.smoker == "Yes").groupby(truth.day).mean().to_dict())
    again = ask("yes", t, mem)
    assert GENERIC not in again["answer"]
