"""Words of a question that are VALUES (company names, category values) or ordinary words are not missing columns.

Two failures on a company-review table (one record per company, 4,000+ names; a company-type category; reviews
written "(26.3k Reviews)"):
  "Show the distribution of public and private companies."  -> group by the category the values belong to and count;
      it used to say the data had nothing called "distribution" / "companies".
  "Compare TCS and Accenture based on their ratings and reviews." -> both names found in the long name column and the
      named measures set side by side; the name right after "Compare" was not read, and the scope kept only the first
      of two names from a long column, so it said there was nothing called "accenture" / "based".
The fixture is synthetic (no dataset is special-cased); expected values come from pandas on the fixture, never from an
answer. No LLM is called.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator, planner, schema_profile


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))


def companies() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    names = ["TCS", "Accenture", "Infosys", "Wipro", "TCS iON", "TCS eServe"] + [f"Firm{i:03d} Labs" for i in range(114)]
    n = len(names)
    types = rng.choice(["Private", "Public", "Partnership", "LLP"], n, p=[0.6, 0.25, 0.1, 0.05]).astype(object)
    types[[7, 19, 33]] = None                                               # missing company type
    types[:4] = ["Public", "Public", "Public", "Public"]
    raw = pd.DataFrame({
        "name": names,
        "rating": np.round(rng.uniform(2.5, 4.8, n), 1),
        "review": [f"({x})" for x in ["26.3k Reviews", "20.5k Reviews", "18.7k Reviews", "17.1k Reviews"]]
                  + [f"({int(k)} Reviews)" for k in rng.integers(20, 900, n - 4)],
        "company_type": types,
    })
    raw.loc[0, "rating"], raw.loc[1, "rating"] = 3.9, 4.0
    return schema_profile.apply(raw)          # read as the app reads an upload ("26.3k" -> 26300)


def _ask(q, df):
    return orchestrator.ask(q, df, {}, use_model=False, filename="companies.csv")


@pytest.mark.parametrize("q", ["Show the distribution of public and private companies.",
                               "What is the breakdown of private vs public companies?"])
def test_distribution_of_named_category_values_is_a_group_count(q):
    df = companies()
    truth = df["company_type"].value_counts()
    out = _ask(q, df)
    a = out["answer"]
    assert "nothing called" not in a, a
    call = next(c for c in out["calls"] if c["tool"] == "count_by")
    assert call["args"]["group"] == "company_type"
    res = call["result"]
    assert res["counts"] == {"Public": int(truth["Public"]), "Private": int(truth["Private"])}
    # every record is accounted for: named values + the column's other values + missing
    assert res["counts"]["Public"] + res["counts"]["Private"] + res["other"] + res["missing"] == len(df)
    assert res["missing"] == int(df["company_type"].isna().sum())
    assert f"| **Public** | {int(truth['Public']):,} |" in a and f"| **Private** | {int(truth['Private']):,} |" in a
    assert out["charts"] and out["charts"][0]["type"] == "bar"


@pytest.mark.parametrize("q", ["Compare TCS and Accenture based on their ratings and reviews.",
                               "Compare the ratings and reviews of TCS and Accenture"])
def test_two_named_companies_are_compared_on_the_named_measures(q):
    df = companies()
    tcs, acc = df[df["name"] == "TCS"].iloc[0], df[df["name"] == "Accenture"].iloc[0]
    out = _ask(q, df)
    a = out["answer"]
    assert "nothing called" not in a, a
    call = next(c for c in out["calls"] if c["tool"] == "compare_records")
    res = call["result"]
    assert res["column"] == "name" and res["values"] == ["TCS", "Accenture"]       # exact "TCS", not "TCS iON"
    assert res["measures"] == ["rating", "review"]
    assert res["records"]["TCS"] == {"rating": float(tcs["rating"]), "review": float(tcs["review"])}
    assert res["records"]["Accenture"] == {"rating": float(acc["rating"]), "review": float(acc["review"])}
    assert float(tcs["review"]) == 26300 and float(acc["review"]) == 20500   # read from "(26.3k Reviews)"
    assert "**Accenture** is higher (4 vs 3.9" in a and "**TCS** is higher (26,300 vs 20,500" in a
    assert "TCS or Accenture — 2 of" in a                                       # the scope holds both names


def test_second_company_after_compare_with_is_kept_in_the_order_asked():
    df = companies()
    out = _ask("Compare Infosys with Wipro on rating", df)
    res = next(c for c in out["calls"] if c["tool"] == "compare_records")["result"]
    assert res["values"] == ["Infosys", "Wipro"] and res["measures"] == ["rating"]
    assert res["records"]["Wipro"]["rating"] == float(df.loc[df["name"] == "Wipro", "rating"].iloc[0])


def test_a_name_that_really_is_absent_is_still_reported_and_nothing_is_computed():
    """The fix must not weaken the check: a company that is not in the data is still said to be missing."""
    df = companies()
    out = _ask("Compare TCS and Zorblax on rating", df)
    assert not any(c["tool"] == "compare_records" for c in out["calls"])
    assert "Zorblax" in out["answer"] or "zorblax" in out["answer"]


# ---------------------------------------------- plain look-ups: values, one record, extremes, range --
def companies_full() -> pd.DataFrame:
    df = companies()
    rng = np.random.default_rng(5)
    df["Head_Quarters"] = rng.choice(["Mumbai", "Pune", "Dublin", "Bengaluru"], len(df))
    df["Company_Age"] = rng.integers(1, 120, len(df))
    df.loc[df["name"] == "TCS", "Head_Quarters"] = "Mumbai"
    return df


@pytest.mark.parametrize("q", ["What are the different types of companies?", "List all company types",
                               "Count companies by company type"])
def test_the_values_of_a_category_are_listed_with_their_counts(q):
    df = companies_full()
    vc = df["company_type"].value_counts()
    out = _ask(q, df)
    res = next(c for c in out["calls"] if c["tool"] == "value_counts")["result"]
    assert res["column"] == "company_type" and res["distinct"] == df["company_type"].nunique()
    assert res["top"] == {k: int(v) for k, v in vc.items()} and res["missing"] == int(df["company_type"].isna().sum())


def test_the_different_names_are_listed_from_the_name_column():
    df = companies_full()
    out = _ask("i am asking what are the different company names are in the dataset", df)
    res = next(c for c in out["calls"] if c["tool"] == "value_counts")["result"]
    assert res["column"] == "name" and res["distinct"] == df["name"].nunique()
    assert "nothing called" not in out["answer"]


def test_the_most_common_value_of_a_category():
    df = companies_full()
    vc = df["company_type"].value_counts()
    a = _ask("What is the most common company type?", df)["answer"]
    assert f"is **{vc.index[0]}** — {int(vc.iloc[0]):,} of {len(df):,} records" in a


def test_a_column_of_one_named_record_and_its_whole_record():
    df = companies_full()
    out = _ask("What are the headquarters of TCS?", df)
    call = next(c for c in out["calls"] if c["tool"] == "lookup")
    assert call["args"] == {"column": "name", "value": "TCS", "columns": ["Head_Quarters"]}
    assert "| TCS | Mumbai |" in out["answer"]
    out = _ask("Show the details of Infosys", df)
    call = next(c for c in out["calls"] if c["tool"] == "lookup")
    assert call["args"]["value"] == "Infosys" and set(call["args"]["columns"]) == set(df.columns) - {"name"}


def test_an_adjective_names_the_extreme_of_the_matching_column():
    df = companies_full()
    top = df.loc[df["Company_Age"].idxmax()]
    out = _ask("What is the oldest company?", df)
    call = next(c for c in out["calls"] if c["tool"] == "extreme")
    assert call["args"]["column"] == "Company_Age" and call["result"]["Company_Age"] == float(top["Company_Age"])
    assert df.loc[df["name"] == call["result"]["value"], "Company_Age"].max() == top["Company_Age"]


def test_the_range_of_a_measure():
    df = companies_full()
    out = _ask("What is the range of ratings?", df)
    res = next(c for c in out["calls"] if c["tool"] == "range")["result"]
    assert res["min"] == float(df["rating"].min()) and res["max"] == float(df["rating"].max())


# ------------------------------- exploration, comparatives, several conditions, comparing categories --
def test_column_names_are_the_schema_not_a_column_s_values():
    df = companies_full()
    out = _ask("What are the column names?", df)
    call = next(c for c in out["calls"] if c["tool"] == "schema")
    assert call["result"]["columns"] == [str(c) for c in df.columns]
    assert not any(c["tool"] == "value_counts" for c in out["calls"])


def test_first_rows_are_shown_as_stored():
    df = companies_full()
    out = _ask("Show the first 3 rows", df)
    call = next(c for c in out["calls"] if c["tool"] == "rows")
    assert call["args"] == {"which": "first", "n": 3} and call["result"]["rows"] == 3
    assert f"| {df.iloc[0]['name']} |" in out["answer"] and f"| {df.iloc[3]['name']} |" not in out["answer"]


def test_percentiles_and_quartiles():
    df = companies_full()
    res = next(c for c in _ask("What is the 90th percentile of rating?", df)["calls"] if c["tool"] == "percentiles")["result"]
    assert res == {"90": float(np.percentile(df["rating"].dropna(), 90))}
    res = next(c for c in _ask("What are the quartiles of company age?", df)["calls"] if c["tool"] == "percentiles")["result"]
    assert res == {str(q): float(np.percentile(df["Company_Age"].dropna(), q)) for q in (25, 50, 75)}


def test_older_than_is_a_condition_on_the_age_column():
    df = companies_full()
    out = _ask("How many companies are older than 50 years?", df)
    assert "Company Age > 50" in out["answer"] or "Company_Age > 50" in out["answer"]
    assert f"**{int((df['Company_Age'] > 50).sum())}**" in out["answer"]


@pytest.mark.parametrize("q, truth", [
    ("Which companies are older than 30 years and have a rating above 4?", lambda d: (d.Company_Age > 30) & (d.rating > 4)),
    ("List companies with rating above 3 and more than 500 reviews", lambda d: (d.rating > 3) & (d.review > 500)),
])
def test_records_meeting_several_conditions_are_listed(q, truth):
    df = companies_full()
    out = _ask(q, df)
    res = next(c for c in out["calls"] if c["tool"] == "filter_records")["result"]
    assert res["matching"] == int(truth(df).sum()) and res["records"] == len(df)


def test_two_categories_are_compared_on_every_named_measure():
    df = companies_full()
    out = _ask("Compare public and private companies based on average rating and company age.", df)
    calls = [c for c in out["calls"] if c["tool"] == "compare_groups"]
    assert {c["args"]["outcome"] for c in calls} == {"rating", "Company_Age"}
    for c in calls:
        m = c["args"]["outcome"]
        got = {row["group"]: row["mean"] for row in c["result"]["summary"]}
        truth = df[df.company_type.isin(["Public", "Private"])].groupby("company_type")[m].mean()
        assert got.keys() == {"Public", "Private"}
        assert all(abs(got[k] - truth[k]) < 1e-9 for k in got)
    assert "Not completed" not in out["answer"] and "did not pass" not in out["answer"]
