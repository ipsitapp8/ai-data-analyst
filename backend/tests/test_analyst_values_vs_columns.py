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
