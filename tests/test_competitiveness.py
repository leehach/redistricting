"""
Tests for redistricting/competitiveness.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_competitiveness.py

Toy state: 8 precincts of 50 votes each, statewide 200 dem of 400 = 50%.

    plan_a district dem shares: 0.30, 0.49, 0.51, 0.70  (two close districts)
    plan_b district dem shares: 0.30, 0.35, 0.65, 0.70  (all safe districts)

A district is competitive when |dem - rep| / (dem + rep) <= threshold.
Expected values are worked out by hand in each test.
"""

import numpy as np
import pandas as pd
import pytest

from redistricting.competitiveness import competitive_districts, join_plans, read_plans


def make_votes():
    """Toy precinct table with two plans."""
    return pd.DataFrame({
        "precinct_id": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "dem":    [10, 20, 25, 24, 10, 41, 41, 29],
        "rep":    [40, 30, 25, 26, 40,  9,  9, 21],
        "plan_a": [1, 1, 2, 2, 3, 3, 4, 4],
        "plan_b": [1, 2, 1, 3, 2, 4, 3, 4],
    })


def count(result, plan):
    """Competitive districts for one plan from the output table."""
    return result.loc[result["plan"] == plan, "competitive_districts"].iloc[0]


# ---------------------------------------------------------------------------
# Metric values
# ---------------------------------------------------------------------------

def test_close_districts_are_competitive():
    # plan_a margins: 0.40 0.02 0.02 0.40 -> 2 at or below 0.05
    observed = count(competitive_districts(make_votes()), "plan_a")
    expected = 2
    assert observed == expected


def test_safe_districts_are_not_competitive():
    # plan_b margins: 0.40 0.30 0.30 0.40 -> none at or below 0.05
    observed = count(competitive_districts(make_votes()), "plan_b")
    expected = 0
    assert observed == expected


def test_wider_threshold():
    # plan_b margins: 0.40 0.30 0.30 0.40 -> 2 at or below 0.30
    observed = count(competitive_districts(make_votes(), threshold=0.30), "plan_b")
    expected = 2
    assert observed == expected


def test_margin_equal_to_threshold_counts():
    # district 1: 525 - 475 = 50 of 1000 -> margin 0.05, exactly the threshold
    # district 2: 526 - 474 = 52 of 1000 -> margin 0.052, just over
    votes = pd.DataFrame({
        "precinct_id": ["01", "02"],
        "dem":  [525, 526],
        "rep":  [475, 474],
        "plan": [1, 2],
    })
    observed = count(competitive_districts(votes), "plan")
    expected = 1
    assert observed == expected


def test_unassigned_precinct_warns_and_is_left_out():
    # precinct 01 has no district; district 1 is precinct 02 alone (margin 0.20)
    # margins: 0.20 0.02 0.02 0.40 -> 2 competitive
    votes = make_votes().drop(columns=["plan_a", "plan_b"])
    votes["plan"] = [np.nan, 1, 2, 2, 3, 3, 4, 4]
    with pytest.warns(UserWarning, match="no district"):
        result = competitive_districts(votes)
    observed = count(result, "plan")
    expected = 2
    assert observed == expected


# ---------------------------------------------------------------------------
# Output shape and input handling
# ---------------------------------------------------------------------------

def test_one_row_per_plan():
    observed = competitive_districts(make_votes())["plan"].tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_columns():
    observed = competitive_districts(make_votes()).columns.tolist()
    expected = ["plan", "competitive_districts"]
    assert observed == expected


def test_input_not_changed():
    votes = make_votes()
    competitive_districts(votes)
    assert votes.equals(make_votes())


# ---------------------------------------------------------------------------
# read_plans and join_plans
# ---------------------------------------------------------------------------

def test_read_plans_keeps_leading_zeros(tmp_path):
    pd.DataFrame({"GEOID20": ["01", "02"], "District": ["1", "2"]}).to_csv(
        tmp_path / "plan_b.csv", index=False)
    pd.DataFrame({"GEOID20": ["01", "02"], "District": ["2", "1"]}).to_csv(
        tmp_path / "plan_a.csv", index=False)
    plans = read_plans(tmp_path)
    assert list(plans) == ["plan_a", "plan_b"]  # sorted by file name
    assert plans["plan_a"]["GEOID20"].tolist() == ["01", "02"]


def test_join_plans_shape_and_missing_precinct():
    votes = pd.DataFrame({"precinct_id": ["01", "02", "03"],
                          "county": ["A", "A", "B"],
                          "dem": [10, 20, 30], "rep": [30, 20, 10]})
    # plan_x does not list precinct 03
    plans = {"plan_x": pd.DataFrame({"GEOID20": ["01", "02"], "District": ["1", "2"]})}
    df = join_plans(votes, plans)
    assert df.columns.tolist() == ["precinct_id", "dem", "rep", "plan_x"]
    assert df["plan_x"].tolist()[:2] == ["1", "2"]
    assert pd.isna(df["plan_x"].iloc[2])


def test_join_plans_duplicate_precinct_raises():
    votes = pd.DataFrame({"precinct_id": ["01", "02"], "dem": [10, 20], "rep": [30, 20]})
    plans = {"plan_x": pd.DataFrame({"GEOID20": ["01", "01"], "District": ["1", "2"]})}
    with pytest.raises(ValueError):
        join_plans(votes, plans)


def test_read_join_score_end_to_end(tmp_path):
    # plan_a from make_votes, written to a CSV and read back as text
    toy = make_votes()
    pd.DataFrame({"GEOID20": toy["precinct_id"], "District": toy["plan_a"]}).to_csv(
        tmp_path / "plan_a.csv", index=False)
    df = join_plans(toy[["precinct_id", "dem", "rep"]], read_plans(tmp_path))
    observed = count(competitive_districts(df), "plan_a")
    expected = 2
    assert observed == expected


# ---------------------------------------------------------------------------
# exclude
# ---------------------------------------------------------------------------

def test_exclude_list_leaves_out_columns():
    votes = make_votes()
    votes["county"] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = competitive_districts(votes, exclude=["county", "area"])["plan"].tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_single_string():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = competitive_districts(votes, exclude="area")["plan"].tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_does_not_change_counts():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = competitive_districts(votes, exclude="area")
    expected = competitive_districts(make_votes())
    assert observed.equals(expected)


def test_exclude_unknown_column_raises():
    with pytest.raises(ValueError):
        competitive_districts(make_votes(), exclude=["countey"])


def test_without_exclude_extra_column_is_scored_as_plan():
    # documents why exclude exists: an unlisted column is treated as a plan
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = competitive_districts(votes)["plan"].tolist()
    expected = ["plan_a", "plan_b", "area"]
    assert observed == expected
