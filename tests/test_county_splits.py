"""
Tests for redistricting/county_split.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_county_splits.py

Toy state: 3 counties of 100 people each, 2 precincts per county.

    county  precinct  pop  plan_whole  plan_split
    A       a1         50  1           1
    A       a2         50  1           2
    B       b1         60  2           2
    B       b2         40  2           2
    C       c1         70  3           3
    C       c2         30  3           2

plan_whole keeps every county whole. plan_split cuts A 50/50 and C 70/30,
and keeps B whole. Expected values are worked out by hand in each test.
"""

import pandas as pd
import pytest

from redistricting.county_split import (all_metrics, county_hhi,
                                        county_split_intersections, score_plan,
                                        spill_ratio, split_counties)

TOLERANCE = 1e-9


def make_plans():
    """Toy precinct table with two plans."""
    return pd.DataFrame({
        "precinct_id": ["a1", "a2", "b1", "b2", "c1", "c2"],
        "county":      ["A", "A", "B", "B", "C", "C"],
        "pop":         [50, 50, 60, 40, 70, 30],
        "plan_whole":  [1, 1, 2, 2, 3, 3],
        "plan_split":  [1, 2, 2, 2, 3, 2],
    })


def value(result, plan, metric):
    """One metric for one plan from the output table."""
    return result.loc[result["plan"] == plan, metric].iloc[0]


# ---------------------------------------------------------------------------
# split_counties
# ---------------------------------------------------------------------------

def test_split_counties_whole_plan():
    observed = value(split_counties(make_plans()), "plan_whole", "split_counties")
    expected = 0
    assert observed == expected


def test_split_counties_split_plan():
    # A and C are split, B is whole
    observed = value(split_counties(make_plans()), "plan_split", "split_counties")
    expected = 2
    assert observed == expected


# ---------------------------------------------------------------------------
# county_split_intersections
# ---------------------------------------------------------------------------

def test_intersections_whole_plan():
    observed = value(county_split_intersections(make_plans()), "plan_whole",
                     "county_split_intersections")
    expected = 0
    assert observed == expected


def test_intersections_split_plan():
    # A in 2 districts + C in 2 districts; whole B adds 0 -> 4
    observed = value(county_split_intersections(make_plans()), "plan_split",
                     "county_split_intersections")
    expected = 4
    assert observed == expected


def test_intersections_tell_two_way_from_three_way():
    # one county cut into 3 pieces: 1 split county, but 3 pieces
    plans = pd.DataFrame({
        "precinct_id": ["x1", "x2", "x3"],
        "county":      ["X", "X", "X"],
        "pop":         [10, 10, 10],
        "plan":        [1, 2, 3],
    })
    observed = [value(split_counties(plans), "plan", "split_counties"),
                value(county_split_intersections(plans), "plan", "county_split_intersections")]
    expected = [1, 3]
    assert observed == expected


# ---------------------------------------------------------------------------
# spill_ratio
# ---------------------------------------------------------------------------

def test_spill_ratio_whole_plan():
    observed = value(spill_ratio(make_plans()), "plan_whole", "spill_ratio")
    expected = 0.0
    assert abs(observed - expected) < TOLERANCE


def test_spill_ratio_split_plan():
    # largest pieces: A 50, B 100, C 70 -> 220 of 300 stay; 80 spill
    # 1 - 220 / 300 = 0.2667
    observed = value(spill_ratio(make_plans()), "plan_split", "spill_ratio")
    expected = 80 / 300
    assert abs(observed - expected) < TOLERANCE


# ---------------------------------------------------------------------------
# county_hhi
# ---------------------------------------------------------------------------

def test_county_hhi_whole_plan():
    observed = value(county_hhi(make_plans()), "plan_whole", "county_hhi")
    expected = 1.0
    assert abs(observed - expected) < TOLERANCE


def test_county_hhi_split_plan():
    # A: 0.5^2 + 0.5^2 = 0.50   B: 1.00   C: 0.7^2 + 0.3^2 = 0.58
    # counties are all 100 people, so the weighted mean is (0.50 + 1 + 0.58) / 3
    observed = value(county_hhi(make_plans()), "plan_split", "county_hhi")
    expected = (0.50 + 1.00 + 0.58) / 3
    assert abs(observed - expected) < TOLERANCE


# ---------------------------------------------------------------------------
# Empty precincts
# ---------------------------------------------------------------------------

def test_empty_precinct_is_not_a_split():
    # x2 has no people, so the line through county X crosses nobody
    plans = pd.DataFrame({
        "precinct_id": ["x1", "x2"],
        "county":      ["X", "X"],
        "pop":         [100, 0],
        "plan":        [1, 2],
    })
    with pytest.warns(UserWarning, match="no population"):
        result = split_counties(plans)
    observed = value(result, "plan", "split_counties")
    expected = 0
    assert observed == expected


# ---------------------------------------------------------------------------
# all_metrics
# ---------------------------------------------------------------------------

def test_all_metrics_columns():
    observed = all_metrics(make_plans()).columns.tolist()
    expected = ["plan", "split_counties", "county_split_intersections",
                "spill_ratio", "county_hhi"]
    assert observed == expected


def test_all_metrics_matches_single_functions():
    result = all_metrics(make_plans())
    observed = [value(result, "plan_split", "split_counties"),
                value(result, "plan_split", "county_split_intersections"),
                value(result, "plan_split", "spill_ratio"),
                value(result, "plan_split", "county_hhi")]
    expected = [2, 4, 80 / 300, (0.50 + 1.00 + 0.58) / 3]
    assert all(abs(o - e) < TOLERANCE for o, e in zip(observed, expected))


def test_all_metrics_subset():
    observed = all_metrics(make_plans(), ["spill_ratio", "split_counties"]).columns.tolist()
    expected = ["plan", "spill_ratio", "split_counties"]
    assert observed == expected


def test_all_metrics_unknown_name_raises():
    with pytest.raises(ValueError):
        all_metrics(make_plans(), ["split_countys"])


# ---------------------------------------------------------------------------
# Output shape and input handling
# ---------------------------------------------------------------------------

def test_one_row_per_plan():
    observed = split_counties(make_plans())["plan"].tolist()
    expected = ["plan_whole", "plan_split"]
    assert observed == expected


def test_input_not_changed():
    plans = make_plans()
    all_metrics(plans)
    assert plans.equals(make_plans())


# ---------------------------------------------------------------------------
# exclude
# ---------------------------------------------------------------------------

def test_exclude_list_leaves_out_columns():
    plans = make_plans()
    plans["district_name"] = ["N", "N", "S", "S", "E", "E"]
    plans["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0]
    observed = all_metrics(plans, exclude=["district_name", "area"])["plan"].tolist()
    expected = ["plan_whole", "plan_split"]
    assert observed == expected


def test_exclude_single_string():
    plans = make_plans()
    plans["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0]
    observed = spill_ratio(plans, exclude="area")["plan"].tolist()
    expected = ["plan_whole", "plan_split"]
    assert observed == expected


@pytest.mark.parametrize("func", [split_counties, county_split_intersections,
                                  spill_ratio, county_hhi, all_metrics])
def test_exclude_works_in_every_function(func):
    plans = make_plans()
    plans["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0]
    observed = func(plans, exclude="area")
    expected = func(make_plans())
    assert observed.equals(expected)


def test_exclude_unknown_column_raises():
    with pytest.raises(ValueError):
        all_metrics(make_plans(), exclude=["aera"])


def test_without_exclude_extra_column_is_scored_as_plan():
    # documents why exclude exists: an unlisted column is treated as a plan
    plans = make_plans()
    plans["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0]
    observed = split_counties(plans)["plan"].tolist()
    expected = ["plan_whole", "plan_split", "area"]
    assert observed == expected


# ---------------------------------------------------------------------------
# Input checks and new options
# ---------------------------------------------------------------------------

def test_exclude_can_list_id_column():
    observed = all_metrics(make_plans(), exclude=["precinct_id"])["plan"].tolist()
    expected = ["plan_whole", "plan_split"]
    assert observed == expected


def test_exclude_can_list_geometry_column():
    plans = make_plans()
    plans["geometry"] = None
    observed = all_metrics(plans, exclude=["geometry"])["plan"].tolist()
    expected = ["plan_whole", "plan_split"]
    assert observed == expected


@pytest.mark.parametrize("func", [split_counties, county_split_intersections,
                                  spill_ratio, county_hhi, all_metrics])
def test_other_column_names_work_in_every_function(func):
    plans = make_plans().rename(columns={"county": "CNTY", "pop": "TotPop"})
    observed = func(plans, county="CNTY", pop="TotPop")
    expected = func(make_plans())
    assert observed.equals(expected)


def test_missing_column_raises():
    plans = make_plans().rename(columns={"pop": "TotPop"})
    with pytest.raises(ValueError, match="pop"):
        all_metrics(plans)


def test_chosen_name_clashing_with_existing_column_raises():
    plans = make_plans()
    plans["TotPop"] = plans["pop"]
    with pytest.raises(ValueError, match="already has"):
        all_metrics(plans, pop="TotPop")


def test_blank_population_raises():
    plans = make_plans().astype({"pop": float})
    plans.loc[0, "pop"] = float("nan")
    with pytest.raises(ValueError, match="blank population"):
        all_metrics(plans)


def test_blank_county_raises():
    plans = make_plans()
    plans.loc[0, "county"] = None
    with pytest.raises(ValueError, match="blank county"):
        all_metrics(plans)


def test_plan_with_no_population_raises():
    plans = pd.DataFrame({"precinct_id": ["x1"], "county": ["X"], "pop": [0], "plan": [1]})
    with pytest.raises(ValueError, match="no county-district piece"), pytest.warns(UserWarning, match="no population"):
        all_metrics(plans)


def test_precinct_with_no_district_warns_and_is_left_out():
    # the 90-person precinct has no district, so only 10 people are scored
    plans = pd.DataFrame({
        "precinct_id": ["x1", "x2"],
        "county":      ["X", "X"],
        "pop":         [10, 90],
        "plan":        [1, None],
    })
    with pytest.warns(UserWarning, match="no district"):
        result = split_counties(plans)
    observed = value(result, "plan", "split_counties")
    expected = 0
    assert observed == expected


def test_score_plan():
    # A is cut 50/50, B is whole, C is cut 70/30
    # split counties: A and C = 2     pieces in split counties: 2 + 2 = 4
    # spill: 1 - (50 + 100 + 70) / 300 = 80 / 300
    # hhi: (0.50 + 1.00 + 0.58) / 3
    observed = score_plan(["A", "A", "B", "B", "C", "C"], [50, 50, 60, 40, 70, 30], [1, 2, 2, 2, 3, 2])
    expected = {"split_counties": 2, "county_split_intersections": 4,
                "spill_ratio": 80 / 300, "county_hhi": (0.50 + 1.00 + 0.58) / 3}
    assert observed == pytest.approx(expected, abs=TOLERANCE)
