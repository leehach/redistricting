"""
Tests for the sprint metrics added to redistricting/metrics_gui/metrics.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_metrics_gui.py

These run the metrics the same way the window does (auto_match, PlanContext,
run_metrics), without opening Qt.

Toy state: 8 precincts of 50 votes and 100 people each, 2 counties, 4 districts.

    precinct  dem  rep  county  district
    01        10   40   A       1
    02        20   30   A       1
    03        25   25   A       2
    04        24   26   A       2
    05        10   40   B       3
    06        41    9   B       3
    07        41    9   B       4
    08        29   21   B       4

District dem shares are 0.30, 0.49, 0.51, 0.70, statewide 50%. Each county is
cut 50/50 between two districts. Expected values are worked out by hand in
each test.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# metrics.py imports "registry" directly, the same way main.py does
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "redistricting" / "metrics_gui"))

import metrics  # noqa: E402,F401  adds every metric to REGISTRY
from registry import REGISTRY, PlanContext, auto_match, needed_fields, run_metrics  # noqa: E402

from redistricting import competitiveness, county_split  # noqa: E402

TOLERANCE = 1e-9

OUR_METRICS = [("Competitiveness", "Competitive districts"),
               ("Responsiveness", "Responsiveness"),
               ("Seats-votes", "Seats at 45%, 50% and 55%"),
               ("County splits", "Split counties"),
               ("County splits", "County split intersections"),
               ("County splits", "Spill ratio"),
               ("County splits", "County HHI")]


def make_plan():
    """Toy precinct table, with column names a user's file might have."""
    return pd.DataFrame({
        "precinct_id": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "dem":      [10, 20, 25, 24, 10, 41, 41, 29],
        "rep":      [40, 30, 25, 26, 40,  9,  9, 21],
        "county":   ["A", "A", "A", "A", "B", "B", "B", "B"],
        "pop":      [100] * 8,
        "district": [1, 1, 2, 2, 3, 3, 4, 4],
    })


def find(group, name):
    """One metric from the registry, by its group and checkbox name."""
    return next(m for m in REGISTRY[group] if m.name == name)


def run(chosen, data=None):
    """Score a plan the way the window does: auto-match columns, then run."""
    data = make_plan() if data is None else data
    columns = auto_match(data, needed_fields(chosen))
    return run_metrics(PlanContext(data, columns), chosen, "toy_plan")


def value(table, column):
    """A plan-wide value, read from the first row of the output table."""
    return table[column].iloc[0]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group, name", OUR_METRICS)
def test_metric_is_in_window(group, name):
    observed = [m.name for m in REGISTRY.get(group, [])]
    assert name in observed


def test_partisan_metrics_need_votes():
    observed = find("Competitiveness", "Competitive districts").needs
    expected = ["district", "dem_votes", "rep_votes"]
    assert observed == expected


def test_county_metrics_need_population():
    observed = find("County splits", "Spill ratio").needs
    expected = ["district", "county", "population"]
    assert observed == expected


def test_auto_match_finds_columns():
    chosen = [find(g, n) for g, n in OUR_METRICS]
    observed = auto_match(make_plan(), needed_fields(chosen))
    expected = {"district": "district", "dem_votes": "dem", "rep_votes": "rep",
                "county": "county", "population": "pop"}
    assert observed == expected


# ---------------------------------------------------------------------------
# Metric values
# ---------------------------------------------------------------------------

def test_competitive_districts():
    # margins: 0.40 0.02 0.02 0.40 -> 2 at or below 0.05
    observed = value(run([find("Competitiveness", "Competitive districts")]),
                     "competitive_districts")
    expected = 2
    assert observed == expected


def test_responsiveness():
    # at 52.5%: shares 0.325 0.515 0.535 0.725 -> 3 of 4 seats = 0.75
    # at 47.5%: shares 0.275 0.465 0.485 0.675 -> 1 of 4 seats = 0.25
    # (0.75 - 0.25) / 0.05 = 10
    observed = value(run([find("Responsiveness", "Responsiveness")]), "responsiveness")
    expected = 10.0
    assert abs(observed - expected) < TOLERANCE


def test_seats_at_three_vote_shares():
    # 45%: shares 0.25 0.44 0.46 0.65 -> 1 seat
    # 50%: shares 0.30 0.49 0.51 0.70 -> 2 seats
    # 55%: shares 0.35 0.54 0.56 0.75 -> 3 seats
    table = run([find("Seats-votes", "Seats at 45%, 50% and 55%")])
    observed = [value(table, "dem_seats_at_45pct"), value(table, "dem_seats_at_50pct"),
                value(table, "dem_seats_at_55pct")]
    expected = [1, 2, 3]
    assert observed == expected


def test_split_counties():
    # A is in districts 1 and 2, B in 3 and 4 -> both split
    observed = value(run([find("County splits", "Split counties")]), "split_counties")
    expected = 2
    assert observed == expected


def test_county_split_intersections():
    # A in 2 districts + B in 2 districts -> 4 pieces
    observed = value(run([find("County splits", "County split intersections")]),
                     "county_split_intersections")
    expected = 4
    assert observed == expected


def test_spill_ratio():
    # largest piece of each county is 200 of 400 -> 400 of 800 stay
    # 1 - 400 / 800 = 0.5
    observed = value(run([find("County splits", "Spill ratio")]), "spill_ratio")
    expected = 0.5
    assert abs(observed - expected) < TOLERANCE


def test_county_hhi():
    # each county: 0.5^2 + 0.5^2 = 0.5
    observed = value(run([find("County splits", "County HHI")]), "county_hhi")
    expected = 0.5
    assert abs(observed - expected) < TOLERANCE


def test_whole_counties():
    # one district per county: nothing split, nothing spills, hhi 1
    data = make_plan()
    data["district"] = [1, 1, 1, 1, 2, 2, 2, 2]
    chosen = [find(g, n) for g, n in OUR_METRICS if g == "County splits" and n != "County splits"]
    table = run(chosen, data)
    observed = [value(table, "split_counties"), value(table, "county_split_intersections"),
                value(table, "spill_ratio"), value(table, "county_hhi")]
    expected = [0, 0, 0.0, 1.0]
    assert all(abs(o - e) < TOLERANCE for o, e in zip(observed, expected))


# ---------------------------------------------------------------------------
# Same answer as the metric modules
# ---------------------------------------------------------------------------

def test_competitive_matches_module():
    data = make_plan()
    module = competitiveness.competitive_districts(
        data[["precinct_id", "dem", "rep", "district"]])
    observed = value(run([find("Competitiveness", "Competitive districts")]),
                     "competitive_districts")
    expected = module["competitive_districts"].iloc[0]
    assert observed == expected


def test_county_metrics_match_module():
    data = make_plan()
    module = county_split.all_metrics(data[["precinct_id", "county", "pop", "district"]])
    chosen = [find(g, n) for g, n in OUR_METRICS if g == "County splits" and n != "County splits"]
    table = run(chosen)
    names = ["split_counties", "county_split_intersections", "spill_ratio", "county_hhi"]
    observed = [value(table, n) for n in names]
    expected = [module[n].iloc[0] for n in names]
    assert all(abs(o - e) < TOLERANCE for o, e in zip(observed, expected))


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_one_row_per_district():
    table = run([find(g, n) for g, n in OUR_METRICS])
    observed = table["district"].tolist()
    expected = [1, 2, 3, 4]
    assert observed == expected


def test_plan_wide_values_repeat_on_every_row():
    table = run([find(g, n) for g, n in OUR_METRICS])
    columns = ["competitive_districts", "responsiveness", "dem_seats_at_45pct",
               "dem_seats_at_50pct", "dem_seats_at_55pct", "split_counties",
               "county_split_intersections", "spill_ratio", "county_hhi"]
    observed = table[columns].nunique().tolist()
    expected = [1] * len(columns)
    assert observed == expected


def test_input_not_changed():
    data = make_plan()
    run([find(g, n) for g, n in OUR_METRICS], data)
    assert data.equals(make_plan())
