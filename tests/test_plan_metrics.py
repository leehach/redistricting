"""
Tests for the county-split options in plan_metrics.py.

Run from the top folder of the project:

    python -m pytest tests/test_plan_metrics.py -v

These use the four hand-made geounits in conftest.py, so they need no
download.
"""

import geopandas as gpd
import pandas as pd
import pytest

from redistricting import plan_metrics
from redistricting.plan_metrics import COUNTY_SPLIT_DIRECTION, DIRECTION


@pytest.fixture
def plan(plans):
    """District 2 takes 50 people from county B."""
    return plans["p1"]


def test_county_split_direction_matches_county_split_module():
    from redistricting import county_split
    assert list(COUNTY_SPLIT_DIRECTION) == list(county_split.METRICS)
    assert COUNTY_SPLIT_DIRECTION["county_hhi"] == "higher"
    assert DIRECTION["split_counties"] == COUNTY_SPLIT_DIRECTION["split_counties"]


def test_measure_plan_returns_requested_county_metrics(geounits, plan):
    row = plan_metrics._measure_plan(plan, geounits, list(COUNTY_SPLIT_DIRECTION))
    # county B (200 people) is split 150 / 50; county A is whole
    assert row["split_counties"] == 1
    assert row["county_split_intersections"] == 2
    # 50 of 400 people live outside their county's largest piece
    assert row["spill_ratio"] == pytest.approx(50 / 400)
    # A scores 1, B scores 0.75^2 + 0.25^2 = 0.625; weighted by 200 each
    assert row["county_hhi"] == pytest.approx((1 + 0.625) / 2)


def test_measure_plan_default_is_split_counties_only(geounits, plan):
    row = plan_metrics._measure_plan(plan, geounits, ["split_counties"])
    assert set(row) == {"max_pop_deviation", "efficiency_gap", "split_counties"}


def test_measure_rejects_unknown_county_metric():
    with pytest.raises(ValueError, match="Unknown county-split metric"):
        plan_metrics.measure(pd.DataFrame(), gpd.GeoDataFrame(), ["bogus"])


def test_measure_default_columns_are_direction(geounits, plans):
    table = plan_metrics.measure(plans, geounits)
    assert list(table.columns) == list(DIRECTION)
    assert list(table.index) == list(plans.columns)


def test_measure_adds_county_metrics_after_direction(geounits, plans):
    table = plan_metrics.measure(plans, geounits, list(COUNTY_SPLIT_DIRECTION))
    # split_counties is in DIRECTION already, so it is not added twice
    assert list(table.columns) == list(DIRECTION) + [
        "county_split_intersections", "spill_ratio", "county_hhi"]


def test_measure_county_values_per_plan(geounits, plans):
    table = plan_metrics.measure(plans, geounits, list(COUNTY_SPLIT_DIRECTION))
    # p1: only county B is split, 150 / 50
    assert table.loc["p1", "split_counties"] == 1
    assert table.loc["p1", "spill_ratio"] == pytest.approx(50 / 400)
    # p3: A is split 100 / 100 and B 150 / 50, so 100 + 50 people spill.
    # HHI: A scores 0.5^2 + 0.5^2 = 0.5, B scores 0.625.
    assert table.loc["p3", "split_counties"] == 2
    assert table.loc["p3", "county_split_intersections"] == 4
    assert table.loc["p3", "spill_ratio"] == pytest.approx(150 / 400)
    assert table.loc["p3", "county_hhi"] == pytest.approx((0.5 + 0.625) / 2)
