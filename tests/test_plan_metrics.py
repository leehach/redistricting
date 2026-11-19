"""
Tests for the county-split options in plan_metrics.py.

Run from the top folder of the project:

    python -m pytest tests/test_plan_metrics.py -v

These use a hand-made table of four geounits, so they need no download.
"""

import geopandas as gpd
import pandas as pd
import pytest

from redistricting import plan_metrics
from redistricting.plan_metrics import COUNTY_SPLIT_DIRECTION, DIRECTION


@pytest.fixture
def geounits():
    """Two counties of 200 people. District 2 takes 50 people from county B."""
    return pd.DataFrame({
        "population": [100, 100, 150, 50],
        "county":     ["A", "A", "B", "B"],
        "dem":        [60, 40, 70, 20],
        "rep":        [40, 60, 80, 30],
    }, index=["g1", "g2", "g3", "g4"])


@pytest.fixture
def plan():
    return pd.Series([1, 1, 2, 1], index=["g1", "g2", "g3", "g4"])


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
