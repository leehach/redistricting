"""Shared fixtures: a tiny, hand-checkable stand-in for the PA geounits."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

GEOIDS = ["g1", "g2", "g3", "g4"]


@pytest.fixture
def geounits():
    """Four unit squares in a row, in two counties of 200 people each.

    Shaped like plan_metrics.load_pa_geounits(), so plan_metrics.measure()
    can run on it without a download.
    """
    return gpd.GeoDataFrame({
        "population": [100, 100, 150, 50],
        "county":     ["A", "A", "B", "B"],
        "dem":        [60, 40, 70, 20],
        "rep":        [40, 60, 80, 30],
    }, geometry=[box(i, 0, i + 1, 1) for i in range(4)],
        index=GEOIDS, crs="EPSG:32618")


@pytest.fixture
def plans():
    """Five two-district plans for ``geounits``, one column per plan.

    p1 splits only county B (150 / 50). p3 splits both counties.
    Every plan splits at least one county, so no metric is 0, as SBM needs.
    """
    return pd.DataFrame({
        "p1": [1, 1, 2, 1],
        "p2": [1, 2, 2, 2],
        "p3": [1, 2, 1, 2],
        "p4": [1, 1, 1, 2],
        "p5": [2, 1, 1, 2],
    }, index=GEOIDS)
