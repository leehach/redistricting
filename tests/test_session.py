"""
Tests for redistricting/metrics_gui/session.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_session.py
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from redistricting.metrics_gui.session import SessionPlans


def make_table():
    return pd.DataFrame({"GEOID20": ["01", "02"], "District": [1, 2]})


def test_add_then_get_returns_the_same_table():
    plans, table = SessionPlans(), make_table()
    plans.add("a", table)
    assert plans.get("a") is table
    assert plans.names == ["a"] and len(plans) == 1 and "a" in plans


def test_names_keep_the_order_they_were_added():
    plans = SessionPlans()
    for name in ["b", "a", "c"]:
        plans.add(name, make_table())
    assert plans.names == ["b", "a", "c"]
    assert [p.name for p in plans] == ["b", "a", "c"]


def test_same_name_twice_raises_unless_replacing():
    plans = SessionPlans()
    plans.add("a", make_table())
    with pytest.raises(KeyError, match="already a plan"):
        plans.add("a", make_table())
    newer = make_table()
    plans.add("a", newer, replace=True)
    assert plans.get("a") is newer and len(plans) == 1


def test_blank_name_raises():
    with pytest.raises(ValueError, match="needs a name"):
        SessionPlans().add("  ", make_table())


def test_name_is_stripped():
    plans = SessionPlans()
    plans.add(" a ", make_table())
    assert plans.names == ["a"]


def test_free_name():
    plans = SessionPlans()
    assert plans.free_name("a") == "a"
    plans.add("a", make_table())
    assert plans.free_name("a") == "a_2"
    plans.add("a_2", make_table())
    assert plans.free_name("a") == "a_3"


def test_remove():
    plans = SessionPlans()
    plans.add("a", make_table())
    plans.remove("a")
    assert plans.names == [] and "a" not in plans


@pytest.mark.parametrize("method", ["get", "plan", "remove"])
def test_unknown_name_raises(method):
    with pytest.raises(KeyError, match="No plan named"):
        getattr(SessionPlans(), method)("nope")


def test_summary_and_notes():
    plans = SessionPlans()
    plan = plans.add("a", make_table(), source="disk", notes=["voting data: 2 of 2 precincts matched"])
    assert plan.summary() == "2 precincts, 2 fields, no shapes"
    assert plan.source == "disk" and plan.notes == ["voting data: 2 of 2 precincts matched"]


def test_summary_with_shapes_does_not_count_geometry_as_a_field():
    gdf = gpd.GeoDataFrame(make_table(), geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    plan = SessionPlans().add("a", gdf)
    assert plan.has_shapes and plan.summary() == "2 precincts, 2 fields, with shapes"
