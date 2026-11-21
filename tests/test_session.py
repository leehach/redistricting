"""
Tests for redistricting/metrics_gui/session.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_session.py
"""

import pandas as pd
import pytest

from redistricting.metrics_gui.session import Plan, SessionPlans


def make_plan(name="a", kind="PAF"):
    width = 11 if kind == "PAF" else 15
    geoids = [str(i).zfill(width) for i in (1, 2, 3)]
    return Plan(name, pd.DataFrame({"GEOID": geoids, "district": [1, 1, 2]}), kind)


def test_add_then_get_returns_the_assignment():
    plans, plan = SessionPlans(), make_plan()
    assert plans.add(plan) is plan
    assert plans.get("a") is plan.assignment and plans.plan("a") is plan
    assert plans.names == ["a"] and len(plans) == 1 and "a" in plans


def test_names_keep_the_order_they_were_added():
    plans = SessionPlans()
    for name in ["b", "a", "c"]:
        plans.add(make_plan(name))
    assert plans.names == ["b", "a", "c"]
    assert [p.name for p in plans] == ["b", "a", "c"]


def test_same_name_twice_raises_unless_replacing():
    plans = SessionPlans()
    plans.add(make_plan())
    with pytest.raises(KeyError, match="already a plan"):
        plans.add(make_plan())
    newer = make_plan()
    plans.add(newer, replace=True)
    assert plans.plan("a") is newer and len(plans) == 1


def test_blank_name_raises():
    with pytest.raises(ValueError, match="needs a name"):
        SessionPlans().add(make_plan("  "))


def test_name_is_stripped():
    plans = SessionPlans()
    plans.add(make_plan(" a "))
    assert plans.names == ["a"]


def test_a_plan_is_only_geoid_and_district():
    plan = make_plan()
    plan.assignment["dem_votes"] = [1, 2, 3]
    with pytest.raises(ValueError, match="exactly the columns"):
        SessionPlans().add(plan)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        SessionPlans().add(make_plan(kind="XYZ"))


def test_free_name():
    plans = SessionPlans()
    assert plans.free_name("a") == "a"
    plans.add(make_plan("a"))
    assert plans.free_name("a") == "a_2"
    plans.add(make_plan("a_2"))
    assert plans.free_name("a") == "a_3"


def test_remove():
    plans = SessionPlans()
    plans.add(make_plan())
    plans.remove("a")
    assert plans.names == [] and "a" not in plans


@pytest.mark.parametrize("method", ["get", "plan", "remove"])
def test_unknown_name_raises(method):
    with pytest.raises(KeyError, match="No plan named"):
        getattr(SessionPlans(), method)("nope")


def test_summary_counts():
    assert make_plan().summary() == "PAF: 3 precincts in 2 districts"
    assert make_plan(kind="BAF").summary() == "BAF: 3 blocks in 2 districts"


def test_long_table_stacks_every_plan():
    plans = SessionPlans()
    plans.add(make_plan("a"))
    plans.add(make_plan("b"))
    table = plans.long_table()
    assert list(table.columns) == ["GEOID", "plan", "district"]
    assert table["plan"].tolist() == ["a"] * 3 + ["b"] * 3


def test_long_table_when_empty():
    assert list(SessionPlans().long_table().columns) == ["GEOID", "plan", "district"]
