import pandas as pd
import pytest

from pa_redistricting import census
from pa_redistricting.demographics import (COUNT_COLUMNS, VEST_COLUMNS, compare_sources,
                                           vest_demographics)

GEOIDS = ["42001000010", "42001000020"]


def tiny_vest() -> pd.DataFrame:
    df = pd.DataFrame({"GEOID": GEOIDS, **{c: [100, 50] for c in VEST_COLUMNS}})
    df["pop_white"] = [70, 30]           # VEST: non-Hispanic White
    df["pre_20_dem_bid"] = [40.5, 20.0]  # vote columns are ignored
    return df


def test_vest_white_becomes_nh_white():
    demo = vest_demographics(tiny_vest())
    assert demo["pop_nh_white"].tolist() == [70, 30]
    assert "pop_white" not in demo.columns
    assert "pre_20_dem_bid" not in demo.columns


def test_vest_gives_every_count_column():
    demo = vest_demographics(tiny_vest())
    assert set(COUNT_COLUMNS) <= set(demo.columns)
    assert all(str(demo[c].dtype) == "Int64" for c in COUNT_COLUMNS)
    assert demo["pct_nh_white"].tolist() == [70.0, 60.0]


def test_vest_missing_column_raises():
    with pytest.raises(ValueError, match="pop_hisp"):
        vest_demographics(tiny_vest().drop(columns="pop_hisp"))


def test_census_shared_columns_match_vest():
    assert list(census.SHARED_VARIABLES.values()) == COUNT_COLUMNS


def test_census_only_columns_do_not_collide():
    assert not set(census.EXTRA_COLUMNS) & set(COUNT_COLUMNS)
    assert "pop_white" not in census.VARIABLES.values()
    assert len(census.VARIABLES) < 50   # Census API limit per request


def test_compare_identical_sources():
    demo = vest_demographics(tiny_vest())
    table, only = compare_sources(demo, demo.copy())
    assert table["vtds_differ"].eq(0).all()
    assert table["difference"].eq(0).all()
    assert only == {"only_vest": [], "only_census": []}


def test_compare_finds_differences_and_missing_geoids():
    vest = vest_demographics(tiny_vest())
    other = vest.copy()
    other.loc[0, "pop_total"] += 5
    other.loc[2] = other.loc[1].copy()
    other.loc[2, "GEOID"] = "42003000010"
    table, only = compare_sources(vest, other)
    row = table.set_index("column").loc["pop_total"]
    assert row["vtds_differ"] == 1
    assert row["difference"] == -5
    assert only == {"only_vest": [], "only_census": ["42003000010"]}


def test_unknown_source_raises():
    pytest.importorskip("geopandas")
    from pa_redistricting.plans import load_demographics
    with pytest.raises(ValueError, match="demographics must be"):
        load_demographics("acs")


def test_vest_plus_census_adds_extra_columns(monkeypatch):
    pytest.importorskip("geopandas")
    from pa_redistricting import plans
    extras = pd.DataFrame({"GEOID": GEOIDS, **{c: [1, 2] for c in census.EXTRA_COLUMNS}})
    monkeypatch.setattr(plans, "fetch_census", lambda key=None: extras)
    demo = plans.load_demographics("vest+census", vest=tiny_vest())
    assert demo["gq_correctional"].tolist() == [1, 2]
    assert "pct_vacant" in demo.columns
    assert demo["pop_nh_white"].tolist() == [70, 30]
