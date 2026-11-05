import pandas as pd
import pytest

shapely_geometry = pytest.importorskip("shapely.geometry")
gpd = pytest.importorskip("geopandas")

from pa_redistricting.plans import Plans  # noqa: E402

GEOIDS = ["42001000010", "42001000020", "42001000030"]


def tiny_plans() -> Plans:
    vtds = gpd.GeoDataFrame(
        {"GEOID": GEOIDS, "VTD_NAME": ["A", "B", "C"], "COUNTYFP": ["001"] * 3,
         "ALAND": [1, 1, 1], "AWATER": [0, 0, 0]},
        geometry=[shapely_geometry.box(i, 0, i + 1, 1) for i in range(3)], crs="EPSG:4269")
    census = pd.DataFrame({"GEOID": GEOIDS, "VTD_NAME_CENSUS": ["A", "B", "C"],
                           "COUNTYFP": ["001"] * 3, "pop_total": [100, 200, 300]})
    assignments = {
        "plan_a": pd.DataFrame({"GEOID": GEOIDS[:2], "district": [1, 2]}),
        "plan_b": pd.DataFrame({"GEOID": GEOIDS, "district": [2, 2, 1]}),
    }
    return Plans(vtds, census, assignments)


def test_get_keeps_every_voting_district():
    df = tiny_plans().get("plan_a", geometry=False)
    assert len(df) == 3
    assert df["district"].isna().sum() == 1
    assert str(df["district"].dtype) == "Int64"


def test_get_joins_census_without_duplicate_columns():
    df = tiny_plans().get("plan_a", geometry=False)
    assert df.set_index("GEOID").loc["42001000020", "pop_total"] == 200
    assert "VTD_NAME_CENSUS" not in df.columns
    assert df.columns.tolist().count("COUNTYFP") == 1


def test_get_column_order():
    gdf = tiny_plans().get("plan_a")
    assert gdf.columns[:3].tolist() == ["GEOID", "plan", "district"]
    assert gdf.columns[-1] == "geometry"
    assert gdf.crs == "EPSG:4269"


def test_geometry_false_returns_plain_dataframe():
    df = tiny_plans().get("plan_a", geometry=False)
    assert not isinstance(df, gpd.GeoDataFrame)
    assert "geometry" not in df.columns


def test_unknown_plan_raises():
    with pytest.raises(KeyError):
        tiny_plans().get("nope")


def test_match_and_all():
    plans = tiny_plans()
    assert plans.match(["_A"]) == ["plan_a"]
    assert len(plans.all()) == 6
    assert sorted(plans.all()["plan"].unique()) == ["plan_a", "plan_b"]
