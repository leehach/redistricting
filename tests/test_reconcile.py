"""
Tests for redistricting/metrics_gui/reconcile.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_reconcile.py

Toy state: 4 precincts 01-04, two districts. Each source file names its id column
differently, the way real files do.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from redistricting.metrics_gui.reconcile import (ReconcileError, clean_key, find_key, read_table,
                                                 reconcile)


def make_assignments():
    return pd.DataFrame({"GEOID20": ["01", "02", "03", "04"], "District": ["1", "1", "2", "2"]})


def make_votes():
    return pd.DataFrame({"geoid": ["01", "02", "03", "04"], "dem": ["10", "20", "30", "40"],
                         "rep": ["40", "30", "20", "10"]})


def make_census():
    return pd.DataFrame({"GEOID": ["01", "02", "03", "04"], "pop_total": [100, 200, 300, 400]})


def make_shapes():
    return gpd.GeoDataFrame({"GEOID20": ["01", "02", "03", "04"]},
                            geometry=[box(i, 0, i + 1, 1) for i in range(4)], crs="EPSG:3857")


def test_joins_all_three_on_differently_named_ids():
    plan = reconcile(make_assignments(), voting=make_votes(), demographics=make_census())
    assert plan.data.columns.tolist() == ["GEOID20", "District", "dem", "rep", "pop_total"]
    assert plan.data["dem"].tolist() == [10, 20, 30, 40]
    assert plan.data["pop_total"].tolist() == [100, 200, 300, 400]


def test_assignments_alone_is_enough():
    plan = reconcile(make_assignments())
    assert len(plan.data) == 4 and plan.notes == []


def test_number_columns_become_numbers_but_ids_stay_text():
    data = reconcile(make_assignments(), voting=make_votes()).data
    assert pd.api.types.is_numeric_dtype(data["dem"])
    assert pd.api.types.is_numeric_dtype(data["District"])
    assert data["GEOID20"].tolist() == ["01", "02", "03", "04"]


def test_keeps_every_assigned_precinct_and_reports_partial_match():
    votes = make_votes().iloc[:3]
    plan = reconcile(make_assignments(), voting=votes)
    assert len(plan.data) == 4
    assert pd.isna(plan.data["dem"].iloc[3])
    assert plan.notes == ["voting data: 3 of 4 precincts matched"]


def test_rows_not_in_the_plan_are_left_out():
    votes = pd.concat([make_votes(), pd.DataFrame({"geoid": ["99"], "dem": ["1"], "rep": ["1"]})])
    plan = reconcile(make_assignments(), voting=votes)
    assert len(plan.data) == 4
    assert plan.notes == ["voting data: 4 of 4 precincts matched, 1 rows not in the plan were left out"]


def test_stripped_leading_zeros_are_put_back():
    votes = make_votes().assign(geoid=[1, 2, 3, 4])  # what a spreadsheet does to "01"
    plan = reconcile(make_assignments(), voting=votes)
    assert plan.data["dem"].tolist() == [10, 20, 30, 40]


def test_ids_read_as_floats_still_match():
    assignments = make_assignments().assign(GEOID20=["11", "12", "13", "14"])
    votes = make_votes().assign(geoid=[11.0, 12.0, 13.0, 14.0])
    assert reconcile(assignments, voting=votes).data["dem"].tolist() == [10, 20, 30, 40]


def test_whitespace_around_ids_is_ignored():
    votes = make_votes().assign(geoid=[" 01", "02 ", "03", "04"])
    assert reconcile(make_assignments(), voting=votes).data["dem"].notna().all()


def test_repeated_ids_in_a_data_file_keep_the_first():
    votes = pd.concat([make_votes(), make_votes().iloc[:1].assign(dem="999")])
    plan = reconcile(make_assignments(), voting=votes)
    assert plan.data["dem"].tolist() == [10, 20, 30, 40]
    assert "voting data: 1 repeated precinct ids, kept the first of each" in plan.notes


def test_column_already_in_the_plan_is_skipped():
    census = make_census().assign(dem=[0, 0, 0, 0])
    plan = reconcile(make_assignments(), voting=make_votes(), demographics=census)
    assert plan.data["dem"].tolist() == [10, 20, 30, 40]
    assert "demographic data: skipped columns already in the plan (dem)" in plan.notes


def test_shapes_from_any_file_make_it_a_geodataframe():
    plan = reconcile(make_assignments(), voting=make_votes(), shapes=make_shapes())
    assert isinstance(plan.data, gpd.GeoDataFrame)
    assert plan.data.crs == "EPSG:3857"
    assert plan.data.geometry.notna().all()


def test_shapes_in_the_assignments_file_are_kept():
    assignments = make_shapes().assign(District=[1, 1, 2, 2])
    plan = reconcile(assignments, voting=make_votes(), shapes=make_shapes())
    assert isinstance(plan.data, gpd.GeoDataFrame)
    assert plan.data.columns.tolist().count("geometry") == 1


def test_precincts_without_a_shape_are_reported():
    plan = reconcile(make_assignments(), shapes=make_shapes().iloc[:3])
    assert "shapes: 1 precincts have no shape" in plan.notes


def test_blank_ids_in_assignments_are_dropped():
    assignments = make_assignments()
    assignments.loc[3, "GEOID20"] = None
    plan = reconcile(assignments)
    assert len(plan.data) == 3
    assert plan.notes == ["district assignments: dropped 1 rows with no precinct id"]


def test_key_can_be_named_instead_of_guessed():
    assignments = make_assignments().rename(columns={"GEOID20": "code"})
    plan = reconcile(assignments, voting=make_votes(), keys={"assignments": "code"})
    assert plan.data["dem"].tolist() == [10, 20, 30, 40]


def test_no_id_column_raises():
    with pytest.raises(ReconcileError, match="couldn't find a precinct id column"):
        reconcile(pd.DataFrame({"a": [1], "b": [2]}))


def test_nothing_matching_raises():
    votes = make_votes().assign(geoid=["a", "b", "c", "d"])
    with pytest.raises(ReconcileError, match="none of its precinct ids match"):
        reconcile(make_assignments(), voting=votes)


def test_precinct_assigned_twice_raises():
    assignments = pd.concat([make_assignments(), make_assignments().iloc[:1]])
    with pytest.raises(ReconcileError, match="more than once"):
        reconcile(assignments)


def test_assignments_with_only_ids_raises():
    with pytest.raises(ReconcileError, match="no districts"):
        reconcile(make_assignments()[["GEOID20"]])


def test_inputs_are_not_changed():
    assignments, votes = make_assignments(), make_votes()
    reconcile(assignments, voting=votes)
    pd.testing.assert_frame_equal(assignments, make_assignments())
    pd.testing.assert_frame_equal(votes, make_votes())


@pytest.mark.parametrize("columns, expected", [
    (["name", "GEOID20", "id"], "GEOID20"),     # best name wins over a vaguer one
    (["Precinct ID", "votes"], "Precinct ID"),  # case and spaces don't matter
    (["vtd_geoid_2020", "votes"], "vtd_geoid_2020"),
])
def test_find_key(columns, expected):
    assert find_key(pd.DataFrame(columns=columns)) == expected


def test_clean_key():
    observed = clean_key(pd.Series([" 01 ", 2.0, None, ""])).tolist()
    assert observed[:2] == ["01", "2"]
    assert pd.isna(observed[2]) and pd.isna(observed[3])


def test_read_table_keeps_leading_zeros(tmp_path):
    path = tmp_path / "plan.csv"
    path.write_text("GEOID20;District\n01;1\n02;2\n", encoding="utf-8")
    assert read_table(path)["GEOID20"].tolist() == ["01", "02"]


def test_read_table_reads_spatial_files(tmp_path):
    path = tmp_path / "shapes.geojson"
    make_shapes().to_file(path, driver="GeoJSON")
    assert isinstance(read_table(path), gpd.GeoDataFrame)
