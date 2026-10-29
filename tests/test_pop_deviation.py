import numpy as np
import pandas as pd
import pytest

from pop_deviation import population_deviation


@pytest.fixture
def precincts():
    # 4 precincts of 100 people each -> total 400
    return pd.DataFrame({
        "GEOID20": ["001", "002", "003", "004"],
        "P0010001": [100, 100, 100, 100],
    })


def make_plan(assignments):
    return pd.DataFrame({
        "GEOID20": list(assignments.keys()),
        "District": list(assignments.values()),
    })


# --- correct results ---------------------------------------------------------

def test_balanced_plan_has_zero_deviation(precincts):
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2})
    result = population_deviation(precincts, plan)

    assert result.attrs["ideal"] == 200
    assert list(result["Population"]) == [200, 200]
    assert list(result["Deviation"]) == [0, 0]
    assert list(result["Deviation %"]) == [0, 0]


def test_unbalanced_plan(precincts):
    plan = make_plan({"001": 1, "002": 1, "003": 1, "004": 2})
    result = population_deviation(precincts, plan)

    assert result.attrs["ideal"] == 200
    assert result.loc[1, "Population"] == 300
    assert result.loc[2, "Population"] == 100
    assert result.loc[1, "Deviation"] == 100
    assert result.loc[2, "Deviation"] == -100
    assert result.loc[1, "Deviation %"] == 50
    assert result.loc[2, "Deviation %"] == -50


def test_deviation_rounding():
    precincts = pd.DataFrame({"GEOID20": ["a", "b", "c"], "P0010001": [1, 1, 1]})
    plan = make_plan({"a": 1, "b": 1, "c": 2})
    result = population_deviation(precincts, plan)

    assert result.attrs["ideal"] == pytest.approx(1.5)
    assert result.loc[1, "Deviation %"] == 33.33
    assert result.loc[2, "Deviation %"] == -33.33


def test_custom_column_names():
    precincts = pd.DataFrame({"id": ["a", "b"], "pop": [50, 150]})
    plan = pd.DataFrame({"id": ["a", "b"], "dist": ["X", "Y"]})
    result = population_deviation(precincts, plan, geoid_col="id",
                                  pop_col="pop", district_col="dist")

    assert result.loc["X", "Deviation %"] == -50
    assert result.loc["Y", "Deviation %"] == 50


def test_float_population_column(precincts):
    # shapefiles often read population as float
    precincts["P0010001"] = precincts["P0010001"].astype(float)
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2})
    result = population_deviation(precincts, plan)

    assert result["Population"].dtype == int
    assert list(result["Deviation %"]) == [0, 0]


def test_num_districts_matches(precincts):
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2})
    result = population_deviation(precincts, plan, num_districts=2)
    assert result.attrs["ideal"] == 200


# --- invalid plans raise instead of giving wrong numbers ---------------------

def test_precinct_missing_from_plan_raises(precincts):
    # Previously gave +33% / -33% with no warning
    plan = make_plan({"001": 1, "002": 1, "003": 2})
    with pytest.raises(ValueError, match="not assigned in plan.*004"):
        population_deviation(precincts, plan)


def test_unknown_precinct_in_plan_raises(precincts):
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2, "999": 2})
    with pytest.raises(ValueError, match="not found in precincts.*999"):
        population_deviation(precincts, plan)


def test_missing_population_raises(precincts):
    precincts.loc[3, "P0010001"] = np.nan
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2})
    with pytest.raises(ValueError, match="no population.*004"):
        population_deviation(precincts, plan)


def test_precinct_assigned_twice_raises(precincts):
    plan = pd.DataFrame({
        "GEOID20": ["001", "002", "003", "004", "004"],
        "District": [1, 1, 2, 2, 1],
    })
    with pytest.raises(ValueError, match="assigned more than once.*004"):
        population_deviation(precincts, plan)


def test_duplicate_precinct_ids_raise(precincts):
    precincts = pd.concat([precincts, precincts.iloc[[0]]], ignore_index=True)
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2})
    with pytest.raises(ValueError, match="duplicate precinct ID.*001"):
        population_deviation(precincts, plan)


def test_precinct_with_no_district_raises(precincts):
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": None})
    with pytest.raises(ValueError, match="no district"):
        population_deviation(precincts, plan)


def test_wrong_number_of_districts_raises(precincts):
    # A plan that leaves district 3 empty would otherwise compute ideal = 400/2
    plan = make_plan({"001": 1, "002": 1, "003": 2, "004": 2})
    with pytest.raises(ValueError, match="2 districts, expected 3"):
        population_deviation(precincts, plan, num_districts=3)


def test_geoid_type_mismatch_is_caught(precincts):
    # e.g. a plan CSV read without dtype=str, so GEOIDs become ints
    plan = make_plan({1: 1, 2: 1, 3: 2, 4: 2})
    with pytest.raises(ValueError, match="not assigned in plan"):
        population_deviation(precincts, plan)
