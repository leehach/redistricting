import pandas as pd
import pytest

from pa_redistricting.population import population_summary


def two_district_plan():
    """Four voting districts: three in Senate districts 1 and 2, one unassigned."""
    return pd.DataFrame({
        "district":     pd.array([1, 1, 2, None], dtype="Int64"),
        "COUNTYFP":     ["001", "003", "001", "005"],
        "pop_total":    [100, 50, 140, 10],
        "pop_hispanic": [10, 20, 14, 0],
        "pop_nh_white": [80, 20, 100, 10],
        "pop_nh_black": [5, 20, 20, 0],
        "vap_total":    [80, 40, 100, 8],
        "vap_hispanic": [8, 16, 10, 0],
        "vap_nh_white": [40, 10, 70, 8],
        "vap_nh_black": [4, 20, 15, 0],
    })


def test_ideal_counts_unassigned_people():
    _, stats = population_summary(two_district_plan(), n_districts=2)
    assert stats["state_pop"] == 300
    assert stats["ideal"] == 150
    assert stats["unassigned_pop"] == 10


def test_deviation_from_ideal():
    out, _ = population_summary(two_district_plan(), n_districts=2)
    assert out["pop_total"].tolist() == [150, 140]
    assert out["deviation"].tolist() == [0, -10]
    assert out["pct_deviation"].tolist() == pytest.approx([0.0, -6.67])


def test_shares_come_from_summed_counts():
    out, _ = population_summary(two_district_plan(), n_districts=2)
    # district 1: (10 + 20) / (100 + 50), not the average of 10% and 40%
    assert out.loc[0, "pct_hispanic"] == 20.0


def test_counts_vtds_and_counties():
    out, _ = population_summary(two_district_plan(), n_districts=2)
    assert out["n_vtds"].tolist() == [2, 1]
    assert out["n_counties"].tolist() == [2, 1]


def test_overall_stats():
    _, stats = population_summary(two_district_plan(), n_districts=2)
    assert stats["districts"] == 2
    assert stats["largest"]["district"] == 1
    assert stats["smallest"]["district"] == 2
    assert stats["overall_range"] == pytest.approx(6.67)
    assert stats["majority_minority_vap"] == 1   # district 1: only 50 of 120 adults are NH white
