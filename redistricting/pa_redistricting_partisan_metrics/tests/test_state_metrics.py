import math

import numpy as np
import pandas as pd
import pytest

from pa_redistricting.demographics import COUNT_COLUMNS
from pa_redistricting.partisan_metrics import efficiency_gap
from pa_redistricting.state_metrics import plan_name, read_plan, score_plan, state_elections

IDS = [f"370010000{i}{i}" for i in range(6)]
STATE = pd.DataFrame({"GEOID": IDS, "COUNTYFP": ["001"] * 3 + ["003"] * 3,
                      **{c: [100] * 6 for c in COUNT_COLUMNS},
                      "ndv": [70, 60, 30, 40, 20, 10], "nrv": [30, 40, 70, 60, 80, 90],
                      "adv_20": [1] * 6, "arv_20": [1] * 6, "adv_16": [2] * 6, "arv_16": [1] * 6,
                      "pre_20_dem_bid": [5] * 6, "pre_20_rep_tru": [5] * 6})
PLAN = pd.DataFrame({"GEOID": IDS[:5], "district": [1, 1, 2, 2, 3]})   # last VTD unassigned


def test_state_elections_finds_composite_yearly_and_presidential():
    assert state_elections(STATE) == {
        "composite": ("ndv", "nrv"),
        "avg_2016": ("adv_16", "arv_16"),
        "avg_2020": ("adv_20", "arv_20"),
        "pres_2020": ("pre_20_dem_bid", "pre_20_rep_tru"),
    }


def test_state_elections_none():
    with pytest.raises(ValueError, match="no election columns"):
        state_elections(STATE[["GEOID", "pop_total"]])


def test_read_plan(tmp_path):
    path = tmp_path / "My Plan (v2).csv"
    path.write_text("﻿GEOID20,District\n" +
                    "".join(f"{g},{d}\n" for g, d in zip(IDS, [1, 1, 2, 2, 3, ""])),
                    encoding="utf-8")
    plan = read_plan(path)
    assert plan["GEOID"].tolist() == IDS[:5]
    assert plan["district"].tolist() == [1, 1, 2, 2, 3]
    assert plan_name(path) == "My_Plan__v2_"


def test_read_plan_lowercase_headers(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(f"geoid,district\n{IDS[0]},1\n")
    assert len(read_plan(path)) == 1


def test_read_plan_rejects_non_numbers(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(f"GEOID20,District\n{IDS[0]},A\n")
    with pytest.raises(ValueError, match="whole numbers"):
        read_plan(path)


def test_score_plan_matches_hand_calculation():
    score = score_plan(PLAN, STATE)
    m = score.metrics.set_index("election").loc["composite"]
    dem, rep = np.array([130.0, 70, 20]), np.array([70.0, 130, 80])
    assert m["n_districts"] == 3 and m["dem_seats"] == 1
    assert math.isclose(m["dem_vote_share"], 220 / 500)
    assert math.isclose(m["efficiency_gap"], efficiency_gap(dem, rep))
    assert score.districts["composite"]["winner"].tolist() == ["D", "R", "R"]
    assert score.population["pop_total"].tolist() == [200, 200, 100]
    assert score.population_stats["unassigned_pop"] == 100


def test_score_plan_ignores_geometry_column():
    score = score_plan(PLAN, STATE.assign(geometry="shape"))
    assert len(score.metrics) == 4


def test_score_plan_from_another_state():
    with pytest.raises(ValueError, match="None of the plan's GEOIDs"):
        score_plan(pd.DataFrame({"GEOID": ["42000000000"], "district": [1]}), STATE)
