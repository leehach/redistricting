import math

import numpy as np
import pandas as pd
import pytest

from pa_redistricting.partisan_metrics import (declination, efficiency_gap, favors, mean_median,
                                               partisan_bias, partisan_summary)

# Three districts of 100 voters. Democrats are packed into district 1 and lose
# districts 2 and 3, so every metric should favor Republicans (positive).
DEM = np.array([75.0, 40.0, 40.0])
REP = np.array([25.0, 60.0, 60.0])
SHARE = DEM / (DEM + REP)


def test_efficiency_gap_worked_by_hand():
    # Dem wasted: 25 (above 50 in district 1) + 40 + 40 = 105
    # Rep wasted: 25 + 10 + 10 = 45  ->  (105 - 45) / 300 = 0.2
    assert efficiency_gap(DEM, REP) == pytest.approx(0.2)


def test_efficiency_gap_is_zero_for_a_mirror_image_plan():
    assert efficiency_gap(np.array([70.0, 30.0]), np.array([30.0, 70.0])) == pytest.approx(0)


def test_tie_counts_as_republican_win():
    # 50-50: the Republican "wins", so Dem waste 50 and Rep waste 0
    assert efficiency_gap(np.array([50.0]), np.array([50.0])) == pytest.approx(0.5)


def test_partisan_bias_worked_by_hand():
    # Statewide Dem share 155/300; shifting to 50-50 leaves Dems 1 of 3 seats
    assert partisan_bias(SHARE, 155 / 300) == pytest.approx(0.5 - 1 / 3)


def test_mean_median_worked_by_hand():
    assert mean_median(SHARE) == pytest.approx(155 / 300 - 0.4)


def test_declination_favors_republicans_when_democrats_are_packed():
    expected = 2 * (math.atan(0.25 / (1 / 6)) - math.atan(0.1 / (2 / 6))) / math.pi
    assert declination(SHARE) == pytest.approx(expected)
    assert declination(SHARE) > 0


def test_declination_is_zero_for_a_mirror_image_plan():
    assert declination(np.array([0.3, 0.7])) == pytest.approx(0)


def test_declination_is_nan_when_one_party_wins_every_seat():
    assert math.isnan(declination(np.array([0.6, 0.7])))


@pytest.mark.parametrize("value, party", [(0.1, "Republicans"), (-0.1, "Democrats"),
                                          (0.0, "balanced"), (float("nan"), "n/a")])
def test_favors(value, party):
    assert favors(value) == party


# ---------------------------------------------------------------------------
# partisan_summary: joining a plan to votes
# ---------------------------------------------------------------------------
def votes():
    return pd.DataFrame({
        "GEOID": ["42001000010", "42001000020", "42001000030", "4200300F437"],
        "ndv": [75, 20, 20, 999], "nrv": [25, 30, 30, 1],
        "pre_20_dem_bid": [60, 30, 30, 0], "pre_20_rep_tru": [40, 70, 70, 0],
    })


def plan():
    """The last voting district is left unassigned."""
    return pd.DataFrame({"GEOID": ["42001000010", "42001000020", "42001000030"],
                         "district": [1, 2, 2]})


ELECTIONS = {"composite": ("ndv", "nrv"), "pres_2020": ("pre_20_dem_bid", "pre_20_rep_tru")}


def test_summary_has_one_row_per_election():
    summary, _ = partisan_summary(plan(), votes(), ELECTIONS)
    assert summary["election"].tolist() == ["composite", "pres_2020"]
    assert summary.loc[0, "n_districts"] == 2
    assert summary.loc[0, "dem_seats"] == 1
    assert summary.loc[0, "efficiency_gap_favors"] == "Republicans"


def test_district_votes_are_summed_and_unassigned_votes_left_out():
    _, districts = partisan_summary(plan(), votes(), ELECTIONS)
    table = districts["composite"]
    assert table["district"].tolist() == [1, 2]
    assert table["dem"].tolist() == [75, 40]        # the unassigned 999 is not counted
    assert table["rep"].tolist() == [25, 60]
    assert table["winner"].tolist() == ["D", "R"]


def test_works_with_a_plan_from_plans_get():
    # Plans.get gives a nullable Int64 district with blanks for unassigned VTDs
    df = pd.DataFrame({"GEOID": votes()["GEOID"],
                       "district": pd.array([1, 2, 2, None], dtype="Int64")})
    summary, _ = partisan_summary(df, votes(), ELECTIONS)
    assert summary.loc[0, "dem_vote_share"] == pytest.approx(115 / 200)


def test_elections_default_to_those_in_the_file():
    summary, _ = partisan_summary(plan(), votes())
    assert summary["election"].tolist() == ["composite", "pres_2020"]
