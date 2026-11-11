"""
Tests for redistricting/responsiveness.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_responsiveness.py

Toy state: 8 precincts of 50 votes each, statewide 200 dem of 400 = 50%.

    plan_a district dem shares: 0.30, 0.49, 0.51, 0.70  (two close districts)
    plan_b district dem shares: 0.30, 0.35, 0.65, 0.70  (all safe districts)

Expected values are worked out by hand in each test.
"""

import pandas as pd
import pytest

from redistricting.responsiveness import responsiveness, score_plan

TOLERANCE = 1e-9


def make_votes():
    """Toy precinct table with two plans."""
    return pd.DataFrame({
        "precinct_id": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "dem":    [10, 20, 25, 24, 10, 41, 41, 29],
        "rep":    [40, 30, 25, 26, 40,  9,  9, 21],
        "plan_a": [1, 1, 2, 2, 3, 3, 4, 4],
        "plan_b": [1, 2, 1, 3, 2, 4, 3, 4],
    })


def score(result, plan):
    """Responsiveness for one plan from the output table."""
    return result.loc[result["plan"] == plan, "responsiveness"].iloc[0]


# ---------------------------------------------------------------------------
# Metric values
# ---------------------------------------------------------------------------

def test_close_districts_are_responsive():
    # plan_a at 47.5%: 0.275 0.465 0.485 0.675 -> 1 of 4 seats (0.25)
    #        at 52.5%: 0.325 0.515 0.535 0.725 -> 3 of 4 seats (0.75)
    #        (0.75 - 0.25) / 0.05 = 10
    observed = score(responsiveness(make_votes()), "plan_a")
    expected = 10.0
    assert abs(observed - expected) < TOLERANCE


def test_safe_districts_score_zero():
    # plan_b wins 2 of 4 at both 47.5% and 52.5%
    observed = score(responsiveness(make_votes()), "plan_b")
    expected = 0.0
    assert abs(observed - expected) < TOLERANCE


def test_wider_window():
    # plan_a at 45%: 0.25 0.44 0.46 0.65 -> 1 of 4 seats (0.25)
    #        at 55%: 0.35 0.54 0.56 0.75 -> 3 of 4 seats (0.75)
    #        (0.75 - 0.25) / 0.10 = 5
    observed = score(responsiveness(make_votes(), window=0.05), "plan_a")
    expected = 5.0
    assert abs(observed - expected) < TOLERANCE


def test_tie_counts_as_republican_win():
    # 3 districts of 400 votes: dem shares 0.6, 0.4, 0.5375; statewide 0.5125
    # at 47.5% (shift -0.0375): 0.5625 0.3625 0.5 -> tie goes R -> 1 of 3
    # at 52.5% (shift +0.0125): 0.6125 0.4125 0.55 -> 2 of 3
    # (2/3 - 1/3) / 0.05 = 6.667; if ties went D it would be 0
    votes = pd.DataFrame({
        "precinct_id": ["01", "02", "03"],
        "dem":  [240, 160, 215],
        "rep":  [160, 240, 185],
        "plan": [1, 2, 3],
    })
    observed = score(responsiveness(votes), "plan")
    expected = (2 / 3 - 1 / 3) / 0.05
    assert abs(observed - expected) < TOLERANCE


# ---------------------------------------------------------------------------
# Output shape and input handling
# ---------------------------------------------------------------------------

def test_one_row_per_plan():
    observed = responsiveness(make_votes())["plan"].tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_input_not_changed():
    votes = make_votes()
    responsiveness(votes)
    assert votes.equals(make_votes())


@pytest.mark.parametrize("window", [0, -0.1, 0.5, 1])
def test_bad_window_raises(window):
    with pytest.raises(ValueError):
        responsiveness(make_votes(), window=window)


# ---------------------------------------------------------------------------
# exclude
# ---------------------------------------------------------------------------

def test_exclude_list_leaves_out_columns():
    votes = make_votes()
    votes["county"] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = responsiveness(votes, exclude=["county", "area"])["plan"].tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_single_string():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = responsiveness(votes, exclude="area")["plan"].tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_does_not_change_scores():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    result = responsiveness(votes, exclude="area")
    observed = [score(result, "plan_a"), score(result, "plan_b")]
    expected = [10.0, 0.0]
    assert all(abs(o - e) < TOLERANCE for o, e in zip(observed, expected))


def test_exclude_unknown_column_raises():
    with pytest.raises(ValueError):
        responsiveness(make_votes(), exclude=["countey"])


def test_without_exclude_extra_column_is_scored_as_plan():
    # documents why exclude exists: an unlisted column is treated as a plan
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = responsiveness(votes)["plan"].tolist()
    expected = ["plan_a", "plan_b", "area"]
    assert observed == expected


# ---------------------------------------------------------------------------
# Input checks and new options
# ---------------------------------------------------------------------------

def test_exclude_can_list_id_column():
    observed = responsiveness(make_votes(), exclude=["precinct_id"])["plan"].unique().tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_can_list_geometry_column():
    votes = make_votes()
    votes["geometry"] = None
    observed = responsiveness(votes, exclude=["geometry"])["plan"].unique().tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_other_vote_column_names_give_same_result():
    votes = make_votes().rename(columns={"dem": "D_votes", "rep": "R_votes"})
    observed = responsiveness(votes, dem="D_votes", rep="R_votes")
    expected = responsiveness(make_votes())
    assert observed.equals(expected)


def test_missing_vote_column_raises():
    votes = make_votes().rename(columns={"dem": "D_votes"})
    with pytest.raises(ValueError, match="dem"):
        responsiveness(votes)


def test_chosen_name_clashing_with_existing_column_raises():
    votes = make_votes()
    votes["D_votes"] = votes["dem"]
    with pytest.raises(ValueError, match="already has"):
        responsiveness(votes, dem="D_votes")


def test_blank_vote_cell_raises():
    votes = make_votes().astype({"dem": float})
    votes.loc[0, "dem"] = float("nan")
    with pytest.raises(ValueError, match="blank dem"):
        responsiveness(votes)


def test_plan_with_no_districts_raises():
    votes = make_votes()
    votes["plan_blank"] = float("nan")
    with pytest.raises(ValueError, match="no district has any votes"), pytest.warns(UserWarning, match="no district"):
        responsiveness(votes)


def test_score_plan():
    # dem shares 0.6, 0.4, 0.5375; statewide 0.5125
    # at 47.5%: 0.5625 0.3625 0.5 -> tie goes R -> 1 of 3
    # at 52.5%: 0.6125 0.4125 0.55 -> 2 of 3
    # (2/3 - 1/3) / 0.05 = 6.667
    observed = score_plan([240, 160, 215], [160, 240, 185], [1, 2, 3])
    expected = {"responsiveness": 20 / 3}
    assert observed == pytest.approx(expected, abs=TOLERANCE)
