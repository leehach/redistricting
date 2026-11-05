"""
Tests for redistricting/seats_votes.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_seats_votes_curve.py

Toy state: 8 precincts of 50 votes each, statewide 200 dem of 400 = 50%.

    plan_a district dem shares: 0.30, 0.49, 0.51, 0.70  (two close districts)
    plan_b district dem shares: 0.30, 0.35, 0.65, 0.70  (all safe districts)

Expected values are worked out by hand in each test.
"""

import pandas as pd
import pytest

from redistricting.seats_votes import plot_seats_votes, seats_votes_app, seats_votes_curve

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


def row_at(curve, plan, vote_share):
    """The curve row for one plan at one statewide vote share."""
    match = (curve["plan"] == plan) & ((curve["dem_vote_share"] - vote_share).abs() < TOLERANCE)
    return curve[match].iloc[0]


# ---------------------------------------------------------------------------
# Curve values
# ---------------------------------------------------------------------------

def test_close_districts_seats():
    # plan_a at 45%: 0.25 0.44 0.46 0.65 -> 1 seat
    #        at 50%: 0.30 0.49 0.51 0.70 -> 2 seats
    #        at 55%: 0.35 0.54 0.56 0.75 -> 3 seats
    curve = seats_votes_curve(make_votes())
    observed = [row_at(curve, "plan_a", v)["dem_seats"] for v in (0.45, 0.5, 0.55)]
    expected = [1, 2, 3]
    assert observed == expected


def test_safe_districts_seats():
    # plan_b at 45%: 0.25 0.30 0.60 0.65 -> 2 seats
    #        at 55%: 0.35 0.40 0.70 0.75 -> 2 seats
    curve = seats_votes_curve(make_votes())
    observed = [row_at(curve, "plan_b", v)["dem_seats"] for v in (0.45, 0.5, 0.55)]
    expected = [2, 2, 2]
    assert observed == expected


def test_seat_share_is_seats_over_districts():
    # plan_a at 55%: 3 of 4 districts
    curve = seats_votes_curve(make_votes())
    observed = row_at(curve, "plan_a", 0.55)["dem_seat_share"]
    expected = 0.75
    assert abs(observed - expected) < TOLERANCE


def test_tie_counts_as_republican_win():
    # 2 districts of 400 votes: dem shares 0.45, 0.55; statewide 0.50
    # at 45% (shift -0.05): 0.40 0.50 -> tie goes R -> 0 seats
    # at 55% (shift +0.05): 0.50 0.60 -> tie goes R -> 1 seat
    votes = pd.DataFrame({
        "precinct_id": ["01", "02"],
        "dem":  [180, 220],
        "rep":  [220, 180],
        "plan": [1, 2],
    })
    curve = seats_votes_curve(votes, low=0.45, high=0.55, step=0.05)
    observed = curve["dem_seats"].tolist()
    expected = [0, 1, 1]
    assert observed == expected


def test_seats_never_fall_as_votes_rise():
    curve = seats_votes_curve(make_votes())
    for plan, rows in curve.groupby("plan"):
        assert (rows["dem_seats"].diff().dropna() >= 0).all(), plan


# ---------------------------------------------------------------------------
# Output shape and input handling
# ---------------------------------------------------------------------------

def test_columns():
    observed = seats_votes_curve(make_votes()).columns.tolist()
    expected = ["plan", "dem_vote_share", "dem_seats", "dem_seat_share"]
    assert observed == expected


def test_default_range():
    # 0.35 to 0.65 in steps of 0.005 -> 61 vote shares per plan
    curve = seats_votes_curve(make_votes())
    rows = curve[curve["plan"] == "plan_a"]
    assert len(rows) == 61
    assert abs(rows["dem_vote_share"].iloc[0] - 0.35) < TOLERANCE
    assert abs(rows["dem_vote_share"].iloc[-1] - 0.65) < TOLERANCE


def test_custom_range():
    curve = seats_votes_curve(make_votes(), low=0.4, high=0.6, step=0.1)
    observed = curve.loc[curve["plan"] == "plan_a", "dem_vote_share"].tolist()
    expected = [0.4, 0.5, 0.6]
    assert all(abs(o - e) < TOLERANCE for o, e in zip(observed, expected))
    assert len(observed) == len(expected)


def test_input_not_changed():
    votes = make_votes()
    seats_votes_curve(votes)
    assert votes.equals(make_votes())


@pytest.mark.parametrize("low, high, step", [
    (0.6, 0.4, 0.005),   # low above high
    (0.5, 0.5, 0.005),   # low equal to high
    (-0.1, 0.5, 0.005),  # low below 0
    (0.5, 1.1, 0.005),   # high above 1
    (0.4, 0.6, 0),       # step of 0
    (0.4, 0.6, -0.01),   # negative step
])
def test_bad_range_raises(low, high, step):
    with pytest.raises(ValueError):
        seats_votes_curve(make_votes(), low=low, high=high, step=step)


# ---------------------------------------------------------------------------
# exclude
# ---------------------------------------------------------------------------

def test_exclude_list_leaves_out_columns():
    votes = make_votes()
    votes["county"] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    curve = seats_votes_curve(votes, exclude=["county", "area"])
    observed = curve["plan"].unique().tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_single_string():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    curve = seats_votes_curve(votes, exclude="area")
    observed = curve["plan"].unique().tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_does_not_change_curve():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = seats_votes_curve(votes, exclude="area")
    expected = seats_votes_curve(make_votes())
    assert observed.equals(expected)


def test_exclude_unknown_column_raises():
    with pytest.raises(ValueError):
        seats_votes_curve(make_votes(), exclude=["countey"])


def test_without_exclude_extra_column_is_scored_as_plan():
    # documents why exclude exists: an unlisted column is treated as a plan
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = seats_votes_curve(votes)["plan"].unique().tolist()
    expected = ["plan_a", "plan_b", "area"]
    assert observed == expected


# ---------------------------------------------------------------------------
# Input checks and new options
# ---------------------------------------------------------------------------

def test_plot_seats_votes_draws_every_plan_by_default():
    pytest.importorskip("plotly")
    fig = plot_seats_votes(seats_votes_curve(make_votes()))
    observed = [trace.name for trace in fig.data]
    expected = ["proportional", "plan_a", "plan_b"]
    assert observed == expected


def test_plot_seats_votes_keeps_plan_color_when_others_are_hidden():
    pytest.importorskip("plotly")
    curve = seats_votes_curve(make_votes())
    observed = plot_seats_votes(curve, ["plan_b"]).data[1].line.color
    expected = plot_seats_votes(curve).data[2].line.color
    assert observed == expected


def test_seats_votes_app_checklist_lists_every_plan():
    pytest.importorskip("dash")
    app = seats_votes_app(seats_votes_curve(make_votes()))
    observed = [app.layout["plans"].options, app.layout["plans"].value]
    expected = [["plan_a", "plan_b"], ["plan_a", "plan_b"]]
    assert observed == expected


def test_seats_votes_app_callback_draws_ticked_plans():
    pytest.importorskip("dash")
    app = seats_votes_app(seats_votes_curve(make_votes()))
    update = app.callback_map["curve.figure"]["callback"].__wrapped__
    observed = [trace.name for trace in update(["plan_a"]).data]
    expected = ["proportional", "plan_a"]
    assert observed == expected


def test_seats_votes_app_callback_handles_no_plans_ticked():
    pytest.importorskip("dash")
    app = seats_votes_app(seats_votes_curve(make_votes()))
    update = app.callback_map["curve.figure"]["callback"].__wrapped__
    observed = [trace.name for trace in update([]).data]
    expected = ["proportional"]
    assert observed == expected


def test_exclude_can_list_id_column():
    observed = seats_votes_curve(make_votes(), exclude=["precinct_id"])["plan"].unique().tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_exclude_can_list_geometry_column():
    votes = make_votes()
    votes["geometry"] = None
    observed = seats_votes_curve(votes, exclude=["geometry"])["plan"].unique().tolist()
    expected = ["plan_a", "plan_b"]
    assert observed == expected


def test_other_vote_column_names_give_same_result():
    votes = make_votes().rename(columns={"dem": "D_votes", "rep": "R_votes"})
    observed = seats_votes_curve(votes, dem="D_votes", rep="R_votes")
    expected = seats_votes_curve(make_votes())
    assert observed.equals(expected)


def test_missing_vote_column_raises():
    votes = make_votes().rename(columns={"dem": "D_votes"})
    with pytest.raises(ValueError, match="dem"):
        seats_votes_curve(votes)


def test_chosen_name_clashing_with_existing_column_raises():
    votes = make_votes()
    votes["D_votes"] = votes["dem"]
    with pytest.raises(ValueError, match="already has"):
        seats_votes_curve(votes, dem="D_votes")


def test_blank_vote_cell_raises():
    votes = make_votes().astype({"dem": float})
    votes.loc[0, "dem"] = float("nan")
    with pytest.raises(ValueError, match="blank dem"):
        seats_votes_curve(votes)


def test_plan_with_no_districts_raises():
    votes = make_votes()
    votes["plan_blank"] = float("nan")
    with pytest.raises(ValueError, match="no district has any votes"), pytest.warns(UserWarning, match="no district"):
        seats_votes_curve(votes)


@pytest.mark.parametrize("low, high, step", [
    (0.4, 0.6, 0.03),  # does not fit the range evenly
    (0.4, 0.6, 0.5),   # bigger than the range
])
def test_step_that_does_not_fit_range_raises(low, high, step):
    with pytest.raises(ValueError, match="evenly"):
        seats_votes_curve(make_votes(), low=low, high=high, step=step)
