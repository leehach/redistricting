"""
Tests for redistricting/seats_votes.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_seats_votes_curve.py

Input is VTD data in the layout of the ALARM Project's 2020 files
(https://alarm-redist.org/posts/2021-08-10-census-2020/), with plan columns
added. Votes come from the 2020 presidential race (``pre_20_dem_bid`` and
``pre_20_rep_tru``); the table also has ALARM's other columns, which must not
be scored as plans.

Toy state: 8 VTDs of 50 votes each, statewide 200 dem of 400 = 50%.

    plan_a district dem shares: 0.30, 0.49, 0.51, 0.70  (two close districts)
    plan_b district dem shares: 0.30, 0.35, 0.65, 0.70  (all safe districts)

Expected values are worked out by hand in each test.

The real-data section uses ALARM's Pennsylvania VTD file, downloaded once and
cached; those tests skip when offline.

The chart tests need PyQt5 and pyqtgraph and skip without them. They run
without showing a window (Qt's "offscreen" platform), as in the Qt DEA
explorer tests.
"""

import os

# must be set before Qt starts: draw into memory instead of on screen
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from redistricting.seats_votes import (plot_seats_votes, score_plan, seats_votes_curve,
                                       seats_votes_window)

PA_VTD_URL = ("https://raw.githubusercontent.com/alarm-redist/census-2020/main/"
              "census-vest-2020/pa_2020_vtd.csv")


@pytest.fixture(scope="session")
def pa_vtd_path(request):
    """Path to a local copy of ALARM's pa_2020_vtd.csv (about 1.8 MB), downloaded if needed."""
    cache = getattr(request.config, "cache", None)
    folder = Path(cache.mkdir("alarm_vtd") if cache else Path(tempfile.gettempdir()) / "alarm_vtd")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "pa_2020_vtd.csv"
    if not path.is_file():
        partial = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(PA_VTD_URL, partial)
        except (urllib.error.URLError, OSError) as e:
            pytest.skip(f"could not download the ALARM PA VTD file: {e}")
        partial.replace(path)  # only a finished download is kept
    return path


@pytest.fixture
def pa_vtd(pa_vtd_path):
    """A fresh copy of the PA VTD table for each test; ids stay text."""
    return pd.read_csv(pa_vtd_path, dtype={"GEOID20": str, "state": str, "county": str, "vtd": str})


TOLERANCE = 1e-9
DEM, REP = "pre_20_dem_bid", "pre_20_rep_tru"
VOTES = {"dem": DEM, "rep": REP}   # which ALARM race the curve uses
PLANS = ["plan_a", "plan_b"]


def make_votes():
    """Toy VTD table in the ALARM layout with two plans."""
    return pd.DataFrame({
        "GEOID20":  [f"42001{i:06d}" for i in range(10, 90, 10)],
        "state":    ["PA"] * 8,
        "county":   ["Adams County"] * 4 + ["Bucks County"] * 4,
        "vtd":      [f"{i:06d}" for i in range(10, 90, 10)],
        "pop":      [120, 130, 110, 125, 140, 115, 105, 135],
        "vap":      [90, 100, 85, 95, 105, 88, 80, 102],
        DEM:        [10, 20, 25, 24, 10, 41, 41, 29],
        REP:        [40, 30, 25, 26, 40,  9,  9, 21],
        "pre_16_dem_cli": [12, 18, 24, 22, 11, 40, 39, 27],
        "pre_16_rep_tru": [38, 32, 26, 28, 39, 10, 11, 23],
        "ndv":      [11.0, 19.0, 24.5, 23.0, 10.5, 40.5, 40.0, 28.0],
        "nrv":      [39.0, 31.0, 25.5, 27.0, 39.5, 9.5, 10.0, 22.0],
        "plan_a":   [1, 1, 2, 2, 3, 3, 4, 4],
        "plan_b":   [1, 2, 1, 3, 2, 4, 3, 4],
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
    curve = seats_votes_curve(make_votes(), **VOTES)
    observed = [row_at(curve, "plan_a", v)["dem_seats"] for v in (0.45, 0.5, 0.55)]
    expected = [1, 2, 3]
    assert observed == expected


def test_safe_districts_seats():
    # plan_b at 45%: 0.25 0.30 0.60 0.65 -> 2 seats
    #        at 55%: 0.35 0.40 0.70 0.75 -> 2 seats
    curve = seats_votes_curve(make_votes(), **VOTES)
    observed = [row_at(curve, "plan_b", v)["dem_seats"] for v in (0.45, 0.5, 0.55)]
    expected = [2, 2, 2]
    assert observed == expected


def test_seat_share_is_seats_over_districts():
    # plan_a at 55%: 3 of 4 districts
    curve = seats_votes_curve(make_votes(), **VOTES)
    observed = row_at(curve, "plan_a", 0.55)["dem_seat_share"]
    expected = 0.75
    assert abs(observed - expected) < TOLERANCE


def test_tie_counts_as_republican_win():
    # 2 districts of 400 votes: dem shares 0.45, 0.55; statewide 0.50
    # at 45% (shift -0.05): 0.40 0.50 -> tie goes R -> 0 seats
    # at 55% (shift +0.05): 0.50 0.60 -> tie goes R -> 1 seat
    votes = pd.DataFrame({
        "GEOID20": ["42001000010", "42001000020"],
        DEM:       [180, 220],
        REP:       [220, 180],
        "plan":    [1, 2],
    })
    curve = seats_votes_curve(votes, low=0.45, high=0.55, step=0.05, **VOTES)
    observed = curve["dem_seats"].tolist()
    expected = [0, 1, 1]
    assert observed == expected


def test_seats_never_fall_as_votes_rise():
    curve = seats_votes_curve(make_votes(), **VOTES)
    for plan, rows in curve.groupby("plan"):
        assert (rows["dem_seats"].diff().dropna() >= 0).all(), plan


def test_other_alarm_race_gives_its_own_curve():
    # 2016 race: plan_a district dem shares 0.30, 0.46, 0.51, 0.66; statewide 193 / 400
    # at 50% (shift +0.0175): 0.3175 0.4775 0.5275 0.6775 -> 2 seats
    curve = seats_votes_curve(make_votes(), dem="pre_16_dem_cli", rep="pre_16_rep_tru")
    observed = row_at(curve, "plan_a", 0.5)["dem_seats"]
    expected = 2
    assert observed == expected


# ---------------------------------------------------------------------------
# Output shape and input handling
# ---------------------------------------------------------------------------

def test_columns():
    observed = seats_votes_curve(make_votes(), **VOTES).columns.tolist()
    expected = ["plan", "dem_vote_share", "dem_seats", "dem_seat_share"]
    assert observed == expected


def test_alarm_columns_are_not_scored_as_plans():
    # state, county, vtd, pop, vap, the other race and the averages are all ALARM data
    observed = seats_votes_curve(make_votes(), **VOTES)["plan"].unique().tolist()
    expected = PLANS
    assert observed == expected


def test_alarm_columns_do_not_change_curve():
    votes = make_votes()
    bare = votes[["GEOID20", DEM, REP, *PLANS]]
    observed = seats_votes_curve(votes, **VOTES)
    expected = seats_votes_curve(bare, **VOTES)
    assert observed.equals(expected)


def test_default_range():
    # 0.35 to 0.65 in steps of 0.005 -> 61 vote shares per plan
    curve = seats_votes_curve(make_votes(), **VOTES)
    rows = curve[curve["plan"] == "plan_a"]
    assert len(rows) == 61
    assert abs(rows["dem_vote_share"].iloc[0] - 0.35) < TOLERANCE
    assert abs(rows["dem_vote_share"].iloc[-1] - 0.65) < TOLERANCE


def test_custom_range():
    curve = seats_votes_curve(make_votes(), low=0.4, high=0.6, step=0.1, **VOTES)
    observed = curve.loc[curve["plan"] == "plan_a", "dem_vote_share"].tolist()
    expected = [0.4, 0.5, 0.6]
    assert all(abs(o - e) < TOLERANCE for o, e in zip(observed, expected))
    assert len(observed) == len(expected)


def test_input_not_changed():
    votes = make_votes()
    seats_votes_curve(votes, **VOTES)
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
        seats_votes_curve(make_votes(), low=low, high=high, step=step, **VOTES)


# ---------------------------------------------------------------------------
# exclude
# ---------------------------------------------------------------------------

def test_exclude_list_leaves_out_columns():
    votes = make_votes()
    votes["district_name"] = ["N", "N", "S", "S", "E", "E", "W", "W"]
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    curve = seats_votes_curve(votes, exclude=["district_name", "area"], **VOTES)
    observed = curve["plan"].unique().tolist()
    expected = PLANS
    assert observed == expected


def test_exclude_single_string():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    curve = seats_votes_curve(votes, exclude="area", **VOTES)
    observed = curve["plan"].unique().tolist()
    expected = PLANS
    assert observed == expected


def test_exclude_does_not_change_curve():
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = seats_votes_curve(votes, exclude="area", **VOTES)
    expected = seats_votes_curve(make_votes(), **VOTES)
    assert observed.equals(expected)


def test_exclude_unknown_column_raises():
    with pytest.raises(ValueError):
        seats_votes_curve(make_votes(), exclude=["countey"], **VOTES)


def test_without_exclude_extra_column_is_scored_as_plan():
    # documents why exclude exists: a column that is not ALARM data is treated as a plan
    votes = make_votes()
    votes["area"] = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5]
    observed = seats_votes_curve(votes, **VOTES)["plan"].unique().tolist()
    expected = [*PLANS, "area"]
    assert observed == expected


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def needs_qt():
    """Skip the test if PyQt5 or pyqtgraph is not installed."""
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")


def line_names(widget):
    """Names of the named lines on a chart, bottom to top (the legend entries)."""
    return [item.name() for item in widget.getPlotItem().listDataItems() if item.name()]


def line_color(widget, name):
    """Color of the named line on a chart, as a hex string."""
    for item in widget.getPlotItem().listDataItems():
        if item.name() == name:
            return item.opts["pen"].color().name()
    raise KeyError(name)


def test_plot_seats_votes_draws_every_plan_by_default():
    needs_qt()
    widget = plot_seats_votes(seats_votes_curve(make_votes(), **VOTES))
    observed = line_names(widget)
    expected = ["proportional", *PLANS]
    assert observed == expected


def test_plot_seats_votes_keeps_plan_color_when_others_are_hidden():
    needs_qt()
    curve = seats_votes_curve(make_votes(), **VOTES)
    observed = line_color(plot_seats_votes(curve, ["plan_b"]), "plan_b")
    expected = line_color(plot_seats_votes(curve), "plan_b")
    assert observed == expected


def test_plot_seats_votes_steps_match_curve():
    # one step per vote share, so the line has one more edge than seat values
    needs_qt()
    curve = seats_votes_curve(make_votes(), **VOTES)
    widget = plot_seats_votes(curve, ["plan_a"])
    line = [item for item in widget.getPlotItem().listDataItems() if item.name() == "plan_a"][0]
    rows = curve[curve["plan"] == "plan_a"]
    assert len(line.xData) == len(rows) + 1
    assert line.yData.tolist() == rows["dem_seat_share"].tolist()
    assert abs(line.xData[0] - 0.35) < TOLERANCE
    assert abs(line.xData[-1] - 0.65) < TOLERANCE


def test_plot_seats_votes_x_ticks_cover_both_ends():
    # default range 35% to 65%, a tick every 5 points; 0.35 / 0.05 is not exactly 7
    needs_qt()
    widget = plot_seats_votes(seats_votes_curve(make_votes(), **VOTES))
    ticks = widget.getPlotItem().getAxis("bottom")._tickLevels[0]
    observed = [label for _, label in ticks]
    expected = ["35%", "40%", "45%", "50%", "55%", "60%", "65%"]
    assert observed == expected


def test_plot_seats_votes_redraw_clears_old_lines():
    needs_qt()
    curve = seats_votes_curve(make_votes(), **VOTES)
    widget = plot_seats_votes(curve)
    plot_seats_votes(curve, ["plan_a"], widget=widget)
    observed = line_names(widget)
    expected = ["proportional", "plan_a"]
    assert observed == expected


def test_seats_votes_window_checkboxes_list_every_plan():
    needs_qt()
    window = seats_votes_window(seats_votes_curve(make_votes(), **VOTES))
    observed = [list(window.checkboxes), [box.isChecked() for box in window.checkboxes.values()]]
    expected = [PLANS, [True, True]]
    assert observed == expected


def test_seats_votes_window_draws_ticked_plans():
    needs_qt()
    window = seats_votes_window(seats_votes_curve(make_votes(), **VOTES))
    window.checkboxes["plan_b"].setChecked(False)
    observed = line_names(window.chart)
    expected = ["proportional", "plan_a"]
    assert observed == expected


def test_seats_votes_window_handles_no_plans_ticked():
    needs_qt()
    window = seats_votes_window(seats_votes_curve(make_votes(), **VOTES))
    for box in window.checkboxes.values():
        box.setChecked(False)
    observed = line_names(window.chart)
    expected = ["proportional"]
    assert observed == expected


# ---------------------------------------------------------------------------
# Input checks and options
# ---------------------------------------------------------------------------

def test_exclude_can_list_id_column():
    observed = seats_votes_curve(make_votes(), exclude=["GEOID20"], **VOTES)["plan"].unique().tolist()
    expected = PLANS
    assert observed == expected


def test_exclude_can_list_geometry_column():
    votes = make_votes()
    votes["geometry"] = None
    observed = seats_votes_curve(votes, exclude=["geometry"], **VOTES)["plan"].unique().tolist()
    expected = PLANS
    assert observed == expected


def test_default_vote_column_names_give_same_result():
    votes = make_votes().rename(columns={DEM: "dem", REP: "rep"})
    observed = seats_votes_curve(votes)
    expected = seats_votes_curve(make_votes(), **VOTES)
    assert observed.equals(expected)


def test_missing_vote_column_raises():
    # ALARM files have no column named "dem", so the race must be chosen
    with pytest.raises(ValueError, match="dem"):
        seats_votes_curve(make_votes())


def test_chosen_name_clashing_with_existing_column_raises():
    votes = make_votes()
    votes["dem"] = votes[DEM]
    with pytest.raises(ValueError, match="already has"):
        seats_votes_curve(votes, **VOTES)


def test_blank_vote_cell_raises():
    votes = make_votes().astype({DEM: float})
    votes.loc[0, DEM] = float("nan")
    with pytest.raises(ValueError, match="blank dem"):
        seats_votes_curve(votes, **VOTES)


def test_plan_with_no_districts_raises():
    votes = make_votes()
    votes["plan_blank"] = float("nan")
    with pytest.raises(ValueError, match="no district has any votes"), pytest.warns(UserWarning, match="no district"):
        seats_votes_curve(votes, **VOTES)


@pytest.mark.parametrize("low, high, step", [
    (0.4, 0.6, 0.03),  # does not fit the range evenly
    (0.4, 0.6, 0.5),   # bigger than the range
])
def test_step_that_does_not_fit_range_raises(low, high, step):
    with pytest.raises(ValueError, match="evenly"):
        seats_votes_curve(make_votes(), low=low, high=high, step=step, **VOTES)


def test_score_plan():
    # dem shares 0.6, 0.4, 0.5375; statewide 0.5125
    # at 45%: 0.5375 0.3375 0.475 -> 1 seat
    # at 50%: 0.5875 0.3875 0.525 -> 2 seats
    # at 55%: 0.6375 0.4375 0.575 -> 2 seats
    observed = score_plan([240, 160, 215], [160, 240, 185], [1, 2, 3])
    expected = {"dem_seats_at_45pct": 1, "dem_seats_at_50pct": 2, "dem_seats_at_55pct": 2}
    assert observed == expected


def test_score_plan_on_vtd_columns():
    # plan_a from the toy table: 1, 2, 3 seats at 45%, 50%, 55%
    votes = make_votes()
    observed = score_plan(votes[DEM], votes[REP], votes["plan_a"])
    expected = {"dem_seats_at_45pct": 1, "dem_seats_at_50pct": 2, "dem_seats_at_55pct": 3}
    assert observed == expected


# ---------------------------------------------------------------------------
# Real data: ALARM 2020 Pennsylvania VTDs (downloaded once, skipped offline)
# ---------------------------------------------------------------------------

# Biden's two-party share of the 2020 PA vote in the ALARM file:
# 3,457,443 / (3,457,443 + 3,376,900). The certified count is 50.59% too.
PA_BIDEN_SHARE = 3_457_443 / (3_457_443 + 3_376_900)
PA_BIDEN_COUNTIES = 13   # counties Biden carried in 2020


def add_real_plans(vtd):
    """
    Two plans on the real VTDs.

    plan_by_county   every county is its own district (67 districts)
    plan_statewide   one district
    """
    vtd["plan_by_county"] = vtd["GEOID20"].str[2:5]
    vtd["plan_statewide"] = 1
    return vtd


def test_pa_vtd_statewide_share(pa_vtd):
    observed = pa_vtd[DEM].sum() / (pa_vtd[DEM].sum() + pa_vtd[REP].sum())
    expected = PA_BIDEN_SHARE
    assert abs(observed - expected) < TOLERANCE


def test_pa_vtd_only_plan_columns_are_scored(pa_vtd):
    # the table keeps all 44 ALARM columns; none of them is a plan
    observed = seats_votes_curve(add_real_plans(pa_vtd), **VOTES)["plan"].unique().tolist()
    expected = ["plan_by_county", "plan_statewide"]
    assert observed == expected


def test_pa_vtd_counties_won_at_actual_vote_share(pa_vtd):
    # at Biden's real statewide share there is no swing, so the seats are the
    # counties he actually carried
    curve = seats_votes_curve(add_real_plans(pa_vtd), low=PA_BIDEN_SHARE,
                              high=PA_BIDEN_SHARE + 0.1, step=0.1, **VOTES)
    observed = row_at(curve, "plan_by_county", round(PA_BIDEN_SHARE, 6))["dem_seats"]
    expected = PA_BIDEN_COUNTIES
    assert observed == expected


def test_pa_vtd_one_district_flips_at_half(pa_vtd):
    # a single statewide district goes D only above 50% (a tie goes R)
    curve = seats_votes_curve(add_real_plans(pa_vtd), low=0.45, high=0.55, step=0.05, **VOTES)
    observed = curve.loc[curve["plan"] == "plan_statewide", "dem_seats"].tolist()
    expected = [0, 0, 1]
    assert observed == expected


def test_pa_vtd_seats_never_fall_as_votes_rise(pa_vtd):
    curve = seats_votes_curve(add_real_plans(pa_vtd), **VOTES)
    rows = curve[curve["plan"] == "plan_by_county"]
    assert (rows["dem_seats"].diff().dropna() >= 0).all()


def test_pa_vtd_vote_averages_work(pa_vtd):
    # ndv / nrv average every race in the file. Swinging to 50% shifts every
    # county by 0.5 - V, so a county wins exactly when its own share beats V
    vtd = add_real_plans(pa_vtd)
    by_county = vtd.groupby("plan_by_county")[["ndv", "nrv"]].sum()
    statewide = by_county["ndv"].sum() / by_county.to_numpy().sum()
    curve = seats_votes_curve(vtd, dem="ndv", rep="nrv")
    observed = row_at(curve, "plan_by_county", 0.5)["dem_seats"]
    expected = int((by_county["ndv"] / by_county.sum(axis=1) > statewide).sum())
    assert observed == expected


def test_pa_vtd_score_plan_matches_curve(pa_vtd):
    vtd = add_real_plans(pa_vtd)
    curve = seats_votes_curve(vtd, **VOTES)
    observed = score_plan(vtd[DEM], vtd[REP], vtd["plan_by_county"])
    expected = {f"dem_seats_at_{pct}pct": row_at(curve, "plan_by_county", pct / 100)["dem_seats"]
                for pct in (45, 50, 55)}
    assert observed == expected
