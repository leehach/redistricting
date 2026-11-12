"""
Seats-votes curve for redistricting plans.

The curve shows the share of seats Democrats would win at each statewide
Democratic vote share. It is built by uniform swing: every district's
Democratic share is shifted by the same amount, and the districts above 50%
are counted at each step:

    seats_votes_curve   seat share at each statewide vote share, per plan
    plot_seats_votes    chart of the curve, one line per plan
    seats_votes_window  desktop window for choosing which plans to compare
    show_seats_votes    opens that window from a script and waits until it is closed
    score_plan          seats at chosen vote shares for one plan, as the metrics GUI uses it

Input is one table with one row per precinct, the same shape the
competitiveness module uses:

    precinct_id  dem  rep  plan_a  plan_b
    42001000010  420  380  33      12
    42001000100  390  350  33      13

The first column is the precinct id (any name). ``dem`` and ``rep`` are vote
counts for the two leading candidates; use the ``dem`` and ``rep`` arguments if
your columns have other names. Every other column is a plan, with district assignments as values. A GeoDataFrame also works; its
geometry column is ignored.

The chart functions need ``PyQt5`` and ``pyqtgraph``, the same libraries as the
metrics GUI and the Qt DEA frontier explorer. They are imported inside those
functions, so ``seats_votes_curve`` and ``score_plan`` only need pandas and numpy.

References
----------
Tufte, E. R. (1973). The Relationship between Seats and Votes in Two-Party
Systems. American Political Science Review, 67(2), 540-554.

Gelman, A. and King, G. (1994). A Unified Method of Evaluating Electoral
Systems and Redistricting Plans. American Journal of Political Science, 38(2),
514-554. https://doi.org/10.2307/2111417

PySAL redistricting notebook, The Historical Seats-Votes Curve:
https://github.com/pysal/redistricting/blob/master/notebooks/00%20-%20The%20Historical%20Seats-Votes%20Curve.ipynb

pyqtgraph documentation. https://pyqtgraph.readthedocs.io/
"""

import sys
import warnings

import numpy as np
import pandas as pd

# columns that are not plans
INFO_COLUMNS = ["cell_id", "dem", "rep", "total", "geometry"]

# one color per plan, in the order plans appear (the palette the Dash chart used)
PLAN_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
               "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
REFERENCE_GRAY = "#9ca3af"   # proportional line
GUIDE_GRAY = "#d1d5db"       # 50% vote and seat lines
INK = "#1f2933"              # text and axes, as in the Qt DEA explorer


# ---------------------------------------------------------------------------
# Data prep (private)
# ---------------------------------------------------------------------------

def _prepare_df(df, dem="dem", rep="rep"):
    """
    Standardize the input table.

    Renames the first column to ``cell_id`` and the vote columns to ``dem`` and
    ``rep``, drops a geometry column if one is present, sets the vote columns to
    numbers, and adds a two-party ``total`` column (``dem + rep``), replacing
    any existing ``total``.

    A blank vote cell raises a ValueError instead of counting as 0. A blank may
    mean missing data rather than no votes, and treating it as 0 can change a
    result, so the analyst enters 0 where no votes were cast. This matches
    ``population_deviation``, which raises on a blank population.
    """
    missing = [col for col in (dem, rep) if col not in df.columns]
    if missing:
        raise ValueError(f"vote column(s) not in df: {', '.join(missing)}")
    taken = [new for old, new in ((dem, "dem"), (rep, "rep")) if old != new and new in df.columns]
    if taken:
        raise ValueError(f"df already has column(s) named {', '.join(taken)}; "
                         "exclude or rename them before choosing other vote columns")

    # copy so the caller's table is not altered
    df = pd.DataFrame(df).copy()
    df = df.rename(columns={df.columns[0]: "cell_id", dem: "dem", rep: "rep"})
    df = df.drop(columns="geometry", errors="ignore")
    df["cell_id"] = df["cell_id"].astype(str)

    for col in ["dem", "rep"]:
        df[col] = pd.to_numeric(df[col]).astype(float)
        blank = df.loc[df[col].isna(), "cell_id"].tolist()
        if blank:
            raise ValueError(f"{len(blank)} precinct(s) have a blank {col} cell: {blank[:10]}. "
                             "Enter 0 if no votes were cast.")
    # two-party total: denominator is dem + rep, not all votes cast
    df["total"] = df["dem"] + df["rep"]

    return df


def _check_exclude(df, exclude):
    """
    Check the ``exclude`` argument against the input table.

    Runs before any columns are renamed or dropped, so any column in the
    caller's table can be listed, including the id and geometry columns.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Input table, as the caller passed it.
    exclude : str or list of str or None
        Columns that are not plans.

    Returns
    -------
    list of str
        The excluded column names.

    Raises
    ------
    ValueError
        If ``exclude`` names a column that is not in ``df``.
    """
    if exclude is None:
        return []
    if isinstance(exclude, str):
        exclude = [exclude]

    # a misspelled name would leave the real column in as a plan
    missing = [col for col in exclude if col not in df.columns]
    if missing:
        raise ValueError(f"exclude names column(s) not in df: {', '.join(missing)}")
    return list(exclude)


def _plan_columns(df, exclude):
    """
    List the plan columns in the input table.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table, after ``_prepare_df``.
    exclude : list of str
        Columns that are not plans, from ``_check_exclude``. ``cell_id``,
        ``dem``, ``rep``, and ``total`` are always left out, so they do not need
        to be listed.

    Returns
    -------
    list of str
        Names of the plan columns, in the order they appear in ``df``.
    """
    non_plan = INFO_COLUMNS + exclude
    return [col for col in df.columns if col not in non_plan]


def _district_totals(df, plan):
    """
    Vote totals for every district in one plan.

    Returns a DataFrame with columns: district, dem, rep, total.
    """
    plan_df = df[[plan, "dem", "rep", "total"]].rename(columns={plan: "district"})

    # groupby skips blank districts, so say how many precincts that leaves out
    missing = int(plan_df["district"].isna().sum())
    if missing:
        warnings.warn(f"{plan}: left out {missing} precinct(s) with no district")
    totals = plan_df.groupby("district", as_index=False)[["dem", "rep", "total"]].sum()

    empty = totals.loc[totals["total"] == 0, "district"].astype(str).tolist()
    if empty:
        warnings.warn(f"{plan}: left out {len(empty)} district(s) with no votes: "
                      f"{', '.join(empty)}")
    totals = totals[totals["total"] > 0]
    if totals.empty:
        raise ValueError(f"{plan}: no district has any votes")
    return totals


# ---------------------------------------------------------------------------
# Metric math (private)
# ---------------------------------------------------------------------------

def _swing_seats(district_totals, targets):
    """
    Democratic seats at each target statewide share, by uniform swing.

    Returns a list of seat counts, one per target.
    """
    share = district_totals["dem"] / district_totals["total"]
    statewide = district_totals["dem"].sum() / district_totals["total"].sum()

    seats = []
    for target in targets:
        # round so a share that should be exactly 0.5 is not 0.5000000001
        shifted = (share + (target - statewide)).round(10)
        # a tie (exactly 0.5) counts as a Republican win, as in partisan_bias (PR #40)
        seats.append(int((shifted > 0.5).sum()))
    return seats


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def seats_votes_curve(df, low=0.35, high=0.65, step=0.005, exclude=None, dem="dem", rep="rep"):
    """
    Calculate the seats-votes curve for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include Democratic and
        Republican vote columns. All other columns are plans, with values
        indicating district assignments.
    low, high : float, optional
        Range of statewide Democratic vote shares to cover. Defaults to 0.35
        and 0.65.
    step : float, optional
        Distance between vote shares. Defaults to 0.005 (half a percentage
        point). Must divide the range from ``low`` to ``high`` evenly.
    exclude : str or list of str, optional
        Columns that are not plans, such as ``county`` or ``area``. The id,
        ``dem``, ``rep``, and geometry columns are always left out.
    dem, rep : str, optional
        Names of the Democratic and Republican vote columns. Default to ``dem``
        and ``rep``.

    Returns
    -------
    pandas.DataFrame
        One row per plan per vote share, with columns ``plan``,
        ``dem_vote_share``, ``dem_seats``, and ``dem_seat_share``.

    Raises
    ------
    ValueError
        If the range or step is invalid, a vote column is missing or has a blank cell, ``exclude`` names a
        column that is not in ``df``, or a plan has no district with votes.

    Notes
    -----
    For a target statewide share :math:`V'`, every district's Democratic
    share :math:`d_i` is shifted by the same amount:

    .. math::
        d_i' = d_i + (V' - V)

    Where :math:`V` is the plan's actual statewide Democratic share of the
    two-party vote. Democrats win the districts where :math:`d_i' > 0.5`.

    Analyst notes: uniform swing assumes every district moves by the same
    amount, which is a simplification; real swings vary by district (Gelman
    and King 1994). The seats-votes relationship follows Tufte (1973). Shares
    are taken after summing votes by district, using the two-party vote, so
    third-party votes do not count. A tie counts as a Republican win, to
    match ``partisan_bias`` in the PlanScore metrics (PR #40).
    """
    if not 0 <= low < high <= 1:
        raise ValueError("low and high must satisfy 0 <= low < high <= 1")
    if step <= 0:
        raise ValueError("step must be above 0")

    # a step that doesn't fit the range would otherwise be changed without notice
    steps = (high - low) / step
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("step must divide the range from low to high evenly")

    # linspace, not arange, so the last share is exactly `high`
    n = int(round(steps)) + 1
    targets = np.round(np.linspace(low, high, n), 6)

    exclude = _check_exclude(df, exclude)
    df = _prepare_df(df, dem, rep)
    results = []
    for plan in _plan_columns(df, exclude):
        district_totals = _district_totals(df, plan)
        n_districts = len(district_totals)
        for target, seats in zip(targets, _swing_seats(district_totals, targets)):
            results.append({"plan": plan,
                            "dem_vote_share": float(target),
                            "dem_seats": seats,
                            "dem_seat_share": seats / n_districts})
    return pd.DataFrame(results)


def _one_plan_table(dem, rep, district):
    """Build the input table for one plan from separate columns."""
    dem = pd.Series(dem)
    return pd.DataFrame({
        "cell_id": dem.index.astype(str),
        "dem": dem.values,
        "rep": pd.Series(rep).values,
        "plan": pd.Series(district).values,
    })


def score_plan(dem, rep, district, vote_shares=(0.45, 0.5, 0.55)):
    """
    Calculate Democratic seats at chosen statewide vote shares for one plan.

    Parameters
    ----------
    dem, rep, district : array-like
        One value per precinct, in the same order: Democratic votes, Republican
        votes, and the precinct's district in this plan.
    vote_shares : sequence of float, optional
        Statewide Democratic vote shares to report. Defaults to 45%, 50% and 55%.

    Returns
    -------
    dict
        One entry per vote share, such as ``{"dem_seats_at_50pct": 2}``.

    Raises
    ------
    ValueError
        If a vote share is outside 0 to 1, the inputs differ in length, a vote
        value is blank, or the plan has no district with votes.

    Notes
    -----
    This summarizes the curve in plan-wide numbers, the format the metrics GUI
    saves. Seats come from the same uniform swing as ``seats_votes_curve``; use
    that function and ``seats_votes_window`` to see the whole curve.

    Examples
    --------
    In ``metrics_gui/metrics.py``:

    >>> @metric("Seats-votes", "Seats at 45%, 50% and 55%", needs=PARTISAN_NEEDS)
    ... def seats(ctx):
    ...     return seats_votes.score_plan(ctx.column("dem_votes"), ctx.column("rep_votes"),
    ...                                   ctx.district)
    """
    if not all(0 <= share <= 1 for share in vote_shares):
        raise ValueError("vote_shares must be between 0 and 1")
    df = _prepare_df(_one_plan_table(dem, rep, district))
    seats = _swing_seats(_district_totals(df, "plan"), vote_shares)
    return {f"dem_seats_at_{share * 100:g}pct": count for share, count in zip(vote_shares, seats)}


_QT_APP = []   # holds the QApplication this module starts, so it is not garbage collected


def _qt_app():
    """
    Return the running QApplication, starting one if there is none.

    Qt widgets can only be made once a QApplication exists. Inside the metrics
    GUI one is already running, so that one is reused. One started here is
    kept in ``_QT_APP``; if Python deleted it, the next widget would crash Qt.
    """
    # PyQt5 before pyqtgraph, so pyqtgraph uses PyQt5 and not another Qt binding
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        _QT_APP.append(app)
    return app


def _step_edges(x):
    """
    Edges for a stepped line with one step centered on each vote share.

    Each seat count holds from halfway before its vote share to halfway after,
    and the first and last steps stop at the ends of the range. This draws the
    same steps as the Dash chart's ``hvh`` line shape.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.r_[x, x]
    middles = (x[:-1] + x[1:]) / 2
    return np.r_[x[0], middles, x[-1]]


def _percent_ticks(low, high, every):
    """
    Tick positions from ``low`` to ``high``, at each multiple of ``every``.

    Rounds before ``ceil`` and ``floor``, since 0.35 / 0.05 is
    7.000000000000001 in floating point and would otherwise skip the 35% tick.
    """
    first = np.ceil(round(low / every, 6))
    last = np.floor(round(high / every, 6))
    return np.round(np.arange(first, last + 1) * every, 6)


def plot_seats_votes(curve, plans=None, widget=None):
    """
    Draw the seats-votes curve as a pyqtgraph chart.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of ``seats_votes_curve``.
    plans : list of str, optional
        Plans to draw. Defaults to every plan in ``curve``.
    widget : pyqtgraph.PlotWidget, optional
        Chart to draw on. Anything already on it is cleared first. Defaults to
        a new chart.

    Returns
    -------
    pyqtgraph.PlotWidget
        The chart. ``seats_votes_window`` puts it in a window with a checkbox
        per plan; ``widget.show()`` opens the chart on its own.

    Notes
    -----
    One stepped line per plan. The dashed diagonal is proportional
    representation (x% of votes wins x% of seats). Where a plan's line
    crosses the 50% vote line shows its partisan bias; how steep it is there
    shows its responsiveness. Hovering over a point on a line shows its vote
    and seat shares. A plan keeps its color when other plans are hidden.
    Scroll to zoom, drag to pan, and click the small "A" in the corner to
    reset the view.
    """
    # imported here so seats_votes_curve works without Qt installed
    _qt_app()
    import pyqtgraph as pg
    from PyQt5.QtCore import Qt

    all_plans = curve["plan"].unique().tolist()
    if plans is None:
        plans = all_plans

    if widget is None:
        widget = pg.PlotWidget(background="w")
        widget.resize(650, 600)
    plot = widget.getPlotItem()
    plot.clear()   # also empties the legend
    if plot.legend is None:
        plot.addLegend(offset=(10, -10), labelTextColor=INK,
                       brush=pg.mkBrush(255, 255, 255, 230))

    # reference lines go first so the plan lines draw on top
    for angle in (0, 90):
        plot.addItem(pg.InfiniteLine(pos=0.5, angle=angle, pen=pg.mkPen(GUIDE_GRAY, width=1)))
    plot.addItem(pg.PlotDataItem([0, 1], [0, 1], name="proportional",
                                 pen=pg.mkPen(REFERENCE_GRAY, width=1, style=Qt.DashLine)))

    for plan in plans:
        rows = curve[curve["plan"] == plan]
        x = rows["dem_vote_share"].to_numpy(float)
        y = rows["dem_seat_share"].to_numpy(float)
        color = PLAN_COLORS[all_plans.index(plan) % len(PLAN_COLORS)]
        plot.addItem(pg.PlotDataItem(_step_edges(x), y, stepMode="center", name=str(plan),
                                     pen=pg.mkPen(color, width=2)))
        # hidden points carry the hover text and appear under the mouse;
        # no name, so the legend has one entry per plan
        plot.addItem(pg.ScatterPlotItem(
            x=x, y=y, size=8, brush=pg.mkBrush(0, 0, 0, 0), pen=None,
            hoverable=True, hoverBrush=pg.mkBrush(color), hoverPen=pg.mkPen("w", width=1),
            tip=lambda x, y, data, plan=plan: f"{plan}\n{x:.1%} of votes: {y:.0%} of seats"))

    plot.setTitle("Seats-votes curve", color=INK)
    plot.setLabel("bottom", "Statewide Democratic vote share")
    plot.setLabel("left", "Democratic seat share")
    # the tick labels are already percents; stop pyqtgraph adding "(x0.001)"
    for side in ("bottom", "left"):
        plot.getAxis(side).enableAutoSIPrefix(False)

    # percent tick labels, every 5 points on x and 10 points on y
    low, high = curve["dem_vote_share"].min(), curve["dem_vote_share"].max()
    plot.getAxis("bottom").setTicks([[(v, f"{v:.0%}") for v in _percent_ticks(low, high, 0.05)]])
    plot.getAxis("left").setTicks([[(v, f"{v:.0%}") for v in _percent_ticks(0, 1, 0.1)]])
    # keep the first and last labels, which hang past the ends of the axis
    plot.getAxis("bottom").setStyle(hideOverlappingLabels=False)
    plot.setRange(xRange=(low, high), yRange=(0, 1), padding=0.02)
    return widget


def seats_votes_window(curve):
    """
    Build a desktop window that compares seats-votes curves.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of ``seats_votes_curve``.

    Returns
    -------
    PyQt5.QtWidgets.QWidget
        The window, not yet shown. ``window.checkboxes`` maps each plan to its
        checkbox and ``window.chart`` is the pyqtgraph chart.

    Notes
    -----
    A row of checkboxes above the chart shows or hides each plan. The chart
    itself comes from ``plot_seats_votes``. See that function for how to read it.

    The window has no parent, so it opens on its own beside the metrics GUI
    instead of inside it. From code that is already running Qt, such as the
    metrics GUI, keep a reference so Python does not close the window::

        self.seats_votes = seats_votes_window(curve)
        self.seats_votes.show()

    From a script, use ``show_seats_votes`` instead.
    """
    _qt_app()
    from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

    plans = curve["plan"].unique().tolist()
    window = QWidget()
    window.setWindowTitle("Seats-votes curve")
    window.resize(700, 680)
    layout = QVBoxLayout(window)

    boxes = QHBoxLayout()
    window.checkboxes = {}
    for plan in plans:
        box = QCheckBox(str(plan))
        box.setChecked(True)
        boxes.addWidget(box)
        window.checkboxes[plan] = box
    boxes.addStretch(1)
    layout.addLayout(boxes)

    window.chart = plot_seats_votes(curve)
    layout.addWidget(window.chart, stretch=1)

    def redraw():
        """Redraw the chart with the plans ticked in the checkboxes."""
        ticked = [plan for plan, box in window.checkboxes.items() if box.isChecked()]
        plot_seats_votes(curve, ticked, widget=window.chart)

    for box in window.checkboxes.values():
        box.toggled.connect(redraw)
    return window


def show_seats_votes(curve):
    """
    Open the seats-votes window and wait until it is closed.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of ``seats_votes_curve``.

    Returns
    -------
    int
        Qt's exit code.

    Notes
    -----
    For scripts and ``python -m redistricting.seats_votes``. Code that is
    already running Qt, such as the metrics GUI, should call
    ``seats_votes_window(curve).show()`` instead, since this starts Qt's
    event loop and waits.
    """
    app = _qt_app()
    window = seats_votes_window(curve)
    window.show()
    return app.exec_()


# ---------------------------------------------------------------------------
# Quick check: python -m redistricting.seats_votes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 8 precincts of 50 votes each; statewide 200 dem of 400 = exactly 50%.
    # plan_a district dem shares: 0.30, 0.49, 0.51, 0.70  (two close districts)
    # plan_b district dem shares: 0.30, 0.35, 0.65, 0.70  (all safe districts)
    test = pd.DataFrame({
        "precinct_id": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "dem":    [10, 20, 25, 24, 10, 41, 41, 29],
        "rep":    [40, 30, 25, 26, 40,  9,  9, 21],
        "plan_a": [1, 1, 2, 2, 3, 3, 4, 4],
        "plan_b": [1, 2, 1, 3, 2, 4, 3, 4],
    })

    curve = seats_votes_curve(test)
    print(curve[curve["dem_vote_share"].isin([0.45, 0.5, 0.55])].to_string(index=False))

    def seats_at(plan, vote_share):
        row = curve[(curve["plan"] == plan) & (curve["dem_vote_share"] == vote_share)]
        return int(row["dem_seats"].iloc[0])

    # at 45%: plan_a 0.25 0.44 0.46 0.65 -> 1 seat; plan_b 0.25 0.30 0.60 0.65 -> 2 seats
    # at 50%: both plans win 2 of 4
    # at 55%: plan_a 0.35 0.54 0.56 0.75 -> 3 seats; plan_b 0.35 0.40 0.70 0.75 -> 2 seats
    assert [seats_at("plan_a", v) for v in (0.45, 0.5, 0.55)] == [1, 2, 3]
    assert [seats_at("plan_b", v) for v in (0.45, 0.5, 0.55)] == [2, 2, 2]
    print("seats_votes_curve test passed")

    # opens the chart in a window; close the window to stop
    show_seats_votes(curve)
