"""
Seats-votes curve for redistricting plans.

The curve shows the share of seats Democrats would win at each statewide
Democratic vote share. It is built by uniform swing: every district's
Democratic share is shifted by the same amount, and the districts above 50%
are counted at each step:

    seats_votes_curve   seat share at each statewide vote share, per plan
    plot_seats_votes    chart of the curve, one line per plan
    seats_votes_app     Dash web app for choosing which plans to compare

Input is one table with one row per precinct, the same shape the
competitiveness module uses:

    precinct_id  dem  rep  plan_a  plan_b
    42001000010  420  380  33      12
    42001000100  390  350  33      13

The first column is the precinct id (any name). ``dem`` and ``rep`` are vote
counts for the two leading candidates, found by name. Every other column is a
plan, with district assignments as values. A GeoDataFrame also works; its
geometry column is ignored.

The chart functions need ``plotly`` and ``dash``. They are imported inside those
functions, so ``seats_votes_curve`` itself only needs pandas and numpy.

References
----------
Tufte, E. R. (1973). The Relationship between Seats and Votes in Two-Party
Systems. American Political Science Review, 67(2), 540-554.

Gelman, A. and King, G. (1994). A Unified Method of Evaluating Electoral
Systems and Redistricting Plans. American Journal of Political Science, 38(2),
514-554. https://doi.org/10.2307/2111417

PySAL redistricting notebook, The Historical Seats-Votes Curve:
https://github.com/pysal/redistricting/blob/master/notebooks/00%20-%20The%20Historical%20Seats-Votes%20Curve.ipynb

Plotly Technologies Inc. Dash Python documentation. https://dash.plotly.com/
"""

import warnings

import numpy as np
import pandas as pd

# columns that are not plans
INFO_COLUMNS = ["cell_id", "dem", "rep", "total", "geometry"]


# ---------------------------------------------------------------------------
# Data prep (private)
# ---------------------------------------------------------------------------

def _prepare_df(df):
    """
    Standardize the input table.

    Renames the first column to ``cell_id``, drops a geometry column if one is
    present, sets ``dem`` and ``rep`` to numbers, and adds a two-party
    ``total`` column (``dem + rep``), replacing any existing ``total``.
    """
    # copy so the caller's table is not altered
    df = pd.DataFrame(df).copy()
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df = df.drop(columns="geometry", errors="ignore")

    for col in ["dem", "rep"]:
        df[col] = pd.to_numeric(df[col]).fillna(0).astype(float)
    # two-party total: denominator is dem + rep, not all votes cast
    df["total"] = df["dem"] + df["rep"]
    df["cell_id"] = df["cell_id"].astype(str)

    return df


def _plan_columns(df, exclude=None):
    """
    List the plan columns in the input table.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table, after ``_prepare_df``.
    exclude : str or list of str, optional
        Columns that are not plans, such as ``county`` or ``area``.
        ``cell_id``, ``dem``, ``rep``, and ``total`` are always left out, so
        they do not need to be listed.

    Returns
    -------
    list of str
        Names of the plan columns, in the order they appear in ``df``.

    Raises
    ------
    ValueError
        If ``exclude`` names a column that is not in ``df``.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    # a misspelled name would leave the real column in as a plan
    missing = [col for col in exclude if col not in df.columns]
    if missing:
        raise ValueError(f"exclude names column(s) not in df: {', '.join(missing)}")

    non_plan = INFO_COLUMNS + list(exclude)
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
    return totals[totals["total"] > 0]


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

def seats_votes_curve(df, low=0.35, high=0.65, step=0.005, exclude=None):
    """
    Calculate the seats-votes curve for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``dem`` and ``rep``
        vote columns. All other columns are plans, with values indicating
        district assignments.
    low, high : float, optional
        Range of statewide Democratic vote shares to cover. Defaults to 0.35
        and 0.65.
    step : float, optional
        Distance between vote shares. Defaults to 0.005 (half a percentage
        point).
    exclude : str or list of str, optional
        Columns that are not plans, such as ``county`` or ``area``. The id,
        ``dem``, ``rep``, and geometry columns are always left out.

    Returns
    -------
    pandas.DataFrame
        One row per plan per vote share, with columns ``plan``,
        ``dem_vote_share``, ``dem_seats``, and ``dem_seat_share``.

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

    # linspace, not arange, so the last share is exactly `high`
    n = int(round((high - low) / step)) + 1
    targets = np.round(np.linspace(low, high, n), 6)

    df = _prepare_df(df)
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


def plot_seats_votes(curve, plans=None):
    """
    Draw the seats-votes curve as a Plotly figure.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of ``seats_votes_curve``.
    plans : list of str, optional
        Plans to draw. Defaults to every plan in ``curve``.

    Returns
    -------
    plotly.graph_objects.Figure
        The chart. ``seats_votes_app`` displays it in Dash, and
        ``fig.write_html("curve.html")`` saves a copy.

    Notes
    -----
    One stepped line per plan. The dashed diagonal is proportional
    representation (x% of votes wins x% of seats). Where a plan's line
    crosses the 50% vote line shows its partisan bias; how steep it is there
    shows its responsiveness. Hovering over a line shows its vote and seat
    shares. A plan keeps its color when other plans are hidden.
    """
    # imported here so seats_votes_curve works without plotly installed
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    all_plans = curve["plan"].unique().tolist()
    if plans is None:
        plans = all_plans

    fig = go.Figure()
    # reference lines go below so the plan lines draw on top
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="proportional",
                             line={"dash": "dash", "color": "gray", "width": 1},
                             hoverinfo="skip"))
    fig.add_hline(y=0.5, line={"color": "lightgray", "width": 1}, layer="below")
    fig.add_vline(x=0.5, line={"color": "lightgray", "width": 1}, layer="below")

    colors = qualitative.Plotly
    for plan in plans:
        rows = curve[curve["plan"] == plan]
        fig.add_trace(go.Scatter(
            x=rows["dem_vote_share"], y=rows["dem_seat_share"], mode="lines",
            line_shape="hvh", name=str(plan),
            line={"color": colors[all_plans.index(plan) % len(colors)]},
            hovertemplate="%{x:.1%} of votes: %{y:.0%} of seats"))

    fig.update_layout(title="Seats-votes curve", template="simple_white",
                      width=650, height=600,
                      xaxis_title="Statewide Democratic vote share",
                      yaxis_title="Democratic seat share")
    fig.update_xaxes(range=[curve["dem_vote_share"].min(), curve["dem_vote_share"].max()],
                     tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def seats_votes_app(curve):
    """
    Build a Dash app that compares seats-votes curves.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of ``seats_votes_curve``.

    Returns
    -------
    dash.Dash
        The app. ``seats_votes_app(curve).run()`` serves it on this computer
        at http://127.0.0.1:8050 until stopped with Ctrl+C.

    Notes
    -----
    A checklist above the chart shows or hides each plan. The chart itself
    comes from ``plot_seats_votes``. See that function for how to read it.
    Built with Dash (Plotly Technologies Inc.); see https://dash.plotly.com/.
    """
    # imported here so seats_votes_curve works without dash installed
    from dash import Dash, Input, Output, dcc, html

    plans = curve["plan"].unique().tolist()
    app = Dash(__name__, title="Seats-votes curve")
    app.layout = html.Div([
        dcc.Checklist(id="plans", options=plans, value=plans, inline=True),
        dcc.Graph(id="curve", figure=plot_seats_votes(curve)),
    ])

    @app.callback(Output("curve", "figure"), Input("plans", "value"))
    def _update_curve(selected):
        """Redraw the chart with the plans ticked in the checklist."""
        return plot_seats_votes(curve, selected)

    return app


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

    # serves the chart at http://127.0.0.1:8050; Ctrl+C stops it
    seats_votes_app(curve).run()
