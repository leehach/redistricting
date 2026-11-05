"""
Responsiveness metric for redistricting plans.

Responsiveness is how much a plan's seat share changes when the statewide
vote shifts, measured near a 50-50 vote:

    responsiveness   change in Democratic seat share per change in vote share

A responsiveness of 2 means a 1-point swing in the statewide vote moves about
2% of the seats. A plan of all safe districts scores 0.

Input is one table with one row per precinct, the same shape the
competitiveness module uses:

    precinct_id  dem  rep  plan_a  plan_b
    42001000010  420  380  33      12
    42001000100  390  350  33      13

The first column is the precinct id (any name). ``dem`` and ``rep`` are vote
counts for the two leading candidates, found by name. Every other column is a
plan, with district assignments as values. A GeoDataFrame also works; its
geometry column is ignored.

References
----------
Tufte, E. R. (1973). The Relationship between Seats and Votes in Two-Party
Systems. American Political Science Review, 67(2), 540-554.

Gelman, A. and King, G. (1994). A Unified Method of Evaluating Electoral
Systems and Redistricting Plans. American Journal of Political Science, 38(2),
514-554. https://doi.org/10.2307/2111417
"""

import warnings

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

def _seat_share_at(district_totals, target):
    """Democratic seat share if the statewide vote were `target`, by uniform swing."""
    share = district_totals["dem"] / district_totals["total"]
    statewide = district_totals["dem"].sum() / district_totals["total"].sum()
    # round so a share that should be exactly 0.5 is not 0.5000000001
    shifted = (share + (target - statewide)).round(10)
    # a tie (exactly 0.5) counts as a Republican win, as in partisan_bias (PR #40)
    return float((shifted > 0.5).mean())


def _responsiveness(district_totals, window):
    """Slope of the seats-votes curve between 50% - window and 50% + window."""
    low = _seat_share_at(district_totals, 0.5 - window)
    high = _seat_share_at(district_totals, 0.5 + window)
    return (high - low) / (2 * window)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def responsiveness(df, window=0.025, exclude=None):
    """
    Calculate responsiveness for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``dem`` and ``rep``
        vote columns. All other columns are plans, with values indicating
        district assignments.
    window : float, optional
        How far either side of a 50% statewide vote to measure, as a fraction
        of the two-party vote. Defaults to 0.025 (47.5% to 52.5%).
    exclude : str or list of str, optional
        Columns that are not plans, such as ``county`` or ``area``. The id,
        ``dem``, ``rep``, and geometry columns are always left out.

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``responsiveness``.

    Notes
    -----
    Responsiveness is the slope of the seats-votes curve around 50%:

    .. math::
        R = \\frac{S(0.5 + w) - S(0.5 - w)}{2w}

    Where :math:`S(v)` is the Democratic seat share when the statewide
    Democratic vote share is :math:`v`, and :math:`w` is the window. Seat
    shares come from uniform swing: every district's Democratic share is
    shifted by the same amount until the statewide share is :math:`v`, and
    districts above 50% count as Democratic seats.

    Analyst notes: this slope is the swing ratio discussed by Tufte (1973).
    Uniform swing assumes every district moves by the same amount, which is a
    simplification (Gelman and King 1994). A small window reacts to one or two
    districts near 50%; a larger window is smoother but less local. A tie
    counts as a Republican win, to match ``partisan_bias`` in the PlanScore
    metrics (PR #40). Whether higher is better for DEA is a team choice, since
    very high values mean winner-take-all outcomes.
    """
    if not 0 < window < 0.5:
        raise ValueError("window must be between 0 and 0.5")

    df = _prepare_df(df)
    results = []
    for plan in _plan_columns(df, exclude):
        district_totals = _district_totals(df, plan)
        results.append({"plan": plan,
                        "responsiveness": _responsiveness(district_totals, window)})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Quick check: python -m redistricting.responsiveness
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

    result = responsiveness(test)
    print(result.to_string(index=False))

    # plan_a: at 47.5% wins 1 of 4 (0.25), at 52.5% wins 3 of 4 (0.75)
    #         (0.75 - 0.25) / 0.05 = 10
    # plan_b: wins 2 of 4 at both ends -> 0
    assert result["responsiveness"].round(6).tolist() == [10.0, 0.0]
    print("responsiveness test passed")
