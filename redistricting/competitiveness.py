"""
Competitiveness metric for redistricting plans.

A district is competitive when its Democratic and Republican shares of the
two-party vote are within 5 percentage points of each other:

    competitive_districts   number of competitive districts in a plan

Plans are precinct assignment files, one CSV per plan, as in the CCFD plan
collection (for example, its ``pa-senate`` folder):

    GEOID20      District
    42001000010  33
    42001000100  33

Votes are one table with one row per precinct, using the same precinct ids:

    GEOID20      dem  rep
    42001000010  420  380
    42001000100  390  350

``dem`` and ``rep`` are vote counts for the two leading candidates, found by
name. To score a folder of plans:

    plans = read_plans("pa-senate")
    df = join_plans(votes, plans)
    competitive_districts(df)

``join_plans`` returns one table: the precinct id first, then ``dem`` and
``rep``, then one column per plan, with district assignments as values. A
table already in that shape can go straight to ``competitive_districts``. A
GeoDataFrame also works; its geometry column is ignored.

References
----------
Gopalan, R., Hachadoorian, L., Kimbrough, S., Murphy, F., and Villere, J.
(2026). Finding Pareto Dominant Redistricting Plans Using DEA: A Comparison
With Ranking by Human Judges. American Politics Research, 54(6), 686-703.
https://doi.org/10.1177/1532673X261451261

Plan files: https://github.com/ccfdpa/pa-citizen-redistricting-plans-2020
"""

import warnings
from pathlib import Path

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
        df[col] = df[col].fillna(0).astype(float)
    # two-party total: denominator is dem + rep, not all votes cast
    df["total"] = df["dem"] + df["rep"]
    df["cell_id"] = df["cell_id"].astype(str)

    return df


def _plan_columns(df):
    """List the plan columns: every column that is not an id or vote column."""
    return [col for col in df.columns if col not in INFO_COLUMNS]


def _district_totals(df, plan):
    """
    Vote totals for every district in one plan.

    Returns a DataFrame with columns: district, dem, rep, total.
    """
    plan_df = df[[plan, "dem", "rep", "total"]].rename(columns={plan: "district"})

    # groupby skips blank districts, so say how many precincts that leaves out
    missing = int(plan_df["district"].isna().sum())
    if missing:
        warnings.warn(f"{plan}: {missing} precincts have no district and are "
                      f"left out of the metric")
    totals =plan_df.groupby("district", as_index=False)[["dem", "rep", "total"]].sum()

    empty = totals.loc[totals["total"] == 0, "district"].astype(str).tolist()
    if empty:
        warnings.warn(f"{plan}: {len(empty)} districts have no votes and "
                      f"are left out of the metric: {', '.join(empty)}")
    return totals[totals["total"] > 0]


# ---------------------------------------------------------------------------
# Metric math (private)
# ---------------------------------------------------------------------------

def _competitive_districts(district_totals, threshold):
    """Count districts where |dem - rep| / total is at or below the threshold."""
    margin = (district_totals["dem"] - district_totals["rep"]).abs() / district_totals["total"]
    return int((margin <= threshold).sum())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def read_plans(folder):
    """
    Read every plan CSV in a folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder of precinct assignment files, one CSV per plan. In each file
        the first column is the precinct id (``GEOID20``) and the second is
        the district (``District``).

    Returns
    -------
    dict
        ``{plan name: DataFrame}``, where the plan name is the file name
        without ``.csv``.
    """
    plans = {}
    for path in sorted(Path(folder).glob("*.csv")):
        # read as text so ids keep their leading zeros
        plans[path.stem] = pd.read_csv(path, dtype=str)
    return plans


def join_plans(votes, plans):
    """
    Combine precinct votes and plan assignments into one table.

    Parameters
    ----------
    votes : pandas.DataFrame or geopandas.GeoDataFrame
        One row per precinct. First column is the precinct id. Must include
        ``dem`` and ``rep`` vote columns; any other columns are dropped.
    plans : dict
        ``{plan name: DataFrame}``, as returned by ``read_plans``. In each
        DataFrame the first column is the precinct id and the second is the
        district.

    Returns
    -------
    pandas.DataFrame
        Precinct id, ``dem``, ``rep``, then one column per plan. A precinct
        that a plan does not list gets a blank district for that plan.
    """
    id_col = votes.columns[0]
    df = pd.DataFrame(votes)[[id_col, "dem", "rep"]].copy()
    ids = df[id_col].astype(str)

    for name, plan in plans.items():
        plan_ids = plan.iloc[:, 0].astype(str)
        if plan_ids.duplicated().any():
            raise ValueError(f"{name}: some precincts are listed more than once")
        # look up each precinct's district in this plan
        districts = pd.Series(plan.iloc[:, 1].values, index=plan_ids)
        df[name] = ids.map(districts)
    return df


def competitive_districts(df, threshold=0.05):
    """
    Calculate the number of competitive districts for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``dem`` and ``rep``
        vote columns. All other columns are plans, with values indicating
        district assignments.
    threshold : float, optional
        Largest gap between Democratic and Republican shares, as a fraction
        of the two-party vote, for a district to count as competitive.
        Defaults to 0.05 (5 percentage points).

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``competitive_districts``.

    Notes
    -----
    A district is competitive when:

    .. math::
        \\frac{|D - R|}{D + R} \\le 0.05

    Where :math:`D` and :math:`R` are the district's Democratic and
    Republican votes, summed over its precincts. A margin of exactly the
    threshold counts as competitive.

    Analyst notes: the denominator is the two-party vote, not all votes cast,
    so third-party candidates who have no chance of winning do not affect the
    score. The function requires two-party vote counts: the analyst decides
    which two candidates go in ``dem`` and ``rep``. Usually these are the
    Democratic and Republican candidates, but if a third-party candidate
    finished second, the analyst must know the state well enough to choose.
    ``total`` is computed as ``dem + rep``; any ``total`` column in the input
    is replaced.
    """
    df = _prepare_df(df)
    results = []
    for plan in _plan_columns(df):
        district_totals = _district_totals(df, plan)
        results.append({"plan": plan,
                        "competitive_districts": _competitive_districts(district_totals, threshold)})
    return pd.DataFrame(results)
