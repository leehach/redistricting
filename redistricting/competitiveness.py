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

``dem`` and ``rep`` are vote counts for the two leading candidates. Use the
``dem`` and ``rep`` arguments if your columns have other names. To score a
folder of plans:

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


def join_plans(votes, plans, dem="dem", rep="rep"):
    """
    Combine precinct votes and plan assignments into one table.

    Parameters
    ----------
    votes : pandas.DataFrame or geopandas.GeoDataFrame
        One row per precinct. First column is the precinct id. Must include
        Democratic and Republican vote columns; any other columns are dropped.
    plans : dict
        ``{plan name: DataFrame}``, as returned by ``read_plans``. In each
        DataFrame the first column is the precinct id and the second is the
        district.
    dem, rep : str, optional
        Names of the Democratic and Republican vote columns. Default to ``dem``
        and ``rep``.

    Returns
    -------
    pandas.DataFrame
        Precinct id, ``dem``, ``rep``, then one column per plan.

    Raises
    ------
    ValueError
        If a precinct id is blank or repeated, a plan has a blank district, or
        a plan and the votes table do not list the same precincts.

    Notes
    -----
    Analyst notes: every mismatch stops with an error, in both directions, as
    ``population_deviation`` does. A precinct missing from either table would
    otherwise change a district's score with no sign of it. A common cause is
    reading the votes file without ``dtype=str``: one blank id makes pandas store
    every id as a float (``42001000010.0``), and then no id matches the plans.
    """
    id_col = votes.columns[0]
    df = pd.DataFrame(votes)[[id_col, dem, rep]].rename(columns={dem: "dem", rep: "rep"})

    if df[id_col].isna().any():
        raise ValueError(f"{int(df[id_col].isna().sum())} row(s) in votes have a blank precinct id")
    ids = df[id_col].astype(str)
    if ids.duplicated().any():
        raise ValueError("votes: some precincts are listed more than once")

    for name, plan in plans.items():
        plan_ids = plan.iloc[:, 0].astype(str)
        if plan_ids.duplicated().any():
            raise ValueError(f"{name}: some precincts are listed more than once")
        if plan.iloc[:, 1].isna().any():
            raise ValueError(f"{name}: {int(plan.iloc[:, 1].isna().sum())} precinct(s) have no district")

        not_in_plan = sorted(set(ids) - set(plan_ids))
        not_in_votes = sorted(set(plan_ids) - set(ids))
        if not_in_plan or not_in_votes:
            raise ValueError(f"{name}: precinct ids do not match votes. "
                             f"{len(not_in_plan)} in votes but not the plan: {not_in_plan[:10]}. "
                             f"{len(not_in_votes)} in the plan but not votes: {not_in_votes[:10]}. "
                             "If ids look like 42001000010.0, read the votes file with dtype=str.")

        # look up each precinct's district in this plan
        districts = pd.Series(plan.iloc[:, 1].values, index=plan_ids)
        df[name] = ids.map(districts)
    return df


def competitive_districts(df, threshold=0.05, exclude=None, dem="dem", rep="rep"):
    """
    Calculate the number of competitive districts for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include Democratic and
        Republican vote columns. All other columns are plans, with values
        indicating district assignments.
    threshold : float, optional
        Largest gap between Democratic and Republican shares, as a fraction
        of the two-party vote, for a district to count as competitive.
        Defaults to 0.05 (5 percentage points).
    exclude : str or list of str, optional
        Columns that are not plans, such as ``county`` or ``area``. The id,
        ``dem``, ``rep``, and geometry columns are always left out.
    dem, rep : str, optional
        Names of the Democratic and Republican vote columns. Default to ``dem``
        and ``rep``.

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``competitive_districts``.

    Raises
    ------
    ValueError
        If a vote column is missing or has a blank cell, ``exclude`` names a
        column that is not in ``df``, or a plan has no district with votes.

    Notes
    -----
    A district is competitive when:

    .. math::
        \\frac{|D - R|}{D + R} \\le 0.05

    Where :math:`D` and :math:`R` are the district's Democratic and
    Republican votes, summed over its precincts. A margin of exactly the
    threshold counts as competitive.

    Analyst notes: the 5-point cutoff comes from the contest studied by Gopalan
    et al. (2026), which compared party registration percentages. This function
    applies it to vote counts. The denominator is the two-party vote, not all votes cast,
    so third-party candidates who have no chance of winning do not affect the
    score. The function requires two-party vote counts: the analyst decides
    which two candidates go in ``dem`` and ``rep``. Usually these are the
    Democratic and Republican candidates, but if a third-party candidate
    finished second, the analyst must know the state well enough to choose.
    ``total`` is computed as ``dem + rep``; any ``total`` column in the input
    is replaced.
    """
    exclude = _check_exclude(df, exclude)
    df = _prepare_df(df, dem, rep)
    results = []
    for plan in _plan_columns(df, exclude):
        district_totals = _district_totals(df, plan)
        results.append({"plan": plan,
                        "competitive_districts": _competitive_districts(district_totals, threshold)})
    return pd.DataFrame(results)
