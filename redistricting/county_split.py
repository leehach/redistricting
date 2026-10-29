"""
County-splitting metrics for redistricting plans.

Four metrics measure how a plan divides counties between districts:

    split_counties               number of counties split between 2+ districts
    county_split_intersections   number of district pieces inside split counties
    spill_ratio                  share of people outside their county's largest district piece
    county_hhi                   how concentrated each county's population is in one district

The first two count splits; the last two measure how many people the splits
affect. Each metric has a public function, and ``all_metrics`` runs any
combination of them at once.

Input is one table with one row per precinct (or block):

    precinct_id  county  pop   plan_a1  plan_a2
    42001000001  42001   1200  22       23
    42001000002  42001   850   23       23

The first column is the precinct id (any name). ``county`` and ``pop`` are
found by name. Every other column is a plan, with district assignments as
values. A GeoDataFrame also works; its geometry column is ignored.

References
----------
Wachspress, J. and Adler, W. T. (2021). Split Decisions: Guidance for
Measuring Locality Preservation in District Maps. Center for Democracy &
Technology. https://github.com/jacobwachspress/locality-splitting

Waggoner, P. D. (2018). hhi: Calculate and Visualize the
Herfindahl-Hirschman Index. https://github.com/pdwaggoner/hhi

Herfindahl-Hirschman index. Wikipedia.
https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index
"""

import warnings

import pandas as pd

# columns that are not plans
INFO_COLUMNS = ["cell_id", "county", "pop", "geometry"]


# ---------------------------------------------------------------------------
# Data prep (private)
# ---------------------------------------------------------------------------

def _prepare_df(df):
    """
    Standardize the input table and note any empty precincts.

    Renames the first column to ``cell_id``, drops a geometry column if one is
    present, and sets ``county`` to text and ``pop`` to numbers. Precincts
    with no population are listed in a warning; they add nothing to any
    metric.
    """
    # copy so the caller's table is not altered
    df = pd.DataFrame(df).copy()
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df = df.drop(columns="geometry", errors="ignore")

    df["cell_id"] = df["cell_id"].astype(str)
    df["county"] = df["county"].astype(str)
    df["pop"] = df["pop"].fillna(0).astype(float)

    empty = df.loc[df["pop"] == 0, "cell_id"].tolist()
    if empty:
        warnings.warn(f"{len(empty)} precincts have no population and are "
                      f"left out of the metrics: {', '.join(empty)}")
    return df


def _plan_columns(df):
    """List the plan columns: every column that is not an id, county, or pop column."""
    return [col for col in df.columns if col not in INFO_COLUMNS]


def _county_district_pop(df, plan):
    """
    Population of every county-district piece for one plan.

    Pieces with no population are dropped, so a district line that only
    crosses empty precincts does not count as a split.

    Returns a DataFrame with columns: county, district, pop.
    """
    plan_df = df[["county", plan, "pop"]].rename(columns={plan: "district"})
    pieces = plan_df.groupby(["county", "district"], as_index=False)["pop"].sum()
    return pieces[pieces["pop"] > 0]


# ---------------------------------------------------------------------------
# Metric math (private): each takes the pieces table for one plan
# and returns one number
# ---------------------------------------------------------------------------

def _split_counties(pieces):
    """Count counties that have population in two or more districts."""
    districts_per_county = pieces.groupby("county")["district"].nunique()
    return int((districts_per_county > 1).sum())


def _county_split_intersections(pieces):
    """Sum the number of district pieces in each split county; whole counties add 0."""
    districts_per_county = pieces.groupby("county")["district"].nunique()
    return int(districts_per_county[districts_per_county > 1].sum())


def _spill_ratio(pieces):
    """Share of people living outside their county's largest district piece."""
    county_pop = pieces.groupby("county")["pop"].sum()
    largest_piece = pieces.groupby("county")["pop"].max()
    return float(1 - largest_piece.sum() / county_pop.sum())


def _county_hhi(pieces):
    """Population-weighted mean of each county's sum of squared district shares."""
    county_pop = pieces.groupby("county")["pop"].sum()
    share = pieces["pop"] / pieces.groupby("county")["pop"].transform("sum")
    hhi_per_county = (share ** 2).groupby(pieces["county"]).sum()
    return float((hhi_per_county * county_pop).sum() / county_pop.sum())


# metric name -> private function that calculates it
METRICS = {
    "split_counties": _split_counties,
    "county_split_intersections": _county_split_intersections,
    "spill_ratio": _spill_ratio,
    "county_hhi": _county_hhi,
}


def _score_plans(df, metric_names):
    """Run the named metrics on every plan; return one row per plan."""
    unknown = [name for name in metric_names if name not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown}. Choose from {list(METRICS)}")

    df = _prepare_df(df)
    results = []
    for plan in _plan_columns(df):
        pieces = _county_district_pop(df, plan)
        row = {"plan": plan}
        for name in metric_names:
            row[name] = METRICS[name](pieces)
        results.append(row)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def split_counties(df):
    """
    Calculate the number of split counties for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``county`` and
        ``pop`` columns. All other columns are plans, with values indicating
        district assignments.

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``split_counties``.

    Notes
    -----
    A county is split when its population falls in two or more districts:

    .. math::
        SC = \\sum_c [k_c > 1]

    Where :math:`k_c` is the number of districts holding population from
    county :math:`c`. This treats a county cut in 2 the same as one cut in 5;
    see ``county_split_intersections`` for a count that tells them apart.

    Analyst notes: precincts with no population are ignored, so a district
    line that only crosses empty precincts is not a split. Equivalent to
    ``splits_pop`` in the locality-splitting package (Wachspress and Adler
    2021).
    """
    return _score_plans(df, ["split_counties"])


def county_split_intersections(df):
    """
    Calculate the number of district pieces within split counties for each plan.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``county`` and
        ``pop`` columns. All other columns are plans, with values indicating
        district assignments.

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``county_split_intersections``.

    Notes
    -----
    For each split county, the number of districts it touches is counted, and
    these counts are summed:

    .. math::
        CSI = \\sum_{c \\,:\\, k_c > 1} k_c

    Where :math:`k_c` is the number of districts holding population from
    county :math:`c`. Whole counties add 0, so a county cut in 4 adds 4.

    Analyst notes: adapted from ``locality_intersections`` in the
    locality-splitting package (Wachspress and Adler 2021), which also counts
    each whole county as 1 piece. Precincts with no population are ignored.
    """
    return _score_plans(df, ["county_split_intersections"])


def spill_ratio(df):
    """
    Calculate the spill ratio for each plan: the share of people who live
    outside their county's largest district piece.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``county`` and
        ``pop`` columns. All other columns are plans, with values indicating
        district assignments.

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``spill_ratio``. 0 means no
        county is split; higher values mean more people are separated from
        the rest of their county.

    Notes
    -----
    For each county, the district holding the most of its population is the
    dominant district. Everyone else in the county has "spilled":

    .. math::
        SR = 1 - \\frac{\\sum_c \\max_d p_{cd}}{\\sum_c p_c}

    Where :math:`p_{cd}` is the population of county :math:`c` in district
    :math:`d` and :math:`p_c` is the county's total population. This equals
    the population-weighted mean of each county's
    :math:`1 - \\max_d p_{cd} / p_c`.

    Analyst notes: weighting county scores by population is our choice; an
    unweighted mean would treat a small county the same as a large one. The
    PA LRC used prisoner-adjusted populations, so results can differ slightly
    from official figures. Built from the county-district population table
    described by Wachspress and Adler (2021).
    """
    return _score_plans(df, ["spill_ratio"])


def county_hhi(df):
    """
    Calculate the county Herfindahl index for each plan: how concentrated
    each county's population is within a single district.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``county`` and
        ``pop`` columns. All other columns are plans, with values indicating
        district assignments.

    Returns
    -------
    pandas.DataFrame
        One row per plan with columns ``plan`` and ``county_hhi``. 1 means no
        county is split; lower values mean county populations are spread
        across more districts.

    Notes
    -----
    Each county's HHI is the sum of its squared district shares:

    .. math::
        HHI_c = \\sum_d \\left( \\frac{p_{cd}}{p_c} \\right)^2

    Where :math:`p_{cd}` is the population of county :math:`c` in district
    :math:`d` and :math:`p_c` is the county's total population. A whole county
    scores 1, a county split evenly in 2 scores 0.5, and in 3 about 0.33. The
    plan score is the population-weighted mean of :math:`HHI_c`.

    Analyst notes: the per-county formula matches ``hhi()`` in the
    pdwaggoner/hhi R package (Waggoner 2018) and is related to effective
    splits in the locality-splitting package by :math:`ES_c = 1 / HHI_c - 1`.
    Weighting county scores by population is our choice.
    """
    return _score_plans(df, ["county_hhi"])


def all_metrics(df, metrics=None):
    """
    Calculate several county-splitting metrics for every plan at once.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        First column contains precinct ids. Must include ``county`` and
        ``pop`` columns. All other columns are plans, with values indicating
        district assignments.
    metrics : list of str, optional
        Metrics to calculate, any of ``"split_counties"``,
        ``"county_split_intersections"``, ``"spill_ratio"``, ``"county_hhi"``.
        Defaults to all four.

    Returns
    -------
    pandas.DataFrame
        One row per plan with a ``plan`` column and one column per metric.

    Notes
    -----
    Two plans can split the same number of counties but divide very
    different numbers of residents, so counting metrics and population
    metrics are best reported together. See each metric's function for its
    formula.

    Examples
    --------
    >>> all_metrics(plans)                                   # all four
    >>> all_metrics(plans, ["split_counties", "spill_ratio"])  # pick some
    """
    if metrics is None:
        metrics = list(METRICS)
    return _score_plans(df, metrics)