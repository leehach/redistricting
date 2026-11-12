"""One row of metrics per plan, ready for DEA.

    from redistricting import plan_metrics
    from redistricting.DEA import dea

    geounits = plan_metrics.load_pa_geounits()
    plans = plan_metrics.combine_plans({"plan_a": table_a, "wide": table_b})
    metrics = plan_metrics.measure(plans, geounits)
    scores = dea.score(metrics, plan_metrics.DIRECTION, "sbm")

The tables follow compactness.py:

- ``plans``: one row per geounit, indexed by geoid, and one column per
  plan that maps each geounit to its district.
- ``geounits``: one row per geounit, indexed by geoid, with a projected
  shape and the ``population``, ``county``, ``dem`` and ``rep`` columns.

For now the geounits are Pennsylvania 2020 voting districts (VTDs), so a
plan must assign VTDs. Dave's Redistricting App calls this export
"Precinct Assignment". Its default export assigns census blocks, which
this module does not read yet.
"""

import geopandas as gpd
import pandas as pd

from pa_redistricting.election_results import fetch_elections
from pa_redistricting.partisan_metrics import score_districts
from pa_redistricting.voting_districts import load_vtds

from redistricting import compactness, county_split

# Metric name -> "higher" or "lower" is better, in the form dea.score takes.
DIRECTION = {
    "polsby_popper": "higher",
    "max_pop_deviation": "lower",
    "split_counties": "lower",
    "efficiency_gap": "lower",
}

GEOID_COLUMNS = ["GEOID20", "GEOID"]
BLOCK_GEOID_LENGTH = 15


def load_pa_geounits() -> gpd.GeoDataFrame:
    """Download the Pennsylvania VTDs with population, county and votes.

    Shapes come from the Census TIGER/Line files. Population and votes come
    from the ALARM Project's VTD file, so no Census API key is needed.
    ``dem`` and ``rep`` are ALARM's average over every statewide race.
    """
    vtds = load_vtds()
    votes = fetch_elections()[["GEOID", "pop", "ndv", "nrv"]]

    geounits = vtds.merge(votes, on="GEOID", how="left").rename(
        columns={"pop": "population", "COUNTYFP": "county",
                 "ndv": "dem", "nrv": "rep"})
    geounits = geounits.set_index("GEOID")[
        ["population", "county", "dem", "rep", "geometry"]]

    # Areas in degrees are wrong, so compactness needs a projected CRS.
    return geounits.to_crs(geounits.estimate_utm_crs())


def combine_plans(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join plan tables into one plans table.

    Each key names its table. A table needs a geoid column (``GEOID20`` or
    ``GEOID``). With one other column, the table is one plan, named by its
    key: the "GEOID20,District" export of Dave's Redistricting App. With
    more, each other column is one plan, named by its column.
    """
    plans = []
    for name, table in tables.items():
        geoid = _geoid_column(table, name)
        table = table.assign(**{geoid: table[geoid].astype(str).str.strip()})
        table = table.set_index(geoid)

        if table.index.duplicated().any():
            raise ValueError(f"{name}: some geoids appear more than once")
        if len(table.columns) == 1:
            table.columns = [name]
        plans.append(table)

    combined = pd.concat(plans, axis=1)
    repeated = combined.columns[combined.columns.duplicated()]
    if len(repeated):
        raise ValueError(f"Plan names used twice: {sorted(set(repeated))}")
    return combined


def measure(plans: pd.DataFrame, geounits: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return one row per plan and one column per metric in DIRECTION.

    Every metric is 0 or more, as DEA needs:

    - ``polsby_popper``: mean over the districts. 1 is a circle.
    - ``max_pop_deviation``: the largest district's distance from the ideal
      population, in percent.
    - ``split_counties``: how many counties have more than one district.
    - ``efficiency_gap``: its absolute value, so it does not say which
      party the plan favors.
    """
    _check_geoids(plans, geounits)
    plans = plans.reindex(geounits.index)

    rows = {
        name: _measure_plan(plans[name], geounits) for name in plans.columns
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["polsby_popper"] = compactness.measure_plans(
        plans, geounits, "polsby_popper")
    return table[list(DIRECTION)]


def _measure_plan(plan: pd.Series, geounits: gpd.GeoDataFrame) -> dict:
    population = geounits["population"].groupby(plan).sum()
    ideal = population.sum() / len(population)

    votes = geounits[["dem", "rep"]].groupby(plan).sum()
    partisan = score_districts(votes)

    splits = county_split.score_plan(
        geounits["county"], geounits["population"], plan,
        metrics=["split_counties"])

    return {
        "max_pop_deviation": float(
            (population - ideal).abs().max() / ideal * 100),
        "split_counties": splits["split_counties"],
        "efficiency_gap": abs(partisan["efficiency_gap"]),
    }


def _geoid_column(table: pd.DataFrame, name: str) -> str:
    for column in GEOID_COLUMNS:
        if column in table.columns:
            return column
    raise ValueError(
        f"{name}: no geoid column. Expected one of {GEOID_COLUMNS}, "
        f"found {list(table.columns)}")


def _check_geoids(plans: pd.DataFrame, geounits: gpd.GeoDataFrame) -> None:
    if (plans.index.str.len() == BLOCK_GEOID_LENGTH).any():
        raise ValueError(
            "These plans assign census blocks. Only voting districts work "
            "for now: in Dave's Redistricting App, export "
            "\"Precinct Assignment\" instead.")

    unknown = plans.index.difference(geounits.index)
    if len(unknown):
        raise ValueError(
            f"{len(unknown)} geoids are not Pennsylvania voting districts, "
            f"for example {list(unknown[:3])}")

    unassigned = plans.reindex(geounits.index).isna().sum()
    unassigned = unassigned[unassigned > 0]
    if len(unassigned):
        raise ValueError(
            "Some plans leave voting districts with no district: "
            + ", ".join(f"{name} ({n})" for name, n in unassigned.items()))
