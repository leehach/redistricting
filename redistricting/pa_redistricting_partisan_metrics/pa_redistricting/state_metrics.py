"""PlanScore metrics and population balance for a plan in ANY state.

The same calculations as the PA Senate workflow (partisan_metrics.py and
population.py), for a state loaded with state_data.load_state:

    from pa_redistricting.plan_assignments import read_plans
    from pa_redistricting.state_data import load_state
    from pa_redistricting.state_metrics import score_plan
    gdf, _ = load_state("NC")
    for plan in read_plans("nc_plans/"):          # a CSV, a folder or a GitHub folder
        score = score_plan(plan.assignment, gdf)
    score.metrics      # one row per election: seats, efficiency gap, partisan bias, ...
    score.districts    # {election: votes and winner per district}
    score.population   # population and deviation per district

Plans are read by plan_assignments.read_plans (one row per voting district, or
Census block in CA, HI, OR; see that module for the accepted formats).
"""

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from .partisan_metrics import partisan_summary
from .plan_assignments import plan_name, read_plan  # noqa: F401  (kept importable here)
from .population import population_summary

log = logging.getLogger(__name__)


@dataclass
class PlanScore:
    metrics: pd.DataFrame                      # one row per election
    districts: dict[str, pd.DataFrame]         # election -> one row per district
    population: pd.DataFrame                   # one row per district
    population_stats: dict
    plan_geoids_not_in_state: list[str] = field(default_factory=list)


def state_elections(votes: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Every election the state's file has both party columns for:

    composite   ndv / nrv: average of all statewide races, all years
    avg_YY      adv_YY / arv_YY: average of all statewide races in 20YY
    pres_20YY   the presidential race, e.g. pre_20_dem_bid / pre_20_rep_tru

    composite comes first; the overview uses the first election.
    """
    cols = set(votes.columns)
    elections = {}
    if {"ndv", "nrv"} <= cols:
        elections["composite"] = ("ndv", "nrv")
    for yy in sorted({m[1] for c in cols if (m := re.fullmatch(r"adv_(\d\d)", c))}):
        if f"arv_{yy}" in cols:
            elections[f"avg_20{yy}"] = (f"adv_{yy}", f"arv_{yy}")
    for yy in sorted({m[1] for c in cols if (m := re.fullmatch(r"pre_(\d\d)_dem_\w+", c))}):
        dem = sorted(c for c in cols if re.fullmatch(rf"pre_{yy}_dem_\w+", c))
        rep = sorted(c for c in cols if re.fullmatch(rf"pre_{yy}_rep_\w+", c))
        if len(dem) == 1 and len(rep) == 1:
            elections[f"pres_20{yy}"] = (dem[0], rep[0])
    if not elections:
        raise ValueError("This state's data has no election columns, so partisan metrics "
                         "can't be calculated")
    return elections


def score_plan(plan: pd.DataFrame, state: pd.DataFrame,
               elections: dict[str, tuple[str, str]] | None = None) -> PlanScore:
    """Partisan metrics (efficiency gap, partisan bias, mean-median, declination)
    and population balance for one plan.

    plan:  GEOID and district (a Plan's assignment from read_plans)
    state: the joined table from load_state (with or without geometry)
    """
    data = pd.DataFrame(state.drop(columns="geometry", errors="ignore"))
    elections = elections or state_elections(data)
    if plan.empty:
        raise ValueError("The plan assigns no voting districts")

    plan_fips = plan["GEOID"].str[:2].mode()[0]
    state_fips = data["GEOID"].str[:2].mode()[0]
    if plan_fips != state_fips:
        raise ValueError(f"This plan's GEOIDs start with {plan_fips} but the state's start "
                         f"with {state_fips}: is it a plan for a different state?")

    plan_ids, data_ids = set(plan["GEOID"]), set(data["GEOID"])
    missing = sorted(plan_ids - data_ids)
    if missing:
        log.warning("%d of the plan's GEOIDs are not in this state's data, e.g. %s. "
                    "Is the plan at the right level (VTD or block)?",
                    len(missing), missing[0])
    if not plan_ids & data_ids:
        raise ValueError("None of the plan's GEOIDs are in this state's data")

    metrics, districts = partisan_summary(plan, data, elections)

    joined = data.merge(plan, on="GEOID", how="left")
    population, stats = population_summary(joined, n_districts=plan["district"].nunique())
    return PlanScore(metrics, districts, population, stats, missing)
