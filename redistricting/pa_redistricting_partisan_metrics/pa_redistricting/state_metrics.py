"""PlanScore metrics and population balance for a plan in ANY state.

The same calculations as the PA Senate workflow (partisan_metrics.py and
population.py), for a state loaded with state_data.load_state:

    from pa_redistricting.state_data import load_state
    from pa_redistricting.state_metrics import read_plan, score_plan
    gdf, _ = load_state("NC")
    score = score_plan(read_plan("nc_congress.csv"), gdf)
    score.metrics      # one row per election: seats, efficiency gap, partisan bias, ...
    score.districts    # {election: votes and winner per district}
    score.population   # population and deviation per district

A plan is a CSV with one row per voting district (Census block in CA, HI, OR):
a GEOID column (GEOID20 or GEOID) and a district column (District or district).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .partisan_metrics import partisan_summary
from .population import population_summary

log = logging.getLogger(__name__)

GEOID_NAMES = ["geoid20", "geoid", "geoid_20"]
DISTRICT_NAMES = ["district", "district_id", "districtid", "cd", "dist"]


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


def _column(df: pd.DataFrame, names: list[str], what: str) -> str:
    by_lower = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n in by_lower:
            return by_lower[n]
    raise ValueError(f"No {what} column; expected one of {names}, found {list(df.columns)}")


def read_plan(path: str | Path) -> pd.DataFrame:
    """Read a plan CSV into GEOID (text) and district (whole number).
    Rows with a blank district are treated as unassigned and dropped."""
    raw = pd.read_csv(path, dtype=str, skipinitialspace=True, encoding="utf-8-sig")
    geoid = _column(raw, GEOID_NAMES, "GEOID")
    district = _column(raw, DISTRICT_NAMES, "district")
    df = raw[[geoid, district]].set_axis(["GEOID", "district"], axis=1)
    df["GEOID"] = df["GEOID"].str.strip()
    df = df[df["district"].str.strip().fillna("") != ""]
    try:
        df["district"] = df["district"].astype(int)
    except ValueError:
        raise ValueError(f"{path}: district values must be whole numbers, e.g. 1, 2, 3")
    return df.reset_index(drop=True)


def plan_name(path: str | Path) -> str:
    """File name without extension, with anything but letters, digits, _ and - as _."""
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in Path(path).stem)


def score_plan(plan: pd.DataFrame, state: pd.DataFrame,
               elections: dict[str, tuple[str, str]] | None = None) -> PlanScore:
    """Partisan metrics (efficiency gap, partisan bias, mean-median, declination)
    and population balance for one plan.

    plan:  GEOID and district (read_plan)
    state: the joined table from load_state (with or without geometry)
    """
    data = pd.DataFrame(state.drop(columns="geometry", errors="ignore"))
    elections = elections or state_elections(data)

    missing = sorted(set(plan["GEOID"]) - set(data["GEOID"]))
    if missing:
        log.warning("%d of the plan's GEOIDs are not in this state's data, e.g. %s. "
                    "Is the plan for this state, and at the right level (VTD or block)?",
                    len(missing), missing[0])
    if len(missing) == len(plan):
        raise ValueError("None of the plan's GEOIDs are in this state's data")

    metrics, districts = partisan_summary(plan, data, elections)

    joined = data.merge(plan, on="GEOID", how="left")
    population, stats = population_summary(joined, n_districts=plan["district"].nunique())
    return PlanScore(metrics, districts, population, stats, missing)
