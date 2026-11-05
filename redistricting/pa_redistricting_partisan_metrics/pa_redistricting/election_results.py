"""Election results for each Pennsylvania voting district, from the ALARM Project.

    from pa_redistricting.election_results import fetch_elections, available_elections
    votes = fetch_elections()               # one row per VTD, keyed by GEOID
    elections = available_elections(votes)  # the ELECTIONS found in the file

Columns follow ALARM's off_yr_par_can pattern (e.g. pre_20_dem_bid). ndv / nrv
average the Democratic and Republican vote across every statewide race.
Source: https://github.com/alarm-redist/census-2020
"""

import io
import logging

import pandas as pd
import requests

from .config import ELECTIONS_URL

log = logging.getLogger(__name__)

# Elections to score: name -> (Democratic column, Republican column).
# Statewide races stand in for Senate voting: many Senate races are uncontested
# and only half the seats are up each cycle.
ELECTIONS = {
    "composite": ("ndv", "nrv"),
    "pres_2020": ("pre_20_dem_bid", "pre_20_rep_tru"),
    "pres_2016": ("pre_16_dem_cli", "pre_16_rep_tru"),
    "ussen_2018": ("uss_18_dem_cas", "uss_18_rep_bar"),
}


def fetch_elections() -> pd.DataFrame:
    """VTD election results with GEOID20 renamed to GEOID."""
    log.info("Downloading VTD election results from the ALARM Project ...")
    r = requests.get(ELECTIONS_URL, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), dtype={"GEOID20": str})
    return df.rename(columns={"GEOID20": "GEOID"}).assign(GEOID=lambda d: d["GEOID"].str.strip())


def vote_columns(votes: pd.DataFrame) -> list[str]:
    """Every Democratic/Republican vote column in the file."""
    return [c for c in votes.columns if "_dem_" in c or "_rep_" in c
            or c[:3] in ("adv", "arv", "ndv", "nrv")]


def available_elections(votes: pd.DataFrame,
                        elections: dict[str, tuple[str, str]] = ELECTIONS
                        ) -> dict[str, tuple[str, str]]:
    """The elections whose two columns are both in the file.
    Raises ValueError if none are."""
    found = {k: v for k, v in elections.items() if set(v) <= set(votes.columns)}
    missing = [k for k in elections if k not in found]
    if missing:
        log.warning("Skipping elections not in the file: %s. Vote columns available: %s",
                    ", ".join(missing), ", ".join(vote_columns(votes)))
    if not found:
        raise ValueError(f"None of the requested elections are in the file. "
                         f"Vote columns available: {', '.join(vote_columns(votes))}")
    return found
