"""2020 Census redistricting data (PL 94-171) for Pennsylvania voting districts.

    from pa_redistricting.census import fetch_census, fetch_codebook
    census = fetch_census(api_key="...")   # or set CENSUS_API_KEY
    codebook = fetch_codebook()            # Census variable code and label per column

Nothing here prompts for input. A missing API key raises MissingAPIKeyError.
"""

import logging
import os

import pandas as pd
import requests

from .config import CENSUS_API_URL, CENSUS_KEY_SIGNUP_URL, STATE_FIPS
from .demographics import RACES, add_percentages

log = logging.getLogger(__name__)

PA_COUNTIES = [f"{i:03d}" for i in range(1, 134, 2)]  # 001, 003, ... 133

# Counts VEST also has (demographics.COUNT_COLUMNS).
# P2 / P4: Hispanic or Latino, and not Hispanic by race (total / 18+).
# _001N total, _002N Hispanic, _005N.._011N not Hispanic: White .. two or more races
SHARED_VARIABLES = {
    "P2_001N": "pop_total", "P2_002N": "pop_hispanic",
    **{f"P2_{i:03d}N": f"pop_nh_{r}" for i, r in zip(range(5, 12), RACES)},
    "P4_001N": "vap_total", "P4_002N": "vap_hispanic",
    **{f"P4_{i:03d}N": f"vap_nh_{r}" for i, r in zip(range(5, 12), RACES)},
}

# Only in the Census API
EXTRA_VARIABLES = {
    # P1 / P3: race alone, INCLUDING Hispanics (_003N White .. _009N two or more)
    **{f"P1_{i:03d}N": f"pop_{r}_incl_hisp" for i, r in zip(range(3, 10), RACES)},
    **{f"P3_{i:03d}N": f"vap_{r}_incl_hisp" for i, r in zip(range(3, 10), RACES)},
    # P5: group quarters (correctional = adult correctional facilities)
    "P5_001N": "gq_total",
    "P5_003N": "gq_correctional",
    # H1: housing occupancy
    "H1_001N": "housing_total",
    "H1_002N": "housing_occupied",
    "H1_003N": "housing_vacant",
}

VARIABLES = {**SHARED_VARIABLES, **EXTRA_VARIABLES}   # 37 codes; the API allows 50 per request
EXTRA_COLUMNS = list(EXTRA_VARIABLES.values())

KEY_HELP = f"""No Census API key found.

Pass one as fetch_census(api_key=...) or set the CENSUS_API_KEY environment
variable. Keys are free: {CENSUS_KEY_SIGNUP_URL}

To save it in the gus8066 conda environment (one-time setup):
  conda env config vars set CENSUS_API_KEY=PASTE_KEY_HERE -n gus8066
  conda deactivate
  conda activate gus8066"""


class MissingAPIKeyError(RuntimeError):
    """No Census API key was passed in or found in CENSUS_API_KEY."""


def get_api_key(api_key: str | None = None) -> str:
    """The key passed in, else CENSUS_API_KEY. Raises MissingAPIKeyError if neither."""
    key = (api_key or os.environ.get("CENSUS_API_KEY", "")).strip()
    if not key:
        raise MissingAPIKeyError(KEY_HELP)
    return key


def fetch_vtd_data(api_key: str | None = None) -> pd.DataFrame:
    """Raw counts for every PA voting district, requested one county at a time."""
    api_key = get_api_key(api_key)
    frames = []
    for county in PA_COUNTIES:
        url = (f"{CENSUS_API_URL}?get=NAME,{','.join(VARIABLES)}"
               f"&for=voting%20district:*&in=state:{STATE_FIPS}&in=county:{county}"
               f"&key={api_key}")
        r = requests.get(url, timeout=120)
        try:
            rows = r.json()
        except ValueError:
            raise RuntimeError(f"County {county}: the Census API did not return data "
                               f"(HTTP {r.status_code}). It said:\n{r.text[:800]}")
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
        log.debug("county %s: %d voting districts", county, len(rows) - 1)

    df = pd.concat(frames, ignore_index=True)
    df["GEOID"] = df["state"] + df["county"] + df["voting district"]
    df = df.rename(columns=VARIABLES).rename(columns={"NAME": "VTD_NAME_CENSUS",
                                                      "county": "COUNTYFP"})
    for col in VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def fetch_census(api_key: str | None = None) -> pd.DataFrame:
    """One row per PA voting district: GEOID, names, counts and percentages.
    api_key defaults to the CENSUS_API_KEY environment variable."""
    api_key = get_api_key(api_key)
    log.info("Requesting PA voting-district data from the Census API ...")
    df = add_percentages(fetch_vtd_data(api_key))

    front = ["GEOID", "VTD_NAME_CENSUS", "COUNTYFP"]
    rest = [c for c in df.columns if c not in front + ["state", "voting district"]]
    return df[front + rest].sort_values("GEOID").reset_index(drop=True)


def fetch_codebook() -> pd.DataFrame:
    """The official Census label for each column in fetch_census()."""
    r = requests.get(f"{CENSUS_API_URL}/variables.json", timeout=120)
    r.raise_for_status()
    labels = r.json()["variables"]
    return pd.DataFrame([{"column": name, "census_code": code,
                          "label": labels.get(code, {}).get("label", ""),
                          "table": labels.get(code, {}).get("concept", "")}
                         for code, name in VARIABLES.items()])
