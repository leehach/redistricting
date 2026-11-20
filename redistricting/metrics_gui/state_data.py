"""Precinct data for any state: shapes, VEST election + demographic data, Census demographics.
no Qt in here so it can be used (and tested) without the window

    from state_data import load_shapes, load_vest, load_census
    shapes = load_shapes("NC")   # GeoDataFrame: GEOID, NAME, COUNTYFP, ALAND, AWATER, geometry
    vest = load_vest("NC")       # ALARM Project file: votes + demographics, keyed by GEOID
    census = load_census("NC")   # Census API PL 94-171 counts (needs CENSUS_API_KEY)

Everything is by 2020 voting district, except CA, HI, ME and OR, which are by Census block
(ALARM only has blocks for those). Downloads are cached in ~/.redistricting_cache (or
REDISTRICTING_CACHE); refresh=True downloads again.

Sources:
  shapes  https://www2.census.gov/geo/tiger/TIGER2020PL/
  VEST    https://github.com/alarm-redist/census-2020 (census-vest-2020)
  Census  https://api.census.gov/data/2020/dec/pl
"""

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# our package sits next to this folder in the repo, no install needed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pa_redistricting_partisan_metrics"))
from pa_redistricting.census import (VARIABLES, MissingAPIKeyError, add_percentages,  # noqa: F401
                                     get_api_key)

# abbreviation -> (name, FIPS code)
STATES = {
    "AL": ("Alabama", "01"), "AK": ("Alaska", "02"), "AZ": ("Arizona", "04"),
    "AR": ("Arkansas", "05"), "CA": ("California", "06"), "CO": ("Colorado", "08"),
    "CT": ("Connecticut", "09"), "DE": ("Delaware", "10"),
    "DC": ("District of Columbia", "11"), "FL": ("Florida", "12"),
    "GA": ("Georgia", "13"), "HI": ("Hawaii", "15"), "ID": ("Idaho", "16"),
    "IL": ("Illinois", "17"), "IN": ("Indiana", "18"), "IA": ("Iowa", "19"),
    "KS": ("Kansas", "20"), "KY": ("Kentucky", "21"), "LA": ("Louisiana", "22"),
    "ME": ("Maine", "23"), "MD": ("Maryland", "24"), "MA": ("Massachusetts", "25"),
    "MI": ("Michigan", "26"), "MN": ("Minnesota", "27"), "MS": ("Mississippi", "28"),
    "MO": ("Missouri", "29"), "MT": ("Montana", "30"), "NE": ("Nebraska", "31"),
    "NV": ("Nevada", "32"), "NH": ("New Hampshire", "33"), "NJ": ("New Jersey", "34"),
    "NM": ("New Mexico", "35"), "NY": ("New York", "36"),
    "NC": ("North Carolina", "37"), "ND": ("North Dakota", "38"), "OH": ("Ohio", "39"),
    "OK": ("Oklahoma", "40"), "OR": ("Oregon", "41"), "PA": ("Pennsylvania", "42"),
    "RI": ("Rhode Island", "44"), "SC": ("South Carolina", "45"),
    "SD": ("South Dakota", "46"), "TN": ("Tennessee", "47"), "TX": ("Texas", "48"),
    "UT": ("Utah", "49"), "VT": ("Vermont", "50"), "VA": ("Virginia", "51"),
    "WA": ("Washington", "53"), "WV": ("West Virginia", "54"),
    "WI": ("Wisconsin", "55"), "WY": ("Wyoming", "56"),
}
BLOCK_STATES = {"CA", "HI", "ME", "OR"}  # ALARM's file is by Census block for these

TIGER_ROOT = "https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/"
TIGER_LAYER = {"vtd": "vtd20", "block": "tabblock20"}
SHAPE_COLUMNS = {"GEOID20": "GEOID", "NAME20": "NAME", "COUNTYFP20": "COUNTYFP",
                 "ALAND20": "ALAND", "AWATER20": "AWATER"}  # same in both layers
ALARM_URL = ("https://raw.githubusercontent.com/alarm-redist/census-2020/main/"
             "census-vest-2020/{abbr}_2020_{level}.csv")
CENSUS_URL = "https://api.census.gov/data/2020/dec/pl"

# Census P1/P3 race counts include Hispanics, VEST's pop_white etc. don't. renamed so they don't clash
INCL_HISP = {name: f"{name}_incl_hisp" for code, name in VARIABLES.items()
             if code[:3] in ("P1_", "P3_") and not code.endswith("_001N")}


def _quiet(text):  # default for say=
    pass


def state_abbr(state: str) -> str:
    """'pa', 'Pennsylvania' or '42' -> 'PA'. raises ValueError"""
    text = str(state).strip()
    if text.upper() in STATES:
        return text.upper()
    for abbr, (name, fips) in STATES.items():
        if text.lower() == name.lower() or (text.isdigit() and text.zfill(2) == fips):
            return abbr
    raise ValueError(f"Unknown state: {state!r}")


def geography(state: str) -> str:
    """'block' for CA, HI, ME, OR, 'vtd' (2020 voting districts) everywhere else"""
    return "block" if state_abbr(state) in BLOCK_STATES else "vtd"


def unit_name(state: str) -> str:
    return "Census blocks" if geography(state) == "block" else "voting districts"


# --- downloading, with a cache ---

def cache_dir() -> Path:
    return Path(os.environ.get("REDISTRICTING_CACHE", Path.home() / ".redistricting_cache"))


def _download(url: str, path: Path) -> None:
    """to a .partial file first, so an interrupted download never leaves a broken file behind"""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    partial.replace(path)


def _cached(url: str, filename: str, refresh: bool) -> Path:
    path = cache_dir() / filename
    if refresh or not path.exists():
        _download(url, path)
    return path


# --- shapes ---

def tiger_url(state: str, folder: str | None = None) -> str:
    abbr = state_abbr(state)
    name, fips = STATES[abbr]
    folder = folder or f"{fips}_{name.upper().replace(' ', '_')}"
    return f"{TIGER_ROOT}{folder}/{fips}/tl_2020_{fips}_{TIGER_LAYER[geography(abbr)]}.zip"


def find_tiger_folder(state: str) -> str:
    """the state's folder name from the Census directory listing, for when the usual one 404s"""
    fips = STATES[state_abbr(state)][1]
    r = requests.get(TIGER_ROOT, timeout=120)
    r.raise_for_status()
    found = re.findall(rf'href="({fips}_[A-Za-z_]+)/?"', r.text)
    if not found:
        raise RuntimeError(f"No folder for FIPS {fips} at {TIGER_ROOT}")
    return found[0]


def load_shapes(state: str, refresh: bool = False, say: Callable[[str], None] = _quiet
                ) -> gpd.GeoDataFrame:
    """2020 TIGER/Line voting districts (blocks for CA, HI, ME, OR):
    GEOID, NAME, COUNTYFP, ALAND, AWATER, geometry"""
    abbr = state_abbr(state)
    name, fips = STATES[abbr]
    filename = f"tl_2020_{fips}_{TIGER_LAYER[geography(abbr)]}.zip"
    say(f"Getting {name} {unit_name(abbr)} shapes...")
    try:
        path = _cached(tiger_url(abbr), filename, refresh)
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise
        path = _cached(tiger_url(abbr, find_tiger_folder(abbr)), filename, refresh)
    gdf = gpd.read_file(path).rename(columns=SHAPE_COLUMNS)
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.strip()
    return gdf[[*SHAPE_COLUMNS.values(), "geometry"]]


# --- VEST (ALARM Project) ---

def load_vest(state: str, refresh: bool = False, say: Callable[[str], None] = _quiet
              ) -> pd.DataFrame:
    """ALARM's census + VEST election file: one row per voting district (block for CA, HI, ME,
    OR), GEOID20 renamed GEOID. votes (ndv, nrv, pre_20_dem_bid ...) and demographics (pop ...)"""
    abbr = state_abbr(state)
    level = geography(abbr)
    say(f"Getting {STATES[abbr][0]} VEST election + demographic data...")
    path = _cached(ALARM_URL.format(abbr=abbr.lower(), level=level),
                   f"{abbr.lower()}_2020_{level}.csv", refresh)
    df = pd.read_csv(path, dtype={"GEOID20": str, "state": str, "county": str})
    return df.rename(columns={"GEOID20": "GEOID"}).assign(GEOID=lambda d: d["GEOID"].str.strip())


# --- Census API ---

def _census_rows(params: str, api_key: str, what: str) -> pd.DataFrame:
    url = f"{CENSUS_URL}?get=NAME,{','.join(VARIABLES)}{params}&key={api_key}"
    r = requests.get(url, timeout=300)
    try:
        rows = r.json()
    except ValueError:
        raise RuntimeError(f"{what}: the Census API did not return data "
                           f"(HTTP {r.status_code}). It said:\n{r.text[:800]}") from None
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_census(state: str, api_key: str | None = None, say: Callable[[str], None] = _quiet
                 ) -> pd.DataFrame:
    """PL 94-171 counts from the Census API, at the same level as the state's VEST file.
    voting districts come in one request; blocks one county at a time"""
    abbr = state_abbr(state)
    name, fips = STATES[abbr]
    api_key = get_api_key(api_key)  # MissingAPIKeyError if there isn't one
    if geography(abbr) == "vtd":
        say(f"Getting {name} Census data...")
        df = _census_rows(f"&for=voting%20district:*&in=state:{fips}&in=county:*", api_key, name)
        df["GEOID"] = df["state"] + df["county"] + df["voting district"]
    else:
        counties = _census_rows(f"&for=county:*&in=state:{fips}", api_key, name)["county"]
        frames = []
        for i, county in enumerate(sorted(counties), 1):
            say(f"Getting {name} Census blocks, county {i} of {len(counties)}...")
            frames.append(_census_rows(f"&for=block:*&in=state:{fips}&in=county:{county}"
                                       "&in=tract:*", api_key, f"{name} county {county}"))
        df = pd.concat(frames, ignore_index=True)
        df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block"]

    df = df.rename(columns=VARIABLES).rename(columns={"NAME": "NAME_CENSUS",
                                                      "county": "COUNTYFP"})
    for col in VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = add_percentages(df).rename(columns=INCL_HISP)
    drop = {"state", "voting district", "tract", "block"}
    front = ["GEOID", "NAME_CENSUS", "COUNTYFP"]
    rest = [c for c in df.columns if c not in front and c not in drop]
    return df[front + rest].sort_values("GEOID").reset_index(drop=True)


def load_census(state: str, refresh: bool = False, api_key: str | None = None,
                say: Callable[[str], None] = _quiet) -> pd.DataFrame:
    """fetch_census, saved to the cache after the first time"""
    abbr = state_abbr(state)
    path = cache_dir() / f"{abbr.lower()}_2020_{geography(abbr)}_census.csv"
    if path.exists() and not refresh:
        say(f"Getting {STATES[abbr][0]} Census data...")
        return pd.read_csv(path, dtype={"GEOID": str, "COUNTYFP": str})
    df = fetch_census(abbr, api_key, say)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def has_census_key() -> bool:
    try:
        get_api_key()
        return True
    except MissingAPIKeyError:
        return False


# what the window's download dialog offers: key -> (checkbox text, loader)
SOURCES = {
    "shapes": ("Shapes (2020 Census TIGER/Line)", load_shapes),
    "vest": ("VEST election results + demographics (ALARM Project)", load_vest),
    "census": ("Census demographics (2020 PL 94-171, needs CENSUS_API_KEY)", load_census),
}
LABELS = {"shapes": "{abbr} shapes", "vest": "{abbr} VEST data", "census": "{abbr} Census data"}
