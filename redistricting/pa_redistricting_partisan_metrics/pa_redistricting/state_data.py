"""Shapes, demographics and votes for ANY state, joined on GEOID.

The same join the PA code does (TIGER shapes + ALARM/VEST file), for a state
the user picks:

    from pa_redistricting.state_data import pick_state, load_state
    abbr = pick_state()                 # asks in the terminal which state
    gdf, report = load_state(abbr)      # GeoDataFrame, one row per voting district

Downloads are kept in ~/.redistricting_cache (set REDISTRICTING_CACHE to use
another folder): each state downloads once, then loads from the cache, offline
too. Pass refresh=True to download again.

California, Hawaii and Oregon have no 2020 voting districts, so for them both
the shapes and the ALARM data are Census blocks. Those downloads are large.

Sources:
  shapes  https://www2.census.gov/geo/tiger/TIGER2020PL/
  data    https://alarm-redist.org/posts/2021-08-10-census-2020/
"""

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from . import config
from .config import ALARM_STATE_URL, TIGER_STATE_ROOT
from .demographics import COUNT_COLUMNS, vest_demographics
from .election_results import vote_columns
from .voting_districts import VTD_COLUMNS

log = logging.getLogger(__name__)

# Abbreviation -> (name, FIPS code)
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

BLOCK_STATES = {"CA", "HI", "OR"}          # no 2020 voting districts: use Census blocks
TIGER_LAYER = {"vtd": "vtd20", "block": "tabblock20"}
MIN_MATCH = 0.9     # warn when fewer of the data's GEOIDs than this match the shapes


# --- Choosing a state -------------------------------------------------------

def find_state(text: str) -> str | None:
    """The abbreviation for a state given as an abbreviation, name or FIPS code
    ("pa", "Pennsylvania", "42"). None if nothing matches."""
    text = text.strip()
    if text.upper() in STATES:
        return text.upper()
    for abbr, (name, fips) in STATES.items():
        if text.lower() == name.lower() or (text.isdigit() and text.zfill(2) == fips):
            return abbr
    return None


def geography(abbr: str) -> str:
    """"block" for CA, HI and OR, "vtd" (voting districts) everywhere else."""
    return "block" if abbr.upper() in BLOCK_STATES else "vtd"


def state_menu(columns: int = 3) -> str:
    """Every state as a text menu, in columns."""
    items = [f"{abbr}  {name}" for abbr, (name, _) in STATES.items()]
    rows = -(-len(items) // columns)
    return "\n".join("".join(f"{items[j]:<26}" for j in range(i, len(items), rows)).rstrip()
                     for i in range(rows))


def pick_state(ask: Callable[[str], str] = input) -> str:
    """Show the states and ask until the user names one. Returns its abbreviation.
    ask reads the answer (input by default; pass another function from a GUI or test)."""
    print(state_menu())
    while True:
        answer = ask("\nWhich state? Type an abbreviation, name or FIPS code: ")
        abbr = find_state(answer)
        if abbr:
            note = (" (no 2020 voting districts, so Census blocks: a large download)"
                    if abbr in BLOCK_STATES else "")
            print(f"{STATES[abbr][0]} selected{note}.")
            return abbr
        print(f"No state matches {answer.strip()!r}. Try e.g. PA, Pennsylvania or 42.")


def _state(state: str) -> str:
    abbr = find_state(state)
    if abbr is None:
        raise ValueError(f"Unknown state: {state!r}")
    return abbr


# --- Downloading, with a cache ----------------------------------------------

def cache_dir() -> Path:
    """Where downloaded files are kept."""
    return Path(os.environ.get("REDISTRICTING_CACHE", Path.home() / ".redistricting_cache"))


def _download(url: str, path: Path) -> None:
    """Save url to path. Writes to a .partial file first, so a failed or
    interrupted download never leaves a broken file in the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    partial.replace(path)


def _cached(url: str, filename: str, refresh: bool, what: str) -> Path:
    path = cache_dir() / filename
    if path.exists() and not refresh:
        log.info("Using saved %s: %s", what, path)
    else:
        log.info("Downloading %s (first time only) ...", what)
        _download(url, path)
    return path


def tiger_url(abbr: str, folder: str | None = None) -> str:
    """URL of the state's 2020 TIGER/Line voting-district (or block) shapes.
    folder is the state's folder on the Census site, e.g. 42_PENNSYLVANIA."""
    name, fips = STATES[abbr]
    folder = folder or f"{fips}_{name.upper().replace(' ', '_')}"
    return f"{TIGER_STATE_ROOT}{folder}/{fips}/tl_2020_{fips}_{TIGER_LAYER[geography(abbr)]}.zip"


def find_tiger_folder(abbr: str) -> str:
    """Look up the state's folder name in the Census directory listing."""
    fips = STATES[abbr][1]
    r = requests.get(TIGER_STATE_ROOT, timeout=120)
    r.raise_for_status()
    found = re.findall(rf'href="({fips}_[A-Za-z_]+)/?"', r.text)
    if not found:
        raise RuntimeError(f"No folder for FIPS {fips} at {TIGER_STATE_ROOT}")
    return found[0]


def load_shapes(state: str, refresh: bool = False) -> gpd.GeoDataFrame:
    """The state's 2020 voting districts (blocks for CA, HI, OR) with the same
    columns as voting_districts.load_vtds(): GEOID, VTD_NAME, COUNTYFP, ALAND,
    AWATER, geometry."""
    abbr = _state(state)
    name, fips = STATES[abbr]
    filename = f"tl_2020_{fips}_{TIGER_LAYER[geography(abbr)]}.zip"
    what = f"{name} shapes"
    try:
        path = _cached(tiger_url(abbr), filename, refresh, what)
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise
        log.info("Not at the usual address; looking up %s's folder on the Census site", name)
        path = _cached(tiger_url(abbr, find_tiger_folder(abbr)), filename, refresh, what)

    gdf = gpd.read_file(path).rename(columns=VTD_COLUMNS)
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.strip()
    return gdf[[*VTD_COLUMNS.values(), "geometry"]]


def load_vest(state: str, refresh: bool = False, source: str | Path | None = None
              ) -> pd.DataFrame:
    """The state's census + VEST election data, with GEOID20 renamed to GEOID and
    kept as text. Same shape as election_results.fetch_elections().

    From source (your own ALARM-format CSV), else config.VEST_SOURCE, else the
    ALARM Project's file (downloaded once, then loaded from the cache)."""
    abbr = _state(state)
    source = source if source is not None else config.VEST_SOURCE
    if source is not None:
        return read_vest_file(source, abbr)

    level = geography(abbr)
    url = ALARM_STATE_URL.format(abbr=abbr.lower(), level=level)
    path = _cached(url, f"{abbr.lower()}_2020_{level}.csv", refresh,
                   f"{STATES[abbr][0]} census and election data")
    df = pd.read_csv(path, dtype={"GEOID20": str, "state": str, "county": str})
    return df.rename(columns={"GEOID20": "GEOID"}).assign(GEOID=lambda d: d["GEOID"].str.strip())


def read_vest_file(path: str | Path, state: str) -> pd.DataFrame:
    """Your own ALARM-format CSV for a state, checked: a GEOID column, GEOIDs
    from this state, no duplicates, and every demographic column."""
    abbr = _state(state)
    name, fips = STATES[abbr]
    path = Path(str(path).strip().strip('"'))            # "Copy as path" adds quotes
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")

    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    by_lower = {c.strip().lower(): c for c in header}
    geoid = next((by_lower[n] for n in config.GEOID_COLUMNS if n in by_lower), None)
    if geoid is None:
        raise ValueError(f"{path.name}: no GEOID column; expected one of "
                         f"{config.GEOID_COLUMNS}, found {list(header)}")

    text_cols = {c: str for c in (geoid, "state", "county") if c in header}
    df = (pd.read_csv(path, dtype=text_cols, encoding="utf-8-sig")
          .rename(columns={geoid: "GEOID"}))
    df["GEOID"] = df["GEOID"].str.strip().str.zfill(15 if geography(abbr) == "block" else 11)

    other_state = ~df["GEOID"].str.startswith(fips)
    if other_state.all():
        raise ValueError(f"{path.name}: GEOIDs start with {df['GEOID'].str[:2].mode()[0]}, "
                         f"not {fips} ({name}). Is it a file for another state?")
    if other_state.any():
        log.warning("%s: dropping %d rows whose GEOIDs aren't in %s",
                    path.name, other_state.sum(), name)
        df = df[~other_state]

    dupes = df.loc[df["GEOID"].duplicated(), "GEOID"]
    if len(dupes):
        raise ValueError(f"{path.name}: {len(dupes)} GEOIDs appear more than once, "
                         f"e.g. {dupes.iloc[0]}")

    missing = [c for c in COUNT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing ALARM demographic columns "
                         f"{', '.join(missing)}. The file needs ALARM's format "
                         f"(see the census-vest-2020 files)")

    log.info("Using your data for %s: %s (%d rows)", name, path, len(df))
    return df.reset_index(drop=True)


def pick_vest(state: str, ask: Callable[[str], str] = input, refresh: bool = False
              ) -> pd.DataFrame:
    """config.VEST_SOURCE if it is set; otherwise ask for your own ALARM-format
    CSV, or Enter for the ALARM Project's file. Asks again if a file is unusable.
    ask reads the answer (input by default; pass another function from a GUI or test)."""
    abbr = _state(state)
    if config.VEST_SOURCE is not None:
        return load_vest(abbr, refresh)
    while True:
        answer = ask("\nYour own data CSV in ALARM's format "
                     "(press Enter to use the ALARM Project's data): ").strip()
        if not answer:
            return load_vest(abbr, refresh)
        try:
            return read_vest_file(answer, abbr)
        except (ValueError, FileNotFoundError) as problem:
            print(f"{problem}. Try again.")


# --- Joining ------------------------------------------------------------------

def add_vote_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Two-party Democratic share, as a fraction from 0 to 1 like partisan_metrics:
    dem_share from ndv/nrv (all years averaged), dem_share_YY from adv_YY/arv_YY.
    Blank where there are no votes."""
    pairs = {}
    if {"ndv", "nrv"} <= set(df.columns):
        pairs["dem_share"] = ("ndv", "nrv")
    for col in sorted(df.columns):
        m = re.fullmatch(r"adv_(\d\d)", col)
        if m and f"arv_{m[1]}" in df.columns:
            pairs[f"dem_share_{m[1]}"] = (col, f"arv_{m[1]}")

    def share(d: str, r: str) -> pd.Series:
        total = df[d] + df[r]
        return (df[d] / total.where(total > 0)).round(4)

    return df.assign(**{name: share(d, r) for name, (d, r) in pairs.items()})


def join_state(shapes: gpd.GeoDataFrame, vest: pd.DataFrame,
               unit: str = "voting districts") -> tuple[gpd.GeoDataFrame, dict]:
    """Join demographics and votes onto the shapes by GEOID, as Plans does for PA.

    Every shape is kept; shapes without data get blanks. Returns
    (GeoDataFrame, report). The report says how well the two sources line up:
    GEOIDs only in one of them, and the population that has no shape.
    unit names the shapes in the low-match warning ("voting districts" or
    "Census blocks").
    """
    data = vest_demographics(vest)
    votes = vote_columns(vest)
    if votes:
        data = data.merge(vest[["GEOID", *votes]].drop_duplicates("GEOID"), on="GEOID",
                          how="left")
        data = add_vote_shares(data)
    else:
        log.warning("No election columns in this state's file: demographics only")

    dupes = data["GEOID"].duplicated()
    if dupes.any():
        log.warning("%d duplicate GEOIDs in the ALARM data; keeping the first of each",
                    dupes.sum())
        data = data[~dupes]

    shape_ids, data_ids = set(shapes["GEOID"]), set(data["GEOID"])
    only_shapes = sorted(shape_ids - data_ids)
    only_data = sorted(data_ids - shape_ids)
    lost_pop = int(data.loc[data["GEOID"].isin(only_data), "pop"].sum())

    joined = shapes.merge(data, on="GEOID", how="left")
    report = {
        "shapes": len(shapes),
        "data_rows": len(data),
        "matched": len(shape_ids & data_ids),
        "only_in_shapes": only_shapes,
        "only_in_data": only_data,
        "pop_total": int(data["pop"].sum()),
        "pop_without_shape": lost_pop,
        "election_columns": votes,
    }
    if only_shapes:
        log.warning("%d shapes have no ALARM data, e.g. %s", len(only_shapes), only_shapes[0])
    if only_data:
        log.warning("%d ALARM rows have no shape (population %s), e.g. %s",
                    len(only_data), f"{lost_pop:,}", only_data[0])
    if data_ids and report["matched"] / len(data_ids) < MIN_MATCH:
        log.warning("Only %.0f%% of the data's GEOIDs match the shapes. Is the file on "
                    "2020 %s?", 100 * report["matched"] / len(data_ids), unit)

    cols = [c for c in joined.columns if c != "geometry"] + ["geometry"]
    return gpd.GeoDataFrame(joined[cols], geometry="geometry", crs=shapes.crs), report


def load_state(state: str, refresh: bool = False, vest_source: str | Path | None = None,
               vest: pd.DataFrame | None = None) -> tuple[gpd.GeoDataFrame, dict]:
    """Download (or load from the cache) a state's shapes and census + election
    data and join them. Returns (GeoDataFrame, report); see join_state.

    vest_source: your own ALARM-format CSV (default config.VEST_SOURCE, else
    the ALARM Project's file). vest: that table already loaded (e.g. by pick_vest).
    """
    abbr = _state(state)
    if vest is None:
        vest = load_vest(abbr, refresh, vest_source)
    unit = "Census blocks" if geography(abbr) == "block" else "voting districts"
    return join_state(load_shapes(abbr, refresh), vest, unit)
