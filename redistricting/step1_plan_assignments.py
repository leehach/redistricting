"""
STEP 1 (in-memory version of plan_pull_geoid(Step1).py)

Pulls the PA Senate citizen plans (GEOID -> district) and the 2020 Census
voting-district shapes, and RETURNS them as Python objects. Nothing is written
to or read from disk: downloads go straight into memory.

Sources:
  - Plans (GEOID -> district):  github.com/ccfdpa/pa-citizen-redistricting-plans-2020 (pa-senate)
  - Shapes (GEOID20 -> polygon): 2020 Census TIGER/Line voting districts for PA

Use from another script / notebook:
  from step1_plan_assignments import fetch_plan_assignments, load_vtds
  assignments = fetch_plan_assignments()   # {plan name: DataFrame[GEOID, district]}
  vtds = load_vtds()                       # GeoDataFrame, one row per voting district

Requirements:  pip install geopandas pandas requests pyogrio
"""

import io
from pathlib import PurePosixPath
from urllib.parse import quote

import geopandas as gpd
import pandas as pd
import requests

OWNER, REPO, BRANCH, FOLDER = "ccfdpa", "pa-citizen-redistricting-plans-2020", "main", "pa-senate"
VTD_URL = "https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/42_PENNSYLVANIA/42/tl_2020_42_vtd20.zip"


# 1. List and download the plan CSVs from GitHub
def list_plan_files() -> list[str]:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return sorted(i["path"] for i in r.json()["tree"]
                  if i["type"] == "blob" and i["path"].startswith(FOLDER + "/")
                  and i["path"].lower().endswith((".csv", ".txt")))


def download_text(repo_path: str) -> str:
    url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{quote(repo_path)}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content.decode("utf-8-sig")


def plan_name(repo_path: str) -> str:
    """File stem with anything other than letters, digits, _ and - turned into _
    (the same names the old GeoPackage layers used)."""
    stem = PurePosixPath(repo_path).stem
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in stem)


# 2. Read one plan CSV into a clean GEOID -> district table
EXPECTED_HEADER = ["GEOID20", "District"]


def read_assignment(text: str) -> pd.DataFrame:
    """Parse a plan CSV with the header 'GEOID20,District'.
    Rows with a blank District are treated as unassigned and dropped."""
    df = pd.read_csv(io.StringIO(text), dtype={"GEOID20": str}, skipinitialspace=True)
    if list(df.columns) != EXPECTED_HEADER:
        raise ValueError(f"expected header {EXPECTED_HEADER}, found {list(df.columns)}")

    df = df.rename(columns={"GEOID20": "GEOID", "District": "district"})
    return df.dropna(subset=["district"]).astype({"district": int}).reset_index(drop=True)


def fetch_plan_assignments() -> dict[str, pd.DataFrame]:
    """Every plan in the repo as {plan name: DataFrame[GEOID, district]}."""
    paths = list_plan_files()
    print(f"Found {len(paths)} plan files")

    assignments = {}
    for path in paths:
        try:
            assignments[plan_name(path)] = read_assignment(download_text(path))
        except ValueError as e:
            e.add_note(f"while reading {path}")
            raise
    return assignments


# 3. Load the Census voting-district shapes straight from the download (no cache file)
def load_vtds() -> gpd.GeoDataFrame:
    print("Downloading 2020 Census voting districts for PA ...")
    r = requests.get(VTD_URL, timeout=600)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))
    gdf = gdf.rename(columns={"GEOID20": "GEOID", "NAME20": "VTD_NAME",
                              "COUNTYFP20": "COUNTYFP", "ALAND20": "ALAND", "AWATER20": "AWATER"})
    return gdf[["GEOID", "VTD_NAME", "COUNTYFP", "ALAND", "AWATER", "geometry"]]


def main() -> None:
    vtds = load_vtds()
    print(f"{len(vtds):,} voting districts loaded")
    for plan, assign in fetch_plan_assignments().items():
        n_missing = (~vtds["GEOID"].isin(assign["GEOID"])).sum()
        print(f"{plan}: {len(assign):,} GEOIDs assigned"
              + (f" ({n_missing} voting districts unassigned)" if n_missing else ""))


if __name__ == "__main__":
    main()