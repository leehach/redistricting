"""Citizen redistricting plans: which Senate district each voting district is in.

    from pa_redistricting.plan_assignments import fetch_plan_assignments, read_assignment
    assignments = fetch_plan_assignments()     # {plan name: DataFrame[GEOID, district]}
    one = read_assignment(csv_text)            # parse a single plan CSV
"""

import io
import logging
from pathlib import PurePosixPath
from urllib.parse import quote

import pandas as pd
import requests

from .config import (GEOID_LENGTH, PLAN_HEADER, PLANS_BRANCH, PLANS_FOLDER,
                     PLANS_OWNER, PLANS_REPO)

log = logging.getLogger(__name__)


def list_plan_files() -> list[str]:
    """Repo paths of every plan CSV in the plans folder."""
    url = (f"https://api.github.com/repos/{PLANS_OWNER}/{PLANS_REPO}"
           f"/git/trees/{PLANS_BRANCH}?recursive=1")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return sorted(i["path"] for i in r.json()["tree"]
                  if i["type"] == "blob" and i["path"].startswith(PLANS_FOLDER + "/")
                  and i["path"].lower().endswith(".csv"))


def download_text(repo_path: str) -> str:
    """Raw text of one file in the plans repo."""
    url = (f"https://raw.githubusercontent.com/{PLANS_OWNER}/{PLANS_REPO}"
           f"/{PLANS_BRANCH}/{quote(repo_path)}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content.decode("utf-8-sig")


def plan_name(repo_path: str) -> str:
    """File stem with anything other than letters, digits, _ and - turned into _."""
    stem = PurePosixPath(repo_path).stem
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in stem)


def read_assignment(text: str) -> pd.DataFrame:
    """Parse a plan CSV with the header 'GEOID20,District'.
    Rows with a blank District are treated as unassigned and dropped."""
    df = pd.read_csv(io.StringIO(text), dtype={"GEOID20": str}, skipinitialspace=True)
    if list(df.columns) != PLAN_HEADER:
        raise ValueError(f"expected header {PLAN_HEADER}, found {list(df.columns)}")

    bad = ~df["GEOID20"].str.fullmatch(rf"[0-9A-Z]{{{GEOID_LENGTH}}}", na=False)
    if bad.any():
        raise ValueError(f"{bad.sum()} GEOID20 values are not {GEOID_LENGTH}-character VTD codes, "
                         f"e.g. {df.loc[bad, 'GEOID20'].iloc[0]!r}")

    df = df.rename(columns={"GEOID20": "GEOID", "District": "district"})
    return df.dropna(subset=["district"]).astype({"district": int}).reset_index(drop=True)


def fetch_plan_assignments() -> dict[str, pd.DataFrame]:
    """Every plan in the repo as {plan name: DataFrame[GEOID, district]}."""
    paths = list_plan_files()
    log.info("Found %d plan files", len(paths))

    assignments = {}
    for path in paths:
        try:
            assignments[plan_name(path)] = read_assignment(download_text(path))
        except ValueError as e:
            e.add_note(f"while reading {path}")
            raise
    return assignments
