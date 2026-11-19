"""Redistricting plans for any state: which district each voting district
(or Census block) is in.

    from pa_redistricting.plan_assignments import read_plans
    plans = read_plans("my_plans/")        # a CSV, a folder, or a GitHub folder URL
    plans[0].name, plans[0].assignment     # DataFrame[GEOID, district]

Sources (config.PLANS_SOURCE is the default):
    a CSV file       one plan (GEOID, district), or several in long format:
                     GEOID, plan, district, one row per voting district per plan
    a folder         every CSV in it (files ending in _metrics are skipped)
    a GitHub folder  https://github.com/<owner>/<repo>/tree/<branch>/<folder>

Column names are matched case-insensitively against config.GEOID_COLUMNS,
DISTRICT_COLUMNS and PLAN_NAME_COLUMNS.
"""

import io
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

METRICS_SUFFIX = "_metrics"        # output files to skip when reading a folder
GITHUB_FOLDER = re.compile(r"https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+?)/?")


@dataclass
class Plan:
    name: str                  # cleaned name used in every output
    source: str                # file path or URL it came from
    assignment: pd.DataFrame   # GEOID (text), district (int)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def clean_name(text: str) -> str:
    """Anything other than letters, digits, _ and - turned into _."""
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(text).strip())


def plan_name(path: str | Path) -> str:
    """A file's name without extension, cleaned: 'My Plan (v2).csv' -> 'My_Plan__v2_'."""
    return clean_name(Path(str(path)).stem)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_plans(source: str | Path | None = None) -> list[Plan]:
    """Every plan at source (default: config.PLANS_SOURCE): a CSV, a folder of
    CSVs, or a GitHub folder URL. Works for any state."""
    if source is None:
        source = config.PLANS_SOURCE
    if source is None:
        raise ValueError("No plans source: pass a CSV, a folder or a GitHub folder URL, "
                         "or set PLANS_SOURCE in config.py")

    text = str(source).strip().strip('"')           # "Copy as path" adds quotes
    if m := GITHUB_FOLDER.fullmatch(text):
        plans = _read_github_folder(*m.groups())
    elif text.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported URL {text}; use a GitHub folder link like "
                         "https://github.com/<owner>/<repo>/tree/<branch>/<folder>")
    else:
        plans = _read_local(Path(text))
    log.info("Read %d plan(s) from %s", len(plans), text)
    return _unique_names(plans)


def fetch_plan_assignments(source: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """{plan name: DataFrame[GEOID, district]} from source, else
    config.PLANS_SOURCE, else the example PA Senate plans."""
    source = source or config.PLANS_SOURCE or config.EXAMPLE_PLANS_SOURCE
    return {p.name: p.assignment for p in read_plans(source)}


def read_plan(path: str | Path) -> pd.DataFrame:
    """The one plan in a CSV, as DataFrame[GEOID, district]. Raises if it holds several."""
    plans = read_plan_file(path)
    if len(plans) > 1:
        raise ValueError(f"{Path(path).name} holds {len(plans)} plans; use read_plans()")
    return plans[0].assignment


def read_plan_file(path: str | Path) -> list[Plan]:
    """The plan(s) in one local CSV."""
    path = Path(path)
    return read_plan_text(path.read_text(encoding="utf-8-sig"), path.stem, str(path))


def read_plan_text(text: str, file_stem: str, source: str) -> list[Plan]:
    """The plan(s) in one CSV's text: one plan, or several when it has a
    plan-name column (long format)."""
    raw = pd.read_csv(io.StringIO(text.lstrip("﻿")), dtype=str, skipinitialspace=True)
    label = f"{file_stem}.csv"
    geoid = _column(raw, config.GEOID_COLUMNS, "GEOID", label)
    district = _column(raw, config.DISTRICT_COLUMNS, "district", label)

    plan_col = _find(raw, config.PLAN_NAME_COLUMNS)
    if plan_col is None:                                               # one plan
        return [Plan(clean_name(file_stem), source, _clean(raw[[geoid, district]], label))]

    names = raw[plan_col].fillna("").str.strip()                       # long: several plans
    if names.eq("").any():
        raise ValueError(f"{label}: {names.eq('').sum()} rows have no plan name")
    return [Plan(clean_name(name), source,
                 _clean(group[[geoid, district]], f"{label} [{name}]"))
            for name, group in raw.groupby(names, sort=False)]


def _read_local(path: Path) -> list[Plan]:
    if path.is_file():
        return read_plan_file(path)
    if path.is_dir():
        files = [f for f in sorted(path.glob("*.csv")) if not f.stem.endswith(METRICS_SUFFIX)]
        return _read_each(files, read_plan_file, where=str(path))
    raise FileNotFoundError(f"No file or folder at {path}")


def _read_github_folder(owner: str, repo: str, branch: str, folder: str) -> list[Plan]:
    """Every CSV in one folder of a public GitHub repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    files = sorted(i["path"] for i in r.json()["tree"]
                   if i["type"] == "blob" and i["path"].startswith(folder + "/")
                   and i["path"].lower().endswith(".csv"))

    def read(repo_path: str) -> list[Plan]:
        raw_url = (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
                   f"{quote(repo_path)}")
        resp = requests.get(raw_url, timeout=120)
        resp.raise_for_status()
        return read_plan_text(resp.content.decode("utf-8-sig"), Path(repo_path).stem, raw_url)

    return _read_each(files, read, where=f"{owner}/{repo}/{folder}")


def _read_each(items: list, read: Callable[[object], list[Plan]], where: str) -> list[Plan]:
    """Read every file; skip (with a warning) any that isn't a valid plan."""
    if not items:
        raise ValueError(f"No plan CSVs in {where}")
    plans = []
    for item in items:
        try:
            plans += read(item)
        except ValueError as e:
            log.warning("Skipping %s: %s", Path(str(item)).name, e)
    if not plans:
        raise ValueError(f"None of the CSVs in {where} could be read as plans")
    return plans


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------
def _find(df: pd.DataFrame, names: list[str]) -> str | None:
    """The first column whose name (case-insensitive) is in names, or None."""
    by_lower = {c.strip().lower(): c for c in df.columns}
    return next((by_lower[n] for n in names if n in by_lower), None)


def _column(df: pd.DataFrame, names: list[str], what: str, label: str) -> str:
    found = _find(df, names)
    if found is None:
        raise ValueError(f"{label}: no {what} column; expected one of {names}, "
                         f"found {list(df.columns)}")
    return found


def _clean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """GEOID + district columns -> checked GEOIDs and whole-number districts.
    Rows with a blank district are unassigned and dropped."""
    df = df.set_axis(["GEOID", "district"], axis=1).apply(lambda s: s.str.strip())
    df = df[df["district"].fillna("") != ""].copy()
    if df.empty:
        raise ValueError(f"{label}: no voting district has a district number")
    if df["GEOID"].fillna("").eq("").any():
        raise ValueError(f"{label}: some rows have a district but no GEOID")

    numbers = pd.to_numeric(df["district"], errors="coerce")           # accepts "1.0"
    if numbers.isna().any() or (numbers % 1 != 0).any():
        raise ValueError(f"{label}: district values must be whole numbers, e.g. 1, 2, 3")
    df["district"] = numbers.astype(int)

    width = 15 if df["GEOID"].str.len().max() > 12 else 11              # block or VTD
    df["GEOID"] = df["GEOID"].str.zfill(width)                          # Excel drops leading 0s

    dupes = df.loc[df["GEOID"].duplicated(), "GEOID"]
    if len(dupes):
        raise ValueError(f"{label}: {len(dupes)} GEOIDs appear more than once, "
                         f"e.g. {dupes.iloc[0]}")
    return df.reset_index(drop=True)


def _unique_names(plans: list[Plan]) -> list[Plan]:
    """Plans that share a name (e.g. 'plan_a' in two files) get the file name
    added: <file>__<plan>."""
    counts = pd.Series([p.name for p in plans]).value_counts()
    return [Plan(f"{plan_name(unquote(p.source))}__{p.name}", p.source, p.assignment)
            if counts[p.name] > 1 else p for p in plans]
