"""Reads plan files: PAFs (precinct assignment files) and BAFs (block assignment files).
no Qt in here so it can be used (and tested) without the window

The convention, the minimum a plan needs:
    column 1   GEOID, whatever its header says (11 characters for a precinct/VTD, 15 for a block)
    column 2   the district it's assigned to, a whole number
anything after that is ignored. a file with no header row works too, and so does one with a pandas
index column in front (to_csv's default).

    plans, problems = read_plan_files(["plans/"])   # files and/or folders
    plans[0].name, plans[0].kind, plans[0].assignment
    problems    # [(path, "why it couldn't be read"), ...] for files that were skipped

The GEOID and district checks follow the ones in pa_redistricting/plan_assignments.py.
"""

import csv
import re
from pathlib import Path

import pandas as pd

try:
    from .session import Plan  # imported as redistricting.metrics_gui.plan_files (tests)
except ImportError:
    from session import Plan   # run from this folder (python app.py)

SUFFIXES = {".csv", ".txt", ".tsv"}
SEPARATORS = ",\t;|"
OUTPUT_SUFFIX = "_metrics"  # files the Metrics window wrote; skipped when reading a folder
FILE_FILTER = "Plan files (*.csv *.txt *.tsv);;All files (*)"

# GEOID length -> file kind. 10 and 14 = a spreadsheet dropped the leading zero (states 01-09)
WIDTHS = {10: "PAF", 11: "PAF", 14: "BAF", 15: "BAF"}
FULL_WIDTH = {"PAF": 11, "BAF": 15}
# state + county digits, then a 6-character VTD code that can have letters, '-', '.' or '/'
# (10001001-28 in DE). case matters (MS has both 28059505-1B and -1b), so never upper-case
PATTERN = {"PAF": re.compile(r"\d{5}[^\s,;|]{6}"),
           "BAF": re.compile(r"\d{15}")}  # state + county + tract + block
NUMBER_READ_AS_FLOAT = r"^(\d+)\.0+$"     # '42001000010.0' -> '42001000010'


class PlanFileError(ValueError):
    """the file isn't a usable plan; the message says why"""


def looks_like_geoid(text) -> bool:
    text = str(text).strip()
    kind = WIDTHS.get(len(text))
    return kind is not None and bool(PATTERN[kind].fullmatch(text.zfill(FULL_WIDTH[kind])))


def plan_files_in(folder) -> list[Path]:
    """every plan file directly in folder, sorted, without the Metrics window's outputs"""
    return sorted(f for f in Path(folder).iterdir()
                  if f.is_file() and f.suffix.lower() in SUFFIXES
                  and not f.stem.endswith(OUTPUT_SUFFIX))


def _read_text(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise PlanFileError("the file is empty")
    # only real separators: left to itself the sniffer can pick a digit and split GEOIDs apart
    try:
        sep = csv.Sniffer().sniff(text[:20_000], delimiters=SEPARATORS).delimiter
    except csv.Error:
        raise PlanFileError("needs at least 2 columns: GEOID first, then district") from None

    def read(header):
        # text so ids keep their leading zeros
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, header=header,
                           skipinitialspace=True)

    raw = read("infer")
    if len(raw.columns) and looks_like_geoid(raw.columns[0]):  # no header row: that was data
        raw = read(None)
    return raw


def _is_index_column(raw: pd.DataFrame) -> bool:
    """a blank-headed first column counting 0, 1, 2 ... (what to_csv writes by default)"""
    first = str(raw.columns[0])
    if len(raw.columns) < 3 or not (first.startswith("Unnamed: ") or not first.strip()):
        return False
    numbers = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    return numbers.notna().all() and (numbers.diff().dropna() == 1).all()


def clean_assignment(raw: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """first two columns -> (DataFrame[GEOID, district], "PAF" or "BAF", notes). raises PlanFileError"""
    notes = []
    if len(raw.columns) and _is_index_column(raw):
        raw = raw.iloc[:, 1:]
        notes.append("skipped an index column in front of the GEOIDs")
    if len(raw.columns) < 2:
        raise PlanFileError("needs at least 2 columns: GEOID first, then district")
    if len(raw.columns) > 2:
        notes.append(f"ignored {len(raw.columns) - 2} column(s) after the district")

    df = raw.iloc[:, :2].copy()
    df.columns = ["GEOID", "district"]
    df = df.apply(lambda s: s.astype("string").str.strip()).replace("", pd.NA)
    df = df.dropna(how="all")

    unassigned = df["district"].isna() & df["GEOID"].notna()
    if unassigned.any():
        notes.append(f"{unassigned.sum():,} GEOIDs have no district (left out as unassigned)")
        df = df[~unassigned]
    if df.empty:
        raise PlanFileError("no rows with both a GEOID and a district")
    if df["GEOID"].isna().any():
        raise PlanFileError(f"{df['GEOID'].isna().sum():,} rows have a district but no GEOID")

    # GEOIDs: '42001000010.0' (an id read as a number) -> '42001000010', then put back stripped zeros
    geoid = df["GEOID"].str.replace(NUMBER_READ_AS_FLOAT, r"\1", regex=True)
    kinds = geoid.str.len().map(WIDTHS)
    if kinds.isna().any():
        bad = geoid[kinds.isna()].iloc[0]
        raise PlanFileError(f"column 1 should be GEOIDs (11 characters for precincts, 15 for "
                            f"blocks) but has {bad!r}")
    if kinds.nunique() > 1:
        raise PlanFileError("column 1 mixes precinct (11 character) and block (15 character) GEOIDs")
    kind = kinds.iloc[0]
    geoid = geoid.str.zfill(FULL_WIDTH[kind])
    bad = ~geoid.str.fullmatch(PATTERN[kind].pattern)
    if bad.any():
        raise PlanFileError(f"column 1 should be GEOIDs but has {geoid[bad].iloc[0]!r}")
    dupes = geoid[geoid.duplicated()]
    if len(dupes):
        raise PlanFileError(f"{len(dupes):,} GEOIDs are listed more than once, e.g. {dupes.iloc[0]}")

    district = pd.to_numeric(df["district"], errors="coerce")  # accepts "3" and "3.0"
    bad = district.isna() | (district % 1 != 0)
    if bad.any():
        raise PlanFileError(f"column 2 should be whole district numbers but has "
                            f"{df.loc[bad, 'district'].iloc[0]!r}")

    out = pd.DataFrame({"GEOID": geoid.astype(str).to_numpy(),
                        "district": district.astype(int).to_numpy()})
    return out, kind, notes


def read_plan_file(path) -> Plan:
    """one PAF/BAF -> a Plan named after the file. raises PlanFileError"""
    path = Path(str(path).strip().strip('"'))  # "Copy as path" adds quotes
    if not path.is_file():
        raise PlanFileError("no such file")
    assignment, kind, notes = clean_assignment(_read_text(path))
    return Plan(path.stem, assignment, kind, str(path), notes)


def read_plan_files(paths) -> tuple[list[Plan], list[tuple[str, str]]]:
    """files and folders -> (plans, problems). a file that can't be read is skipped and listed in
    problems as (path, reason) instead of stopping the rest"""
    files, problems = [], []
    for p in paths:
        p = Path(str(p).strip().strip('"'))
        if p.is_dir():
            found = plan_files_in(p)
            if not found:
                problems.append((str(p), "no .csv, .txt or .tsv files in this folder"))
            files += found
        else:
            files.append(p)

    plans = []
    for f in files:
        try:
            plans.append(read_plan_file(f))
        except PlanFileError as e:
            problems.append((str(f), str(e)))
        except Exception as e:  # unreadable encoding, permissions, ...
            problems.append((str(f), f"{type(e).__name__}: {e}"))
    return plans, problems
