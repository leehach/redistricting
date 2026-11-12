"""Joins district assignments, voting data and demographic data from different files into
one table, 1 row per precinct. no Qt in here so it can be used (and tested) without the window

    plan = reconcile(assignments, voting=votes, demographics=census)
    plan.data    # the joined table (GeoDataFrame if any of the files had shapes)
    plan.notes   # what happened, ex. "voting data: 8,921 of 9,178 precincts matched"
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

SPATIAL = {".shp", ".gpkg", ".geojson", ".json", ".zip"}  # read with geopandas, everything else as csv

# precinct id column names we look for, best first (case/spaces/underscores don't matter)
KEY_NAMES = ["geoid20", "geoid", "geoid10", "vtdst20", "vtdst", "vtd", "precinctid", "precinct",
             "unitid", "id"]


class ReconcileError(ValueError):
    """the files can't be joined (no id column, nothing matches, ...)"""


@dataclass
class Reconciled:
    data: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def has_shapes(data: pd.DataFrame) -> bool:
    return isinstance(data, gpd.GeoDataFrame) and data.active_geometry_name is not None


def read_table(path) -> pd.DataFrame:
    """csv/tsv/txt or a spatial file. csv columns come in as text so ids keep their leading
    zeros, reconcile() turns the number columns back into numbers"""
    if Path(path).suffix.lower() in SPATIAL:
        return gpd.read_file(path)
    # sep=None lets pandas figure out comma vs tab vs semicolon
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)


def _norm(name) -> str:  # "Geo ID_20" -> "geoid20"
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_key(data: pd.DataFrame, label: str = "file") -> str:
    """the precinct id column: exact name from KEY_NAMES first, then a name containing 'geoid'"""
    shapes = data.active_geometry_name if has_shapes(data) else None
    columns = [c for c in data.columns if c != shapes]
    for name in KEY_NAMES:
        for c in columns:
            if _norm(c) == name:
                return c
    for c in columns:
        if "geoid" in _norm(c):
            return c
    raise ReconcileError(f"{label}: couldn't find a precinct id column (looked for GEOID20, GEOID, "
                         f"VTD, precinct_id ...). Columns in the file: {', '.join(map(str, columns))}")


def clean_key(values: pd.Series) -> pd.Series:
    """ids as stripped text. 42001000010.0 (id read as a float) -> '42001000010'"""
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    return text.mask(text.isin(["", "nan", "None", "<NA>"])).astype(object)


def _pad_like(keys: pd.Series, target: pd.Series) -> pd.Series:
    """put back leading zeros a spreadsheet stripped: if the plan's ids are all N digits and
    this file's are digits but shorter, zero-fill them to N"""
    lengths = target.dropna().str.len().unique()
    if len(lengths) != 1 or not target.dropna().str.isdigit().all():
        return keys
    present = keys.dropna()
    if present.empty or not present.str.isdigit().all() or (present.str.len() >= lengths[0]).all():
        return keys
    return keys.str.zfill(int(lengths[0]))


def _numbers(data: pd.DataFrame, skip: list[str]) -> pd.DataFrame:
    """text columns that are really numbers -> numbers (so metrics can add them up)"""
    data = data.copy()
    for c in data.columns:
        if c in skip or not (pd.api.types.is_object_dtype(data[c]) or pd.api.types.is_string_dtype(data[c])):
            continue
        converted = pd.to_numeric(data[c], errors="coerce")
        if converted.notna().any() and converted.notna().sum() == data[c].notna().sum():
            data[c] = converted
    return data


def _prepare(data: pd.DataFrame, label: str, key: str | None) -> tuple[pd.DataFrame, str]:
    if key is None:
        key = find_key(data, label)
    elif key not in data.columns:
        raise ReconcileError(f"{label}: no column named {key!r}")
    skip = [key] + ([data.active_geometry_name] if has_shapes(data) else [])
    data = _numbers(data, skip)
    data[key] = clean_key(data[key])
    return data, key


def reconcile(assignments: pd.DataFrame, voting: pd.DataFrame | None = None,
              demographics: pd.DataFrame | None = None, shapes: pd.DataFrame | None = None,
              keys: dict[str, str] | None = None) -> Reconciled:
    """Left-join everything onto the assignments by precinct id, so every assigned precinct is kept.

    keys = {"assignments": "GEOID20", "voting": ...} to name an id column instead of guessing.
    Raises ReconcileError if a file has no id column or none of its ids are in the plan.
    """
    keys = keys or {}
    notes = []
    base, base_key = _prepare(assignments, "district assignments", keys.get("assignments"))
    if len(base.columns) - has_shapes(base) < 2:
        raise ReconcileError("district assignments: the file only has an id column, no districts")

    blank = base[base_key].isna()
    if blank.any():
        notes.append(f"district assignments: dropped {blank.sum():,} rows with no precinct id")
        base = base[~blank]
    dupes = base[base_key].duplicated()
    if dupes.any():
        raise ReconcileError(f"district assignments: {dupes.sum():,} precinct ids are listed more "
                             f"than once, e.g. {base.loc[dupes, base_key].iloc[0]!r}")
    if base.empty:
        raise ReconcileError("district assignments: no rows with a precinct id")
    base = base.reset_index(drop=True)
    crs = base.crs if has_shapes(base) else None

    extras = [("voting data", "voting", voting), ("demographic data", "demographics", demographics),
              ("shapes", "shapes", shapes)]
    for label, name, other in extras:
        if other is None:
            continue
        other, key = _prepare(other, label, keys.get(name))
        other = other[other[key].notna()].copy()
        other[key] = _pad_like(other[key], base[base_key])
        dupes = other[key].duplicated()
        if dupes.any():
            notes.append(f"{label}: {dupes.sum():,} repeated precinct ids, kept the first of each")
            other = other[~dupes]

        matched = base[base_key].isin(other[key])
        if not matched.any():
            raise ReconcileError(
                f"{label}: none of its precinct ids match the district assignments. "
                f"Plan ids look like {base[base_key].iloc[0]!r}, {label} ids look like "
                f"{other[key].iloc[0] if len(other) else '(empty file)'!r}")
        note = f"{label}: {matched.sum():,} of {len(base):,} precincts matched"
        unused = (~other[key].isin(base[base_key])).sum()
        if unused:
            note += f", {unused:,} rows not in the plan were left out"
        notes.append(note)

        # only one set of shapes, and a column that's already there wins over a later one
        if has_shapes(other):
            if has_shapes(base):
                other = pd.DataFrame(other.drop(columns=other.active_geometry_name))
            else:
                crs = other.crs
                other = pd.DataFrame(other).rename(columns={other.active_geometry_name: "geometry"})
        repeats = [c for c in other.columns if c in base.columns and c != key]
        if repeats:
            notes.append(f"{label}: skipped columns already in the plan ({', '.join(map(str, repeats))})")
        other = other.drop(columns=repeats).rename(columns={key: base_key})
        base = pd.DataFrame(base).merge(other, on=base_key, how="left") if not has_shapes(base) \
            else base.merge(other, on=base_key, how="left")

    if "geometry" in base.columns and not has_shapes(base):
        base = gpd.GeoDataFrame(base, geometry="geometry", crs=crs)
    if has_shapes(base):
        missing = base.geometry.isna().sum()
        if missing:
            notes.append(f"shapes: {missing:,} precincts have no shape")
    return Reconciled(base, notes)
