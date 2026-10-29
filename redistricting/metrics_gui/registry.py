"""Plug-in stuff. field() and @metric register things, the GUI builds itself from FIELDS and REGISTRY.
a metric returns a Series/DataFrame indexed by district, or a dict of plan-wide numbers
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import numpy as np
import pandas as pd

from pa_redistricting.partisan_metrics import score_districts  # our group's partisan math

GEOMETRY = "geometry"  # put in needs if the metric uses shapes


# a column a metric needs from the user's file (ex. Democratic votes)
@dataclass
class Field:
    key: str
    label: str        # what the window shows
    names: list[str]  # column names auto-match looks for
    numeric: bool = False


# one checkbox in the window
@dataclass
class Metric:
    group: str
    name: str
    func: Callable
    needs: list[str]


FIELDS: dict[str, Field] = {}           # every field, in the order they show up
REGISTRY: dict[str, list[Metric]] = {}  # group name -> its metrics


def field(key: str, label: str, names: list[str], numeric: bool = False) -> None:
    FIELDS[key] = Field(key, label, names, numeric)


# @metric("Group", "Name", needs=[...]) above a function adds it to the window
def metric(group: str, name: str, needs: list[str] = ()):
    def add(func):
        REGISTRY.setdefault(group, []).append(Metric(group, name, func, ["district", *needs]))
        return func
    return add


# every metric needs district so it lives here instead of metrics.py
field("district", "District", ["district", "districtid", "district_n", "dist", "sldust", "cd"])


def has_shapes(data: pd.DataFrame) -> bool:
    return isinstance(data, gpd.GeoDataFrame) and data.active_geometry_name is not None


# columns the user can pick from (everything but the geometry column)
def table_columns(data: pd.DataFrame) -> list[str]:
    shapes = data.active_geometry_name if has_shapes(data) else None
    return [c for c in data.columns if c != shapes]


# which fields the checked metrics need, no duplicates
def needed_fields(chosen: list[Metric]) -> list[str]:
    needs = {n for m in chosen for n in m.needs}
    return [k for k in FIELDS if k in needs]


def auto_match(data: pd.DataFrame, keys: list[str], already: dict[str, str] | None = None
               ) -> dict[str, str]:
    """exact name first, then partial. numeric fields only match numeric columns"""
    def norm(s):  # "Dem Votes" -> "demvotes" so case/spaces don't matter
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    found = dict(already or {})
    used = set(found.values())  # so one column doesn't get matched to two fields
    columns = table_columns(data)
    for key in keys:
        if key in found:
            continue
        f = FIELDS[key]
        options = [c for c in columns if c not in used
                   and (not f.numeric or pd.api.types.is_numeric_dtype(data[c]))]
        # pass 1: exact match, pass 2: name appears somewhere in the column name
        for same in (lambda col, name: col == name, lambda col, name: name in col):
            hit = next((c for name in f.names for c in options if same(norm(c), norm(name))), None)
            if hit is not None:
                found[key] = hit
                used.add(hit)
                break
    return found


# what every metric function gets as ctx: the user's data + which column is which field
class PlanContext:
    def __init__(self, data: pd.DataFrame, columns: dict[str, str]):
        self.data = data
        self.columns = columns

    # the user's column for a field, turned into numbers if the field is numeric
    def column(self, key: str) -> pd.Series:
        f, name = FIELDS[key], self.columns[key]
        values = self.data[name]
        if not f.numeric:
            return values
        numbers = pd.to_numeric(values, errors="coerce")
        if numbers.isna().all():
            raise ValueError(f"{f.label}: column {name!r} has no numbers in it")
        return numbers.fillna(0)

    @property
    def district(self) -> pd.Series:
        return self.column("district")

    # adds up precincts into districts (does nothing if it's already 1 row per district)
    def sum_by_district(self, *keys: str) -> pd.DataFrame:
        return pd.DataFrame({k: self.column(k) for k in keys}).groupby(self.district).sum()

    # cached_property = only calculated once even if 5 partisan metrics ask for it
    @cached_property
    def partisan(self) -> tuple[pd.Series, pd.DataFrame]:
        d = self.sum_by_district("dem_votes", "rep_votes").rename(
            columns={"dem_votes": "dem", "rep_votes": "rep"})
        scores = pd.Series(score_districts(d))
        d["dem_share"] = (d["dem"] / (d["dem"] + d["rep"])).round(4)
        d["winner"] = np.where(d["dem_share"] > 0.5, "D", "R")
        return scores, d

    # merges precinct shapes into one shape per district
    @cached_property
    def district_shapes(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame({"district": self.district.values},
                               geometry=self.data.geometry.values, crs=self.data.crs)
        shapes = gdf.dissolve(by="district")
        # lat/long degrees give wrong areas, so reproject to meters
        if shapes.crs is not None and shapes.crs.is_geographic:
            shapes = shapes.to_crs(shapes.estimate_utm_crs())
        return shapes


# runs each checked metric and glues the results into one table, 1 row per district
def run_metrics(ctx: PlanContext, chosen: list[Metric], plan_name: str = "") -> pd.DataFrame:
    table = pd.DataFrame(index=pd.Index(sorted(ctx.district.dropna().unique()), name="district"))
    for m in chosen:
        result = m.func(ctx)
        if isinstance(result, dict):  # plan-wide number, same on every row
            for col, value in result.items():
                table[col] = value
        else:
            table = table.join(result)
    table = table.reset_index()
    table.insert(0, "plan", plan_name)
    return table
