"""Plans: every citizen plan joined to voting-district shapes and Census data.

    from pa_redistricting import Plans
    plans = Plans.load()                     # or Plans.load(census_api_key="...")
    plans.names                              # list of plan names
    gdf = plans.get("allentown_map_by_as_jan-4-2022")                  # GeoDataFrame
    df = plans.get("allentown_map_by_as_jan-4-2022", geometry=False)   # DataFrame
    everything = plans.all(geometry=False)   # every plan stacked in one table

Each plan has one row per voting district with the columns
GEOID, plan, district, VTD_NAME, COUNTYFP, ALAND, AWATER, <Census fields>, geometry.
Voting districts a plan leaves out keep a blank district.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from .plan_assignments import fetch_plan_assignments
from .census import fetch_census
from .voting_districts import load_vtds

log = logging.getLogger(__name__)

# Census columns that duplicate fields already in the voting-district shapes
DROP_FROM_CENSUS = ["COUNTYFP", "VTD_NAME_CENSUS"]


class Plans:
    def __init__(self, vtds: gpd.GeoDataFrame, census: pd.DataFrame,
                 assignments: dict[str, pd.DataFrame]):
        self.vtds = vtds.assign(GEOID=vtds["GEOID"].astype(str).str.strip())
        self.census = self._clean_census(census)
        self.assignments = assignments

    @classmethod
    def load(cls, census_api_key: str | None = None) -> Plans:
        """Download the shapes, the Census data and every plan.
        census_api_key defaults to the CENSUS_API_KEY environment variable."""
        return cls(load_vtds(), fetch_census(census_api_key), fetch_plan_assignments())

    @staticmethod
    def _clean_census(census: pd.DataFrame) -> pd.DataFrame:
        census = census.drop(columns=[c for c in DROP_FROM_CENSUS if c in census.columns])
        census = census.assign(GEOID=census["GEOID"].astype(str).str.strip())
        dupes = census["GEOID"].duplicated().sum()
        if dupes:
            log.warning("%d duplicate GEOIDs in the Census data; keeping the first of each", dupes)
            census = census.drop_duplicates("GEOID")
        return census

    @property
    def names(self) -> list[str]:
        return list(self.assignments)

    def match(self, words: list[str]) -> list[str]:
        """Plan names containing any of the words (case-insensitive)."""
        return [p for p in self.names if any(w.lower() in p.lower() for w in words)]

    def get(self, plan: str, geometry: bool = True) -> gpd.GeoDataFrame | pd.DataFrame:
        """One plan joined to the shapes and Census data, one row per voting district.

        geometry=True  -> GeoDataFrame (for maps and spatial work)
        geometry=False -> DataFrame (faster; for tables and summaries)
        """
        if plan not in self.assignments:
            raise KeyError(f"No plan named {plan!r}")

        base = self.vtds if geometry else pd.DataFrame(self.vtds.drop(columns="geometry"))
        df = base.merge(self.assignments[plan], on="GEOID", how="left")
        df["district"] = df["district"].astype("Int64")
        df.insert(1, "plan", plan)
        df.insert(2, "district", df.pop("district"))
        df = df.merge(self.census, on="GEOID", how="left")
        df = df.sort_values(["district", "GEOID"]).reset_index(drop=True)

        unmatched = df["pop_total"].isna().sum()
        if unmatched:
            log.warning("%s: %d voting districts without Census data", plan, unmatched)

        if geometry:
            cols = [c for c in df.columns if c != "geometry"] + ["geometry"]
            return gpd.GeoDataFrame(df[cols], geometry="geometry", crs=self.vtds.crs)
        return df

    def all(self, geometry: bool = False) -> gpd.GeoDataFrame | pd.DataFrame:
        """Every plan stacked into one table (one row per plan per voting district)."""
        df = pd.concat([self.get(p, geometry=geometry) for p in self.names], ignore_index=True)
        if geometry:
            return gpd.GeoDataFrame(df, geometry="geometry", crs=self.vtds.crs)
        return df

    def __len__(self) -> int:
        return len(self.assignments)

    def __repr__(self) -> str:
        return f"<Plans: {len(self)} plans, {len(self.vtds):,} voting districts>"
