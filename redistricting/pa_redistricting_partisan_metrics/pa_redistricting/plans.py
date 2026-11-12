"""Plans: every citizen plan joined to voting-district shapes and demographic data.

    from pa_redistricting import Plans
    plans = Plans.load()                     # VEST demographics, no API key needed
    plans = Plans.load("vest+census")        # plus group quarters and housing (needs a key)
    plans.names                              # list of plan names
    gdf = plans.get("allentown_map_by_as_jan-4-2022")                  # GeoDataFrame
    df = plans.get("allentown_map_by_as_jan-4-2022", geometry=False)   # DataFrame
    everything = plans.all(geometry=False)   # every plan stacked in one table

Each plan has one row per voting district with the columns
GEOID, plan, district, VTD_NAME, COUNTYFP, ALAND, AWATER, <demographic fields>, geometry.
Voting districts a plan leaves out keep a blank district.
"""

from __future__ import annotations

import logging
from typing import Literal

import geopandas as gpd
import pandas as pd

from .census import EXTRA_COLUMNS, fetch_census
from .demographics import add_percentages, vest_demographics
from .election_results import fetch_elections
from .plan_assignments import fetch_plan_assignments
from .voting_districts import load_vtds

log = logging.getLogger(__name__)

DemographicsSource = Literal["vest", "census", "vest+census"]
DEMOGRAPHICS_SOURCES = ("vest", "census", "vest+census")

# Census columns that duplicate fields already in the voting-district shapes
DROP_FROM_DEMOGRAPHICS = ["COUNTYFP", "VTD_NAME_CENSUS"]


def load_demographics(source: DemographicsSource = "vest", census_api_key: str | None = None,
                      vest: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per voting district of demographic counts.

    "vest"         VEST counts; no API key needed
    "census"       Census API only (needs a key)
    "vest+census"  VEST counts plus the Census-only columns: group quarters,
                   housing, race including Hispanics (needs a key)
    vest: the table from fetch_elections(), if already downloaded.
    census_api_key defaults to the CENSUS_API_KEY environment variable.
    """
    if source not in DEMOGRAPHICS_SOURCES:
        raise ValueError(f"demographics must be one of {DEMOGRAPHICS_SOURCES}, not {source!r}")
    if source == "census":
        return fetch_census(census_api_key)

    demo = vest_demographics(vest if vest is not None else fetch_elections())
    if source == "vest+census":
        extras = fetch_census(census_api_key)[["GEOID", *EXTRA_COLUMNS]]
        demo = add_percentages(demo.merge(extras, on="GEOID", how="left"))
    return demo


class Plans:
    def __init__(self, vtds: gpd.GeoDataFrame, demographics: pd.DataFrame,
                 assignments: dict[str, pd.DataFrame]):
        self.vtds = vtds.assign(GEOID=vtds["GEOID"].astype(str).str.strip())
        self.demographics = self._clean_demographics(demographics)
        self.assignments = assignments

    @classmethod
    def load(cls, demographics: DemographicsSource = "vest", census_api_key: str | None = None,
             vest: pd.DataFrame | None = None) -> Plans:
        """Download the shapes, the demographic data and every plan.
        See load_demographics for the choices. Pass vest (from fetch_elections())
        to avoid downloading the VEST file twice."""
        return cls(load_vtds(), load_demographics(demographics, census_api_key, vest),
                   fetch_plan_assignments())

    @staticmethod
    def _clean_demographics(demo: pd.DataFrame) -> pd.DataFrame:
        demo = demo.drop(columns=[c for c in DROP_FROM_DEMOGRAPHICS if c in demo.columns])
        demo = demo.assign(GEOID=demo["GEOID"].astype(str).str.strip())
        dupes = demo["GEOID"].duplicated().sum()
        if dupes:
            log.warning("%d duplicate GEOIDs in the demographic data; keeping the first of each",
                        dupes)
            demo = demo.drop_duplicates("GEOID")
        return demo

    @property
    def names(self) -> list[str]:
        return list(self.assignments)

    def match(self, words: list[str]) -> list[str]:
        """Plan names containing any of the words (case-insensitive)."""
        return [p for p in self.names if any(w.lower() in p.lower() for w in words)]

    def get(self, plan: str, geometry: bool = True) -> gpd.GeoDataFrame | pd.DataFrame:
        """One plan joined to the shapes and demographic data, one row per voting district.

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
        df = df.merge(self.demographics, on="GEOID", how="left")
        df = df.sort_values(["district", "GEOID"]).reset_index(drop=True)

        unmatched = df["pop_total"].isna().sum()
        if unmatched:
            log.warning("%s: %d voting districts without demographic data", plan, unmatched)

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
