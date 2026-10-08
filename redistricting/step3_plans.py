"""
STEP 3 (in-memory version of Join_data(Step3).py)

The Plans object: holds the voting-district shapes (step 1), every plan's
GEOID -> district table (step 1) and the Census data (step 2) in memory, and
joins them on GEOID whenever you ask for a plan. Nothing is written to or read
from disk.

Each plan comes back with the same columns the old Joined_Data.gpkg layers had:
  GEOID, plan, district, VTD_NAME, COUNTYFP, ALAND, AWATER, <Census fields>, geometry

Use from another script / notebook:
  from step3_plans import Plans
  plans = Plans.load()                              # downloads everything once
  plans.names                                       # list of plan names
  gdf = plans.get("allentown_map_by_as_jan-4-2022")                  # GeoDataFrame
  df  = plans.get("allentown_map_by_as_jan-4-2022", geometry=False)  # pandas DataFrame
  everything = plans.all(geometry=False)            # every plan stacked in one table

Requirements:  pip install geopandas pandas requests pyogrio
"""

import geopandas as gpd
import pandas as pd

from step1_plan_assignments import fetch_plan_assignments, load_vtds
from step2_census_data import fetch_census

# Census columns that duplicate fields already in the voting-district shapes
DROP_FROM_CENSUS = ["COUNTYFP", "VTD_NAME_CENSUS"]


class Plans:
    def __init__(self, vtds: gpd.GeoDataFrame, census: pd.DataFrame,
                 assignments: dict[str, pd.DataFrame]):
        self.vtds = vtds.assign(GEOID=vtds["GEOID"].astype(str).str.strip())
        self.census = self._clean_census(census)
        self.assignments = assignments

    @classmethod
    def load(cls) -> Plans:
        """Download the shapes, the Census data and every plan (steps 1 and 2)."""
        return cls(load_vtds(), fetch_census(), fetch_plan_assignments())

    @staticmethod
    def _clean_census(census: pd.DataFrame) -> pd.DataFrame:
        census = census.drop(columns=[c for c in DROP_FROM_CENSUS if c in census.columns])
        census = census.assign(GEOID=census["GEOID"].astype(str).str.strip())
        dupes = census["GEOID"].duplicated().sum()
        if dupes:
            print(f"Warning: {dupes} duplicate GEOIDs in the Census data; keeping the first of each")
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

        geometry=True  -> GeoDataFrame (for maps / spatial work)
        geometry=False -> plain pandas DataFrame (faster; for tables and summaries)
        """
        if plan not in self.assignments:
            raise KeyError(f"No plan named {plan!r}")

        base = self.vtds if geometry else pd.DataFrame(self.vtds.drop(columns="geometry"))
        df = base.merge(self.assignments[plan], on="GEOID", how="left")   # keeps every voting district
        df["district"] = df["district"].astype("Int64")                  # blanks stay blank, numbers stay whole
        df.insert(1, "plan", plan)
        df.insert(2, "district", df.pop("district"))
        df = df.merge(self.census, on="GEOID", how="left")
        df = df.sort_values(["district", "GEOID"]).reset_index(drop=True)

        unmatched = df["pop_total"].isna().sum()
        if unmatched:
            print(f"  {plan}: {unmatched} voting districts without Census data")

        if geometry:
            cols = [c for c in df.columns if c != "geometry"] + ["geometry"]   # geometry last
            return gpd.GeoDataFrame(df[cols], geometry="geometry", crs=self.vtds.crs)
        return df

    def all(self, geometry: bool = False) -> gpd.GeoDataFrame | pd.DataFrame:
        """Every plan stacked into one long table (one row per plan per voting district)."""
        frames = [self.get(p, geometry=geometry) for p in self.names]
        df = pd.concat(frames, ignore_index=True)
        if geometry:
            return gpd.GeoDataFrame(df, geometry="geometry", crs=self.vtds.crs)
        return df

    def __len__(self) -> int:
        return len(self.assignments)

    def __repr__(self) -> str:
        return f"<Plans: {len(self)} plans, {len(self.vtds):,} voting districts>"


def choose_plans(all_plans: list[str], args: list[str]) -> list[str]:
    """Pick plans from command-line words, --all, or an on-screen menu
    (shared by steps 4 and 5)."""
    if "--all" in args:
        return all_plans

    if args:  # match each word against plan names (case-insensitive)
        chosen = [p for p in all_plans if any(a.lower() in p.lower() for a in args)]
        if not chosen:
            print(f"No plan names contain: {', '.join(args)}")
        return chosen

    # No words given: show a numbered menu
    for i, p in enumerate(all_plans, 1):
        print(f"{i:>3}. {p}")
    answer = input("\nType plan number(s) separated by commas (or 'all'): ").strip()
    if answer.lower() == "all":
        return all_plans
    picks = []
    for part in answer.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(all_plans):
            picks.append(all_plans[int(part) - 1])
        elif part:
            print(f"  Ignoring '{part}' (not a number from the list)")
    return picks


def main() -> None:
    plans = Plans.load()
    print(f"\n{plans}")
    for i, name in enumerate(plans.names, 1):
        df = plans.get(name, geometry=False)
        n_missing = df["district"].isna().sum()
        print(f"[{i}/{len(plans)}] {name}: {len(df):,} voting districts"
              + (f", {n_missing} unassigned" if n_missing else ""))


if __name__ == "__main__":
    main()
