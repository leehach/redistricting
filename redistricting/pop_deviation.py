import sys
import warnings

import geopandas as gpd
import pandas as pd


def population_deviation(precincts, plan, geoid_col="GEOID20",
                         pop_col="P0010001", district_col="District"):
    """
    Population and deviation from ideal for each district in a plan.

    precincts : (Geo)DataFrame with one row per precinct, including
                geoid_col and pop_col (e.g. the PA voting district shapefile).
    plan      : DataFrame assigning each precinct (geoid_col) to a district
                (district_col).

    Returns a DataFrame indexed by district with columns
    Population, Deviation (people), and Deviation %.
    """
    merged = plan[[geoid_col, district_col]].merge(
        precincts[[geoid_col, pop_col]], on=geoid_col, how="left")

    unmatched = merged[pop_col].isna().sum()
    if unmatched:
        warnings.warn(f"{unmatched} precinct(s) in the plan have no population match")

    district_pop = merged.groupby(district_col)[pop_col].sum()
    ideal = district_pop.mean()

    return pd.DataFrame({
        "Population": district_pop.astype(int),
        "Deviation": (district_pop - ideal).round(1),
        "Deviation %": ((district_pop - ideal) / ideal * 100).round(2),
    })


if __name__ == "__main__":
    # Usage: python pop_deviation.py WP_VotingDistricts.shp plan1.csv plan2.csv ...
    precincts = gpd.read_file(sys.argv[1])

    for plan_file in sys.argv[2:]:
        plan = pd.read_csv(plan_file, dtype={"GEOID20": str})
        result = population_deviation(precincts, plan)
        ideal = result["Population"].mean()

        print(f"\n{plan_file}")
        print(result)
        print(f"Ideal population: {ideal:,.0f}")
        print(f"Overall range: {result['Deviation %'].max() - result['Deviation %'].min():.2f}%")
