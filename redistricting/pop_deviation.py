import sys

import geopandas as gpd
import pandas as pd


def population_deviation(precincts, plan, geoid_col="GEOID20",
                         pop_col="P0010001", district_col="District",
                         num_districts=None):
    """
    Population and deviation from ideal for each district in a plan.

    precincts     : (Geo)DataFrame with one row per precinct, including
                    geoid_col and pop_col (e.g. the PA voting district shapefile).
    plan          : DataFrame assigning each precinct (geoid_col) to a district
                    (district_col).
    num_districts : expected number of districts. If given, the plan must use
                    exactly this many; if None, the number of distinct
                    districts in the plan is used.

    The ideal population is the total population of all precincts divided by
    the number of districts. The plan must assign every precinct exactly once,
    and every precinct must have a population, otherwise a ValueError is raised.

    Returns a DataFrame indexed by district with columns
    Population, Deviation (people), and Deviation %. The ideal population is
    stored in result.attrs["ideal"].
    """
    missing_pop = precincts[pop_col].isna()
    if missing_pop.any():
        raise ValueError(
            f"{missing_pop.sum()} precinct(s) have no population: "
            f"{list(precincts.loc[missing_pop, geoid_col][:10])}")

    dup_precincts = precincts[geoid_col].duplicated()
    if dup_precincts.any():
        raise ValueError(
            f"{dup_precincts.sum()} duplicate precinct ID(s) in precincts: "
            f"{list(precincts.loc[dup_precincts, geoid_col][:10])}")

    dup_plan = plan[geoid_col].duplicated()
    if dup_plan.any():
        raise ValueError(
            f"{dup_plan.sum()} precinct(s) assigned more than once in plan: "
            f"{list(plan.loc[dup_plan, geoid_col][:10])}")

    if plan[district_col].isna().any():
        raise ValueError(
            f"{plan[district_col].isna().sum()} precinct(s) in plan have no district")

    precinct_ids = set(precincts[geoid_col])
    plan_ids = set(plan[geoid_col])

    not_in_plan = precinct_ids - plan_ids
    if not_in_plan:
        raise ValueError(
            f"{len(not_in_plan)} precinct(s) not assigned in plan: "
            f"{sorted(not_in_plan)[:10]}")

    unknown = plan_ids - precinct_ids
    if unknown:
        raise ValueError(
            f"{len(unknown)} precinct(s) in plan not found in precincts: "
            f"{sorted(unknown)[:10]}")

    merged = plan[[geoid_col, district_col]].merge(
        precincts[[geoid_col, pop_col]], on=geoid_col, how="left")
    district_pop = merged.groupby(district_col)[pop_col].sum()

    if num_districts is None:
        num_districts = len(district_pop)
    elif len(district_pop) != num_districts:
        raise ValueError(
            f"plan has {len(district_pop)} districts, expected {num_districts}")

    ideal = precincts[pop_col].sum() / num_districts

    result = pd.DataFrame({
        "Population": district_pop.astype(int),
        "Deviation": (district_pop - ideal).round(1),
        "Deviation %": ((district_pop - ideal) / ideal * 100).round(2),
    })
    result.attrs["ideal"] = ideal
    return result


if __name__ == "__main__":
    # Usage: python pop_deviation.py WP_VotingDistricts.shp plan1.csv plan2.csv ...
    precincts = gpd.read_file(sys.argv[1])

    for plan_file in sys.argv[2:]:
        plan = pd.read_csv(plan_file, dtype={"GEOID20": str})
        result = population_deviation(precincts, plan)

        print(f"\n{plan_file}")
        print(result)
        print(f"Ideal population: {result.attrs['ideal']:,.0f}")
        print(f"Overall range: {result['Deviation %'].max() - result['Deviation %'].min():.2f}%")
