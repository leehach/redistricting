import os
import sys
import pandas as pd

# Usage: python pop_deviation.py plan1.csv plan2.csv ...
# Free Census API key: https://api.census.gov/data/key_signup.html
CENSUS_KEY = ""
POP_FILE = "pa_vtd_population_2020.csv"

# Download 2020 Census population for every PA voting district (once)
if not os.path.exists(POP_FILE):
    url = ("https://api.census.gov/data/2020/dec/pl?get=P1_001N"
           "&for=voting%20district:*&in=state:42%20county:*&key=" + CENSUS_KEY)
    data = pd.read_json(url, dtype=False)
    data.columns = data.iloc[0]
    data = data[1:]
    data["GEOID"] = data["state"] + data["county"] + data["voting district"]
    data["Population"] = data["P1_001N"].astype(int)
    data[["GEOID", "Population"]].to_csv(POP_FILE, index=False)

pop = pd.read_csv(POP_FILE, dtype={"GEOID": str})

for plan_file in sys.argv[1:]:
    plan = pd.read_csv(plan_file, dtype={"GEOID20": str}).rename(columns={"GEOID20": "GEOID"})
    unmatched = (~plan["GEOID"].isin(pop["GEOID"])).sum()
    plan = plan.merge(pop, on="GEOID")
    district_pop = plan.groupby("District")["Population"].sum()
    ideal = district_pop.mean()
    deviation = (district_pop - ideal) / ideal * 100

    print(f"\n{plan_file}")
    print(f"Unmatched GEOIDs: {unmatched}")
    print(pd.DataFrame({"Population": district_pop, "Deviation %": deviation.round(2)}))
    print(f"Ideal population: {ideal:,.0f}")
    print(f"Overall range: {deviation.max() - deviation.min():.2f}%")
