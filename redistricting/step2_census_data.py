"""
STEP 2 (in-memory version of plan_pull_censusdata(Step2).py)

Pulls 2020 Census redistricting data (PL 94-171) for every Pennsylvania
voting district (VTD) and RETURNS it as a DataFrame keyed by the same 11-digit
GEOID that step 1 uses. Nothing is written to disk.

Source: Census Data API, 2020 Decennial Redistricting Data (dataset "dec/pl")
        https://api.census.gov/data/2020/dec/pl

Use from another script / notebook:
  from step2_census_data import fetch_census, fetch_codebook
  census = fetch_census()       # one row per voting district, friendly column names
  codebook = fetch_codebook()   # each column's Census variable code and official label

API key (required): pass it as fetch_census(api_key=...) or set the
CENSUS_API_KEY environment variable. If neither is given, MissingAPIKeyError
is raised with setup instructions. Nothing in this module prompts for input,
so it works the same from a script, a notebook or a web application.

Requirements:  pip install pandas requests
"""

import os

import pandas as pd
import requests

API_URL = "https://api.census.gov/data/2020/dec/pl"
STATE_FIPS = "42"  # Pennsylvania
KEY_SIGNUP_URL = "https://api.census.gov/data/key_signup.html"

KEY_HELP = f"""No Census API key found.

Pass one as fetch_census(api_key=...) or set the CENSUS_API_KEY environment
variable. Keys are free: {KEY_SIGNUP_URL}

To save it in the gus8066 conda environment (one-time setup):
  conda env config vars set CENSUS_API_KEY=PASTE_KEY_HERE -n gus8066
  conda deactivate
  conda activate gus8066"""


class MissingAPIKeyError(RuntimeError):
    """No Census API key was passed in or found in CENSUS_API_KEY."""


def get_api_key(api_key: str | None = None) -> str:
    """The key passed in, else CENSUS_API_KEY. Raises MissingAPIKeyError if neither."""
    key = (api_key or os.environ.get("CENSUS_API_KEY", "")).strip()
    if not key:
        raise MissingAPIKeyError(KEY_HELP)
    return key


PA_COUNTIES = [f"{i:03d}" for i in range(1, 134, 2)]  # PA's 67 county codes: 001, 003, ... 133

# Census variable code -> friendly column name
VARIABLES = {
    # P1: Race (total population)
    "P1_001N": "pop_total",
    "P1_003N": "pop_white",
    "P1_004N": "pop_black",
    "P1_005N": "pop_aian",
    "P1_006N": "pop_asian",
    "P1_007N": "pop_nhpi",
    "P1_008N": "pop_other",
    "P1_009N": "pop_two_plus",
    # P2: Hispanic or Latino
    "P2_002N": "pop_hispanic",
    "P2_005N": "pop_nh_white",
    "P2_006N": "pop_nh_black",
    # P3 / P4: Voting-age population (18+)
    "P3_001N": "vap_total",
    "P3_003N": "vap_white",
    "P3_004N": "vap_black",
    "P4_002N": "vap_hispanic",
    "P4_005N": "vap_nh_white",
    "P4_006N": "vap_nh_black",
    # P5: Group quarters
    "P5_001N": "gq_total",
    # H1: Housing occupancy
    "H1_001N": "housing_total",
    "H1_002N": "housing_occupied",
    "H1_003N": "housing_vacant",
}


def fetch_vtd_data(api_key: str | None = None) -> pd.DataFrame:
    """Request each PA county separately and stack the results."""
    api_key = get_api_key(api_key)
    frames = []
    for county in PA_COUNTIES:
        url = (f"{API_URL}?get=NAME,{','.join(VARIABLES)}"
               f"&for=voting%20district:*&in=state:{STATE_FIPS}&in=county:{county}"
               f"&key={api_key}")

        r = requests.get(url, timeout=120)
        try:
            rows = r.json()
        except ValueError:
            raise RuntimeError(
                f"County {county}: the Census API did not return data "
                f"(HTTP {r.status_code}). It said:\n{r.text[:800]}"
            )
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
        print(f"  county {county}: {len(rows) - 1} voting districts")

    df = pd.concat(frames, ignore_index=True)

    # Build the 11-digit GEOID: state (2) + county (3) + voting district (6)
    df["GEOID"] = df["state"] + df["county"] + df["voting district"]

    df = df.rename(columns=VARIABLES).rename(columns={"NAME": "VTD_NAME_CENSUS",
                                                      "county": "COUNTYFP"})
    for col in VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def add_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """A few handy shares. Blank where the voting district has zero people."""
    def pct(num: str, den: str) -> pd.Series:
        d = df[den].astype("float").where(df[den] > 0)
        return (100 * df[num].astype("float") / d).round(2)

    df["pct_hispanic"] = pct("pop_hispanic", "pop_total")
    df["pct_nh_white"] = pct("pop_nh_white", "pop_total")
    df["pct_nh_black"] = pct("pop_nh_black", "pop_total")
    df["pct_vap"] = pct("vap_total", "pop_total")
    df["pct_vacant"] = pct("housing_vacant", "housing_total")
    return df


def fetch_census(api_key: str | None = None) -> pd.DataFrame:
    """One row per PA voting district: GEOID, names, counts and percentages.
    api_key defaults to the CENSUS_API_KEY environment variable."""
    api_key = get_api_key(api_key)
    print("Requesting PA voting-district data from the Census API ...")
    df = add_percentages(fetch_vtd_data(api_key))

    front = ["GEOID", "VTD_NAME_CENSUS", "COUNTYFP"]
    rest = [c for c in df.columns if c not in front + ["state", "voting district"]]
    return df[front + rest].sort_values("GEOID").reset_index(drop=True)


def fetch_codebook() -> pd.DataFrame:
    """The official label for each variable so you can check the meanings."""
    r = requests.get(f"{API_URL}/variables.json", timeout=120)
    r.raise_for_status()
    labels = r.json()["variables"]
    rows = [{"column": name, "census_code": code,
             "label": labels.get(code, {}).get("label", ""),
             "table": labels.get(code, {}).get("concept", "")}
            for code, name in VARIABLES.items()]
    return pd.DataFrame(rows)


def main() -> None:
    df = fetch_census()
    print(f"\n{len(df):,} voting districts, {len(df.columns)} columns")
    print(f"PA total population: {df['pop_total'].sum():,}")
    print(df[["GEOID", "VTD_NAME_CENSUS", "pop_total", "vap_total"]].head())
    print("\nCodebook:")
    print(fetch_codebook().to_string(index=False))


if __name__ == "__main__":
    main()