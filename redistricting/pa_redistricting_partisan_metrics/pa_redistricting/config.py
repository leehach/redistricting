"""Shared constants: Pennsylvania Senate settings and data-source locations."""

STATE_FIPS = "42"          # Pennsylvania
N_DISTRICTS = 50           # PA Senate seats
GEOID_LENGTH = 11          # 2020 VTD GEOID: state (2) + county (3) + VTD (6)

# Citizen plan CSVs (GEOID20 -> District), one file per plan
PLANS_OWNER = "ccfdpa"
PLANS_REPO = "pa-citizen-redistricting-plans-2020"
PLANS_BRANCH = "main"
PLANS_FOLDER = "pa-senate"
PLAN_HEADER = ["GEOID20", "District"]

# 2020 Census TIGER/Line voting-district shapes for PA
VTD_URL = ("https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/"
           "42_PENNSYLVANIA/42/tl_2020_42_vtd20.zip")

# 2020 Census redistricting data (PL 94-171)
CENSUS_API_URL = "https://api.census.gov/data/2020/dec/pl"
CENSUS_KEY_SIGNUP_URL = "https://api.census.gov/data/key_signup.html"

# Votes per 2020 voting district (ALARM Project)
ELECTIONS_URL = ("https://raw.githubusercontent.com/alarm-redist/census-2020/"
                 "main/census-vest-2020/pa_2020_vtd.csv")
