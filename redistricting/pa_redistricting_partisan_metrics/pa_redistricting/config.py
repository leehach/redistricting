"""Shared constants: Pennsylvania Senate settings and data-source locations."""

STATE_FIPS = "42"          # Pennsylvania
N_DISTRICTS = 50           # PA Senate seats

# --- Plans (any state) ---------------------------------------------------------
# Where plans are read from. One of:
#   a CSV file       one plan (GEOID, district) or several in long format (GEOID, plan, district)
#   a folder         every CSV in it
#   a GitHub folder  https://github.com/<owner>/<repo>/tree/<branch>/<folder>
# None: the scripts ask for a path when they run.
PLANS_SOURCE = None

# The PA Senate citizen plans used for testing
EXAMPLE_PLANS_SOURCE = ("https://github.com/ccfdpa/pa-citizen-redistricting-plans-2020/"
                        "tree/main/pa-senate")

# --- Census + election data (any state) ------------------------------------------
# None: download the ALARM Project's file for the state (census + VEST elections).
# Or a path to your own CSV in ALARM's format: one row per 2020 voting district
# (Census block in CA, HI, OR), a GEOID20 or GEOID column, ALARM's demographic
# columns (pop, pop_hisp, ..., vap_two) and its election columns (ndv, nrv, ...).
VEST_SOURCE = None

# Accepted plan column names (case-insensitive); the first match is used
GEOID_COLUMNS = ["geoid20", "geoid", "geoid_20"]
DISTRICT_COLUMNS = ["district", "district_id", "districtid", "cd", "dist"]
PLAN_NAME_COLUMNS = ["plan", "plan_name", "map", "map_name"]

# 2020 Census TIGER/Line voting-district shapes for PA
VTD_URL = ("https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/"
           "42_PENNSYLVANIA/42/tl_2020_42_vtd20.zip")

# 2020 Census redistricting data (PL 94-171)
CENSUS_API_URL = "https://api.census.gov/data/2020/dec/pl"
CENSUS_KEY_SIGNUP_URL = "https://api.census.gov/data/key_signup.html"

# Votes per 2020 voting district (ALARM Project)
ELECTIONS_URL = ("https://raw.githubusercontent.com/alarm-redist/census-2020/"
                 "main/census-vest-2020/pa_2020_vtd.csv")

# --- Any state (state_data.py) ----------------------------------------------
# Census TIGER/Line 2020 shapes: <root><fips>_<STATE_NAME>/<fips>/tl_2020_<fips>_<layer>.zip
# layer is vtd20 (voting districts) or tabblock20 (Census blocks)
TIGER_STATE_ROOT = "https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/"

# ALARM Project census + VEST election data: <abbr>_2020_<level>.csv, level vtd or block
ALARM_STATE_URL = ("https://raw.githubusercontent.com/alarm-redist/census-2020/"
                   "main/census-vest-2020/{abbr}_2020_{level}.csv")
