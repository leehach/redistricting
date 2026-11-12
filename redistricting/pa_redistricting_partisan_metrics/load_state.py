"""Pick a state, download its voting-district shapes and VEST data, and join them.

Open this file in VS Code and press Run, then type a state in the terminal
(PA, Pennsylvania or 42 all work). The first run for a state downloads the
data; later runs use the saved copy, so they are fast and work offline.

Results go to results/states/<state>/:
    <state>_2020_<vtd or block>.gpkg      shapes + data (open in QGIS or geopandas)
    <state>_2020_<vtd or block>.csv       the same table without shapes, with
                                          plain-English column names (see below)
    <state>_unmatched_geoids.csv          GEOIDs found in only one source, if any
    <state>_dem_share.png                 map of the Democratic two-party vote share

CSV columns, in order:
    GEOID, voting_district_name, county_fips   which voting district
    democratic_vote_share, republican_vote_share
        share of the two-party vote (0 to 1, so 0.646 = 64.6%), averaged over
        every statewide race in every year; then the same for each year
        (democratic_vote_share_2020, ...). Blank where there were no votes.
    total_population, hispanic_population, white_non_hispanic_population, ...
    voting_age_population, voting_age_hispanic, ...    18 and over
    percent_hispanic, percent_white_non_hispanic, ...  0 to 100
    democratic_votes_avg_all_elections, democratic_votes_avg_2020, ...
        average votes per statewide race (estimates, so they can be fractions)
    president_2020_democratic_bid_votes, ...  votes for one candidate
        (office, year, party, first three letters of the candidate's name)
    land_area_sq_meters, water_area_sq_meters
"""

import logging
import re
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from pa_redistricting.state_data import STATES, geography, load_state, pick_state

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
OUT = Path(__file__).parent / "results" / "states"     # output folder, next to this file
MAKE_MAP = True                                         # slow for CA, HI and OR (blocks)

# ---------------------------------------------------------------------------


def print_report(abbr: str, report: dict) -> None:
    unit = "Census blocks" if geography(abbr) == "block" else "voting districts"
    print(f"\n{STATES[abbr][0]}: {report['shapes']:,} {unit} in the shapes, "
          f"{report['data_rows']:,} in the ALARM data, {report['matched']:,} matched")
    print(f"Population: {report['pop_total']:,}")
    if report["only_in_shapes"]:
        print(f"!! {len(report['only_in_shapes']):,} shapes have no data")
    if report["only_in_data"]:
        print(f"!! {len(report['only_in_data']):,} data rows have no shape "
              f"({report['pop_without_shape']:,} people)")
    if not report["only_in_shapes"] and not report["only_in_data"]:
        print("Every GEOID matched.")
    if not report["election_columns"]:
        print("No election data for this state, only demographics.")


# ---------------------------------------------------------------------------
# Plain-English column names for the CSV
# ---------------------------------------------------------------------------
COLUMN_NAMES = {
    "VTD_NAME": "voting_district_name",
    "COUNTYFP": "county_fips",
    "ALAND": "land_area_sq_meters",
    "AWATER": "water_area_sq_meters",
    "dem_share": "democratic_vote_share",
    "rep_share": "republican_vote_share",
    "pop_total": "total_population",
    "pop_hispanic": "hispanic_population",
    "vap_total": "voting_age_population",
    "vap_hispanic": "voting_age_hispanic",
    "pct_hispanic": "percent_hispanic",
    "pct_nh_white": "percent_white_non_hispanic",
    "pct_nh_black": "percent_black_non_hispanic",
    "pct_vap": "percent_voting_age",
    "ndv": "democratic_votes_avg_all_elections",
    "nrv": "republican_votes_avg_all_elections",
}
RACES = {"aian": "american_indian_alaska_native", "nhpi": "native_hawaiian_pacific_islander",
         "other": "other_race", "two_plus": "two_or_more_races"}
PARTIES = {"dem": "democratic", "rep": "republican"}
OFFICES = {"pre": "president", "uss": "us_senate", "gov": "governor",
           "ltg": "lieutenant_governor", "atg": "attorney_general",
           "sos": "secretary_of_state", "tre": "treasurer", "aud": "auditor",
           "con": "controller", "agr": "agriculture_commissioner",
           "ins": "insurance_commissioner", "lab": "labor_commissioner",
           "spi": "superintendent_of_public_instruction"}


def readable_name(column: str) -> str:
    """The plain-English name for one column; unknown columns keep their name."""
    if column in COLUMN_NAMES:
        return COLUMN_NAMES[column]

    if m := re.fullmatch(r"(dem|rep)_share_(\d\d)", column):          # dem_share_20
        return f"{PARTIES[m[1]]}_vote_share_20{m[2]}"

    if m := re.fullmatch(r"(pop|vap)_nh_(\w+)", column):              # pop_nh_white
        race = RACES.get(m[2], m[2])
        if m[1] == "pop":
            return f"{race}_non_hispanic_population"
        return f"voting_age_{race}_non_hispanic"

    if m := re.fullmatch(r"a(d|r)v_(\d\d)", column):                  # adv_20
        party = "democratic" if m[1] == "d" else "republican"
        return f"{party}_votes_avg_20{m[2]}"

    if m := re.fullmatch(r"([a-z]+)_(\d\d)_(dem|rep)_(\w+)", column):  # pre_20_dem_bid
        office = OFFICES.get(m[1], m[1])
        return f"{office}_20{m[2]}_{PARTIES[m[3]]}_{m[4]}_votes"

    return column


def column_order(column: str) -> int:
    """Which group a column belongs to, so related columns sit together."""
    if column in ("GEOID", "VTD_NAME", "COUNTYFP"):
        return 0                                        # which voting district
    if re.fullmatch(r"(dem|rep)_share(_\d\d)?", column):
        return 1                                        # vote shares
    if column.startswith("pop_"):
        return 2                                        # population
    if column.startswith("vap_"):
        return 3                                        # voting-age population
    if column.startswith("pct_"):
        return 4                                        # percentages
    if column in ("ALAND", "AWATER"):
        return 6                                        # area, last
    return 5                                            # vote counts


def readable_table(gdf) -> pd.DataFrame:
    """The joined data as a table for the CSV: no shapes, a Republican share next
    to every Democratic share, related columns together, plain-English names."""
    table = pd.DataFrame(gdf.drop(columns="geometry"))

    # republican share = 1 - democratic share (two-party vote), placed right after it
    for dem in [c for c in table.columns if re.fullmatch(r"dem_share(_\d\d)?", c)]:
        rep = dem.replace("dem_", "rep_")
        table.insert(table.columns.get_loc(dem) + 1, rep, (1 - table[dem]).round(4))

    table = table[sorted(table.columns, key=column_order)]     # keeps order within groups

    names = {c: readable_name(c) for c in table.columns}
    repeated = {n for n in names.values() if list(names.values()).count(n) > 1}
    if repeated:                    # never let two columns end up with the same name
        logging.warning("Keeping original names for columns that would clash: %s",
                        ", ".join(sorted(repeated)))
        names = {c: (c if n in repeated else n) for c, n in names.items()}
    return table.rename(columns=names)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def save_map(gdf, title: str, path: Path) -> None:
    gdf = gdf.to_crs("EPSG:5070")    # equal-area, like maps.py
    fig = Figure(figsize=(12, 9))
    ax = fig.subplots()
    gdf.plot(ax=ax, column="dem_share", cmap="RdBu", vmin=0.2, vmax=0.8, legend=True,
             missing_kwds={"color": "#d9d9d9"}, linewidth=0,
             legend_kwds={"label": "Democratic share of the two-party vote", "shrink": 0.6})
    ax.set_title(title, fontsize=14, loc="left")
    ax.set_axis_off()
    fig.savefig(path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # show progress
    abbr = pick_state()
    gdf, report = load_state(abbr)
    print_report(abbr, report)

    folder = OUT / abbr.lower()
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{abbr.lower()}_2020_{geography(abbr)}"
    gdf.to_file(folder / f"{stem}.gpkg", driver="GPKG")
    readable_table(gdf).to_csv(folder / f"{stem}.csv", index=False)

    unmatched = ([{"GEOID": g, "found_in": "shapes only"} for g in report["only_in_shapes"]]
                 + [{"GEOID": g, "found_in": "ALARM data only"} for g in report["only_in_data"]])
    if unmatched:
        pd.DataFrame(unmatched).to_csv(folder / f"{abbr.lower()}_unmatched_geoids.csv",
                                       index=False)

    if MAKE_MAP and "dem_share" in gdf.columns:
        print("Drawing the map ...")
        unit = "Census block" if geography(abbr) == "block" else "voting district"
        save_map(gdf, f"{STATES[abbr][0]}: Democratic two-party vote share, by {unit}",
                 folder / f"{abbr.lower()}_dem_share.png")
    print(f"\nSaved to {folder}")
