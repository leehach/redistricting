"""Pick a state, download its voting-district shapes and VEST data, and join them.

Open this file in VS Code and press Run, then type a state in the terminal
(PA, Pennsylvania or 42 all work). The first run for a state downloads the
data; later runs use the saved copy, so they are fast and work offline.

Results go to results/states/<state>/:
    <state>_2020_<vtd or block>.gpkg      shapes + data (open in QGIS or geopandas)
    <state>_2020_<vtd or block>.csv       the same table without shapes
    <state>_unmatched_geoids.csv          GEOIDs found in only one source, if any
    <state>_dem_share.png                 map of the Democratic two-party vote share
"""

import logging
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
    pd.DataFrame(gdf.drop(columns="geometry")).to_csv(folder / f"{stem}.csv", index=False)

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
