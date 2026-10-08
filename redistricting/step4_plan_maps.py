"""
STEP 4 (in-memory version of PNG_Plan_Creation(Step4).py)

Draws one map per plan, straight from the in-memory Plans object (step 3).

- Every voting district is drawn with its outline (no merging).
- Fill color comes from the 'district' field, and each district number gets
  the SAME color in every plan, so maps can be compared side by side.
- Each map is saved as maps/<plan name>.png (the final output).

How to run:
  python step4_plan_maps.py                  -> every plan, saved to maps/
  python step4_plan_maps.py allentown ccfd   -> only plans whose names contain those words

From a notebook:
  from step3_plans import Plans
  from step4_plan_maps import make_map
  plans = Plans.load()
  fig = make_map(plans.get("allentown_map_by_as_jan-4-2022"), "allentown_map_by_as_jan-4-2022")

Requirements:  pip install geopandas matplotlib pyogrio
"""

import random
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from step3_plans import Plans

OUT_DIR = Path("maps")
DISTRICT_FIELD = "district"
N_DISTRICTS = 50

MAP_CRS = "EPSG:5070"      # equal-area projection, so PA isn't stretched
DPI = 200
FIGSIZE = (16, 9)          # inches -> 3200 x 1800 pixels at 200 dpi
OUTLINE_COLOR = "#ffffff"  # voting-district outlines
OUTLINE_WIDTH = 0.15
LABEL_DISTRICTS = True     # write the district number on each district
UNASSIGNED_COLOR = "#d9d9d9"


def district_colors(n: int) -> dict[int, str]:
    """A fixed color for each district number 1..n, identical in every plan.

    Pulls 60 distinct colors from matplotlib's tab20, tab20b and tab20c sets,
    then shuffles them with a fixed seed so consecutive numbers (which are
    often neighbors on the map) don't get look-alike shades.
    """
    colors = []
    for name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(name)
        colors += [to_hex(cmap(i)) for i in range(cmap.N)]
    random.Random(42).shuffle(colors)
    return {d: colors[(d - 1) % len(colors)] for d in range(1, n + 1)}


COLORS = district_colors(N_DISTRICTS)


def label_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One label spot per district without merging shapes: the voting district
    closest to the middle of that district's voting districts."""
    pts = gdf.geometry.representative_point()
    rows = []
    for d, idx in gdf.groupby(DISTRICT_FIELD).groups.items():
        p = pts.loc[idx]
        cx, cy = p.x.mean(), p.y.mean()
        best = ((p.x - cx) ** 2 + (p.y - cy) ** 2).idxmin()
        rows.append({"district": d, "geometry": pts.loc[best]})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def make_map(gdf: gpd.GeoDataFrame, title: str) -> Figure:
    """Map of one plan (a GeoDataFrame from Plans.get). Returns the Figure."""
    gdf = gdf.to_crs(MAP_CRS)

    # District numbers as whole numbers; blanks become "unassigned"
    gdf[DISTRICT_FIELD] = gdf[DISTRICT_FIELD].astype("Int64")
    gdf["fill"] = gdf[DISTRICT_FIELD].map(COLORS).fillna(UNASSIGNED_COLOR)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    gdf.plot(ax=ax, color=gdf["fill"], edgecolor=OUTLINE_COLOR, linewidth=OUTLINE_WIDTH)

    if LABEL_DISTRICTS:
        for _, row in label_points(gdf.dropna(subset=[DISTRICT_FIELD])).iterrows():
            ax.annotate(str(row["district"]), xy=(row.geometry.x, row.geometry.y),
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Legend: one swatch per district present in this plan
    present = sorted(int(d) for d in gdf[DISTRICT_FIELD].dropna().unique())
    handles = [Patch(facecolor=COLORS.get(d, UNASSIGNED_COLOR), edgecolor="none", label=str(d))
               for d in present]
    if gdf[DISTRICT_FIELD].isna().any():
        handles.append(Patch(facecolor=UNASSIGNED_COLOR, edgecolor="none", label="Unassigned"))
    ax.legend(handles=handles, title="District", loc="center left", bbox_to_anchor=(1.0, 0.5),
              ncol=2, fontsize=8, title_fontsize=9, frameon=False)

    ax.set_title(title.replace("_", " "), fontsize=14, loc="left")
    ax.text(0, -0.02, "PA Senate citizen plan · voting districts colored by Senate district · "
            "2020 Census TIGER/Line", transform=ax.transAxes, fontsize=8, color="#666666")
    ax.set_axis_off()
    return fig


def main() -> None:
    plt.switch_backend("Agg")  # draw straight to files, no pop-up windows

    plans = Plans.load()
    args = sys.argv[1:]
    chosen = plans.match(args) if args else plans.names
    if not chosen:
        print(f"No plan names contain: {', '.join(args)}")
        return

    OUT_DIR.mkdir(exist_ok=True)
    for i, plan in enumerate(chosen, 1):
        try:
            fig = make_map(plans.get(plan), plan)
            out = OUT_DIR / f"{plan}.png"
            fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
            plt.close(fig)  # free memory before the next map
            print(f"[{i}/{len(chosen)}] saved {out}")
        except Exception as e:
            print(f"[{i}/{len(chosen)}] !! skipped {plan}: {e}")

    print(f"\nDone. Maps are in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
