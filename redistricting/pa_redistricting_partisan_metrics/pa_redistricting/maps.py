"""Maps of a plan: voting districts filled by Senate district.

    from pa_redistricting import Plans, plot_plan
    plans = Plans.load()
    fig = plot_plan(plans.get("allentown_map_by_as_jan-4-2022"), "Allentown map")
    fig.savefig("allentown.png", dpi=200)

Each district number gets the same color in every plan, so maps can be
compared side by side. Figures are built without pyplot, so this works in
scripts, notebooks and web servers alike.
"""

import random

import geopandas as gpd
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .config import N_DISTRICTS

MAP_CRS = "EPSG:5070"      # equal-area projection, so PA isn't stretched
FIGSIZE = (16, 9)          # inches
OUTLINE_COLOR = "#ffffff"  # voting-district outlines
OUTLINE_WIDTH = 0.15
UNASSIGNED_COLOR = "#d9d9d9"


def _is_grey(hex_color: str) -> bool:
    return len({hex_color[1:3], hex_color[3:5], hex_color[5:7]}) == 1


def district_colors(n: int = N_DISTRICTS) -> dict[int, str]:
    """A fixed color for each district number 1..n, identical in every plan.

    Takes the non-grey colors from matplotlib's tab20, tab20b and tab20c sets
    (greys would look like "unassigned"), then shuffles them with a fixed seed
    so consecutive numbers, which are often neighbors, don't get similar shades.
    """
    colors = [to_hex(colormaps[name](i))
              for name in ["tab20", "tab20b", "tab20c"]
              for i in range(colormaps[name].N)]
    colors = [c for c in colors if not _is_grey(c)]
    random.Random(42).shuffle(colors)
    return {d: colors[(d - 1) % len(colors)] for d in range(1, n + 1)}


COLORS = district_colors()


def label_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One label spot per district: the voting district closest to the middle
    of that district's voting districts (no shapes are merged)."""
    pts = gdf.geometry.representative_point()
    rows = []
    for d, idx in gdf.groupby("district").groups.items():
        p = pts.loc[idx]
        cx, cy = p.x.mean(), p.y.mean()
        best = ((p.x - cx) ** 2 + (p.y - cy) ** 2).idxmin()
        rows.append({"district": d, "geometry": pts.loc[best]})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def plot_plan(gdf: gpd.GeoDataFrame, title: str, label_districts: bool = True) -> Figure:
    """Map of one plan (a GeoDataFrame from Plans.get). Returns the Figure."""
    gdf = gdf.to_crs(MAP_CRS)
    gdf["district"] = gdf["district"].astype("Int64")
    fill = gdf["district"].map(COLORS).fillna(UNASSIGNED_COLOR)

    fig = Figure(figsize=FIGSIZE)
    ax = fig.subplots()
    gdf.plot(ax=ax, color=fill, edgecolor=OUTLINE_COLOR, linewidth=OUTLINE_WIDTH)

    if label_districts:
        for _, row in label_points(gdf.dropna(subset=["district"])).iterrows():
            ax.annotate(str(row["district"]), xy=(row.geometry.x, row.geometry.y),
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    present = sorted(int(d) for d in gdf["district"].dropna().unique())
    handles = [Patch(facecolor=COLORS.get(d, UNASSIGNED_COLOR), edgecolor="none", label=str(d))
               for d in present]
    if gdf["district"].isna().any():
        handles.append(Patch(facecolor=UNASSIGNED_COLOR, edgecolor="none", label="Unassigned"))
    ax.legend(handles=handles, title="District", loc="center left", bbox_to_anchor=(1.0, 0.5),
              ncol=2, fontsize=8, title_fontsize=9, frameon=False)

    ax.set_title(title.replace("_", " "), fontsize=14, loc="left")
    ax.text(0, -0.02, "PA Senate citizen plan · voting districts colored by Senate district · "
            "2020 Census TIGER/Line", transform=ax.transAxes, fontsize=8, color="#666666")
    ax.set_axis_off()
    return fig
