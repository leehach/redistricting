import pandas as pd
import pytest

from pa_redistricting.maps import UNASSIGNED_COLOR, _is_grey, district_colors, plot_plan


def test_every_district_gets_a_different_color():
    colors = district_colors(50)
    assert len(colors) == 50
    assert len(set(colors.values())) == 50


def test_no_district_looks_unassigned():
    colors = district_colors(50).values()
    assert UNASSIGNED_COLOR not in colors
    assert not any(_is_grey(c) for c in colors)


def test_colors_are_the_same_every_time():
    assert district_colors(50) == district_colors(50)


def test_plot_plan_returns_a_figure_with_unassigned_in_legend():
    shapely_geometry = pytest.importorskip("shapely.geometry")
    gpd = pytest.importorskip("geopandas")
    from matplotlib.figure import Figure

    gdf = gpd.GeoDataFrame(
        {"district": pd.array([1, 2, None], dtype="Int64")},
        geometry=[shapely_geometry.box(i, 0, i + 1, 1) for i in range(3)], crs="EPSG:5070")
    fig = plot_plan(gdf, "test_plan")
    assert isinstance(fig, Figure)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["1", "2", "Unassigned"]
