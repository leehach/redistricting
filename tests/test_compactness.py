import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely import MultiPolygon, Polygon, box

from redistricting import compactness

# Area 1, perimeter 4. The minimum bounding circle passes through the
# corners, so its radius is sqrt(2) / 2 and its area is pi / 2.
UNIT_SQUARE = box(0, 0, 1, 1)

# Three unit squares in an L. Area 3, perimeter 8.
L_SHAPE = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])

# Two unit squares with a gap of 2. Area 2, perimeter 8. The convex hull
# is the 4 x 1 box, area 4. The minimum bounding circle has that box's
# diagonal, sqrt(17), as its diameter, so its area is 17 * pi / 4.
TWO_SQUARES = MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])

def test_polsby_popper():
    scores = compactness.polsby_popper([UNIT_SQUARE])
    assert scores == pytest.approx([math.pi / 4])


def test_schwartzberg():
    scores = compactness.schwartzberg([UNIT_SQUARE])
    assert scores == pytest.approx([math.sqrt(math.pi) / 2])


def test_reock():
    scores = compactness.reock([UNIT_SQUARE])
    assert scores == pytest.approx([2 / math.pi])


def test_convex_hull():
    scores = compactness.convex_hull([UNIT_SQUARE])
    assert scores == pytest.approx([1])


def test_scores_follow_input_order():
    scores = compactness.polsby_popper([UNIT_SQUARE, L_SHAPE])
    assert scores == pytest.approx([math.pi / 4, 12 * math.pi / 64])


def test_multipolygon_polsby_popper():
    scores = compactness.polsby_popper([TWO_SQUARES])
    assert scores == pytest.approx([math.pi / 8])


def test_multipolygon_schwartzberg():
    scores = compactness.schwartzberg([TWO_SQUARES])
    assert scores == pytest.approx([math.sqrt(2 * math.pi) / 4])


def test_multipolygon_reock():
    scores = compactness.reock([TWO_SQUARES])
    assert scores == pytest.approx([8 / (17 * math.pi)])


def test_multipolygon_convex_hull():
    scores = compactness.convex_hull([TWO_SQUARES])
    assert scores == pytest.approx([1 / 2])


def test_multipolygon_parts_farther_apart():
    near = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    far = MultiPolygon([box(0, 0, 1, 1), box(9, 0, 10, 1)])

    # Area and perimeter do not change with the gap.
    near_pp, far_pp = compactness.polsby_popper([near, far])
    assert far_pp == pytest.approx(near_pp)

    # The bounding circle and the convex hull grow with the gap.
    near_reock, far_reock = compactness.reock([near, far])
    near_hull, far_hull = compactness.convex_hull([near, far])
    assert far_reock < near_reock
    assert far_hull < near_hull


# A 2 x 2 grid of unit squares, indexed by geoid:
#
#   c d
#   a b
#
# District 1 is a, b, c: the L_SHAPE above. District 2 is d: a unit square.
# d's center sits exactly on the L's hull edge, so population polygon is
# tested on U_GRID below instead. GRID has no population column: the shape
# metrics do not need one.
GRID = gpd.GeoDataFrame(
    geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
    index=pd.Index(["a", "b", "c", "d"], name="geoid"),
)
L_AND_SQUARE = pd.Series([1, 1, 1, 2], index=GRID.index)

# L: area 3, perimeter 8. Its hull is the 2 x 2 box minus one corner
# triangle: area 3.5. Its bounding circle has the diagonal (2, 0)-(0, 2) as
# its diameter: radius sqrt(2), area 2 * pi. Square: see UNIT_SQUARE.
L_AND_SQUARE_SCORES = {
    "polsby_popper": [12 * math.pi / 64, math.pi / 4],
    "schwartzberg": [math.sqrt(12 * math.pi / 64), math.sqrt(math.pi) / 2],
    "reock": [3 / (2 * math.pi), 2 / math.pi],
    "convex_hull": [3 / 3.5, 1],
}


@pytest.mark.parametrize("measure", L_AND_SQUARE_SCORES)
def test_measure_districts(measure):
    scores = compactness.measure_districts(L_AND_SQUARE, GRID, measure)

    assert list(scores.index) == [1, 2]
    assert list(scores) == pytest.approx(L_AND_SQUARE_SCORES[measure])


@pytest.mark.parametrize("measure", L_AND_SQUARE_SCORES)
def test_measure_plan_is_mean_of_districts(measure):
    (l_score, square_score) = L_AND_SQUARE_SCORES[measure]

    score = compactness.measure_plan(L_AND_SQUARE, GRID, measure)

    assert score == pytest.approx((l_score + square_score) / 2)


# Two plans on GRID. plan_l is L_AND_SQUARE. plan_halves splits the grid
# into left (a, c) and right (b, d) halves: two 1 x 2 rectangles, each
# with area 2 and perimeter 6, so Polsby-Popper 4 * pi * 2 / 36 = 2 * pi / 9.
PLANS = pd.DataFrame(
    {"plan_l": L_AND_SQUARE, "plan_halves": pd.Series([1, 2, 1, 2], index=GRID.index)}
)


def test_measure_plans():
    scores = compactness.measure_plans(PLANS, GRID, "polsby_popper")

    assert scores.name == "polsby_popper"
    assert list(scores.index) == ["plan_l", "plan_halves"]
    assert list(scores) == pytest.approx(
        [(12 * math.pi / 64 + math.pi / 4) / 2, 2 * math.pi / 9]
    )


def test_measure_plans_unassigned_unit_names_the_plan():
    plans = PLANS.copy()
    plans.loc["d", "plan_halves"] = math.nan

    with pytest.raises(ValueError, match="plan_halves"):
        compactness.measure_plans(plans, GRID, "polsby_popper")


def test_measure_plans_missing_geoid_raises():
    with pytest.raises(ValueError, match="same geoids"):
        compactness.measure_plans(PLANS.drop("d"), GRID, "polsby_popper")


def test_measure_plan_matches_geoids_not_order():
    shuffled = L_AND_SQUARE[["d", "b", "a", "c"]]

    assert compactness.measure_plan(
        shuffled, GRID, "polsby_popper"
    ) == pytest.approx(compactness.measure_plan(L_AND_SQUARE, GRID, "polsby_popper"))


def test_measure_plan_unknown_measure_raises():
    with pytest.raises(ValueError, match="polsby_popper"):
        compactness.measure_plan(L_AND_SQUARE, GRID, "roundness")


def test_measure_plan_missing_geoid_raises():
    with pytest.raises(ValueError, match="same geoids"):
        compactness.measure_plan(L_AND_SQUARE.drop("d"), GRID, "polsby_popper")


def test_measure_plan_unassigned_unit_raises():
    with pytest.raises(ValueError, match="every geounit"):
        compactness.measure_plan(
            L_AND_SQUARE.replace(2, math.nan), GRID, "polsby_popper"
        )


def test_measure_plan_geographic_crs_raises():
    with pytest.raises(ValueError, match="projected CRS"):
        compactness.measure_plan(
            L_AND_SQUARE, GRID.set_crs("EPSG:4326"), "polsby_popper"
        )


def test_population_polygon_without_population_raises():
    with pytest.raises(ValueError, match="population column"):
        compactness.measure_plan(L_AND_SQUARE, GRID, "population_polygon")


# A 3 x 2 grid of unit squares, indexed by geoid:
#
#   d e f
#   a b c
#
# District 1 is a U: a, b, c, d, f. Its convex hull is the whole 3 x 2 box,
# so it takes in e's center at (1.5, 1.5). District 2 is e alone.
U_GRID = gpd.GeoDataFrame(
    {"population": [10, 10, 10, 10, 40, 10]},
    geometry=[box(x, y, x + 1, y + 1) for y in (0, 1) for x in (0, 1, 2)],
    index=pd.Index(["a", "b", "c", "d", "e", "f"], name="geoid"),
)
U_AND_CENTER = pd.Series([1, 1, 1, 1, 2, 1], index=U_GRID.index)


def test_population_polygon():
    scores = compactness.measure_districts(
        U_AND_CENTER, U_GRID, "population_polygon"
    )

    # U: its 50 people over the 90 in its hull. e: a square is its own hull.
    assert list(scores) == pytest.approx([50 / 90, 1])


def test_measure_plan_population_polygon_is_mean():
    score = compactness.measure_plan(U_AND_CENTER, U_GRID, "population_polygon")

    assert score == pytest.approx((50 / 90 + 1) / 2)
