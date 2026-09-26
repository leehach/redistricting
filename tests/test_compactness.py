import math
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


def test_all_compactness_measures():
    measures = compactness.all_measures([UNIT_SQUARE])

    assert measures == {
        "polsby_popper": pytest.approx([math.pi / 4]),
        "schwartzberg": pytest.approx([math.sqrt(math.pi) / 2]),
        "reock": pytest.approx([2 / math.pi]),
        "convex_hull": pytest.approx([1]),
    }


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

    measures = compactness.all_measures([near, far])

    # Area and perimeter do not change with the gap.
    near_pp, far_pp = measures["polsby_popper"]
    assert far_pp == pytest.approx(near_pp)

    # The bounding circle and the convex hull grow with the gap.
    near_reock, far_reock = measures["reock"]
    near_hull, far_hull = measures["convex_hull"]
    assert far_reock < near_reock
    assert far_hull < near_hull