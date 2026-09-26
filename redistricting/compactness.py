"""Compactness metrics for the districts of a plan.

Each function takes a collection of districts (a list, NumPy array, or
GeoSeries of shapely ``Polygon`` or ``MultiPolygon``) and returns a NumPy
array with one score per district, in the same order. For every metric,
a bigger value is more compact, and 1 is the ideal (a circle, or a convex
shape for the convex hull ratio).

A ``MultiPolygon`` is scored as one shape. Area and perimeter are the
totals over all parts. The bounding circle and the convex hull enclose all
parts, so Reock and the convex hull ratio also drop as the parts move
apart. Polsby-Popper and Schwartzberg do not.

The metrics wrap the pysal functions in ``esda.shape``.
"""

from esda import shape

def polsby_popper(districts):
    """Return each district's area over the area of a circle with the
    same perimeter: 4 * pi * A / P**2."""
    return shape.isoperimetric_quotient(districts)


def schwartzberg(districts):
    """Return the perimeter of a circle with the same area over each
    district's perimeter: 2 * sqrt(pi * A) / P.

    This is the square root of Polsby-Popper, so both rank districts in
    the same order.
    """
    return shape.isoareal_quotient(districts)


def reock(districts):
    """Return each district's area over the area of its minimum bounding
    circle."""
    return shape.minimum_bounding_circle_ratio(districts)


def convex_hull(districts):
    """Return each district's area over the area of its convex hull."""
    return shape.convex_hull_ratio(districts)


def all_measures(districts):
    """Return a dict that maps each metric name to its array of scores."""
    return {
        "polsby_popper": polsby_popper(districts),
        "schwartzberg": schwartzberg(districts),
        "reock": reock(districts),
        "convex_hull": convex_hull(districts),
    }
