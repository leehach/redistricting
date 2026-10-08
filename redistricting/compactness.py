"""Compactness of redistricting plans: one number per plan, for DEA.

``measure_plans`` scores many plans on one metric and returns one value
per plan: a column of the metrics table that DEA scores. ``measure_plan``
scores one plan, and ``measure_districts`` gives the per-district scores
behind it. A plan's score is the mean over its districts.

Inputs
------
A plan assigns each geounit to a district. Geounits are the small
geographic pieces, such as census blocks, tracts, or precincts, that
districts are built from. (DistrictBuilder uses the same term.) Each
geounit has a geoid. Two tables describe a set of plans:

- ``plans``: a DataFrame with one row per geounit, indexed by geoid, and
  one column per plan. Each column maps a geounit to its district. It
  holds nothing else.
- ``geounits``: a GeoDataFrame with one row per geounit, indexed by
  geoid. It holds what each geounit is: its shape and, when needed, data
  such as ``population``. Every plan shares it. Its CRS must be
  projected, not degrees.

Metrics
-------
Pass one name from ``MEASURES``. For every metric, a bigger value is more
compact, and 1 is the ideal: a circle for most, a convex shape for the
convex hull ratio, and a hull with no one outside the district for
population polygon.

- ``polsby_popper``, ``schwartzberg``, ``reock``, ``convex_hull``: from
  each district's shape alone.
- ``population_polygon``: also needs a ``population`` column, because it
  counts the people around the district, not just its shape.

A district made of separate parts (a ``MultiPolygon``) is scored as one
shape. Area and perimeter are the totals over all parts. The bounding
circle and the convex hull enclose all parts, so Reock and the convex
hull ratio also drop as the parts move apart. Polsby-Popper and
Schwartzberg do not.

The per-shape functions (``polsby_popper`` and so on) take a list or
GeoSeries of district shapes and return one score per district, in the
same order. They wrap the pysal functions in ``esda.shape``.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from esda import shape
from shapely import Polygon


def measure_plans(
    plans: pd.DataFrame, geounits: gpd.GeoDataFrame, measure: str
) -> pd.Series:
    """Return one compactness metric for each plan, indexed by plan.

    ``plans`` has one row per geounit, indexed by geoid, and one column per
    plan that maps each geounit to its district. The result has one value
    per plan column: a column of the metrics table that DEA scores.
    Otherwise takes the same inputs as ``measure_plan``.
    """
    _check_inputs(plans.index, geounits, measure)

    scores = {
        name: _score_plan(plans[name], geounits, measure)
        for name in plans.columns
    }
    return pd.Series(scores, name=measure)


def measure_plan(
    plan: pd.Series, geounits: gpd.GeoDataFrame, measure: str
) -> float:
    """Return one compactness metric for one plan: the mean over its
    districts.

    ``plan`` maps each geoid to its district: one column of the plans
    table, indexed by geoid. ``geounits`` holds the shapes the plan
    is built from (see the module docstring), indexed by geoid. It needs a
    ``population`` column only for ``"population_polygon"``. ``measure``
    is one name from ``MEASURES``.
    """
    _check_inputs(plan.index, geounits, measure)
    return _score_plan(plan, geounits, measure)


def measure_districts(
    plan: pd.Series, geounits: gpd.GeoDataFrame, measure: str
) -> pd.Series:
    """Return one compactness metric for each district, indexed by
    district.

    Takes the same inputs as ``measure_plan``.
    """
    _check_inputs(plan.index, geounits, measure)
    return _score_districts(plan, geounits, measure)


# the gear w/ teeth
def polsby_popper(districts: list[Polygon]):
    """Return each district's area over the area of a circle with the
    same perimeter: 4 * pi * A / P**2."""
    return shape.isoperimetric_quotient(districts)

# same as above, different scale - spreads results out more
def schwartzberg(districts: list[Polygon]):
    """Return the perimeter of a circle with the same area over each
    district's perimeter: 2 * sqrt(pi * A) / P.

    This is the square root of Polsby-Popper, so both rank districts in
    the same order.
    """
    return shape.isoareal_quotient(districts)

# min bounding circle
def reock(districts: list[Polygon]):
    """Return each district's area over the area of its minimum bounding
    circle."""
    return shape.minimum_bounding_circle_ratio(districts)

# like a rubber band 
def convex_hull(districts: list[Polygon]):
    """Return each district's area over the area of its convex hull."""
    return shape.convex_hull_ratio(districts)


# "Courts ask for it"
def population_polygon(
    districts: gpd.GeoSeries,
    district_population: pd.Series,
    geounits: gpd.GeoDataFrame,
):
    """Return each district's population over the population inside its
    convex hull.

    1 means the hull takes in no one outside the district. The score drops
    when the district's shape reaches around populated areas it leaves
    out. ``district_population`` is in the same order as ``districts``.
    ``geounits`` needs a ``population`` column.

    A geounit counts as inside a hull when its representative point (a
    point always inside the geounit) is inside the hull. Geounits cut by
    the hull edge count fully or not at all.
    """
    # TODO(Ben): whole geounits by point is a placeholder. Ask Lee whether
    # to weight cut geounits by the share of their area inside the hull.
    points = geounits.representative_point()

    hull_population = np.array(
        [geounits.loc[points.within(hull), "population"].sum()
         for hull in districts.convex_hull]
    )
    return np.asarray(district_population) / hull_population


def _check_inputs(geoids: pd.Index, geounits: gpd.GeoDataFrame, measure: str):
    """Raise ValueError for problems that hold for every plan."""
    if measure not in MEASURES:
        raise ValueError(f"Unknown measure {measure!r}. Use one of: {MEASURES}")

    # assign() in _score_districts matches rows by geoid, so the order
    # does not matter
    if set(geounits.index) != set(geoids):
        raise ValueError("geounits and plans must have the same geoids")

    if geounits.crs is not None and geounits.crs.is_geographic:
        raise ValueError(
            "geounits must use a projected CRS: area and perimeter in "
            "degrees give wrong scores"
        )

    needs_population = measure == "population_polygon"
    if needs_population and "population" not in geounits.columns:
        raise ValueError("population_polygon needs a population column")


def _score_plan(
    plan: pd.Series, geounits: gpd.GeoDataFrame, measure: str
) -> float:
    # TODO(Ben): mean is a placeholder. Ask Lee how DistrictBuilder turned
    # district scores into one plan score for the DEA paper.
    return float(_score_districts(plan, geounits, measure).mean())


def _score_districts(
    plan: pd.Series, geounits: gpd.GeoDataFrame, measure: str
) -> pd.Series:
    # dissolve would silently drop geounits with no district
    if plan.isna().any():
        raise ValueError(
            f"every geounit must have a district ({plan.name})"
        )

    if measure == "population_polygon":
        columns = ["population", geounits.geometry.name]
        districts = geounits[columns].assign(district=plan)
        districts = districts.dissolve(by="district", aggfunc="sum")
        scores = population_polygon(
            districts.geometry, districts["population"], geounits
        )
    else:
        # only the shapes are needed, so leave the other columns behind
        columns = [geounits.geometry.name]
        districts = geounits[columns].assign(district=plan)
        districts = districts.dissolve(by="district")
        scores = SHAPE_MEASURES[measure](districts.geometry)

    return pd.Series(scores, index=districts.index, name=measure)


SHAPE_MEASURES = {
    "polsby_popper": polsby_popper,
    "schwartzberg": schwartzberg,
    "reock": reock,
    "convex_hull": convex_hull,
}

# every name measure_plan accepts
MEASURES = [*SHAPE_MEASURES, "population_polygon"]
