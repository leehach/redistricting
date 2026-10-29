"""Score three made-up plans for compactness, the way DEA will see them.

The state is a 4 x 4 grid of square geounits. The middle 2 x 2 is a dense
city (100 people per geounit); the rest is rural (10 each). Each plan cuts
the grid into 4 districts of 4 geounits:

    quadrants      stripes        ring
    1 1 2 2        1 2 3 4        3 2 2 2
    1 1 2 2        1 2 3 4        3 4 4 2
    3 3 4 4        1 2 3 4        3 4 4 1
    3 3 4 4        1 2 3 4        3 1 1 1

Quadrants are squares, the most compact. Stripes are long and thin, so
their Polsby-Popper is low, but each stripe is its own convex hull, so
population polygon stays 1. In the ring plan, the city is one district
and the countryside wraps around it in hooks; the hooks' hulls take in
city people, so population polygon drops.

Run it from the repo root:
    micromamba run -n redistricting python -m examples.compactness_score_plans
"""

import geopandas as gpd
import pandas as pd
from shapely import box

from redistricting import compactness

# geoids name each geounit by its column and row, like "x2y1"
cells = [(x, y) for y in range(4) for x in range(4)]
geoids = pd.Index([f"x{x}y{y}" for (x, y) in cells], name="geoid")

city = {(1, 1), (2, 1), (1, 2), (2, 2)}
geounits = gpd.GeoDataFrame(
    {"population": [100 if cell in city else 10 for cell in cells]},
    geometry=[box(x, y, x + 1, y + 1) for (x, y) in cells],
    index=geoids,
)


def grid_plan(rows):
    """Turn a picture of districts (top row first) into a plan column."""
    by_cell = {
        (x, y): int(district)
        for (y, row) in enumerate(reversed(rows))
        for (x, district) in enumerate(row.split())
    }
    return [by_cell[cell] for cell in cells]


plans = pd.DataFrame(
    {
        "quadrants": grid_plan(["1 1 2 2", "1 1 2 2", "3 3 4 4", "3 3 4 4"]),
        "stripes": grid_plan(["1 2 3 4", "1 2 3 4", "1 2 3 4", "1 2 3 4"]),
        "ring": grid_plan(["3 2 2 2", "3 4 4 2", "3 4 4 1", "3 1 1 1"]),
    },
    index=geoids,
)

# one column per metric, one row per plan: the table DEA scores
metrics = pd.DataFrame(
    {
        measure: compactness.measure_plans(plans, geounits, measure)
        for measure in ["polsby_popper", "population_polygon"]
    }
)
print(metrics.round(3))
