"""Score redistricting plans with the DEA method of your choice.

    from redistricting.DEA import dea

    dea.score(metrics, direction, "sbm")     # a Series: one SBM score per plan
    dea.score(metrics, direction, "radial")  # a Series: one radial score per plan
    dea.score_all(metrics, direction)        # a DataFrame: one column per method

`metrics` has one row per plan and one column per metric. `direction`
maps every column to "higher" (higher is better) or "lower" (lower is
better).

A method is a function that scores one plan. It takes the higher and
lower metrics and a row number (counted from 1), checks the values it
needs, and returns the score. To add a method, write that function and
add it to METHODS.
"""

from collections.abc import Callable

import pandas as pd

from redistricting.DEA import radial_bisect, slacks_based_measure

DIRECTIONS = {"higher", "lower"}

# (higher_metrics, lower_metrics, row) -> the score of the plan in `row`
ScorePlan = Callable[[pd.DataFrame, pd.DataFrame, int], float]

# every method `score` can use, by name
METHODS: dict[str, ScorePlan] = {
    "sbm": slacks_based_measure.calculate_efficiency,
    "radial": radial_bisect.calculate_radial_efficiency,
}


def score(
    metrics: pd.DataFrame, direction: dict[str, str], method: str = "sbm"
) -> pd.Series:
    """Return the score of each plan, with the same index as `metrics`.

    The Series is named "<method>_score". Raises ValueError for an unknown
    method, a bad `direction`, or values that do not suit the method (for
    example, SBM needs every value above 0).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {sorted(METHODS)}")

    higher_metrics, lower_metrics = _split_by_direction(metrics, direction)
    score_plan = METHODS[method]

    scores = [
        score_plan(higher_metrics, lower_metrics, row)
        for row in range(1, len(metrics) + 1)
    ]

    return pd.Series(
        scores, index=metrics.index, name=f"{method}_score", dtype=float
    )


def score_all(metrics: pd.DataFrame, direction: dict[str, str]) -> pd.DataFrame:
    """Return one column of scores per method in METHODS.

    If one method rejects the data, this raises. Call `score` with a
    method that suits the data instead.
    """
    return pd.concat([score(metrics, direction, name) for name in METHODS], axis=1)


def _split_by_direction(metrics, direction):
    """Check `direction`, then return the (higher, lower) metric tables.

    Raises ValueError if two columns share a name, a column has no
    direction, a direction names no column, or a direction is not "higher"
    or "lower".
    """
    # a repeated name would select every copy each time, so the metric
    # would count more than once
    repeated = metrics.columns[metrics.columns.duplicated()]
    if not repeated.empty:
        raise ValueError(
            f"Metric names must be unique, repeated: {sorted(set(repeated))}"
        )

    missing = set(metrics.columns) - set(direction)
    if missing:
        raise ValueError(f"No direction given for metrics: {sorted(missing)}")

    unknown = set(direction) - set(metrics.columns)
    if unknown:
        raise ValueError(f"Direction given for unknown metrics: {sorted(unknown)}")

    invalid = set(direction.values()) - DIRECTIONS
    if invalid:
        raise ValueError(
            f"Direction must be 'higher' or 'lower', got: {sorted(invalid)}"
        )

    higher = [name for name in metrics.columns if direction[name] == "higher"]
    lower = [name for name in metrics.columns if direction[name] == "lower"]

    return metrics[higher], metrics[lower]
