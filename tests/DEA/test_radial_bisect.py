import math

import pandas as pd
from pytest import approx, raises

from redistricting.DEA import dea
from redistricting.DEA.radial_bisect import calculate_radial_efficiency

# The search stops when its range is 0.0001 wide, and it returns the top of
# that range. So a score can sit up to 0.0001 above the exact answer.
TOLERANCE = 1e-4

# Plans A, B, C from the hand-checked examples in the SBM notes.
mock_metrics = pd.DataFrame(
    {"compactness": [0.40, 0.50, 0.50], "county_splits": [10, 6, 8]},
    index=["A", "B", "C"],
)

mock_direction = {"compactness": "higher", "county_splits": "lower"}

# DEA's own checks (directions, index, at least one metric) are tested in
# test_dea.py. These tests cover the radial math and value rules.


def radial(metrics, direction=mock_direction):
    return dea.score(metrics, direction, "radial")


def test_radial():
    scores = radial(mock_metrics)

    # A: B has 0.50 compactness (0.40 / 0.8) and 6 splits (below 10 * 0.8)
    assert scores["A"] == approx(0.8, abs=TOLERANCE)
    assert scores["B"] == approx(1)  # nothing beats it
    # C already has the best compactness, so no factor below 1 can grow it.
    # Radial misses that B has fewer splits. SBM scores C at 8/9.
    assert scores["C"] == approx(1)


def test_radial_blend_of_plans():
    # D has no single plan to copy. A 1/3 : 2/3 blend of P1 and P2 gives
    # 3 compactness (2 / (2/3)) and 3 splits (4.5 * 2/3), so D scores 2/3.
    metrics = pd.DataFrame({"compactness": [1, 4, 2], "county_splits": [1, 4, 4.5]})

    scores = radial(metrics)

    assert list(scores) == approx([1, 1, 2 / 3], abs=TOLERANCE)


def test_radial_higher_only():
    scores = radial(mock_metrics[["compactness"]], {"compactness": "higher"})

    assert list(scores) == approx([0.8, 1, 1], abs=TOLERANCE)  # A: 0.40 / 0.50


def test_radial_lower_only():
    scores = radial(mock_metrics[["county_splits"]], {"county_splits": "lower"})

    assert list(scores) == approx([0.6, 1, 0.75], abs=TOLERANCE)  # 6/10, 6/6, 6/8


def test_radial_allows_a_zero_metric():
    # Unlike SBM, radial can score a plan with a 0 (here: 0 county splits),
    # as long as the plan has another value above 0. See the next tests.
    # This also guards the PRESOLVE option in radial_bisect.py. With GLPK
    # presolve on, and "other" accepted as "no blend", the 1st plan scores
    # 0.998 instead of 1.
    metrics = pd.DataFrame({"compactness": [0.50, 0.40], "county_splits": [0, 4]})

    scores = radial(metrics)

    assert list(scores) == approx([1, 0.8], abs=TOLERANCE)  # 2nd: 0.40 / 0.50


# Every θ works for these plans, so the search falls to THETA_FLOOR. That
# is not a measured score, so radial returns NaN (see NOT MEASURED in
# radial_bisect.py).


def test_radial_all_zero_lower_plan_scores_nan():
    # 0 splits beats 4 at every θ, so neither plan has a measured score
    metrics = pd.DataFrame({"county_splits": [0, 4]})

    scores = radial(metrics, {"county_splits": "lower"})

    assert scores.isna().all()


def test_radial_all_zero_mixed_plan_scores_nan():
    # the 2nd plan has more compactness but more splits, so it scores 1
    metrics = pd.DataFrame({"compactness": [0, 0.5], "county_splits": [0, 4]})

    scores = radial(metrics)

    assert math.isnan(scores[0])
    assert scores[1] == approx(1)


def test_radial_all_zero_higher_plan_scores_nan():
    metrics = pd.DataFrame({"compactness": [0, 0.5]})

    scores = radial(metrics, {"compactness": "higher"})

    assert math.isnan(scores[0])
    assert scores[1] == approx(1)


def test_radial_nan_raises():
    metrics = mock_metrics.copy()
    metrics.loc["B", "county_splits"] = math.nan

    with raises(ValueError, match="0 or greater"):
        radial(metrics)


def test_radial_negative_raises():
    metrics = mock_metrics.copy()
    metrics.loc["B", "county_splits"] = -6

    with raises(ValueError, match="0 or greater"):
        radial(metrics)


def test_calculate_radial_efficiency_rows_mismatch():
    higher = pd.DataFrame([0.40, 0.50])
    lower = pd.DataFrame([10, 6, 8])

    with raises(
        ValueError, match="Number of rows in higher and lower metrics are not equal!"
    ):
        calculate_radial_efficiency(higher, lower, 1)


def test_calculate_radial_efficiency_negative_raises():
    higher = pd.DataFrame([0.40, 0.50, 0.50])
    lower = pd.DataFrame([10, -6, 8])

    with raises(ValueError, match="0 or greater"):
        calculate_radial_efficiency(higher, lower, 1)
