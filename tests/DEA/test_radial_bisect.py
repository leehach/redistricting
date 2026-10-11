import math

import pandas as pd
from pytest import approx, raises

from redistricting.DEA.radial_bisect import calculate_radial_efficiency, radial_scores

# The search stops when its range is 0.0001 wide, and it returns the top of
# that range. So a score can sit up to 0.0001 above the exact answer.
TOLERANCE = 1e-4

# Plans A, B, C from the hand-checked examples in the SBM notes.
mock_metrics = pd.DataFrame(
    {"compactness": [0.40, 0.50, 0.50], "county_splits": [10, 6, 8]},
    index=["A", "B", "C"],
)

mock_direction = {"compactness": "higher", "county_splits": "lower"}


def test_radial_scores():
    scores = radial_scores(mock_metrics, mock_direction)

    # A: B has 0.50 compactness (0.40 / 0.8) and 6 splits (below 10 * 0.8)
    assert scores["A"] == approx(0.8, abs=TOLERANCE)
    assert scores["B"] == approx(1)  # nothing beats it
    # C already has the best compactness, so no factor below 1 can grow it.
    # Radial misses that B has fewer splits. SBM scores C at 8/9.
    assert scores["C"] == approx(1)


def test_radial_scores_blend_of_plans():
    # D has no single plan to copy. A 1/3 : 2/3 blend of P1 and P2 gives
    # 3 compactness (2 / (2/3)) and 3 splits (4.5 * 2/3), so D scores 2/3.
    metrics = pd.DataFrame({"compactness": [1, 4, 2], "county_splits": [1, 4, 4.5]})

    scores = radial_scores(metrics, mock_direction)

    assert list(scores) == approx([1, 1, 2 / 3], abs=TOLERANCE)


def test_radial_scores_keeps_index():
    scores = radial_scores(mock_metrics.reset_index(drop=True), mock_direction)

    assert list(scores.index) == [0, 1, 2]


def test_radial_scores_higher_only():
    scores = radial_scores(mock_metrics[["compactness"]], {"compactness": "higher"})

    assert list(scores) == approx([0.8, 1, 1], abs=TOLERANCE)  # A: 0.40 / 0.50


def test_radial_scores_lower_only():
    scores = radial_scores(mock_metrics[["county_splits"]], {"county_splits": "lower"})

    assert list(scores) == approx([0.6, 1, 0.75], abs=TOLERANCE)  # 6/10, 6/6, 6/8


def test_radial_scores_allows_zero():
    # Unlike SBM, radial can score a plan with a 0 (here: 0 county splits).
    # This also guards the PRESOLVE option in radial_bisect.py. With GLPK
    # presolve on, and "other" accepted as "no blend", the 1st plan scores
    # 0.998 instead of 1.
    metrics = pd.DataFrame({"compactness": [0.50, 0.40], "county_splits": [0, 4]})

    scores = radial_scores(metrics, mock_direction)

    assert list(scores) == approx([1, 0.8], abs=TOLERANCE)  # 2nd: 0.40 / 0.50


def test_radial_scores_missing_direction_raises():
    with raises(ValueError, match="county_splits"):
        radial_scores(mock_metrics, {"compactness": "higher"})


def test_radial_scores_unknown_metric_raises():
    with raises(ValueError, match="reock"):
        radial_scores(mock_metrics, {**mock_direction, "reock": "higher"})


def test_radial_scores_invalid_direction_raises():
    with raises(ValueError, match="good"):
        radial_scores(mock_metrics, {**mock_direction, "compactness": "good"})


def test_radial_scores_no_metrics_raises():
    with raises(ValueError, match="at least one metric"):
        radial_scores(mock_metrics[[]], {})


def test_radial_scores_nan_raises():
    metrics = mock_metrics.copy()
    metrics.loc["B", "county_splits"] = math.nan

    with raises(ValueError, match="0 or greater"):
        radial_scores(metrics, mock_direction)


def test_radial_scores_negative_raises():
    metrics = mock_metrics.copy()
    metrics.loc["B", "county_splits"] = -6

    with raises(ValueError, match="0 or greater"):
        radial_scores(metrics, mock_direction)


def test_higher_lower_rows_mismatch():
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
