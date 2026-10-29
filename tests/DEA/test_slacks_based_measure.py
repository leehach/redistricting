import math

import pandas as pd
from pytest import approx, raises

from redistricting.DEA import dea
from redistricting.DEA.slacks_based_measure import calculate_efficiency

mock_higher_metrics = pd.DataFrame([0.40, 0.50, 0.50])

mock_lower_metrics = pd.DataFrame([10, 6, 8])

# Plans A, B, C from the hand-checked examples in the SBM notes.
mock_metrics = pd.DataFrame(
    {"compactness": [0.40, 0.50, 0.50], "county_splits": [10, 6, 8]},
    index=["A", "B", "C"],
)

mock_direction = {"compactness": "higher", "county_splits": "lower"}

# DEA's own checks (directions, index, at least one "higher" metric) are
# tested in test_dea.py. These tests cover the SBM math and value rules.


def sbm(metrics, direction=mock_direction):
    return dea.score(metrics, direction, "sbm")


def test_sbm_scores():
    scores = sbm(mock_metrics)

    assert scores["A"] == approx(1 / 1.575)  # beaten on both metrics
    assert scores["B"] == approx(1)  # nothing beats it
    assert scores["C"] == approx(8 / 9)  # beaten on county splits only


def test_sbm_scores_higher_only():
    scores = sbm(mock_metrics[["compactness"]], {"compactness": "higher"})

    assert list(scores) == approx([0.8, 1, 1])  # A: 1 / (1 + 0.10 / 0.40)


def test_sbm_scores_nan_raises():
    metrics = mock_metrics.copy()
    metrics.loc["B", "county_splits"] = math.nan

    with raises(ValueError, match="greater than 0"):
        sbm(metrics)


def test_calculate_efficiency_rows_mismatch():
    with raises(ValueError, match="Number of rows in higher and lower metrics are not equal!"):
        calculate_efficiency(pd.DataFrame([0.40, 0.50]), mock_lower_metrics, 1)

def test_calculate_efficiency_zero_raises():
    lower_with_zero = pd.DataFrame([10, 0, 8])

    with raises(ValueError):
        calculate_efficiency(mock_higher_metrics, lower_with_zero, 1)

    higher_with_zero = pd.DataFrame([0, 0, 8])

    with raises(ValueError):
        calculate_efficiency(higher_with_zero, mock_lower_metrics, 1)

def test_calculate_efficiency_negative_raises():
    lower_with_lt_zero = pd.DataFrame([10, -0.1, 8])

    with raises(ValueError):
        calculate_efficiency(mock_higher_metrics, lower_with_lt_zero, 1)

    higher_with_lt_zero = pd.DataFrame([-0.01, 2, 8])

    with raises(ValueError):
        calculate_efficiency(higher_with_lt_zero, mock_lower_metrics, 1)
