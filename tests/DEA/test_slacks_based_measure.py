import math

import pandas as pd
from pytest import approx, raises

from redistricting.DEA.slacks_based_measure import calculate_efficiency, score_plans

mock_good_plans = pd.DataFrame([0.40, 0.50, 0.50])

mock_bad_plans = pd.DataFrame([10, 6, 8])

# Plans A, B, C from the hand-checked examples in the SBM notes.
mock_metrics = pd.DataFrame(
    {"compactness": [0.40, 0.50, 0.50], "county_splits": [10, 6, 8]},
    index=["A", "B", "C"],
)

mock_direction = {"compactness": "higher", "county_splits": "lower"}


def test_score_plans():
    scores = score_plans(mock_metrics, mock_direction)

    assert scores["A"] == approx(1 / 1.575)  # beaten on both metrics
    assert scores["B"] == approx(1)  # nothing beats it
    assert scores["C"] == approx(8 / 9)  # beaten on county splits only


def test_score_plans_keeps_index():
    scores = score_plans(mock_metrics.reset_index(drop=True), mock_direction)

    assert list(scores.index) == [0, 1, 2]


def test_score_plans_higher_only():
    scores = score_plans(mock_metrics[["compactness"]], {"compactness": "higher"})

    assert list(scores) == approx([0.8, 1, 1])  # A: 1 / (1 + 0.10 / 0.40)


def test_score_plans_lower_only_raises():
    with raises(ValueError, match="higher"):
        score_plans(mock_metrics[["county_splits"]], {"county_splits": "lower"})


def test_score_plans_missing_direction_raises():
    with raises(ValueError, match="county_splits"):
        score_plans(mock_metrics, {"compactness": "higher"})


def test_score_plans_unknown_metric_raises():
    with raises(ValueError, match="reock"):
        score_plans(mock_metrics, {**mock_direction, "reock": "higher"})


def test_score_plans_invalid_direction_raises():
    with raises(ValueError, match="good"):
        score_plans(mock_metrics, {**mock_direction, "compactness": "good"})


def test_score_plans_nan_raises():
    metrics = mock_metrics.copy()
    metrics.loc["B", "county_splits"] = math.nan

    with raises(ValueError, match="greater than 0"):
        score_plans(metrics, mock_direction)


def test_good_bad_rows_mismatch():
    with raises(ValueError, match="Number of rows in good and bad plans are not equal!"):
        calculate_efficiency(pd.DataFrame([0.40, 0.50]), mock_bad_plans, 1)

def test_zero_value_raises():
    bad_with_zero = pd.DataFrame([10, 0, 8])

    with raises(ValueError):
        calculate_efficiency(mock_good_plans, bad_with_zero, 1)

    good_with_zero = pd.DataFrame([0, 0, 8])

    with raises(ValueError):
        calculate_efficiency(good_with_zero, mock_bad_plans, 1)

def test_lt_zero_value_raises():
    bad_with_lt_zero = pd.DataFrame([10, -0.1, 8])

    with raises(ValueError):
        calculate_efficiency(mock_good_plans, bad_with_lt_zero, 1)

    good_with_lt_zero = pd.DataFrame([-0.01, 2, 8])

    with raises(ValueError):
        calculate_efficiency(good_with_lt_zero, mock_bad_plans, 1)

