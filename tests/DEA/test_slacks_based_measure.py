from redistricting.DEA.slacks_based_measure import calculate_efficiency
from pytest import approx, raises

mock_good_plans = {
    (1, 1): 0.40, 
    (2, 1): 0.50, 
    (3, 1): 0.50
}

mock_bad_plans = {
    (1, 1): 10,
    (2, 1): 6,
    (3, 1): 8
}

def test_calculate_efficiency():
    score = calculate_efficiency(mock_good_plans, mock_bad_plans, 3, 1, 3, 1, 1)
    assert score == approx(1/1.575)

    score = calculate_efficiency(mock_good_plans, mock_bad_plans, 3, 1, 3, 1, 2)
    assert score == 1 # 2nd has no slack on good metric; no slack on bad metric

    score = calculate_efficiency(mock_good_plans, mock_bad_plans, 3, 1, 3, 1, 3)
    assert score == approx(8/9)

def test_good_bad_rows_mismatch():
    with raises(ValueError, match="Number of rows in good and bad plans are not equal!"):
        calculate_efficiency({
            (1, 1): 0.40, 
            (2, 1): 0.50,
        }, mock_bad_plans, 2, 1, 3, 1, 1)

def test_zero_value_raises():
    bad_with_zero = {(1, 1): 10, (2, 1): 0, (3, 1): 8}

    with raises(ValueError):
        calculate_efficiency(mock_good_plans, bad_with_zero, 3, 1, 3, 1, 1)

    good_with_zero = {(1, 1): 0, (2, 1): 0, (3, 1): 8}

    with raises(ValueError):
        calculate_efficiency(good_with_zero, mock_bad_plans, 3, 1, 3, 1, 1)

def test_lt_zero_value_raises():
    bad_with_lt_zero = {(1, 1): 10, (2, 1): -0.1, (3, 1): 8}

    with raises(ValueError):
        calculate_efficiency(mock_good_plans, bad_with_lt_zero, 3, 1, 3, 1, 1)

    good_with_lt_zero = {(1, 1): -0.01, (2, 1): 2, (3, 1): 8}

    with raises(ValueError):
        calculate_efficiency(good_with_lt_zero, mock_bad_plans, 3, 1, 3, 1, 1)

