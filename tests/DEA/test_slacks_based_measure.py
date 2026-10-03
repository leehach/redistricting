from redistricting.DEA.slacks_based_measure import calculate_efficiency
from pytest import approx

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
    (score, code) = calculate_efficiency(mock_good_plans, mock_bad_plans, 3, 1, 3, 1, 1)
    assert code == 0
    assert score == approx(1/1.575)

    (score, code) = calculate_efficiency(mock_good_plans, mock_bad_plans, 3, 1, 3, 1, 2)
    assert code == 0
    assert score == 1 # 2nd has no slack on good metric; no slack on bad metric

    (score, code) = calculate_efficiency(mock_good_plans, mock_bad_plans, 3, 1, 3, 1, 3)
    assert code == 0
    assert score == approx(8/9)

# def test_calculate_efficiency_fails():
   # This is a no-op until we fix the optimal-detection.