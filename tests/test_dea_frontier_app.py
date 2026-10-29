import pandas as pd

from redistricting.dea_frontier_app import add_sbm_scores, create_app

# Plans A, B, C from the SBM tests: B beats A and C.
plans = pd.DataFrame({
    "plan_id": ["A", "B", "C"],
    "compactness": [0.40, 0.50, 0.50],
    "county_splits": [10, 6, 8],
})

direction = {"compactness": "higher", "county_splits": "lower"}


def test_add_sbm_scores_marks_frontier():
    results = add_sbm_scores(plans, direction)

    assert list(results["On_Frontier"]) == [False, True, False]


def test_add_sbm_scores_ignores_id_column():
    results = add_sbm_scores(plans, direction)

    assert list(results["plan_id"]) == ["A", "B", "C"]


def test_create_app():
    results = add_sbm_scores(plans, direction)

    app = create_app(results, direction, "plan_id")

    assert app.layout is not None
