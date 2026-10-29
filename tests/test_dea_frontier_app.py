import pandas as pd

from redistricting.dea_frontier_app import (
    add_sbm_scores,
    create_app,
    pareto_2d,
    sbm_hull_2d,
)

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


def test_sbm_hull_2d():
    line = sbm_hull_2d(plans, "compactness", "county_splits", direction)

    assert list(line["plan_id"]) == ["B"]


def test_sbm_hull_2d_with_two_lower_metrics():
    # D trades county splits for population deviation against B
    two_lower = pd.DataFrame({
        "plan_id": ["A", "B", "C", "D"],
        "county_splits": [10, 6, 8, 9],
        "pop_dev": [0.05, 0.03, 0.04, 0.01],
    })

    line = sbm_hull_2d(two_lower, "county_splits", "pop_dev",
                       {"county_splits": "lower", "pop_dev": "lower"})

    assert list(line["plan_id"]) == ["B", "D"]  # sorted by county splits


def test_pareto_2d():
    line = pareto_2d(plans, "compactness", "county_splits", direction)

    assert list(line["plan_id"]) == ["B"]


# Like Figure 2 in Gopalan et al. 2026: D is on the Pareto frontier (no
# plan beats it on both), but above the hull line from C to G. E is
# beaten by C and D.
figure_2 = pd.DataFrame({
    "plan_id": ["A", "C", "D", "E", "G"],
    "pop_dev": [1, 3, 5, 6, 10],
    "splits": [10, 3, 2.9, 5, 1],
})

figure_2_direction = {"pop_dev": "lower", "splits": "lower"}


def test_pareto_2d_keeps_plans_off_the_hull():
    line = pareto_2d(figure_2, "pop_dev", "splits", figure_2_direction)

    assert list(line["plan_id"]) == ["A", "C", "D", "G"]


def test_sbm_hull_2d_drops_plans_off_the_hull():
    line = sbm_hull_2d(figure_2, "pop_dev", "splits", figure_2_direction)

    assert list(line["plan_id"]) == ["A", "C", "G"]


def test_create_app():
    results = add_sbm_scores(plans, direction)

    app = create_app(results, direction, "plan_id")

    assert app.layout is not None
