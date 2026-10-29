import pandas as pd

from redistricting import dea_frontier_app
from redistricting.dea_frontier_app import (
    Dataset,
    add_dea_scores,
    convex_hull_2d,
    create_app,
    pareto_2d,
)

# Plans A, B, C from the SBM tests: B beats A and C.
plans = pd.DataFrame({
    "plan_id": ["A", "B", "C"],
    "compactness": [0.40, 0.50, 0.50],
    "county_splits": [10, 6, 8],
})

direction = {"compactness": "higher", "county_splits": "lower"}


def test_add_dea_scores_marks_frontier():
    results = add_dea_scores(plans, direction)

    assert list(results["On_Frontier"]) == [False, True, False]


def test_add_dea_scores_reads_cache(tmp_path, monkeypatch):
    first = add_dea_scores(plans, direction, tmp_path)

    # a second solve would fail, so this call must read the cache file
    monkeypatch.setattr(dea_frontier_app, "score_plans", _fail_if_called)
    second = add_dea_scores(plans, direction, tmp_path)

    assert list(second["DEA_Score"]) == list(first["DEA_Score"])
    assert len(list(tmp_path.glob("dea_scores_*.csv"))) == 1


def test_add_dea_scores_cache_misses_on_new_data(tmp_path):
    add_dea_scores(plans, direction, tmp_path)
    changed = plans.assign(county_splits=[10, 6, 7])

    results = add_dea_scores(changed, direction, tmp_path)

    assert len(list(tmp_path.glob("dea_scores_*.csv"))) == 2
    assert list(results["On_Frontier"]) == [False, True, False]


def test_add_dea_scores_ignores_id_column():
    results = add_dea_scores(plans, direction)

    assert list(results["plan_id"]) == ["A", "B", "C"]


def test_convex_hull_2d():
    line = convex_hull_2d(plans, "compactness", "county_splits", direction)

    assert list(line["plan_id"]) == ["B"]


def test_convex_hull_2d_with_two_lower_metrics():
    # D trades county splits for population deviation against B
    two_lower = pd.DataFrame({
        "plan_id": ["A", "B", "C", "D"],
        "county_splits": [10, 6, 8, 9],
        "pop_dev": [0.05, 0.03, 0.04, 0.01],
    })

    line = convex_hull_2d(two_lower, "county_splits", "pop_dev",
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


def test_convex_hull_2d_drops_plans_off_the_hull():
    line = convex_hull_2d(figure_2, "pop_dev", "splits", figure_2_direction)

    assert list(line["plan_id"]) == ["A", "C", "G"]


def test_create_app():
    results = add_dea_scores(plans, direction)

    app = create_app({"A": Dataset(results, direction, "plan_id")})

    assert app.layout is not None


def test_create_app_with_two_datasets():
    first = Dataset(add_dea_scores(plans, direction), direction, "plan_id")
    other_plans = pd.DataFrame({"plan_id": ["X", "Y"], "score": [0.3, 0.6]})
    other_direction = {"score": "higher"}
    second = Dataset(add_dea_scores(other_plans, other_direction),
                     other_direction, "plan_id")

    app = create_app({"First": first, "Second": second})

    assert app.layout["data"].options == ["First", "Second"]
    assert app.layout["data"].value == "First"


def _fail_if_called(*args):
    raise AssertionError("score_plans ran instead of reading the cache")
