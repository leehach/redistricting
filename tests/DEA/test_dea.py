import pandas as pd
from pytest import approx, raises

from redistricting.DEA import dea

# Plans A, B, C from the hand-checked examples in the SBM notes.
mock_metrics = pd.DataFrame(
    {"compactness": [0.40, 0.50, 0.50], "county_splits": [10, 6, 8]},
    index=["A", "B", "C"],
)

mock_direction = {"compactness": "higher", "county_splits": "lower"}

# The radial search stops when its range is 0.0001 wide and returns the top.
RADIAL_TOLERANCE = 1e-4


def test_score_sbm():
    scores = dea.score(mock_metrics, mock_direction, "sbm")

    assert scores["A"] == approx(1 / 1.575)  # beaten on both metrics
    assert scores["B"] == approx(1)  # nothing beats it
    assert scores["C"] == approx(8 / 9)  # beaten on county splits only


def test_score_radial():
    scores = dea.score(mock_metrics, mock_direction, "radial")

    assert scores["A"] == approx(0.8, abs=RADIAL_TOLERANCE)
    assert scores["B"] == approx(1)
    assert scores["C"] == approx(1)  # radial cannot see C's leftover gap


def test_score_defaults_to_sbm():
    pd.testing.assert_series_equal(
        dea.score(mock_metrics, mock_direction),
        dea.score(mock_metrics, mock_direction, "sbm"),
    )


def test_score_keeps_index_and_names_the_series():
    sbm = dea.score(mock_metrics, mock_direction, "sbm")
    radial = dea.score(mock_metrics, mock_direction, "radial")

    assert list(sbm.index) == ["A", "B", "C"]
    assert sbm.name == "sbm_score"
    assert radial.name == "radial_score"


def test_score_all():
    scores = dea.score_all(mock_metrics, mock_direction)

    assert list(scores.columns) == ["sbm_score", "radial_score"]
    assert list(scores.index) == ["A", "B", "C"]
    assert list(scores["sbm_score"]) == approx([1 / 1.575, 1, 8 / 9])
    assert list(scores["radial_score"]) == approx([0.8, 1, 1], abs=RADIAL_TOLERANCE)


def test_unknown_method_raises():
    with raises(ValueError, match="Unknown method 'ebm'.*radial.*sbm"):
        dea.score(mock_metrics, mock_direction, "ebm")


def test_missing_direction_raises():
    with raises(ValueError, match="No direction given.*county_splits"):
        dea.score(mock_metrics, {"compactness": "higher"})


def test_unknown_metric_raises():
    with raises(ValueError, match="unknown metrics.*reock"):
        dea.score(mock_metrics, {**mock_direction, "reock": "higher"})


def test_invalid_direction_raises():
    with raises(ValueError, match="'higher' or 'lower'.*good"):
        dea.score(mock_metrics, {**mock_direction, "compactness": "good"})


def test_repeated_metric_name_raises():
    # two files joined with pd.concat can both have a "compactness" column
    metrics = pd.concat([mock_metrics, mock_metrics[["compactness"]]], axis=1)

    with raises(ValueError, match="unique.*compactness"):
        dea.score(metrics, mock_direction)


def test_no_plans_gives_no_scores():
    scores = dea.score(pd.DataFrame(), {}, "radial")

    assert scores.empty
    assert scores.name == "radial_score"
    assert scores.dtype == float


def test_sbm_needs_a_higher_metric():
    metrics = mock_metrics[["county_splits"]]
    direction = {"county_splits": "lower"}

    with raises(ValueError, match="higher"):
        dea.score(metrics, direction, "sbm")

    # radial has no such rule: 6/10, 6/6, 6/8
    radial = dea.score(metrics, direction, "radial")
    assert list(radial) == approx([0.6, 1, 0.75], abs=RADIAL_TOLERANCE)


def test_radial_needs_a_metric():
    with raises(ValueError, match="at least one metric"):
        dea.score(mock_metrics[[]], {}, "radial")


def test_score_all_raises_when_a_method_rejects_the_data():
    # a plan with 0 county splits: radial allows a 0, SBM does not
    metrics = pd.DataFrame({"compactness": [0.50, 0.40], "county_splits": [0, 4]})
    with raises(ValueError, match="greater than 0"):
        dea.score_all(metrics, mock_direction)

    radial = dea.score(metrics, mock_direction, "radial")
    assert list(radial) == approx([1, 0.8], abs=RADIAL_TOLERANCE)


def test_a_new_method_is_one_line(monkeypatch):
    def constant(higher_metrics, lower_metrics, row):
        return 0.5

    monkeypatch.setitem(dea.METHODS, "constant", constant)

    scores = dea.score(mock_metrics, mock_direction, "constant")
    assert list(scores) == [0.5, 0.5, 0.5]

    all_scores = dea.score_all(mock_metrics, mock_direction)
    assert list(all_scores.columns) == [
        "sbm_score",
        "radial_score",
        "constant_score",
    ]
