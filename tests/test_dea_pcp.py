"""
Tests for dea_pcp.py.

Run from the top folder of the project (the one that holds both the
redistricting/ and tests/ folders):

    python -m pytest tests/test_dea_pcp.py -v

The small hand-made examples below are simple enough to check by hand,
so each expected score is worked out in a comment next to the test.
"""

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from redistricting import dea_pcp
from redistricting.dea_pcp import (
    DEFAULT_METRICS,
    make_mock_metrics,
    pareto_nondominated,
    plot_pcp,
    run_dea,
)


# --------------------------------------------------------------------------
# Small hand-checkable data sets
# --------------------------------------------------------------------------
# One "higher is better" metric (Good) and one "lower is better" (Bad).
LINE_METRICS = {"Good": True, "Bad": False}


@pytest.fixture
def line_plans():
    """P1, P2, P3 sit on a straight line, so all three are on the frontier.

    P4 and P5 are each beaten by a frontier plan:
      P4 (Good 1, Bad 3) is beaten by P1 (Good 1, Bad 1)
      P5 (Good 2, Bad 3) is beaten by P2 (Good 2, Bad 2)
    """
    return pd.DataFrame({
        "Plan_ID": ["P1", "P2", "P3", "P4", "P5"],
        "Good":    [1.0, 2.0, 3.0, 1.0, 2.0],
        "Bad":     [1.0, 2.0, 3.0, 3.0, 3.0],
    })


@pytest.fixture
def weak_plans():
    """P2 ties P1 on everything except X2, where it is worse.

    Shrinking all of P2's inputs by the same factor can't help (X1 is
    already tied with P1), so its score is 1, but it still has room to
    improve on X2. That makes it "weakly efficient".
    """
    return pd.DataFrame({
        "Plan_ID": ["P1", "P2"],
        "Out": [1.0, 1.0],
        "X1":  [1.0, 1.0],
        "X2":  [1.0, 2.0],
    })


WEAK_METRICS = {"Out": True, "X1": False, "X2": False}


def by_id(results: pd.DataFrame) -> pd.DataFrame:
    """Index results by plan ID so tests can write results.loc['P4', ...]."""
    return results.set_index("Plan_ID")


# --------------------------------------------------------------------------
# make_mock_metrics
# --------------------------------------------------------------------------
class TestMockData:
    def test_shape_and_columns(self):
        df = make_mock_metrics(n=25)
        assert len(df) == 25
        assert list(df.columns) == ["Plan_ID", *DEFAULT_METRICS]

    def test_same_seed_gives_same_data(self):
        pd.testing.assert_frame_equal(make_mock_metrics(seed=7),
                                      make_mock_metrics(seed=7))

    def test_different_seed_gives_different_data(self):
        assert not make_mock_metrics(seed=1).equals(make_mock_metrics(seed=2))

    def test_values_are_valid_for_dea(self):
        df = make_mock_metrics(n=200)
        assert (df[list(DEFAULT_METRICS)] >= 0).all().all()
        assert df["Compactness"].between(0.05, 0.95).all()
        assert df["Plan_ID"].is_unique


# --------------------------------------------------------------------------
# pareto_nondominated
# --------------------------------------------------------------------------
class TestPareto:
    def test_simple_case(self):
        # columns: higher-better, lower-better
        values = np.array([[3, 1],    # best on both -> kept
                           [2, 2],    # beaten by row 0 -> dropped
                           [1, 3]])   # beaten by row 0 -> dropped
        mask = pareto_nondominated(values, np.array([True, False]))
        assert mask.tolist() == [True, False, False]

    def test_trade_off_keeps_both(self):
        # each plan is better on one metric, so neither beats the other
        values = np.array([[3, 3], [1, 1]])
        mask = pareto_nondominated(values, np.array([True, False]))
        assert mask.tolist() == [True, True]

    def test_identical_plans_both_kept(self):
        # a tie is not a win, so neither copy beats the other
        values = np.array([[2, 2], [2, 2]])
        assert pareto_nondominated(values, np.array([True, True])).all()

    def test_matches_hand_example(self, line_plans):
        values = line_plans[list(LINE_METRICS)].to_numpy()
        mask = pareto_nondominated(values, np.array(list(LINE_METRICS.values())))
        assert mask.tolist() == [True, True, True, False, False]


# --------------------------------------------------------------------------
# run_dea: scores on the hand-made examples
# --------------------------------------------------------------------------
class TestRunDeaScores:
    def test_input_orientation_scores(self, line_plans):
        r = by_id(run_dea(line_plans, metrics=LINE_METRICS))
        # Frontier plans score 1.
        for plan in ["P1", "P2", "P3"]:
            assert r.loc[plan, "DEA_Score"] == pytest.approx(1.0)
        # P4 needs Good >= 1; P1 gets that with Bad = 1, so P4's Bad of 3
        # could shrink to 1/3 of itself.
        assert r.loc["P4", "DEA_Score"] == pytest.approx(1 / 3, abs=1e-5)
        # P5 needs Good >= 2; P2 gets that with Bad = 2, so 2/3.
        assert r.loc["P5", "DEA_Score"] == pytest.approx(2 / 3, abs=1e-5)

    def test_output_orientation_scores(self, line_plans):
        r = by_id(run_dea(line_plans, metrics=LINE_METRICS, orientation="output"))
        # P4 has Bad = 3; P3 reaches Good = 3 with Bad = 3, so P4's Good
        # could triple. Score = 1/3.
        assert r.loc["P4", "DEA_Score"] == pytest.approx(1 / 3, abs=1e-5)
        # P5: Good could go from 2 to 3, so the score is 2/3.
        assert r.loc["P5", "DEA_Score"] == pytest.approx(2 / 3, abs=1e-5)
        assert r.loc["P1", "DEA_Score"] == pytest.approx(1.0)

    def test_status_columns(self, line_plans):
        r = by_id(run_dea(line_plans, metrics=LINE_METRICS))
        assert r["DEA_Status"].to_dict() == {
            "P1": "efficient", "P2": "efficient", "P3": "efficient",
            "P4": "inefficient", "P5": "inefficient"}
        assert r["DEA_Efficient"].tolist() == [1, 1, 1, 0, 0]
        assert r["Pareto_Optimal"].tolist() == [1, 1, 1, 0, 0]

    def test_peers(self, line_plans):
        r = by_id(run_dea(line_plans, metrics=LINE_METRICS))
        # P4 and P5 are each benchmarked against the plan that beats them.
        assert r.loc["P4", "DEA_Peers"] == "P1"
        assert r.loc["P5", "DEA_Peers"] == "P2"
        # efficient plans get no peers
        assert (r.loc[["P1", "P2", "P3"], "DEA_Peers"] == "").all()

    def test_weakly_efficient(self, weak_plans):
        r = by_id(run_dea(weak_plans, metrics=WEAK_METRICS))
        assert r.loc["P2", "DEA_Score"] == pytest.approx(1.0)
        assert r.loc["P2", "DEA_Status"] == "weakly efficient"
        assert r.loc["P2", "DEA_Efficient"] == 0
        assert r.loc["P2", "Pareto_Optimal"] == 0
        assert r.loc["P2", "Slack_X2"] == pytest.approx(1.0)  # X2 could drop 2 -> 1
        assert r.loc["P2", "DEA_Peers"] == "P1"
        assert r.loc["P1", "DEA_Status"] == "efficient"

    def test_slack_columns_added(self, line_plans):
        r = run_dea(line_plans, metrics=LINE_METRICS)
        assert {"Slack_Good", "Slack_Bad"} <= set(r.columns)

    def test_input_not_changed(self, line_plans):
        before = line_plans.copy()
        run_dea(line_plans, metrics=LINE_METRICS)
        pd.testing.assert_frame_equal(line_plans, before)

    def test_custom_id_column(self, line_plans):
        df = line_plans.rename(columns={"Plan_ID": "Name"})
        r = run_dea(df, id_col="Name", metrics=LINE_METRICS)
        assert r.loc[r["Name"] == "P4", "DEA_Peers"].item() == "P1"


# --------------------------------------------------------------------------
# run_dea: general properties on the mock data
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_results():
    return run_dea(make_mock_metrics(n=60, seed=42))


class TestRunDeaMock:
    def test_one_row_per_plan(self, mock_results):
        assert len(mock_results) == 60

    def test_scores_between_0_and_1(self, mock_results):
        assert mock_results["DEA_Score"].between(0, 1).all()

    def test_some_efficient_some_not(self, mock_results):
        n_eff = mock_results["DEA_Efficient"].sum()
        assert 0 < n_eff < len(mock_results)

    def test_efficient_plans_score_1(self, mock_results):
        eff = mock_results[mock_results["DEA_Efficient"] == 1]
        assert (eff["DEA_Score"] >= 1 - 1e-6).all()

    def test_efficient_implies_pareto_optimal(self, mock_results):
        # a plan that another plan beats outright can't be on the frontier
        eff = mock_results[mock_results["DEA_Efficient"] == 1]
        assert (eff["Pareto_Optimal"] == 1).all()

    def test_status_matches_flags(self, mock_results):
        r = mock_results
        assert ((r["DEA_Status"] == "efficient") == (r["DEA_Efficient"] == 1)).all()
        assert set(r["DEA_Status"]) <= {"efficient", "weakly efficient", "inefficient"}

    def test_peers_are_real_pareto_plans(self, mock_results):
        pareto_ids = set(mock_results.loc[mock_results["Pareto_Optimal"] == 1, "Plan_ID"])
        for peers in mock_results["DEA_Peers"]:
            for peer in filter(None, peers.split(", ")):
                assert peer in pareto_ids

    def test_slacks_not_negative(self, mock_results):
        slack_cols = [c for c in mock_results.columns if c.startswith("Slack_")]
        assert len(slack_cols) == len(DEFAULT_METRICS)
        assert (mock_results[slack_cols] >= -1e-9).all().all()

    def test_row_order_does_not_change_scores(self):
        df = make_mock_metrics(n=30, seed=3)
        a = run_dea(df).set_index("Plan_ID")["DEA_Score"]
        b = run_dea(df.sample(frac=1, random_state=0)).set_index("Plan_ID")["DEA_Score"]
        pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), atol=1e-5)


# --------------------------------------------------------------------------
# run_dea: bad input and edge cases
# --------------------------------------------------------------------------
class TestRunDeaEdgeCases:
    def test_missing_column(self, line_plans):
        with pytest.raises(KeyError, match="Missing columns"):
            run_dea(line_plans, metrics={"Good": True, "Nope": False})

    def test_missing_id_column(self, line_plans):
        with pytest.raises(KeyError):
            run_dea(line_plans, id_col="Wrong", metrics=LINE_METRICS)

    def test_bad_orientation(self, line_plans):
        with pytest.raises(ValueError, match="orientation"):
            run_dea(line_plans, metrics=LINE_METRICS, orientation="sideways")

    def test_negative_values_rejected(self, line_plans):
        line_plans.loc[0, "Bad"] = -1
        with pytest.raises(ValueError, match="non-negative"):
            run_dea(line_plans, metrics=LINE_METRICS)

    def test_rows_with_blanks_dropped(self, line_plans, capsys):
        line_plans.loc[4, "Good"] = np.nan
        r = run_dea(line_plans, metrics=LINE_METRICS)
        assert len(r) == 4
        assert "P5" not in r["Plan_ID"].values
        assert "dropped 1 plan" in capsys.readouterr().out

    def test_only_higher_is_better_metrics(self, capsys):
        df = pd.DataFrame({"Plan_ID": ["A", "B", "C"],
                           "M1": [1.0, 2.0, 1.0], "M2": [2.0, 1.0, 1.0]})
        r = by_id(run_dea(df, metrics={"M1": True, "M2": True}))
        assert "switching to output" in capsys.readouterr().out
        assert r.loc["A", "DEA_Efficient"] == 1
        assert r.loc["B", "DEA_Efficient"] == 1
        assert r.loc["C", "DEA_Score"] < 1   # beaten by both A and B

    def test_only_lower_is_better_metrics(self, capsys):
        df = pd.DataFrame({"Plan_ID": ["A", "B", "C"],
                           "M1": [1.0, 2.0, 2.0], "M2": [2.0, 1.0, 2.0]})
        r = by_id(run_dea(df, metrics={"M1": False, "M2": False},
                          orientation="output"))
        assert "switching to input" in capsys.readouterr().out
        assert r.loc["C", "DEA_Score"] < 1   # beaten by both A and B

    def test_single_plan(self):
        df = pd.DataFrame({"Plan_ID": ["Only"], "Good": [1.0], "Bad": [1.0]})
        r = run_dea(df, metrics=LINE_METRICS)
        assert r["DEA_Status"].item() == "efficient"

    def test_default_metrics_used_when_none_given(self):
        r = run_dea(make_mock_metrics(n=10))
        assert {f"Slack_{m}" for m in DEFAULT_METRICS} <= set(r.columns)


# --------------------------------------------------------------------------
# plot_pcp
# --------------------------------------------------------------------------
class TestPlot:
    def test_returns_parcoords_and_table(self, mock_results):
        fig = plot_pcp(mock_results)
        assert isinstance(fig, go.Figure)
        kinds = [type(t).__name__ for t in fig.data]
        assert kinds == ["Parcoords", "Table"]

    def test_axes(self, mock_results):
        dims = plot_pcp(mock_results).data[0].dimensions
        # Plan axis first, one axis per metric, DEA score last
        assert len(dims) == 1 + len(DEFAULT_METRICS) + 1
        assert dims[0].label == "Plan"
        assert dims[-1].label == "DEA Score"

    def test_lower_is_better_axes_are_flipped(self, mock_results):
        dims = plot_pcp(mock_results).data[0].dimensions
        for dim, (col, higher) in zip(dims[1:-1], DEFAULT_METRICS.items()):
            lo, hi = dim.range
            assert (lo < hi) == higher, col
            assert dim.label.endswith("↑" if higher else "↓")

    def test_table_lists_efficient_plans(self, mock_results):
        fig = plot_pcp(mock_results)
        table_ids = list(fig.data[1].cells.values[0])
        eff_ids = sorted(mock_results.loc[mock_results["DEA_Efficient"] == 1, "Plan_ID"])
        assert table_ids == eff_ids

    def test_plan_axis_labels(self, mock_results):
        plan_axis = plot_pcp(mock_results).data[0].dimensions[0]
        n_ineff = int((mock_results["DEA_Efficient"] == 0).sum())
        assert f"({n_ineff} others)" in plan_axis.ticktext

    def test_title_counts(self, mock_results):
        n_eff = int(mock_results["DEA_Efficient"].sum())
        title = plot_pcp(mock_results).layout.title.text
        assert f"{n_eff} of {len(mock_results)} plans" in title

    def test_display_names_override(self, mock_results):
        fig = plot_pcp(mock_results, display_names={"Compactness": "Roundness"})
        labels = [d.label for d in fig.data[0].dimensions]
        assert any(label.startswith("Roundness") for label in labels)

    def test_all_plans_efficient(self, line_plans):
        # no "(n others)" label when every plan is on the frontier
        r = run_dea(line_plans.iloc[:3], metrics=LINE_METRICS)
        fig = plot_pcp(r, metrics=LINE_METRICS)
        assert not any("others" in t for t in fig.data[0].dimensions[0].ticktext)

    def test_one_efficient_plan(self):
        df = pd.DataFrame({"Plan_ID": ["Best", "Worse"],
                           "Good": [2.0, 1.0], "Bad": [1.0, 2.0]})
        r = run_dea(df, metrics=LINE_METRICS)
        fig = plot_pcp(r, metrics=LINE_METRICS)
        assert list(fig.data[1].cells.values[0]) == ["Best"]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
class TestCommandLine:
    def run_main(self, monkeypatch, *args):
        monkeypatch.setattr(sys, "argv", ["dea_pcp.py", "--no-show", *args])
        dea_pcp.main()

    def test_mock_data_run(self, tmp_path, monkeypatch, capsys):
        out_csv, out_html = tmp_path / "r.csv", tmp_path / "p.html"
        self.run_main(monkeypatch, "--out-csv", str(out_csv),
                      "--out-html", str(out_html))
        assert out_csv.exists() and out_html.exists()
        assert len(pd.read_csv(out_csv)) == 60
        assert "DEA status counts" in capsys.readouterr().out

    def test_csv_input(self, tmp_path, monkeypatch):
        src = tmp_path / "metrics.csv"
        make_mock_metrics(n=15, seed=5).rename(
            columns={"Plan_ID": "Name"}).to_csv(src, index=False)
        out_csv = tmp_path / "r.csv"
        self.run_main(monkeypatch, "--csv", str(src), "--id-col", "Name",
                      "--out-csv", str(out_csv),
                      "--out-html", str(tmp_path / "p.html"))
        result = pd.read_csv(out_csv)
        assert len(result) == 15
        assert "DEA_Score" in result.columns

    def test_no_show_does_not_open_browser(self, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(dea_pcp.webbrowser, "open", opened.append)
        self.run_main(monkeypatch, "--out-csv", str(tmp_path / "r.csv"),
                      "--out-html", str(tmp_path / "p.html"))
        assert opened == []
