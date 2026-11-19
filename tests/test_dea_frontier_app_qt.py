"""
Tests for dea_frontier_app_qt.py.

Run from the top folder of the project (the one that holds both the
redistricting/ and tests/ folders):

    python -m pytest tests/test_dea_frontier_app_qt.py -v

The DEA calculation itself (run_dea) is the same code as in dea_pcp.py and
is tested in test_dea_pcp.py. These tests cover what is new in the Qt app:
the chart helpers and the window (controls, chart layers, side panel, and
clicking between plans).

The window tests need PyQt5 and pyqtgraph. They run without showing a
window (Qt's "offscreen" platform), so they also work over SSH or in CI.
If PyQt5 or pyqtgraph is missing, the whole file is skipped.
"""

import os

# Must be set before Qt starts: draw into memory instead of on screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from PyQt5 import sip
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton

from redistricting import dea_frontier_app_qt as app_module
from redistricting.dea_frontier_app_qt import (
    DEFAULT_METRICS,
    MOCK_COUNTY_METHODS,
    STATUS_LABELS,
    FrontierWindow,
    fmt,
    frontier_targets,
    make_mock_metrics,
    near_frontier,
    off_frontier,
    pareto_2d,
    run_dea,
    staircase,
)


# --------------------------------------------------------------------------
# Small hand-checkable data set
# --------------------------------------------------------------------------
# One "higher is better" metric (Good) and one "lower is better" (Bad).
LINE_METRICS = {"Good": True, "Bad": False}


@pytest.fixture
def line_plans():
    """P1, P2, P3 sit on a straight line, so all three are on the frontier.

    P4 and P5 are each beaten by a frontier plan:
      P4 (Good 1, Bad 3) is beaten by P1 (Good 1, Bad 1): score 1/3
      P5 (Good 2, Bad 3) is beaten by P2 (Good 2, Bad 2): score 2/3
    """
    return pd.DataFrame({
        "Plan_ID": ["P1", "P2", "P3", "P4", "P5"],
        "Good":    [1.0, 2.0, 3.0, 1.0, 2.0],
        "Bad":     [1.0, 2.0, 3.0, 3.0, 3.0],
    })


@pytest.fixture
def line_results(line_plans):
    return run_dea(line_plans, metrics=LINE_METRICS)


def row_of(results, plan_id):
    return results[results["Plan_ID"] == plan_id].iloc[0]


# --------------------------------------------------------------------------
# Chart helpers
# --------------------------------------------------------------------------
def test_line_plans_scores(line_results):
    """The helpers below rely on these scores, so check them first."""
    scores = dict(zip(line_results["Plan_ID"], line_results["DEA_Score"]))
    assert scores["P1"] == pytest.approx(1)
    assert scores["P2"] == pytest.approx(1)
    assert scores["P3"] == pytest.approx(1)
    assert scores["P4"] == pytest.approx(1 / 3)
    assert scores["P5"] == pytest.approx(2 / 3)


def test_near_frontier_picks_highest_scoring_inefficient_plans(line_results):
    assert near_frontier(line_results, 1)["Plan_ID"].tolist() == ["P5"]
    assert near_frontier(line_results, 2)["Plan_ID"].tolist() == ["P5", "P4"]


def test_near_frontier_never_includes_efficient_plans(line_results):
    # asking for more plans than exist returns only the inefficient ones
    assert set(near_frontier(line_results, 10)["Plan_ID"]) == {"P4", "P5"}


def test_near_frontier_zero(line_results):
    assert near_frontier(line_results, 0).empty


def test_pareto_2d_keeps_nondominated_plans_in_order(line_results):
    front = pareto_2d(line_results, "Good", "Bad", LINE_METRICS)
    # sorted from worst to best along the x axis (Good, higher is better)
    assert front["Plan_ID"].tolist() == ["P1", "P2", "P3"]


def test_pareto_2d_sorts_lower_is_better_x_axis(line_results):
    front = pareto_2d(line_results, "Bad", "Good", LINE_METRICS)
    # x is Bad (lower is better), so worst-to-best is high Bad to low Bad
    assert front["Plan_ID"].tolist() == ["P3", "P2", "P1"]


def test_staircase_adds_corners():
    xs, ys = staircase(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    # (1,3) -> (2,3) -> (2,2) -> (3,2) -> (3,1)
    assert xs.tolist() == [1, 2, 2, 3, 3]
    assert ys.tolist() == [3, 3, 2, 2, 1]


@pytest.mark.parametrize("n", [0, 1])
def test_staircase_short_input_unchanged(n):
    xs, ys = staircase(np.arange(n, dtype=float), np.arange(n, dtype=float))
    assert len(xs) == len(ys) == n


# run_dea rounds scores and slacks to 6 decimals, so targets built from them
# can be off by about 1e-6.
TARGET_TOL = 1e-4


@pytest.mark.parametrize("plan, target", [
    ("P4", {"Good": 1.0, "Bad": 1.0}),   # P4's target is P1
    ("P5", {"Good": 2.0, "Bad": 2.0}),   # P5's target is P2
    ("P3", {"Good": 3.0, "Bad": 3.0}),   # efficient: its own values
])
def test_frontier_targets(line_results, plan, target):
    got = frontier_targets(row_of(line_results, plan), LINE_METRICS, "input")
    assert got == pytest.approx(target, abs=TARGET_TOL)


def test_frontier_targets_output_orientation(line_plans):
    results = run_dea(line_plans, metrics=LINE_METRICS, orientation="output")
    # Output orientation grows Good while keeping Bad <= 3: P4 and P5 can
    # both reach P3 (Good 3, Bad 3).
    for plan in ("P4", "P5"):
        got = frontier_targets(row_of(results, plan), LINE_METRICS, "output")
        assert got == pytest.approx({"Good": 3.0, "Bad": 3.0}, abs=TARGET_TOL)


@pytest.mark.parametrize("value, text", [
    (12.0, "12"),
    (1234.0, "1,234"),
    (0.5, "0.5"),
    (0.004, "0.004"),
    (0.0, "0"),
    (6.333333, "6.333"),
])
def test_fmt(value, text):
    assert fmt(value) == text


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mock_results():
    return run_dea(make_mock_metrics())


@pytest.fixture
def window(qapp, mock_results):
    w = FrontierWindow(mock_results, n_default=10)
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


@pytest.fixture
def line_window(qapp, line_results):
    w = FrontierWindow(line_results, metrics=LINE_METRICS,
                       display_names={"Good": "Goodness", "Bad": "Badness"},
                       n_default=1, title="Test frontier")
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


def scatter_items(w):
    return [i for i in w.plot.items if isinstance(i, pg.ScatterPlotItem)]


def legend_names(w):
    return [label.text for _sample, label in w.plot.legend.items]


def panel_texts(w):
    return [label.text() for label in w.panelScroll.widget().findChildren(QLabel)]


def peer_buttons(w):
    return w.panelScroll.widget().findChildren(QPushButton)


def first_inefficient(results):
    return str(results.loc[off_frontier(results), "Plan_ID"].iloc[0])


def first_efficient(results):
    return str(results.loc[results["DEA_Efficient"] == 1, "Plan_ID"].iloc[0])


# ---- setup ---------------------------------------------------------------
def test_window_loads_ui_widgets(window):
    for name in ("titleLabel", "summaryLabel", "xCombo", "yCombo", "nSlider",
                 "nValue", "showOthers", "showLine", "plotWidget", "panelScroll"):
        assert hasattr(window, name), name


def test_title_and_summary(line_window):
    assert line_window.windowTitle() == "Test frontier"
    assert line_window.titleLabel.text() == "Test frontier"
    assert line_window.summaryLabel.text().startswith("3 of 5 plans")


def test_combos_use_display_names_and_store_columns(line_window):
    combo = line_window.xCombo
    assert [combo.itemText(i) for i in range(combo.count())] == ["Goodness", "Badness"]
    assert [combo.itemData(i) for i in range(combo.count())] == ["Good", "Bad"]
    assert line_window.xCombo.currentData() == "Good"
    assert line_window.yCombo.currentData() == "Bad"


def test_slider_range_is_number_of_off_frontier_plans(window, mock_results):
    n_ineff = int((mock_results["DEA_Status"] == "inefficient").sum())
    assert window.nSlider.minimum() == 0
    assert window.nSlider.maximum() == n_ineff
    assert window.nSlider.value() == 10
    assert window.nValue.text() == "10"


def test_n_default_capped_at_inefficient_count(line_window):
    # only 2 inefficient plans
    line_window.nSlider.setValue(99)
    assert line_window.nSlider.value() == 2


# ---- chart ---------------------------------------------------------------
def test_chart_layers(window, mock_results):
    n_eff = int(mock_results["DEA_Efficient"].sum())
    n_weak = int((mock_results["DEA_Status"] == "weakly efficient").sum())
    n_others = len(mock_results) - n_eff - n_weak - 10
    assert legend_names(window) == [
        f"Other plans ({n_others})",
        "Frontier for these 2 metrics only",
        "10 closest to frontier",
        f"On frontier, weak ({n_weak})",
        f"DEA frontier ({n_eff})",
    ]


def test_chart_points_match_groups(line_window):
    # n = 1: P5 is near, P4 is "other", P1-P3 are the frontier
    others, near, eff = scatter_items(line_window)
    assert sorted(others.data["data"]) == ["P4"]
    assert sorted(near.data["data"]) == ["P5"]
    assert sorted(eff.data["data"]) == ["P1", "P2", "P3"]


def test_hide_others_and_line(window):
    window.showOthers.setChecked(False)
    window.showLine.setChecked(False)
    names = legend_names(window)
    assert not any(n.startswith("Other plans") for n in names)
    assert "Frontier for these 2 metrics only" not in names
    assert len(scatter_items(window)) == 3   # near, weak, frontier


def test_slider_changes_near_group(window):
    window.nSlider.setValue(3)
    assert window.nValue.text() == "3"
    assert "3 closest to frontier" in legend_names(window)
    window.nSlider.setValue(0)
    assert not any("closest to frontier" in n for n in legend_names(window))


def test_axes_flip_for_lower_is_better(line_window):
    vb = line_window.plot.getViewBox()
    # x = Good (higher better), y = Bad (lower better)
    assert not vb.state["xInverted"]
    assert vb.state["yInverted"]
    line_window.xCombo.setCurrentIndex(1)   # Bad
    line_window.yCombo.setCurrentIndex(0)   # Good
    assert vb.state["xInverted"]
    assert not vb.state["yInverted"]


def test_axis_labels(line_window):
    assert "Goodness" in line_window.plot.getAxis("bottom").labelText
    assert "Badness" in line_window.plot.getAxis("left").labelText


def test_hover_text(line_window):
    tip = line_window._hover_text("Good", "Bad")
    text = tip(2.0, 3.0, "P5")
    assert text.splitlines() == ["P5", "Goodness: 2", "Badness: 3",
                                 "DEA score: 0.667"]


def test_point_click_selects_plan(line_window):
    class FakePoint:
        def data(self):
            return "P4"
    line_window._point_clicked(None, [FakePoint()])
    assert line_window.selected == "P4"


def test_point_click_with_no_points_does_nothing(line_window):
    line_window._point_clicked(None, [])
    assert line_window.selected is None


def test_selected_plan_is_ringed_and_labelled(line_window):
    line_window.select("P4")
    labels = [i for i in line_window.plot.items if isinstance(i, pg.TextItem)]
    assert [label.toPlainText() for label in labels] == ["P4"]
    ring = scatter_items(line_window)[-1]
    assert ring.data["x"].tolist() == [1.0]
    assert ring.data["y"].tolist() == [3.0]


# ---- side panel ----------------------------------------------------------
def test_panel_starts_with_instructions(window):
    assert "Click a plan" in panel_texts(window)
    assert peer_buttons(window) == []


def test_panel_for_inefficient_plan(line_window):
    line_window.select("P4")
    texts = panel_texts(line_window)
    assert "P4" in texts
    assert "off frontier" in texts
    assert "DEA score 0.333" in texts
    # P4 needs Badness 1 to reach the frontier; Goodness is already there
    assert "frontier target 1" in texts
    assert sum(t.startswith("frontier target") for t in texts) == 1
    assert [b.text() for b in peer_buttons(line_window)] == ["P1"]


def test_panel_percentiles(line_window):
    line_window.select("P4")
    texts = panel_texts(line_window)
    # Goodness 1: 2 of 5 plans are <= 1. Badness 3: 3 of 5 plans are >= 3.
    assert "As good as or better than 40% of plans" in texts
    assert "As good as or better than 60% of plans" in texts


def test_panel_for_efficient_plan(window, mock_results):
    plan = first_efficient(mock_results)
    window.select(plan)
    texts = panel_texts(window)
    assert "on frontier" in texts
    assert any("on the DEA frontier" in t for t in texts)
    assert not any(t.startswith("frontier target") for t in texts)
    assert peer_buttons(window) == []


def test_panel_unknown_plan(window):
    window.select("No_Such_Plan")
    assert "No_Such_Plan not found." in panel_texts(window)


def test_peer_buttons_match_dea_peers(window, mock_results):
    plan = first_inefficient(mock_results)
    window.select(plan)
    expected = row_of(mock_results, plan)["DEA_Peers"].split(", ")
    assert [b.text() for b in peer_buttons(window)] == expected


def test_clicking_peer_selects_it(qapp, line_window):
    line_window.select("P5")
    (button,) = peer_buttons(line_window)
    peer = button.text()
    button.click()
    qapp.processEvents()
    assert line_window.selected == peer
    assert peer in panel_texts(line_window)


def test_clicking_peers_repeatedly_does_not_crash(qapp, window, mock_results):
    """Each click rebuilds the panel that holds the clicked button."""
    inefficient = mock_results.loc[off_frontier(mock_results), "Plan_ID"]
    for plan in inefficient.astype(str).head(15):
        window.select(plan)
        old_panel = window.panelScroll.widget()
        peer_buttons(window)[0].click()
        # The old panel is still alive right after the click (deleteLater).
        assert not sip.isdeleted(old_panel)
        # In the app, Qt's event loop deletes it once the click has finished.
        # The tests have no running event loop, so flush deletions by hand.
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(old_panel)
        assert window.selected != plan


def test_deselect(line_window):
    line_window.select("P4")
    line_window.select(None)
    assert "Click a plan" in panel_texts(line_window)
    assert not [i for i in line_window.plot.items if isinstance(i, pg.TextItem)]


# ---- plan status wording --------------------------------------------------
def first_weak(results):
    return str(results.loc[results["DEA_Status"] == "weakly efficient", "Plan_ID"].iloc[0])


def test_status_labels_avoid_efficient():
    assert set(STATUS_LABELS) == {"efficient", "weakly efficient", "inefficient"}
    assert not any("efficien" in label for label in STATUS_LABELS.values())


def test_mock_data_has_weak_frontier_plans(mock_results):
    # radial DEA on the default mock data gives 2 weak frontier plans
    assert (mock_results["DEA_Status"] == "weakly efficient").sum() == 2


def test_weak_plans_are_not_counted_as_closest(mock_results):
    near = near_frontier(mock_results, len(mock_results))
    assert set(near["DEA_Status"]) == {"inefficient"}


def test_weak_plans_drawn_as_own_group(window, mock_results):
    weak = sorted(mock_results.loc[mock_results["DEA_Status"] == "weakly efficient",
                                   "Plan_ID"])
    named = [i for i in scatter_items(window) if i.name()]
    (item,) = [i for i in named if i.name().startswith("On frontier, weak")]
    assert sorted(item.data["data"]) == weak


def test_panel_for_weak_plan(window, mock_results):
    window.select(first_weak(mock_results))
    assert "on frontier (weak)" in panel_texts(window)


# ---- zoom ----------------------------------------------------------------
def view_shares(w):
    """The visible part of each axis, as shares of its full view."""
    (x0, x1), (y0, y1) = w.plot.getViewBox().viewRange()
    x, y = w.xCombo.currentData(), w.yCombo.currentData()
    return w._to_shares(x, x0, x1), w._to_shares(y, y0, y1)


def zoom_to(w, x_shares, y_shares):
    """Zoom as a user would, by moving the view box."""
    x, y = w.xCombo.currentData(), w.yCombo.currentData()
    w.plot.getViewBox().setRange(xRange=w._from_shares(x, x_shares),
                                 yRange=w._from_shares(y, y_shares), padding=0)


def assert_shares(got, expected):
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-6)


def test_starts_at_full_view(window):
    assert_shares(view_shares(window), [(0, 1), (0, 1)])


def test_full_view_pads_each_side(line_window):
    # Good runs 1-3, so the full view adds 10% of 2 on each side
    assert line_window._full_view("Good") == pytest.approx((0.8, 3.2))


def test_shares_measure_from_worst_end(line_window):
    # Bad is lower-is-better: its worst end is the high value
    assert line_window._to_shares("Bad", 2.6, 3.2) == pytest.approx((0, 0.25))
    assert line_window._from_shares("Bad", (0, 0.25)) == pytest.approx((2.6, 3.2))


@pytest.mark.parametrize("axis, index", [("y", 2), ("y", 3), ("x", 1), ("x", 2)])
def test_zoom_kept_when_metric_changes(window, axis, index):
    zoom_to(window, (0.2, 0.6), (0.5, 0.9))
    combo = window.yCombo if axis == "y" else window.xCombo
    combo.setCurrentIndex(index)
    assert_shares(view_shares(window), [(0.2, 0.6), (0.5, 0.9)])


def test_zoom_kept_through_several_changes(window):
    zoom_to(window, (0.1, 0.4), (0.3, 0.7))
    for x, y in [(1, 2), (3, 0), (2, 1), (0, 3)]:
        window.xCombo.setCurrentIndex(x)
        window.yCombo.setCurrentIndex(y)
        assert_shares(view_shares(window), [(0.1, 0.4), (0.3, 0.7)])


def test_zoom_kept_when_controls_change(window):
    zoom_to(window, (0.2, 0.6), (0.5, 0.9))
    window.nSlider.setValue(3)
    window.showOthers.setChecked(False)
    window.select(first_inefficient(window.results))
    assert_shares(view_shares(window), [(0.2, 0.6), (0.5, 0.9)])


# ---- county-split measure -------------------------------------------------
@pytest.fixture
def mock_metrics():
    return make_mock_metrics()


def mock_rescore(df):
    def rescore(metrics):
        return run_dea(df, metrics=metrics)
    return rescore


@pytest.fixture
def county_window(qapp, mock_metrics):
    w = FrontierWindow(run_dea(mock_metrics), county_methods=MOCK_COUNTY_METHODS,
                       rescore=mock_rescore(mock_metrics))
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


def choose_county(w, col):
    w.countyCombo.setCurrentIndex(w.countyCombo.findData(col))


def test_mock_data_has_all_county_measures(mock_metrics):
    for col in MOCK_COUNTY_METHODS:
        assert col in mock_metrics.columns
        assert (mock_metrics[col] >= 0).all()
    # every split county is in at least 2 pieces
    assert (mock_metrics["County_Split_Pieces"] >= 2 * mock_metrics["County_Splits"]).all()
    assert mock_metrics["County_HHI"].between(0, 1).all()


def test_county_combo_hidden_without_choices(window):
    assert not window.countyCombo.isVisibleTo(window)
    assert not window.countyLabel.isVisibleTo(window)


def test_county_combo_lists_measures(county_window):
    combo = county_window.countyCombo
    assert combo.isVisibleTo(county_window)
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Split Counties", "County Splits", "Spill Ratio", "Herfindahl Index"]
    assert combo.currentData() == "County_Splits"


@pytest.mark.parametrize("col", ["County_Split_Pieces", "Spill_Ratio", "County_HHI"])
def test_choosing_county_measure_rescores(county_window, mock_metrics, col):
    choose_county(county_window, col)
    expected_metrics = {(col if m == "County_Splits" else m):
                        (MOCK_COUNTY_METHODS[col] if m == "County_Splits" else hb)
                        for m, hb in DEFAULT_METRICS.items()}
    assert county_window.metrics == expected_metrics
    expected = run_dea(mock_metrics, metrics=expected_metrics)
    assert county_window.results["DEA_Score"].tolist() == expected["DEA_Score"].tolist()
    n_eff = int(expected["DEA_Efficient"].sum())
    assert county_window.summaryLabel.text().startswith(f"{n_eff} of 60 plans")


def test_county_swap_replaces_axis(county_window):
    # y starts on County_Splits, the second metric
    assert county_window.yCombo.currentData() == "County_Splits"
    choose_county(county_window, "County_HHI")
    assert county_window.xCombo.currentData() == "Compactness"
    assert county_window.yCombo.currentData() == "County_HHI"
    names = [county_window.yCombo.itemText(i) for i in range(county_window.yCombo.count())]
    assert "Herfindahl Index" in names and "Split Counties" not in names


def test_county_swap_flips_axis_for_hhi(county_window):
    vb = county_window.plot.getViewBox()
    assert vb.state["yInverted"]            # split counties: lower is better
    choose_county(county_window, "County_HHI")
    assert not vb.state["yInverted"]        # Herfindahl index: higher is better


def test_county_swap_keeps_selection_and_zoom(county_window):
    zoom_to(county_window, (0.2, 0.6), (0.5, 0.9))
    county_window.select("Plan_005")
    choose_county(county_window, "Spill_Ratio")
    assert county_window.selected == "Plan_005"
    assert "Spill Ratio ↓" in panel_texts(county_window)
    assert_shares(view_shares(county_window), [(0.2, 0.6), (0.5, 0.9)])


def test_county_swap_updates_slider_range(county_window):
    choose_county(county_window, "Spill_Ratio")
    n_off = int(off_frontier(county_window.results).sum())
    assert county_window.nSlider.maximum() == n_off
    assert county_window.nSlider.value() <= n_off


def test_failed_rescore_keeps_old_measure(qapp, mock_metrics, monkeypatch):
    warnings = []
    monkeypatch.setattr(app_module.QMessageBox, "warning",
                        lambda *args: warnings.append(args[2]))

    def broken(metrics):
        raise RuntimeError("solver missing")

    w = FrontierWindow(run_dea(mock_metrics), county_methods=MOCK_COUNTY_METHODS,
                       rescore=broken)
    before = w.results.copy()
    choose_county(w, "Spill_Ratio")
    assert warnings == ["solver missing"]
    assert w.countyCombo.currentData() == "County_Splits"
    assert "County_Splits" in w.metrics
    assert w.results.equals(before)
    w.close()
