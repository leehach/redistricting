"""
Interactive Pareto-frontier explorer for DEA results of redistricting plans.

A standalone PyQt5 desktop app: it runs the DEA itself (the same calculation
as the parallel-coordinates script, dea_pcp.py). Its window layout is in
dea_frontier_app_qt.ui, which must sit in the same folder as this file.

  * a scatter plot of any two metrics, with every axis oriented so that
    up and right are better
  * the DEA-frontier plans in crimson
  * the n plans closest to the frontier (highest DEA scores among the
    off-frontier plans) in blue, with a slider for n
  * optionally, all other plans in light gray, and the 2-metric Pareto
    frontier as a dashed staircase line
  * hover over a plan to see its values; click it to see its metrics, its
    percentile on each one, the values it would need to reach the frontier,
    and its frontier peers (click a peer to jump to it)
  * a choice of county-split measure (split counties, county splits, spill
    ratio, or Herfindahl index); changing it rescores every plan
  * zooming keeps the same zoom when you switch metrics: each axis shows the
    same share of its metric's range as before

Note: the DEA frontier uses ALL metrics at once, so a crimson plan can sit
inside the dashed 2-metric line. That's expected: it earns its place on the
frontier with a metric that isn't on the current axes.

Plan status
-----------
The side panel labels each plan with one of the names in STATUS_LABELS.
These avoid the economics term "efficient":

  on frontier          no blend of other plans beats it
  on frontier (weak)   scores 1, but could still improve one metric without
                       losing on any other. Drawn as amber diamonds. Only
                       radial scoring (run_dea, used for mock data and
                       --csv) produces this; the default mock data has 2.
  off frontier         a blend of frontier plans beats it

Real plans loaded from files are scored with SBM (slacks-based measure),
which counts that leftover room in the score itself. Under SBM a "weak"
plan scores below 1, so real plans are only ever on or off the frontier.

Usage
-----
    pip install PyQt5 pyqtgraph scipy pandas numpy
    python dea_frontier_app_qt.py                     # mock data
    python dea_frontier_app_qt.py --csv metrics.csv   # real metrics table
    python dea_frontier_app_qt.py --n 15

Close the window to stop the app.

In code (functions take DataFrames, not file paths):

    results = run_dea(metrics_df)
    show_app(results)

Layout and code
---------------
The .ui file holds the layout: title, controls, chart, and side panel. Edit
it in Qt Designer. The chart is a pyqtgraph PlotWidget, placed in Designer as
a QGraphicsView "promoted" to PlotWidget with header file ``pyqtgraph``.
This file finds the widgets by the names they have in the .ui file
(``xCombo``, ``nSlider``, ``plotWidget``, ...), so keep those names when
editing the layout. The side panel's contents change with each plan, so they
are built here in code.

The data columns keep the standard DEA names (DEA_Efficient, DEA_Status with
values "efficient", "weakly efficient", "inefficient"), as in dea_pcp.py, so
saved results stay compatible. Only the text shown in the window changes.

Requirements: PyQt5, pyqtgraph, numpy, pandas, scipy
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# PyQt5 must be imported before pyqtgraph, so pyqtgraph uses it rather than
# another Qt binding (PyQt6, PySide6) that may also be installed.
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QProgressDialog, QPushButton,
                             QVBoxLayout, QWidget)
import pyqtgraph as pg

from redistricting import plan_metrics
from redistricting.DEA import dea

# Window layout, made in Qt Designer.
UI_FILE = Path(__file__).with_suffix(".ui")

# Column name -> True if higher values are better, False if lower is better.
# Edit this (or pass your own dict) to match your collaborators' output table.
DEFAULT_METRICS: dict[str, bool] = {
    "Compactness": True,      # e.g. mean Polsby-Popper, 0-1
    "County_Splits": False,   # count of split counties
    "Pop_Dev_Pct": False,     # max population deviation, percent
    "Partisan_Gap": False,    # e.g. |efficiency gap| or partisan bias
}

# County-split measures in the mock data (and a --csv table that has these
# columns). Any of them can replace County_Splits in DEFAULT_METRICS.
MOCK_COUNTY_METHODS: dict[str, bool] = {
    "County_Splits": False,        # counties split between 2+ districts
    "County_Split_Pieces": False,  # district pieces inside split counties
    "Spill_Ratio": False,          # share of people outside their county's main district
    "County_HHI": True,            # Herfindahl index; 1 = no county split
}

DISPLAY_NAMES: dict[str, str] = {
    "Compactness": "Compactness (PP)",
    "County_Splits": "Split Counties",
    "County_Split_Pieces": "County Splits",
    "Spill_Ratio": "Spill Ratio",
    "County_HHI": "Herfindahl Index",
    "Pop_Dev_Pct": "Pop. Deviation (%)",
    "Partisan_Gap": "Partisan Gap",
}

TOL = 1e-6

# Display names for the metrics in plan_metrics.DIRECTION and
# plan_metrics.COUNTY_SPLIT_DIRECTION.
PLAN_METRIC_NAMES: dict[str, str] = {
    "polsby_popper": "Polsby-Popper (mean)",
    "max_pop_deviation": "Max Pop. Deviation (%)",
    "split_counties": "Split Counties",
    "county_split_intersections": "County Splits",
    "spill_ratio": "Spill Ratio",
    "county_hhi": "Herfindahl Index",
    "efficiency_gap": "Efficiency Gap (abs)",
}

# DEA_Status value -> what the window calls it. Change the wording here.
STATUS_LABELS: dict[str, str] = {
    "efficient": "on frontier",
    "weakly efficient": "on frontier (weak)",
    "inefficient": "off frontier",
}

# Share of a metric's range added on each side of the full view.
VIEW_PADDING = 0.1


# --------------------------------------------------------------------------
# Mock data
# --------------------------------------------------------------------------
def make_mock_metrics(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic plan metrics with realistic trade-offs.

    Unlike independent random columns, these build in tension between goals
    (more compact plans tend to split more counties, tighter population
    balance costs a little compactness), so the DEA frontier is meaningful.
    """
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0, 1, n)        # overall plan quality
    tradeoff = rng.uniform(0, 1, n)       # compactness-vs-county-splits lean

    compactness = 0.20 + 0.30 * quality * (0.5 + 0.5 * tradeoff) + rng.normal(0, 0.02, n)
    county_splits = np.round(4 + 14 * (1 - quality) * (0.4 + 0.6 * tradeoff)
                             + rng.normal(0, 1.5, n))
    pop_dev = 0.05 * (1 - quality) ** 1.5 + 0.004 * tradeoff + np.abs(rng.normal(0, 0.003, n))
    partisan_gap = np.abs(0.08 * (1 - quality) * rng.uniform(0.3, 1.0, n) + rng.normal(0, 0.01, n))
    splits = np.clip(county_splits, 0, None).astype(int)

    # The other county-split measures, drawn after everything above so the
    # columns above stay the same for a given seed. They follow the split
    # count but not exactly, so each one gives a somewhat different frontier.
    pieces = 2 * splits + rng.poisson(0.35 * splits)     # >= 2 per split county
    spill = 0.0045 * splits * rng.uniform(0.6, 1.4, n)   # 0 when nothing is split
    hhi = 1 - spill * rng.uniform(1.3, 1.7, n)           # 1 when nothing is split

    return pd.DataFrame({
        "Plan_ID": [f"Plan_{i:03d}" for i in range(1, n + 1)],
        "Compactness": np.clip(compactness, 0.05, 0.95).round(4),
        "County_Splits": splits,
        "Pop_Dev_Pct": np.clip(pop_dev, 0.0, None).round(4),
        "Partisan_Gap": partisan_gap.round(4),
        "County_Split_Pieces": pieces,
        "Spill_Ratio": spill.round(4),
        "County_HHI": hhi.round(4),
    })


# --------------------------------------------------------------------------
# DEA
# --------------------------------------------------------------------------
def pareto_nondominated(values: np.ndarray, higher_better: np.ndarray) -> np.ndarray:
    """Boolean mask of plans not dominated by any other plan.

    values: (n, m) array; higher_better: (m,) bool array.
    """
    v = np.where(higher_better, values, -values)  # flip so higher is always better
    n = v.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        ge = np.all(v >= v[i], axis=1)
        gt = np.any(v > v[i], axis=1)
        if np.any(ge & gt):
            mask[i] = False
    return mask


def _solve_bcc(X, Y, x_o, y_o, orientation, x_scale, y_scale):
    """Two-phase BCC model for a single plan against reference set (X, Y).

    X: (k, m_in) inputs of reference plans; Y: (k, m_out) outputs.
    Returns (score, input_slacks, output_slacks, lambdas).
    """
    k, m_in = X.shape
    m_out = Y.shape[1]
    bounds_l = [(0, None)] * k

    # ---- Phase 1: radial score.  Variables: [eff, lambda_1..k]
    c = np.zeros(1 + k)
    A_ub, b_ub = [], []
    if orientation == "input":
        c[0] = 1.0                                      # min theta
        for i in range(m_in):                           # sum l x_i - theta x_io <= 0
            A_ub.append(np.r_[-x_o[i], X[:, i]]); b_ub.append(0.0)
        for r in range(m_out):                          # -sum l y_r <= -y_ro
            A_ub.append(np.r_[0.0, -Y[:, r]]); b_ub.append(-y_o[r])
    else:
        c[0] = -1.0                                     # max phi
        for i in range(m_in):                           # sum l x_i <= x_io
            A_ub.append(np.r_[0.0, X[:, i]]); b_ub.append(x_o[i])
        for r in range(m_out):                          # phi y_ro - sum l y_r <= 0
            A_ub.append(np.r_[y_o[r], -Y[:, r]]); b_ub.append(0.0)
    A_eq = [np.r_[0.0, np.ones(k)]]                     # VRS convexity
    res1 = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=[1.0],
                   bounds=[(None, None)] + bounds_l, method="highs")
    if not res1.success:
        raise RuntimeError(f"Phase 1 LP failed: {res1.message}")
    eff = res1.x[0]

    # ---- Phase 2: max normalized slacks.  Variables: [lambda, s_in, s_out]
    n_var = k + m_in + m_out
    c2 = np.zeros(n_var)
    c2[k:k + m_in] = -1.0 / x_scale
    c2[k + m_in:] = -1.0 / y_scale
    A_eq2, b_eq2 = [], []
    for i in range(m_in):
        row = np.zeros(n_var); row[:k] = X[:, i]; row[k + i] = 1.0
        A_eq2.append(row)
        b_eq2.append(eff * x_o[i] if orientation == "input" else x_o[i])
    for r in range(m_out):
        row = np.zeros(n_var); row[:k] = Y[:, r]; row[k + m_in + r] = -1.0
        A_eq2.append(row)
        b_eq2.append(y_o[r] if orientation == "input" else eff * y_o[r])
    row = np.zeros(n_var); row[:k] = 1.0
    A_eq2.append(row); b_eq2.append(1.0)
    res2 = linprog(c2, A_eq=np.array(A_eq2), b_eq=b_eq2,
                   bounds=[(0, None)] * n_var, method="highs")
    if not res2.success:
        raise RuntimeError(f"Phase 2 LP failed: {res2.message}")

    lam = res2.x[:k]
    s_in = res2.x[k:k + m_in]
    s_out = res2.x[k + m_in:]
    score = eff if orientation == "input" else 1.0 / eff
    return score, s_in, s_out, lam


def run_dea(
    df: pd.DataFrame,
    id_col: str = "Plan_ID",
    metrics: dict[str, bool] | None = None,
    orientation: str = "input",
) -> pd.DataFrame:
    """Run BCC DEA on a table of plan metrics.

    Parameters
    ----------
    df : one row per plan, with an ID column and the metric columns.
    id_col : name of the plan ID column.
    metrics : {column: higher_is_better}. Defaults to DEFAULT_METRICS.
    orientation : "input"  - how far could the lower-is-better metrics shrink?
                  "output" - how far could the higher-is-better metrics grow?

    Returns a copy of df with added columns:
      DEA_Score       radial score, 0-1 (1 = on the frontier)
      DEA_Efficient   1 if score == 1 and all slacks are 0, else 0
      DEA_Status      "efficient", "weakly efficient", or "inefficient";
                      the window shows these as STATUS_LABELS
      Pareto_Optimal  1 if no other single plan beats it on every metric
      DEA_Peers       IDs of the efficient plans that benchmark this plan
      Slack_<metric>  remaining improvement possible on each metric after
                      the radial adjustment
    """
    metrics = metrics or DEFAULT_METRICS
    if orientation not in ("input", "output"):
        raise ValueError("orientation must be 'input' or 'output'")
    missing = [c for c in [id_col, *metrics] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    data = df.dropna(subset=list(metrics)).reset_index(drop=True)
    if len(data) < len(df):
        print(f"Note: dropped {len(df) - len(data)} plan(s) with missing metric values.")

    in_cols = [m for m, hb in metrics.items() if not hb]
    out_cols = [m for m, hb in metrics.items() if hb]
    values = data[list(metrics)].to_numpy(dtype=float)
    if np.any(values < 0):
        raise ValueError("DEA requires non-negative metrics. Use absolute values "
                         "for signed measures like efficiency gap or mean-median.")

    # DEA needs at least one input and one output; pad with a constant if not.
    X_all = data[in_cols].to_numpy(dtype=float) if in_cols else np.ones((len(data), 1))
    Y_all = data[out_cols].to_numpy(dtype=float) if out_cols else np.ones((len(data), 1))
    if not in_cols and orientation == "input":
        print("Note: no lower-is-better metrics; switching to output orientation.")
        orientation = "output"
    if not out_cols and orientation == "output":
        print("Note: no higher-is-better metrics; switching to input orientation.")
        orientation = "input"

    # Reference set: only nondominated plans can span the frontier.
    nd = pareto_nondominated(values, np.array(list(metrics.values())))
    ref_idx = np.flatnonzero(nd)
    X_ref, Y_ref = X_all[ref_idx], Y_all[ref_idx]

    def _scale(a):
        s = a.max(axis=0) - a.min(axis=0)
        return np.where(s > 0, s, 1.0)
    x_scale, y_scale = _scale(X_all), _scale(Y_all)

    ids = data[id_col].astype(str).to_numpy()
    scores, s_ins, s_outs, peers = [], [], [], []
    for o in range(len(data)):
        score, s_in, s_out, lam = _solve_bcc(X_ref, Y_ref, X_all[o], Y_all[o],
                                             orientation, x_scale, y_scale)
        scores.append(score)
        s_ins.append(s_in)
        s_outs.append(s_out)
        peer_ids = [ids[ref_idx[j]] for j in np.flatnonzero(lam > TOL) if ref_idx[j] != o]
        peers.append(", ".join(peer_ids))

    out = data.copy()
    out["DEA_Score"] = np.clip(np.round(scores, 6), 0, 1)
    s_ins, s_outs = np.array(s_ins), np.array(s_outs)
    norm_slack = np.c_[s_ins / x_scale, s_outs / y_scale]
    has_slack = np.any(norm_slack > 1e-5, axis=1)
    on_frontier = out["DEA_Score"].to_numpy() >= 1 - 1e-6

    out["DEA_Efficient"] = (on_frontier & ~has_slack).astype(int)
    out["DEA_Status"] = np.select(
        [on_frontier & ~has_slack, on_frontier & has_slack],
        ["efficient", "weakly efficient"], default="inefficient")
    out["Pareto_Optimal"] = nd.astype(int)
    out["DEA_Peers"] = np.where(out["DEA_Efficient"] == 1, "", peers)
    for i, col in enumerate(in_cols):
        out[f"Slack_{col}"] = np.round(s_ins[:, i], 6)
    for r, col in enumerate(out_cols):
        out[f"Slack_{col}"] = np.round(s_outs[:, r], 6)
    return out


# Same colors as the parallel coordinates plot.
FRONTIER = "#dc143c"   # crimson
NEAR = "#4a78a8"       # steel blue
OTHER = "#c4c8ce"      # light gray
INK = "#1f2933"
MUTED = "#6b7280"
RULE = "#e5e7eb"

STATUS_COLORS = {"efficient": FRONTIER, "weakly efficient": "#b45309",
                 "inefficient": NEAR}

# White chart background. Must be set before any PlotWidget is created.
pg.setConfigOptions(antialias=True, background="w", foreground=INK)


# --------------------------------------------------------------------------
# Real plans
# --------------------------------------------------------------------------
def score_sbm(metrics: pd.DataFrame, direction: dict[str, str],
              id_col: str = "Plan_ID") -> pd.DataFrame:
    """Score plans with SBM DEA, in the columns FrontierWindow reads.

    ``metrics`` has one row per plan, indexed by plan name. A plan is on
    the frontier when its SBM score is 1. SBM gives no peers or slacks, so
    the side panel leaves those out.
    """
    metrics = metrics[list(direction)]
    scores = dea.score(metrics, direction, "sbm")
    efficient = (scores >= 1 - TOL).astype(int)

    results = metrics.rename_axis(id_col).reset_index()
    results["DEA_Score"] = scores.to_numpy()
    results["DEA_Efficient"] = efficient.to_numpy()
    results["DEA_Status"] = np.where(efficient, "efficient", "inefficient")
    return results


def read_plan_files(paths: list[str]) -> dict[str, pd.DataFrame]:
    """Read plan CSVs, each named by its file name without ".csv"."""
    # Geoids are text: a number type would drop leading zeros.
    return {Path(p).stem: pd.read_csv(p, dtype=str) for p in sorted(paths)}


def sbm_rescorer(table: pd.DataFrame, id_col: str = "Plan_ID"
                 ) -> Callable[[dict[str, bool]], pd.DataFrame]:
    """A function that scores ``table`` with SBM for a given set of metrics.

    FrontierWindow calls it when the county-split measure changes.
    """
    def rescore(metrics: dict[str, bool]) -> pd.DataFrame:
        direction = {m: "higher" if hb else "lower" for m, hb in metrics.items()}
        return score_sbm(table, direction, id_col)
    return rescore


# plan_metrics.DIRECTION and COUNTY_SPLIT_DIRECTION as {metric: higher_is_better}
def _higher_is_better(direction: dict[str, str]) -> dict[str, bool]:
    return {m: d == "higher" for m, d in direction.items()}


class PlanLoader(QThread):
    """Reads plan files, measures them and scores them, off the UI thread.

    Measures all four county-split metrics, so the window can switch between
    them without measuring again. Scores with split_counties to start.
    """

    done = pyqtSignal(object)    # (metric table, score_sbm results)
    failed = pyqtSignal(str)     # an error message for the user

    # The geounits download once per app run, then stay in memory.
    _geounits = None

    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths

    def run(self):
        try:
            if PlanLoader._geounits is None:
                PlanLoader._geounits = plan_metrics.load_pa_geounits()
            plans = plan_metrics.combine_plans(read_plan_files(self.paths))
            table = plan_metrics.measure(
                plans, PlanLoader._geounits,
                county_metrics=list(plan_metrics.COUNTY_SPLIT_DIRECTION))
            results = score_sbm(table, plan_metrics.DIRECTION)
            self.done.emit((table, results))
        except Exception as error:  # show any failure to the user
            self.failed.emit(str(error))


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def off_frontier(results: pd.DataFrame) -> pd.Series:
    """True for plans off the frontier (not on it, not even weakly)."""
    return results["DEA_Status"] == "inefficient"


def near_frontier(results: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n off-frontier plans with the highest DEA scores."""
    return results[off_frontier(results)].nlargest(n, "DEA_Score")


def pareto_2d(results: pd.DataFrame, x: str, y: str, metrics: dict) -> pd.DataFrame:
    """Plans on the Pareto frontier for just two metrics, in drawing order."""
    vals = results[[x, y]].to_numpy(dtype=float)
    hb = np.array([metrics[x], metrics[y]])
    front = results[pareto_nondominated(vals, hb)].copy()
    # sort along the x axis from worst to best, so the staircase draws cleanly
    front["_u"] = front[x] if metrics[x] else -front[x]
    front["_v"] = front[y] if metrics[y] else -front[y]
    return front.sort_values(["_u", "_v"], ascending=[True, False])


def staircase(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner points of a step line that goes across, then up or down.

    pyqtgraph draws straight lines between points, so the corners are added
    here: (x0, y0) -> (x1, y0) -> (x1, y1) -> ...
    """
    if len(xs) < 2:
        return xs, ys
    return np.repeat(xs, 2)[1:], np.repeat(ys, 2)[:-1]


def frontier_targets(row: pd.Series, metrics: dict, orientation: str) -> dict:
    """Values this plan would need to reach the DEA frontier.

    These are the metrics of the blend of peer plans that beats it: the radial
    adjustment (DEA score) plus any leftover slack.
    """
    score = float(row["DEA_Score"])
    targets = {}
    for col, higher_better in metrics.items():
        v = float(row[col])
        slack = float(row.get(f"Slack_{col}", 0.0))
        if higher_better:
            base = v / score if (orientation == "output" and score > 0) else v
            targets[col] = base + slack
        else:
            base = v * score if orientation == "input" else v
            targets[col] = base - slack
    return targets


def fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.4g}"


def _label(text: str, style: str = "", wrap: bool = False) -> QLabel:
    """A QLabel with an optional style sheet."""
    label = QLabel(text)
    if style:
        label.setStyleSheet(style)
    label.setWordWrap(wrap)
    return label


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
class FrontierWindow(QMainWindow):
    """The app window: layout from dea_frontier_app_qt.ui, behavior from here."""

    def __init__(
        self,
        results: pd.DataFrame,
        id_col: str = "Plan_ID",
        metrics: dict[str, bool] | None = None,
        display_names: dict[str, str] | None = None,
        orientation: str = "input",
        n_default: int = 10,
        title: str = "DEA Frontier of Prospective Redistricting Plans",
        county_methods: dict[str, bool] | None = None,
        rescore: Callable[[dict[str, bool]], pd.DataFrame] | None = None,
    ):
        """
        county_methods : {column: higher_is_better} of county-split measures
            the user can choose between. One of them must be in ``metrics``.
        rescore : function that takes a metrics dict like ``metrics`` and
            returns new results for it. Needed to switch county measures.
        The county-split dropdown is hidden unless both are given.
        """
        super().__init__()
        # Creates every widget in the .ui file as an attribute of this
        # window, named as in Qt Designer: self.xCombo, self.nSlider, ...
        uic.loadUi(str(UI_FILE), self)

        self.id_col = id_col
        self.names = {**DISPLAY_NAMES, **(display_names or {})}
        self.orientation = orientation
        self.county_methods = county_methods or {}
        self.rescore = rescore
        self.selected: str | None = None   # plan shown in the side panel
        self._axes = None                  # (x, y) last drawn
        # Visible part of each axis, as shares of its metric's full view
        # (0 = worst end, 1 = best end). Kept when the metrics change, so
        # the zoom stays the same. Starts as the full view.
        self._view = ((0.0, 1.0), (0.0, 1.0))
        self._drawing = False              # True while the code moves the view

        # Text
        self.setWindowTitle(title)
        self.titleLabel.setText(title)

        # Controls. Each dropdown item shows the display name and stores the
        # column name, which currentData() returns.
        self._set_results(results, metrics or DEFAULT_METRICS)
        self.yCombo.setCurrentIndex(1 if len(self.metrics) > 1 else 0)
        self.nSlider.setValue(n_default)
        self.nValue.setText(str(self.nSlider.value()))
        self._setup_county_combo()

        # Chart
        self.plot = self.plotWidget.getPlotItem()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        # show real values (0.004), not rescaled ones with a "(x0.001)" label
        for side in ("bottom", "left"):
            self.plot.getAxis(side).enableAutoSIPrefix(False)
        self.plot.addLegend(offset=(10, -10), labelTextColor=INK,
                            brush=pg.mkBrush(255, 255, 255, 230), pen=pg.mkPen(RULE))
        # Remember the zoom whenever the user scrolls, drags or resets it
        self.plot.getViewBox().sigRangeChanged.connect(self._view_changed)

        # Chart gets 2/3 of the width, side panel 1/3
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([860, 400])

        # Redraw whenever a control changes
        self.xCombo.currentIndexChanged.connect(self.redraw)
        self.yCombo.currentIndexChanged.connect(self.redraw)
        self.nSlider.valueChanged.connect(self._n_changed)
        self.showOthers.toggled.connect(self.redraw)
        self.showLine.toggled.connect(self.redraw)

        toolbar = self.addToolBar("Plans")
        toolbar.addAction("Load plan files…", self.load_files)
        toolbar.addAction("Load plan folder…", self.load_folder)
        self._loader: PlanLoader | None = None

        self.redraw()
        self.update_panel()

    # ---- results and county-split measure -------------------------------
    def _set_results(self, results: pd.DataFrame, metrics: dict[str, bool]):
        """Show new scores: refill the axis dropdowns, slider and summary.

        Keeps the axes on the same metrics (or the metric that replaced one)
        and keeps the selected plan. Call redraw() afterwards.
        """
        old = getattr(self, "metrics", {})
        # old metric -> new metric in the same position (the county swap)
        swap = dict(zip(old, metrics))
        x, y = (swap.get(c, c) for c in (self.xCombo.currentData(),
                                          self.yCombo.currentData()))

        self.results = results.reset_index(drop=True)
        self.metrics = metrics
        self._scores = dict(zip(self.results[self.id_col].astype(str),
                                self.results["DEA_Score"]))

        for combo, col in ((self.xCombo, x), (self.yCombo, y)):
            combo.blockSignals(True)   # redraw once at the end, not per item
            combo.clear()
            for c in metrics:
                combo.addItem(self.names.get(c, c), c)
            if col in metrics:
                combo.setCurrentIndex(list(metrics).index(col))
            combo.blockSignals(False)

        n_ineff = int(off_frontier(self.results).sum())
        n_eff = int(self.results["DEA_Efficient"].sum())
        self.nSlider.blockSignals(True)
        self.nSlider.setRange(0, n_ineff)   # also caps the current value
        self.nSlider.blockSignals(False)
        self.nValue.setText(str(self.nSlider.value()))

        self.summaryLabel.setText(
            f"{n_eff} of {len(self.results)} plans are on the DEA frontier (crimson). "
            "Axes are flipped where needed so up and right are always better. "
            "The DEA frontier uses all metrics, so a crimson plan can sit inside "
            "the dashed line, which uses only the two metrics shown. Scroll to zoom, "
            "drag to pan, and click the small \"A\" in the corner to reset the view.")

    def _county_metric(self) -> str | None:
        """The county-split measure in use, if any."""
        return next((m for m in self.metrics if m in self.county_methods), None)

    def _setup_county_combo(self):
        current = self._county_metric()
        usable = self.rescore is not None and current is not None \
            and len(self.county_methods) > 1
        self.countyLabel.setVisible(usable)
        self.countyCombo.setVisible(usable)
        if not usable:
            return
        for col in self.county_methods:
            self.countyCombo.addItem(self.names.get(col, col), col)
        self.countyCombo.setCurrentIndex(list(self.county_methods).index(current))
        self.countyCombo.currentIndexChanged.connect(self._county_changed)

    def _county_changed(self, index: int):
        """Rescore every plan with the chosen county-split measure."""
        new, old = self.countyCombo.itemData(index), self._county_metric()
        if new is None or new == old:
            return
        # Same metrics in the same order, with the county measure swapped
        metrics = {(new if m == old else m): (self.county_methods[new] if m == old else hb)
                   for m, hb in self.metrics.items()}

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = self.rescore(metrics)
        except Exception as error:  # show any failure to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not rescore plans", str(error))
            self.countyCombo.blockSignals(True)
            self.countyCombo.setCurrentIndex(list(self.county_methods).index(old))
            self.countyCombo.blockSignals(False)
            return
        QApplication.restoreOverrideCursor()

        self._drawing = True   # the view is redrawn below; don't record it
        try:
            self._set_results(results, metrics)
        finally:
            self._drawing = False
        self.redraw()
        self.update_panel()

    # ---- loading real plans --------------------------------------------
    def load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load plan files", "", "CSV files (*.csv)")
        if paths:
            self._start_loading(paths)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Load plan folder")
        paths = [str(p) for p in Path(folder).glob("*.csv")] if folder else []
        if paths:
            self._start_loading(paths)
        elif folder:
            QMessageBox.warning(self, "No plans", f"No CSV files in {folder}")

    def _start_loading(self, paths: list[str]):
        self._progress = QProgressDialog(
            f"Measuring and scoring {len(paths)} plan file(s)…", None, 0, 0, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.show()

        self._loader = PlanLoader(paths)
        self._loader.done.connect(self._loaded)
        self._loader.failed.connect(self._load_failed)
        self._loader.start()

    def _loaded(self, loaded: tuple[pd.DataFrame, pd.DataFrame]):
        self._progress.close()
        table, results = loaded
        window = FrontierWindow(
            results, metrics=_higher_is_better(plan_metrics.DIRECTION),
            display_names=PLAN_METRIC_NAMES,
            county_methods=_higher_is_better(plan_metrics.COUNTY_SPLIT_DIRECTION),
            rescore=sbm_rescorer(table))
        # Keep a reference, or Python deletes the new window.
        QApplication.instance().frontier_window = window
        window.show()
        self.close()

    def _load_failed(self, message: str):
        self._progress.close()
        QMessageBox.warning(self, "Could not load plans", message)

    # ---- selection ------------------------------------------------------
    def select(self, plan_id: str | None):
        """Show a plan in the side panel and circle it on the chart."""
        self.selected = plan_id
        self.redraw()
        self.update_panel()

    def _point_clicked(self, _item, points, *_):
        if len(points):
            self.select(str(points[0].data()))

    def _n_changed(self, n: int):
        self.nValue.setText(str(n))
        self.redraw()

    # ---- chart ----------------------------------------------------------
    def redraw(self, *_):
        """Draw the chart for the current controls and selection."""
        x, y = self.xCombo.currentData(), self.yCombo.currentData()
        if x is None or y is None:
            return
        # Drawing moves the view (flipping axes, fitting new metrics); only
        # the user's own zooming should change the remembered zoom.
        self._drawing = True
        try:
            self._draw(x, y)
        finally:
            self._drawing = False

    def _draw(self, x: str, y: str):
        r, m = self.results, self.metrics
        eff = r[r["DEA_Efficient"] == 1]
        weak = r[r["DEA_Status"] == "weakly efficient"]
        near = near_frontier(r, self.nSlider.value())
        others = r.drop(index=eff.index.union(weak.index).union(near.index))

        self.plot.clear()   # also empties the legend

        # Layers from bottom to top, so frontier plans draw over the rest
        if self.showOthers.isChecked() and len(others):
            self._add_points(others, x, y, f"Other plans ({len(others)})", OTHER, 8, "o")
        if self.showLine.isChecked():
            front = pareto_2d(r, x, y, m)
            xs, ys = staircase(front[x].to_numpy(float), front[y].to_numpy(float))
            self.plot.addItem(pg.PlotDataItem(
                xs, ys, pen=pg.mkPen(MUTED, width=2, style=Qt.DashLine),
                name="Frontier for these 2 metrics only"))
        if len(near):
            self._add_points(near, x, y, f"{len(near)} closest to frontier", NEAR, 11, "o")
        if len(weak):
            self._add_points(weak, x, y, f"On frontier, weak ({len(weak)})",
                             STATUS_COLORS["weakly efficient"], 12, "d")
        self._add_points(eff, x, y, f"DEA frontier ({len(eff)})", FRONTIER, 14, "d")
        self._draw_selected(x, y)

        # Flip lower-is-better axes so better is always up and right
        self.plot.invertX(not m[x])
        self.plot.invertY(not m[y])
        self.plot.setLabel("bottom", f"{self.names.get(x, x)}  (better →)")
        self.plot.setLabel("left", f"{self.names.get(y, y)}  (better ↑)")

        # Move the view only when the axes change, so changing n or the
        # checkboxes leaves it alone. New metrics get the same zoom: each
        # axis shows the same share of its metric's range as before.
        if (x, y) != self._axes:
            self._axes = (x, y)
            self.plot.setRange(xRange=self._from_shares(x, self._view[0]),
                               yRange=self._from_shares(y, self._view[1]),
                               padding=0)

    # ---- zoom -----------------------------------------------------------
    def _full_view(self, col: str) -> tuple[float, float]:
        """The range that shows every plan, plus VIEW_PADDING on each side."""
        values = self.results[col].astype(float)
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo if hi > lo else (abs(hi) or 1.0)
        return lo - VIEW_PADDING * span, hi + VIEW_PADDING * span

    def _to_shares(self, col: str, lo: float, hi: float) -> tuple[float, float]:
        """A visible range of ``col`` as shares of its full view, 0 = worst end."""
        a, b = self._full_view(col)
        s0, s1 = (lo - a) / (b - a), (hi - a) / (b - a)
        # lower-is-better axes are flipped, so measure from the other end
        return (s0, s1) if self.metrics[col] else (1 - s1, 1 - s0)

    def _from_shares(self, col: str, shares: tuple[float, float]) -> tuple[float, float]:
        """The inverse of _to_shares: a range of ``col`` in its own units."""
        s0, s1 = shares if self.metrics[col] else (1 - shares[1], 1 - shares[0])
        a, b = self._full_view(col)
        return a + s0 * (b - a), a + s1 * (b - a)

    def _view_changed(self, *_):
        """Remember the user's zoom as shares of each axis."""
        if self._drawing or self._axes is None:
            return
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        x, y = self._axes
        self._view = (self._to_shares(x, x0, x1), self._to_shares(y, y0, y1))

    def _add_points(self, df, x, y, name, color, size, symbol):
        """Add one group of plans as clickable points with hover text."""
        item = pg.ScatterPlotItem(
            x=df[x].to_numpy(float), y=df[y].to_numpy(float),
            data=df[self.id_col].astype(str).tolist(),   # plan id of each point
            size=size, symbol=symbol, brush=pg.mkBrush(color),
            pen=pg.mkPen("w", width=1.5), name=name,
            hoverable=True, tip=self._hover_text(x, y))
        item.sigClicked.connect(self._point_clicked)
        self.plot.addItem(item)

    def _hover_text(self, x_col, y_col):
        """Function pyqtgraph calls for a point's hover text."""
        nx, ny = self.names.get(x_col, x_col), self.names.get(y_col, y_col)

        def tip(x, y, data):
            return (f"{data}\n{nx}: {fmt(x)}\n{ny}: {fmt(y)}\n"
                    f"DEA score: {self._scores[data]:.3f}")
        return tip

    def _draw_selected(self, x, y):
        """Ring and label around the selected plan."""
        if self.selected is None:
            return
        sel = self.results[self.results[self.id_col].astype(str) == self.selected]
        if sel.empty:
            return
        px, py = float(sel[x].iloc[0]), float(sel[y].iloc[0])
        self.plot.addItem(pg.ScatterPlotItem(
            x=[px], y=[py], size=26, symbol="o",
            brush=pg.mkBrush(0, 0, 0, 0), pen=pg.mkPen(INK, width=2)))
        label = pg.TextItem(self.selected, color=INK, anchor=(0.5, 1.7))
        label.setPos(px, py)
        self.plot.addItem(label)

    # ---- side panel -----------------------------------------------------
    def update_panel(self):
        """Rebuild the side panel for the selected plan."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(4)

        if self.selected is None:
            layout.addWidget(_label("Click a plan", "font-size: 17px; font-weight: 600;"))
            layout.addWidget(_label(
                "Click any point to see its metrics, how it ranks against the "
                "other plans, and which frontier plans beat it.",
                f"color: {MUTED};", wrap=True))
        else:
            self._fill_panel(layout)

        layout.addStretch(1)
        # Swap in the new panel. setWidget() alone would delete the old panel
        # immediately, and this method can run from a click on a peer button
        # inside that panel; deleting a button during its own click handler
        # can crash Qt. deleteLater() waits until the click has finished.
        old = self.panelScroll.takeWidget()
        self.panelScroll.setWidget(panel)
        if old is not None:
            old.deleteLater()

    def _fill_panel(self, layout: QVBoxLayout):
        r, id_col = self.results, self.id_col
        match = r[r[id_col].astype(str) == self.selected]
        if match.empty:
            layout.addWidget(_label(f"{self.selected} not found."))
            return
        row = match.iloc[0]
        status = row["DEA_Status"]
        color = STATUS_COLORS.get(status, MUTED)
        efficient = row["DEA_Efficient"] == 1
        # SBM results (score_sbm) have no peers or slacks to show.
        has_peers = "DEA_Peers" in r.columns
        targets = frontier_targets(row, self.metrics, self.orientation)

        # Header: plan id, status badge, score
        header = QHBoxLayout()
        header.addWidget(_label(self.selected, "font-size: 18px; font-weight: 600;"))
        header.addWidget(_label(STATUS_LABELS.get(status, status),
                                f"background: {color}; color: white; "
                                        "border-radius: 9px; padding: 2px 10px; "
                                        "font-size: 12px;"))
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(_label(f"DEA score {row['DEA_Score']:.3f}", f"color: {MUTED};"))
        layout.addSpacing(12)

        # One block per metric: value, percentile bar, frontier target
        for col, higher_better in self.metrics.items():
            v = float(row[col])
            allv = r[col].astype(float)
            pct = float((allv <= v).mean() if higher_better else (allv >= v).mean())
            t = targets[col]
            show_target = (has_peers and not efficient
                           and abs(t - v) > 1e-6 * max(1.0, abs(v)))

            top = QHBoxLayout()
            top.addWidget(_label(f"{self.names.get(col, col)} {'↑' if higher_better else '↓'}",
                                 "font-weight: 600;"))
            top.addStretch(1)
            top.addWidget(_label(fmt(v)))
            layout.addLayout(top)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(round(pct * 100))
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(
                f"QProgressBar {{ background: {RULE}; border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}")
            layout.addWidget(bar)

            bottom = QHBoxLayout()
            small = f"color: {MUTED}; font-size: 12px;"
            bottom.addWidget(_label(f"As good as or better than {pct:.0%} of plans", small))
            bottom.addStretch(1)
            if show_target:
                bottom.addWidget(_label(f"frontier target {fmt(t)}", small))
            layout.addLayout(bottom)
            layout.addSpacing(10)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {RULE};")
        layout.addWidget(line)
        layout.addSpacing(6)

        # Peers
        if efficient:
            layout.addWidget(_label(
                "This plan is on the DEA frontier: no blend of other plans "
                "beats it on every metric.", f"color: {MUTED};", wrap=True))
            return
        if not has_peers:
            return

        layout.addWidget(_label("Frontier peers", "font-weight: 600;"))
        layout.addWidget(_label(
            "A blend of these frontier plans beats this plan. Click one to compare.",
            f"color: {MUTED};", wrap=True))

        peer_ids = [p.strip() for p in str(row["DEA_Peers"]).split(",") if p.strip()]
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, p in enumerate(peer_ids):
            button = QPushButton(p)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(
                f"QPushButton {{ border: 1px solid {FRONTIER}; color: {FRONTIER}; "
                "background: white; border-radius: 6px; padding: 4px 10px; }"
                f"QPushButton:hover {{ background: #fbeaec; }}")
            # p=p keeps each button's own plan id
            button.clicked.connect(lambda _checked=False, p=p: self.select(p))
            grid.addWidget(button, i // 3, i % 3)
        layout.addLayout(grid)

        layout.addWidget(_label(
            "Targets are what that blend achieves. They can be fractional "
            "(e.g. 6.3 county splits) because they average several plans.",
            f"color: {MUTED}; font-size: 12px;", wrap=True))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def show_app(results: pd.DataFrame, **kwargs) -> int:
    """Open the explorer for run_dea() output and wait until it is closed.

    Keyword arguments go to FrontierWindow: id_col, metrics, display_names,
    orientation, n_default, title, county_methods, rescore. Returns Qt's
    exit code.
    """
    app = QApplication.instance()
    if app is None:
        # sharper text on high-resolution screens
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)
    window = FrontierWindow(results, **kwargs)
    window.show()
    return app.exec_()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Interactive DEA frontier explorer")
    p.add_argument("--csv", help="metrics table (one row per plan); omit to use mock data")
    p.add_argument("--id-col", default="Plan_ID")
    p.add_argument("--orientation", choices=["input", "output"], default="input")
    p.add_argument("--n", type=int, default=10, help="plans closest to the frontier to show")
    args = p.parse_args()

    df = pd.read_csv(args.csv) if args.csv else make_mock_metrics()
    results = run_dea(df, id_col=args.id_col, orientation=args.orientation)

    # Offer the county-split measures this table has
    county_methods = {c: hb for c, hb in MOCK_COUNTY_METHODS.items() if c in df.columns}

    def rescore(metrics: dict[str, bool]) -> pd.DataFrame:
        return run_dea(df, id_col=args.id_col, metrics=metrics,
                       orientation=args.orientation)

    sys.exit(show_app(results, id_col=args.id_col, orientation=args.orientation,
                      n_default=args.n, county_methods=county_methods,
                      rescore=rescore))


if __name__ == "__main__":
    main()
