"""
Data Envelopment Analysis (DEA) of prospective redistricting plans,
visualized with a Parallel Coordinates Plot (PCP).

How DEA is applied here
-----------------------
Each redistricting plan is a "decision-making unit" (DMU). Every metric is
classified by direction:

  * lower-is-better metrics (county splits, population deviation, partisan gap)
    are treated as DEA *inputs*  -> a good plan uses "less" of them
  * higher-is-better metrics (compactness) are treated as DEA *outputs*

The model is the BCC (variable-returns-to-scale) envelopment model, solved as
one linear program per plan, in two phases:

  Phase 1  radial score
     input orientation : min theta  s.t.  sum_j l_j x_j <= theta * x_o,
                                          sum_j l_j y_j >= y_o,
                                          sum_j l_j = 1,  l >= 0
     output orientation: max phi    s.t.  sum_j l_j x_j <= x_o,
                                          sum_j l_j y_j >= phi * y_o,
                                          sum_j l_j = 1,  l >= 0
     efficiency score = theta (input) or 1/phi (output); 1.0 = on the frontier.

  Phase 2  slacks
     With the radial score fixed, maximize the (range-normalized) slacks. A plan
     is DEA-efficient only if its score is 1 AND every slack is 0. Plans with
     score 1 but nonzero slacks are "weakly efficient" (they sit on a flat part
     of the frontier and are beaten on at least one metric by a mix of others).

The lambdas from phase 2 name each plan's "peers": the efficient plans that
together outperform it. Those are the plans to look at if you're trying to
improve an inefficient one.

Speed-up: dominated plans can never be part of the frontier, so the reference
set in every LP is limited to the Pareto-nondominated plans. This gives
identical results and keeps large plan ensembles tractable.

Usage
-----
    python dea_redistricting.py                     # mock data demo
    python dea_redistricting.py --csv metrics.csv   # real metrics table

In code (functions take DataFrames, not file paths):

    from dea_redistricting import run_dea, plot_pcp, DEFAULT_METRICS
    results = run_dea(metrics_df, id_col="Plan_ID", metrics=DEFAULT_METRICS)
    fig = plot_pcp(results, id_col="Plan_ID", metrics=DEFAULT_METRICS)
    fig.show()

Requirements: numpy, pandas, scipy, plotly
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog

# Column name -> True if higher values are better, False if lower is better.
# Edit this (or pass your own dict) to match your collaborators' output table.
DEFAULT_METRICS: dict[str, bool] = {
    "Compactness": True,      # e.g. mean Polsby-Popper, 0-1
    "County_Splits": False,   # count of split counties
    "Pop_Dev_Pct": False,     # max population deviation, percent
    "Partisan_Gap": False,    # e.g. |efficiency gap| or partisan bias
}

DISPLAY_NAMES: dict[str, str] = {
    "Compactness": "Compactness (PP)",
    "County_Splits": "County Splits",
    "Pop_Dev_Pct": "Pop. Deviation (%)",
    "Partisan_Gap": "Partisan Gap",
}

TOL = 1e-6


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

    return pd.DataFrame({
        "Plan_ID": [f"Plan_{i:03d}" for i in range(1, n + 1)],
        "Compactness": np.clip(compactness, 0.05, 0.95).round(4),
        "County_Splits": np.clip(county_splits, 0, None).astype(int),
        "Pop_Dev_Pct": np.clip(pop_dev, 0.0, None).round(4),
        "Partisan_Gap": partisan_gap.round(4),
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
      DEA_Score       radial efficiency, 0-1 (1 = on the frontier)
      DEA_Efficient   1 if score == 1 and all slacks are 0, else 0
      DEA_Status      "efficient", "weakly efficient", or "inefficient"
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


# --------------------------------------------------------------------------
# Parallel coordinates plot
# --------------------------------------------------------------------------
def plot_pcp(
    results: pd.DataFrame,
    id_col: str = "Plan_ID",
    metrics: dict[str, bool] | None = None,
    display_names: dict[str, str] | None = None,
    title: str = "DEA of Prospective Redistricting Plans",
) -> go.Figure:
    """Parallel coordinates plot of DEA results.

    Every axis is oriented so that UP IS BETTER (lower-is-better axes are
    flipped), which makes trade-offs read as line crossings and dominated
    plans read as lines sitting below others everywhere. Efficient plans are
    drawn in crimson on top; inefficient plans fade from blue to gray as
    their DEA score drops. Drag along any axis to filter plans.
    """
    metrics = metrics or DEFAULT_METRICS
    names = {**DISPLAY_NAMES, **(display_names or {})}

    # Draw efficient plans last so they sit on top.
    d = results.sort_values(["DEA_Efficient", "DEA_Score"]).reset_index(drop=True)
    eff_mask = (d["DEA_Efficient"] == 1).to_numpy()
    n_eff, n_ineff = int(eff_mask.sum()), int((~eff_mask).sum())

    # "Plan" axis: parallel-coordinates lines have no hover labels, so each
    # frontier plan gets its own labeled tick on the leftmost axis. Inefficient
    # plans are packed, unlabeled, into the bottom of the axis.
    plan_pos = np.empty(len(d))
    plan_pos[~eff_mask] = np.linspace(0.0, 0.3, n_ineff) if n_ineff > 1 else 0.15
    eff_pos = np.linspace(0.4, 1.0, n_eff) if n_eff > 1 else np.array([0.7])
    plan_pos[eff_mask] = eff_pos
    eff_ids = d.loc[eff_mask, id_col].astype(str).tolist()
    tickvals, ticktext = list(eff_pos), eff_ids
    if n_ineff:
        tickvals.append(0.15)
        ticktext.append(f"({n_ineff} others)")
    dims = [dict(label="Plan", values=plan_pos, range=[-0.02, 1.02],
                 tickvals=tickvals, ticktext=ticktext)]

    for col, higher_better in metrics.items():
        lo, hi = float(d[col].min()), float(d[col].max())
        if hi == lo:
            lo, hi = lo - 0.5, hi + 0.5
        rng = [lo, hi] if higher_better else [hi, lo]
        label = names.get(col, col) + ("  ↑" if higher_better else "  ↓")
        dims.append(dict(label=label, values=d[col], range=rng))
    dims.append(dict(label="DEA Score", values=d["DEA_Score"],
                     range=[float(d["DEA_Score"].min()) - 0.01, 1.0]))

    # Color: gray (worst) -> steel blue (near frontier) -> crimson (efficient)
    smin = float(d["DEA_Score"].min())
    # Non-efficient plans (including weakly efficient ones that score 1.0)
    # are capped below the crimson band so only true frontier plans are red.
    color_val = np.where(d["DEA_Efficient"] == 1, 1.0, np.minimum(d["DEA_Score"], 0.97))
    span = max(1.0 - smin, 1e-6)
    cut = (0.98 - smin) / span if 0.98 > smin else 0.0
    colorscale = [[0.0, "#d4d4d4"], [max(cut, 0.0), "#4a78a8"],
                  [min(cut + 1e-3, 1.0), "crimson"], [1.0, "crimson"]]

    fig = go.Figure(go.Parcoords(
        domain=dict(x=[0.0, 1.0], y=[0.42, 1.0]),
        line=dict(color=color_val, colorscale=colorscale, cmin=smin, cmax=1.0,
                  showscale=True,
                  colorbar=dict(title="DEA<br>Score", tickformat=".2f",
                                y=0.71, len=0.55)),
        dimensions=dims,
        labelfont=dict(size=13),
        tickfont=dict(size=11),
    ))

    # Table of frontier plans beneath the chart, so exact values are readable.
    eff_rows = d[eff_mask].sort_values(id_col)
    header = ["Plan"] + [names.get(c, c) for c in metrics] + ["DEA Score"]
    cells = [eff_rows[id_col].astype(str)]
    for col in metrics:
        cells.append(eff_rows[col].map(lambda v: f"{v:.4g}"))
    cells.append(eff_rows["DEA_Score"].map(lambda v: f"{v:.3f}"))
    fig.add_trace(go.Table(
        domain=dict(x=[0.0, 1.0], y=[0.0, 0.33]),
        header=dict(values=[f"<b>{h}</b>" for h in header], fill_color="crimson",
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=cells, fill_color="#fbeaec", align="left",
                   font=dict(size=12), height=24),
    ))

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{n_eff} of {len(d)} plans DEA-efficient (crimson). "
                        f"All axes oriented so up = better. Drag along an axis "
                        f"(e.g. over one name on the Plan axis) to isolate lines; "
                        f"click elsewhere on the axis to clear.</sup>"),
        height=max(850, 650 + 24 * n_eff),
        margin=dict(l=110, r=70, t=160, b=30),
        paper_bgcolor="white",
    )
    return fig


# --------------------------------------------------------------------------
# Demo / CLI
# --------------------------------------------------------------------------
def summarize(results: pd.DataFrame, id_col: str = "Plan_ID") -> None:
    counts = results["DEA_Status"].value_counts()
    print("\nDEA status counts:")
    for status in ["efficient", "weakly efficient", "inefficient"]:
        print(f"  {status:<17} {counts.get(status, 0)}")
    eff = results[results["DEA_Efficient"] == 1]
    print(f"\nFrontier plans: {', '.join(eff[id_col].astype(str))}")
    worst = results.nsmallest(5, "DEA_Score")[[id_col, "DEA_Score", "DEA_Peers"]]
    print("\nLowest-scoring plans and their benchmark peers:")
    print(worst.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--csv", help="metrics table (one row per plan); omit to use mock data")
    p.add_argument("--id-col", default="Plan_ID")
    p.add_argument("--orientation", choices=["input", "output"], default="input")
    p.add_argument("--out-csv", default="dea_results.csv")
    p.add_argument("--out-html", default="dea_pcp.html")
    p.add_argument("--no-show", action="store_true", help="don't open the plot")
    args = p.parse_args()

    df = pd.read_csv(args.csv) if args.csv else make_mock_metrics()
    results = run_dea(df, id_col=args.id_col, orientation=args.orientation)
    summarize(results, args.id_col)

    results.to_csv(args.out_csv, index=False)
    fig = plot_pcp(results, id_col=args.id_col)
    fig.write_html(args.out_html, include_plotlyjs="cdn")
    print(f"\nWrote {args.out_csv} and {args.out_html}")
    if not args.no_show:
        # Open the saved HTML directly rather than calling fig.show(), which in
        # some consoles (IPython, Spyder, editor terminals) prints the whole
        # figure as raw text instead of opening a browser.
        webbrowser.open(Path(args.out_html).resolve().as_uri())


if __name__ == "__main__":
    main()
