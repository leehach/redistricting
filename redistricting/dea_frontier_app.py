"""
Interactive Pareto-frontier explorer for DEA results of redistricting plans.

A standalone Dash app: it runs the DEA itself (the same calculation as the
parallel-coordinates script, dea_pcp.py) and doesn't need any other file.

  * a scatter plot of any two metrics, with every axis oriented so that
    up and right are better
  * the DEA-frontier plans in crimson
  * the n plans closest to the frontier (highest DEA scores among the
    inefficient plans) in blue, with a slider for n
  * optionally, all other plans in light gray, and the 2-metric Pareto
    frontier as a dashed staircase line
  * click any plan to see its metrics, its percentile on each one, the
    values it would need to reach the frontier, and its frontier peers
    (click a peer to jump to it)

Note: the DEA frontier uses ALL metrics at once, so a crimson plan can sit
inside the dashed 2-metric line. That's expected: it earns its place on the
frontier with a metric that isn't on the current axes.

Usage
-----
    pip install dash scipy pandas numpy plotly
    python dea_frontier_app.py                     # mock data
    python dea_frontier_app.py --csv metrics.csv   # real metrics table
    python dea_frontier_app.py --n 15 --port 8051

The app opens in your browser at http://127.0.0.1:8050. Stop it with Ctrl+C.

In code (functions take DataFrames, not file paths):

    results = run_dea(metrics_df)
    app = create_app(results)
    app.run()
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
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


# Same colors as the parallel coordinates plot.
FRONTIER = "#dc143c"   # crimson
NEAR = "#4a78a8"       # steel blue
OTHER = "#c4c8ce"      # light gray
INK = "#1f2933"
MUTED = "#6b7280"
RULE = "#e5e7eb"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

STATUS_COLORS = {"efficient": FRONTIER, "weakly efficient": "#b45309",
                 "inefficient": NEAR}


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def near_frontier(results: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n inefficient plans with the highest DEA scores."""
    return results[results["DEA_Efficient"] == 0].nlargest(n, "DEA_Score")


def pareto_2d(results: pd.DataFrame, x: str, y: str, metrics: dict) -> pd.DataFrame:
    """Plans on the Pareto frontier for just two metrics, in drawing order."""
    vals = results[[x, y]].to_numpy(dtype=float)
    hb = np.array([metrics[x], metrics[y]])
    front = results[pareto_nondominated(vals, hb)].copy()
    # sort along the x axis from worst to best, so the staircase draws cleanly
    front["_u"] = front[x] if metrics[x] else -front[x]
    front["_v"] = front[y] if metrics[y] else -front[y]
    return front.sort_values(["_u", "_v"], ascending=[True, False])


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


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def build_figure(results, metrics, names, id_col, x, y, n,
                 show_others, show_line, selected) -> go.Figure:
    eff = results[results["DEA_Efficient"] == 1]
    near = near_frontier(results, n)
    others = results.drop(index=eff.index.union(near.index))
    nx, ny = names.get(x, x), names.get(y, y)

    hover = ("<b>%{customdata[0]}</b><br>" + nx + ": %{x}<br>" + ny + ": %{y}"
             "<br>DEA score: %{customdata[1]:.3f}<extra></extra>")

    def scatter(df, name, color, size, symbol, opacity):
        return go.Scatter(
            x=df[x], y=df[y], mode="markers", name=name,
            marker=dict(color=color, size=size, symbol=symbol, opacity=opacity,
                        line=dict(color="white", width=1.5)),
            customdata=df[[id_col, "DEA_Score"]].astype(object).to_numpy(),
            hovertemplate=hover,
        )

    fig = go.Figure()
    if show_others and len(others):
        fig.add_trace(scatter(others, f"Other plans ({len(others)})", OTHER, 8, "circle", 0.8))
    if show_line:
        front = pareto_2d(results, x, y, metrics)
        fig.add_trace(go.Scatter(
            x=front[x], y=front[y], mode="lines", line_shape="hv",
            line=dict(color=MUTED, width=2, dash="dash"),
            name=f"Frontier for these 2 metrics only", hoverinfo="skip"))
    if len(near):
        fig.add_trace(scatter(near, f"{len(near)} closest to frontier", NEAR, 11, "circle", 1))
    fig.add_trace(scatter(eff, f"DEA frontier ({len(eff)})", FRONTIER, 14, "diamond", 1))

    if selected is not None:
        sel = results[results[id_col].astype(str) == selected]
        if len(sel):
            fig.add_trace(go.Scatter(
                x=sel[x], y=sel[y], mode="markers+text", name="Selected",
                text=sel[id_col].astype(str), textposition="top center",
                textfont=dict(color=INK, size=12),
                marker=dict(size=24, symbol="circle-open", color=INK, line=dict(width=2)),
                hoverinfo="skip", showlegend=False))

    def axis(col, title_suffix):
        a = dict(title=dict(text=f"{names.get(col, col)}  {title_suffix}",
                            font=dict(size=13, color=INK)),
                 gridcolor=RULE, zeroline=False, linecolor=RULE,
                 tickfont=dict(color=MUTED))
        if not metrics[col]:
            a["autorange"] = "reversed"   # lower is better -> flip so better is up/right
        return a

    fig.update_layout(
        xaxis=axis(x, "(better →)"),
        yaxis=axis(y, "(better ↑)"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT, color=INK),
        legend=dict(orientation="h", yanchor="top", y=-0.13, x=0,
                    font=dict(size=12)),
        margin=dict(l=70, r=20, t=30, b=110),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        uirevision=f"{x}|{y}",   # keep zoom when only n or toggles change
        clickmode="event",
    )
    return fig


# --------------------------------------------------------------------------
# Side panel
# --------------------------------------------------------------------------
def plan_panel(results, metrics, names, id_col, orientation, plan_id):
    if plan_id is None:
        return html.Div([
            html.H3("Click a plan", style={"margin": "0 0 8px"}),
            html.P("Click any point to see its metrics, how it ranks against the "
                   "other plans, and which frontier plans beat it.",
                   style={"color": MUTED, "margin": 0}),
        ])

    match = results[results[id_col].astype(str) == plan_id]
    if match.empty:
        return html.P(f"{plan_id} not found.")
    row = match.iloc[0]
    status = row["DEA_Status"]
    efficient = row["DEA_Efficient"] == 1
    targets = frontier_targets(row, metrics, orientation)

    header = html.Div([
        html.H3(plan_id, style={"margin": "0 8px 0 0", "display": "inline"}),
        html.Span(status, style={
            "background": STATUS_COLORS.get(status, MUTED), "color": "white",
            "borderRadius": "10px", "padding": "2px 10px", "fontSize": "12px",
            "verticalAlign": "middle"}),
        html.Div(f"DEA score {row['DEA_Score']:.3f}",
                 style={"color": MUTED, "marginTop": "6px"}),
    ], style={"marginBottom": "16px"})

    # Metric rows: value, percentile bar, and frontier target if inefficient
    rows = []
    for col, higher_better in metrics.items():
        v = float(row[col])
        allv = results[col].astype(float)
        pct = float((allv <= v).mean() if higher_better else (allv >= v).mean())
        t = targets[col]
        show_target = not efficient and abs(t - v) > 1e-6 * max(1.0, abs(v))
        rows.append(html.Div([
            html.Div([
                html.Span(f"{names.get(col, col)} {'↑' if higher_better else '↓'}",
                          style={"fontWeight": 600}),
                html.Span(fmt(v), style={"float": "right", "fontVariantNumeric": "tabular-nums"}),
            ]),
            html.Div(html.Div(style={
                "width": f"{pct * 100:.0f}%", "height": "6px",
                "background": STATUS_COLORS.get(status, MUTED), "borderRadius": "3px"}),
                style={"background": RULE, "borderRadius": "3px", "margin": "6px 0 4px"}),
            html.Div([
                html.Span(f"As good as or better than {pct:.0%} of plans"),
                html.Span(f"frontier target {fmt(t)}", style={"float": "right"})
                if show_target else None,
            ], style={"fontSize": "12px", "color": MUTED}),
        ], style={"marginBottom": "14px"}))

    # Peers
    if efficient:
        peers_block = html.P("This plan is on the DEA frontier: no blend of other "
                             "plans beats it on every metric.",
                             style={"color": MUTED, "fontSize": "13px"})
    else:
        peer_ids = [p.strip() for p in str(row["DEA_Peers"]).split(",") if p.strip()]
        peers_block = html.Div([
            html.Div("Frontier peers", style={"fontWeight": 600, "marginBottom": "4px"}),
            html.P("A blend of these frontier plans beats this plan. Click one to compare.",
                   style={"color": MUTED, "fontSize": "13px", "margin": "0 0 8px"}),
            html.Div([
                html.Button(p, id={"type": "peer", "plan": p}, n_clicks=0, style={
                    "marginRight": "6px", "marginBottom": "6px", "cursor": "pointer",
                    "border": f"1px solid {FRONTIER}", "color": FRONTIER,
                    "background": "white", "borderRadius": "6px",
                    "padding": "4px 10px", "fontFamily": FONT})
                for p in peer_ids
            ]),
            html.P("Targets are what that blend achieves. They can be fractional "
                   "(e.g. 6.3 county splits) because they average several plans.",
                   style={"color": MUTED, "fontSize": "12px", "marginTop": "8px"}),
        ])

    return html.Div([header, *rows, html.Hr(style={"border": 0, "borderTop": f"1px solid {RULE}"}),
                     peers_block])


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def create_app(
    results: pd.DataFrame,
    id_col: str = "Plan_ID",
    metrics: dict[str, bool] | None = None,
    display_names: dict[str, str] | None = None,
    orientation: str = "input",
    n_default: int = 10,
    title: str = "DEA Frontier of Prospective Redistricting Plans",
) -> Dash:
    """Build the Dash app from run_dea() output."""
    metrics = metrics or DEFAULT_METRICS
    names = {**DISPLAY_NAMES, **(display_names or {})}
    results = results.reset_index(drop=True)
    cols = list(metrics)
    n_ineff = int((results["DEA_Efficient"] == 0).sum())
    n_default = min(n_default, n_ineff)
    n_eff = int(results["DEA_Efficient"].sum())

    options = [{"label": names.get(c, c), "value": c} for c in cols]
    label = {"fontSize": "12px", "color": MUTED, "marginBottom": "4px", "display": "block"}

    app = Dash(__name__, title="DEA Frontier")
    app.layout = html.Div([
        html.H2(title, style={"margin": "0 0 4px"}),
        html.P(f"{n_eff} of {len(results)} plans are on the DEA frontier (crimson). "
               "Axes are flipped where needed so up and right are always better. "
               "The DEA frontier uses all metrics, so a crimson plan can sit inside "
               "the dashed line, which uses only the two metrics shown.",
               style={"color": MUTED, "margin": "0 0 16px", "maxWidth": "900px"}),

        # Controls
        html.Div([
            html.Div([html.Label("X axis", style=label),
                      dcc.Dropdown(id="x", options=options, value=cols[0], clearable=False)],
                     style={"width": "200px"}),
            html.Div([html.Label("Y axis", style=label),
                      dcc.Dropdown(id="y", options=options,
                                   value=cols[1] if len(cols) > 1 else cols[0],
                                   clearable=False)],
                     style={"width": "200px"}),
            html.Div([html.Label("Plans closest to frontier (n)", style=label),
                      dcc.Slider(id="n", min=0, max=max(n_ineff, 1), step=1,
                                 value=n_default,
                                 marks={0: "0", n_default: str(n_default),
                                        n_ineff: str(n_ineff)},
                                 tooltip={"placement": "bottom"})],
                     style={"width": "260px"}),
            dcc.Checklist(id="toggles",
                          options=[{"label": " Show other plans", "value": "others"},
                                   {"label": " Show 2-metric frontier line", "value": "line"}],
                          value=["others", "line"],
                          style={"fontSize": "14px"},
                          labelStyle={"display": "block", "marginBottom": "4px"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap",
                  "alignItems": "flex-end", "marginBottom": "12px"}),

        # Chart + panel
        html.Div([
            dcc.Graph(id="graph", config={"displaylogo": False},
                      style={"flex": "2 1 520px", "height": "620px", "minWidth": "0"}),
            html.Div(id="panel", style={
                "flex": "1 1 300px", "border": f"1px solid {RULE}", "borderRadius": "10px",
                "padding": "16px", "alignSelf": "flex-start", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        dcc.Store(id="selected"),
    ], style={"fontFamily": FONT, "color": INK, "padding": "20px 24px",
              "maxWidth": "1400px", "margin": "0 auto"})

    @app.callback(
        Output("graph", "figure"),
        Input("x", "value"), Input("y", "value"), Input("n", "value"),
        Input("toggles", "value"), Input("selected", "data"),
    )
    def _update_figure(x, y, n, toggles, selected):
        toggles = toggles or []
        return build_figure(results, metrics, names, id_col, x, y, n or 0,
                            "others" in toggles, "line" in toggles, selected)

    @app.callback(
        Output("selected", "data"),
        Input("graph", "clickData"),
        Input({"type": "peer", "plan": ALL}, "n_clicks"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _select(click, _peer_clicks, current):
        trig = ctx.triggered_id
        if trig == "graph":
            if not click or "customdata" not in click["points"][0]:
                return no_update
            return str(click["points"][0]["customdata"][0])
        if isinstance(trig, dict) and trig.get("type") == "peer":
            # peer buttons also "fire" when first drawn, with n_clicks = 0
            if not ctx.triggered[0]["value"]:
                return no_update
            return trig["plan"]
        return no_update

    @app.callback(Output("panel", "children"), Input("selected", "data"))
    def _update_panel(selected):
        return plan_panel(results, metrics, names, id_col, orientation, selected)

    return app


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Interactive DEA frontier explorer")
    p.add_argument("--csv", help="metrics table (one row per plan); omit to use mock data")
    p.add_argument("--id-col", default="Plan_ID")
    p.add_argument("--orientation", choices=["input", "output"], default="input")
    p.add_argument("--n", type=int, default=10, help="plans closest to the frontier to show")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = p.parse_args()

    df = pd.read_csv(args.csv) if args.csv else make_mock_metrics()
    results = run_dea(df, id_col=args.id_col, orientation=args.orientation)
    app = create_app(results, id_col=args.id_col, orientation=args.orientation,
                     n_default=args.n)

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nDEA frontier app running at {url}  (press Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
