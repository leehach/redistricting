"""
Interactive Pareto-frontier explorer for SBM DEA results of redistricting plans.

A Dash app that scores plans with redistricting.DEA.slacks_based_measure:

  * a scatter plot of any two metrics, with every axis oriented so that
    up and right are better
  * the frontier plans (SBM score 1) in crimson
  * the n plans closest to the frontier (highest SBM scores among the
    other plans) in blue, with a slider for n
  * optionally, all other plans in light gray, and the 2-metric Pareto
    frontier as a dashed staircase line
  * click any plan to see its metrics, its percentile on each one, and
    its SBM score

Note: the SBM frontier uses ALL metrics at once, so a crimson plan can sit
inside the dashed 2-metric line. That's expected: it earns its place on the
frontier with a metric that isn't on the current axes.

Usage (see examples/frontier_app/mock_plans_app.py):

    results = add_sbm_scores(plans, direction)
    app = create_app(results, direction, id_col)
    app.run()

`plans` has one row per plan. `direction` maps each metric column to
"higher" or "lower", the same as score_plans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html, no_update

from redistricting.DEA.slacks_based_measure import score_plans

# GLPK returns 0.9999999... for some frontier plans.
TOL = 1e-6


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def add_sbm_scores(df: pd.DataFrame, direction: dict[str, str]) -> pd.DataFrame:
    """Return a copy of df with SBM_Score and On_Frontier columns.

    SBM_Score comes from score_plans over the columns named in `direction`.
    On_Frontier is True when the score is 1.
    """
    results = df.copy()
    results["SBM_Score"] = score_plans(df[list(direction)], direction)
    results["On_Frontier"] = results["SBM_Score"] >= 1 - TOL
    return results


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


# Same colors as the parallel coordinates plot.
FRONTIER = "#dc143c"   # crimson
NEAR = "#4a78a8"       # steel blue
OTHER = "#c4c8ce"      # light gray
INK = "#1f2933"
MUTED = "#6b7280"
RULE = "#e5e7eb"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def near_frontier(results: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n plans off the frontier with the highest SBM scores."""
    return results[~results["On_Frontier"]].nlargest(n, "SBM_Score")


def pareto_2d(results: pd.DataFrame, x: str, y: str,
              direction: dict[str, str]) -> pd.DataFrame:
    """Plans on the Pareto frontier for just two metrics, in drawing order."""
    vals = results[[x, y]].to_numpy(dtype=float)
    hb = np.array([direction[x] == "higher", direction[y] == "higher"])
    front = results[pareto_nondominated(vals, hb)].copy()
    # sort along the x axis from worst to best, so the staircase draws cleanly
    front["_u"] = front[x] if hb[0] else -front[x]
    front["_v"] = front[y] if hb[1] else -front[y]
    return front.sort_values(["_u", "_v"], ascending=[True, False])


def fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.4g}"


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def build_figure(results, direction, names, id_col, x, y, n,
                 show_others, show_line, selected) -> go.Figure:
    eff = results[results["On_Frontier"]]
    near = near_frontier(results, n)
    others = results.drop(index=eff.index.union(near.index))
    nx, ny = names.get(x, x), names.get(y, y)

    hover = ("<b>%{customdata[0]}</b><br>" + nx + ": %{x}<br>" + ny + ": %{y}"
             "<br>SBM score: %{customdata[1]:.3f}<extra></extra>")

    def scatter(df, name, color, size, symbol, opacity):
        return go.Scatter(
            x=df[x], y=df[y], mode="markers", name=name,
            marker=dict(color=color, size=size, symbol=symbol, opacity=opacity,
                        line=dict(color="white", width=1.5)),
            customdata=df[[id_col, "SBM_Score"]].astype(object).to_numpy(),
            hovertemplate=hover,
        )

    fig = go.Figure()
    if show_others and len(others):
        fig.add_trace(scatter(others, f"Other plans ({len(others)})", OTHER, 8, "circle", 0.8))
    if show_line:
        front = pareto_2d(results, x, y, direction)
        fig.add_trace(go.Scatter(
            x=front[x], y=front[y], mode="lines", line_shape="hv",
            line=dict(color=MUTED, width=2, dash="dash"),
            name="Frontier for these 2 metrics only", hoverinfo="skip"))
    if len(near):
        fig.add_trace(scatter(near, f"{len(near)} closest to frontier", NEAR, 11, "circle", 1))
    fig.add_trace(scatter(eff, f"SBM frontier ({len(eff)})", FRONTIER, 14, "diamond", 1))

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
        if direction[col] == "lower":
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
def plan_panel(results, direction, names, id_col, plan_id):
    if plan_id is None:
        return html.Div([
            html.H3("Click a plan", style={"margin": "0 0 8px"}),
            html.P("Click any point to see its metrics and how it ranks against "
                   "the other plans.",
                   style={"color": MUTED, "margin": 0}),
        ])

    match = results[results[id_col].astype(str) == plan_id]
    if match.empty:
        return html.P(f"{plan_id} not found.")
    row = match.iloc[0]
    color = FRONTIER if row["On_Frontier"] else NEAR

    header = html.Div([
        html.H3(plan_id, style={"margin": "0 8px 0 0", "display": "inline"}),
        html.Span("on frontier" if row["On_Frontier"] else "off frontier", style={
            "background": color, "color": "white",
            "borderRadius": "10px", "padding": "2px 10px", "fontSize": "12px",
            "verticalAlign": "middle"}),
        html.Div(f"SBM score {row['SBM_Score']:.3f}",
                 style={"color": MUTED, "marginTop": "6px"}),
    ], style={"marginBottom": "16px"})

    # Metric rows: value and percentile bar
    rows = []
    for col, way in direction.items():
        v = float(row[col])
        allv = results[col].astype(float)
        pct = float((allv <= v).mean() if way == "higher" else (allv >= v).mean())
        rows.append(html.Div([
            html.Div([
                html.Span(f"{names.get(col, col)} {'↑' if way == 'higher' else '↓'}",
                          style={"fontWeight": 600}),
                html.Span(fmt(v), style={"float": "right", "fontVariantNumeric": "tabular-nums"}),
            ]),
            html.Div(html.Div(style={
                "width": f"{pct * 100:.0f}%", "height": "6px",
                "background": color, "borderRadius": "3px"}),
                style={"background": RULE, "borderRadius": "3px", "margin": "6px 0 4px"}),
            html.Div(f"As good as or better than {pct:.0%} of plans",
                     style={"fontSize": "12px", "color": MUTED}),
        ], style={"marginBottom": "14px"}))

    return html.Div([header, *rows])


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def create_app(
    results: pd.DataFrame,
    direction: dict[str, str],
    id_col: str,
    labels: dict[str, str] | None = None,
    n_default: int = 10,
    title: str = "SBM Frontier of Prospective Redistricting Plans",
) -> Dash:
    """Build the Dash app from add_sbm_scores() output."""
    names = labels or {}
    results = results.reset_index(drop=True)
    cols = list(direction)
    n_off = int((~results["On_Frontier"]).sum())
    n_default = min(n_default, n_off)
    n_eff = int(results["On_Frontier"].sum())

    options = [{"label": names.get(c, c), "value": c} for c in cols]
    label = {"fontSize": "12px", "color": MUTED, "marginBottom": "4px", "display": "block"}

    app = Dash(__name__, title="SBM Frontier")
    app.layout = html.Div([
        html.H2(title, style={"margin": "0 0 4px"}),
        html.P(f"{n_eff} of {len(results)} plans are on the SBM frontier (crimson). "
               "Axes are flipped where needed so up and right are always better. "
               "The SBM frontier uses all metrics, so a crimson plan can sit inside "
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
                      dcc.Slider(id="n", min=0, max=max(n_off, 1), step=1,
                                 value=n_default,
                                 marks={0: "0", n_default: str(n_default),
                                        n_off: str(n_off)},
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
        return build_figure(results, direction, names, id_col, x, y, n or 0,
                            "others" in toggles, "line" in toggles, selected)

    @app.callback(
        Output("selected", "data"),
        Input("graph", "clickData"),
        prevent_initial_call=True,
    )
    def _select(click):
        if ctx.triggered_id != "graph" or not click:
            return no_update
        point = click["points"][0]
        if "customdata" not in point:
            return no_update
        return str(point["customdata"][0])

    @app.callback(Output("panel", "children"), Input("selected", "data"))
    def _update_panel(selected):
        return plan_panel(results, direction, names, id_col, selected)

    return app
