"""
Interactive Pareto-frontier explorer for DEA results of redistricting plans.

A Dash app that scores plans with redistricting.DEA.slacks_based_measure:

  * a scatter plot of any two metrics, with every axis oriented so that
    up and right are better
  * for just the two metrics on the axes: the Pareto frontier plans as
    dark points joined by a dashed line, and its convex hull as
    a solid line, as in Gopalan et al. 2026, Figure 2
  * a red x on each plan with DEA score 1 on all metrics
  * optionally, the n plans closest to that frontier (highest DEA scores
    among the other plans) in blue, with a slider for n
  * click any plan to see its metrics, its percentile on each one, and
    its DEA score

Note: a red x uses ALL metrics at once, so it can sit far from the
2-metric lines. That's expected: it earns its place on the frontier with
a metric that isn't on the current axes.

Usage (see examples/frontier_app/mock_plans_app.py):

    results = add_dea_scores(plans, direction)
    app = create_app(results, direction, id_col)
    app.run()

`plans` has one row per plan. `direction` maps each metric column to
"higher" or "lower", the same as score_plans.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html, no_update

from redistricting.DEA.slacks_based_measure import score_plans

# GLPK returns 0.9999999... for some frontier plans.
TOL = 1e-6


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def add_dea_scores(df: pd.DataFrame, direction: dict[str, str]) -> pd.DataFrame:
    """Return a copy of df with DEA_Score and On_Frontier columns.

    DEA_Score comes from score_plans over the columns named in `direction`.
    On_Frontier is True when the score is 1.
    """
    results = df.copy()
    results["DEA_Score"] = score_plans(df[list(direction)], direction)
    results["On_Frontier"] = results["DEA_Score"] >= 1 - TOL
    return results


def convex_hull_2d(results: pd.DataFrame, x: str, y: str,
                   direction: dict[str, str]) -> pd.DataFrame:
    """Plans with DEA score 1 for just metrics x and y, sorted by x.

    These are the corners of the convex hull of the Pareto frontier
    (Gopalan et al. 2026, Figure 2, solid line).

    score_plans needs at least one "higher" metric, so this adds a
    constant "higher" column of 1s. Its weight cancels the model's v >= 1
    floor (the paper's constraint 4), so the result is the plain convex
    hull of Figure 2. It can include plans that score_plans scores below
    1 on the same two metrics, such as a plan that is best on only one.
    """
    metrics = results[[x, y]].copy()
    metrics["_one"] = 1.0
    scores = score_plans(metrics, {x: direction[x], y: direction[y],
                                   "_one": "higher"})
    return results[scores >= 1 - TOL].sort_values([x, y])


def pareto_2d(results: pd.DataFrame, x: str, y: str,
              direction: dict[str, str]) -> pd.DataFrame:
    """Plans that no other plan beats on both x and y, sorted by x.

    This is the bivariate Pareto frontier (Gopalan et al. 2026, Figure 4).
    """
    # flip "lower" metrics so that higher is always better
    x_better = results[x] if direction[x] == "higher" else -results[x]
    y_better = results[y] if direction[y] == "higher" else -results[y]

    # Walk from best x to worst x. A plan is on the frontier when its y
    # beats every plan seen so far, since those all have an x at least
    # as good.
    order = pd.DataFrame({"x": x_better, "y": y_better}).sort_values(
        ["x", "y"], ascending=False)
    keep = []
    best_y = float("-inf")
    for index, y_value in order["y"].items():
        if y_value > best_y:
            keep.append(index)
            best_y = y_value

    return results.loc[keep].sort_values([x, y])


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
    """The n plans off the frontier with the highest DEA scores."""
    return results[~results["On_Frontier"]].nlargest(n, "DEA_Score")


def fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.4g}"


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def build_figure(results, direction, names, id_col, x, y, n,
                 show_others, pareto, hull, selected) -> go.Figure:
    """Draw the scatter plot.

    `pareto` is pareto_2d() output, drawn as bold points joined by a
    dashed line. `hull` is convex_hull_2d() output, drawn as a solid
    line. Pass None to leave either out.
    """
    eff = results[results["On_Frontier"]]
    near = near_frontier(results, n)
    front_index = pareto.index if pareto is not None else pd.Index([])
    others = results.drop(index=near.index.union(front_index))
    nx, ny = names.get(x, x), names.get(y, y)

    hover = ("<b>%{customdata[0]}</b><br>" + nx + ": %{x}<br>" + ny + ": %{y}"
             "<br>DEA score: %{customdata[1]:.3f}<extra></extra>")

    def scatter(df, name, color, size, symbol, opacity, outline=1.5):
        return go.Scatter(
            x=df[x], y=df[y], mode="markers", name=name,
            marker=dict(color=color, size=size, symbol=symbol, opacity=opacity,
                        line=dict(color="white", width=outline)),
            customdata=df[[id_col, "DEA_Score"]].astype(object).to_numpy(),
            hovertemplate=hover,
        )

    # Draw order, back to front: plain plans, the hull, the Pareto
    # frontier, then the all-metric x marks, so nothing important is hidden.
    fig = go.Figure()
    if show_others and len(others):
        fig.add_trace(scatter(others, f"Other plans ({len(others)})", OTHER, 8, "circle", 0.8))
    if len(near):
        fig.add_trace(scatter(near, f"{len(near)} closest to frontier", NEAR, 10, "circle", 1))

    if hull is not None:
        fig.add_trace(go.Scatter(
            x=hull[x], y=hull[y], mode="lines",
            line=dict(color=INK, width=2),
            name=f"Convex hull, these 2 metrics ({len(hull)})",
            hoverinfo="skip"))
    if pareto is not None:
        front = scatter(pareto, f"Pareto frontier, these 2 metrics ({len(pareto)})",
                        INK, 12, "circle", 1)
        front.update(mode="lines+markers",
                     line=dict(color=MUTED, width=2, dash="dash"))
        fig.add_trace(front)

    fig.add_trace(scatter(eff, f"DEA score 1 on all {len(direction)} metrics ({len(eff)})",
                          FRONTIER, 8, "x", 1, outline=0))

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
        html.Div(f"DEA score {row['DEA_Score']:.3f}",
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
    n_default: int = 0,
    title: str = "DEA Frontier of Prospective Redistricting Plans",
) -> Dash:
    """Build the Dash app from add_dea_scores() output."""
    names = labels or {}
    results = results.reset_index(drop=True)
    cols = list(direction)
    n_off = int((~results["On_Frontier"]).sum())
    n_default = min(n_default, n_off)
    n_eff = int(results["On_Frontier"].sum())
    hulls: dict[frozenset[str], pd.DataFrame] = {}  # axis pair -> convex_hull_2d()

    options = [{"label": names.get(c, c), "value": c} for c in cols]
    label = {"fontSize": "12px", "color": MUTED, "marginBottom": "4px", "display": "block"}

    app = Dash(__name__, title="DEA Frontier")
    app.layout = html.Div([
        html.H2(title, style={"margin": "0 0 4px"}),
        html.P("Axes are flipped where needed so up and right are always better. "
               "Dark points on the dashed line: the Pareto frontier for the "
               "two metrics shown (no plan beats them on both). Solid line: "
               "its convex hull. "
               f"Red x: the {n_eff} of {len(results)} plans with DEA score 1 "
               f"on all {len(cols)} metrics. An x can sit far from the lines, "
               "because it is on the frontier through a metric not shown.",
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
                                   {"label": " Show convex hull", "value": "hull"},
                                   {"label": " Show Pareto frontier", "value": "pareto"}],
                          value=["others", "hull", "pareto"],
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
        hull = pareto = None
        if "pareto" in toggles and x != y:
            pareto = pareto_2d(results, x, y, direction)
        if "hull" in toggles and x != y:
            # one GLPK solve per plan, so score each axis pair only once
            pair = frozenset((x, y))
            if pair not in hulls:
                hulls[pair] = convex_hull_2d(results, x, y, direction)
            hull = hulls[pair].sort_values([x, y])
        return build_figure(results, direction, names, id_col, x, y, n or 0,
                            "others" in toggles, pareto, hull, selected)

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
