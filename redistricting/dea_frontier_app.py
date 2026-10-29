"""
Interactive Pareto-frontier explorer for DEA results of redistricting plans.

A Dash app that scores plans with redistricting.DEA.slacks_based_measure:

  * a scatter plot of any two metrics, with every axis oriented so that
    up and right are better
  * for just the two metrics on the axes: the Pareto frontier plans as
    dark points joined by a dashed line, and its convex hull as
    a solid line, as in Gopalan et al. 2026, Figure 2
  * a red x on each plan on the DEA frontier, which uses all metrics
    together (DEA score 1)
  * optionally, the n plans closest to that frontier (highest DEA scores
    among the other plans) in blue, with a slider for n
  * click any plan to see its metrics, its percentile on each one, and
    its DEA score

Note: a red x uses ALL metrics at once, so it can sit far from the
2-metric lines. That's expected: it earns its place on the frontier with
a metric that isn't on the current axes.

Usage (see examples/frontier_app/app.py):

    results = add_dea_scores(plans, direction)
    app = create_app({"My plans": Dataset(results, direction, id_col)})
    app.run()

Pass more than one Dataset to switch between them with the "Data" dropdown.

`plans` has one row per plan. `direction` maps each metric column to
"higher" or "lower", the same as score_plans.
"""

# NOTES
# get rid of this annotations statement
# scale of plot should encompass entire set, even when subset is shown so it doesn't jump around

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html, no_update

from redistricting.DEA import slacks_based_measure
from redistricting.DEA.slacks_based_measure import score_plans

# GLPK returns 0.9999999... for some frontier plans.
TOL = 1e-6


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def add_dea_scores(df: pd.DataFrame, direction: dict[str, str],
                   cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Return a copy of df with DEA_Score and On_Frontier columns.

    DEA_Score comes from score_plans over the columns named in `direction`.
    On_Frontier is True when the score is 1.

    score_plans runs one solve per plan, which takes minutes for thousands
    of plans. With `cache_dir`, the scores are saved there and read back on
    the next call with the same metric values, directions, and DEA code.
    """
    metrics = df[list(direction)]

    if cache_dir is None:
        scores = score_plans(metrics, direction).to_numpy()
    else:
        path = Path(cache_dir) / f"dea_scores_{_cache_key(metrics, direction)}.csv"
        if path.exists():
            scores = pd.read_csv(path)["DEA_Score"].to_numpy()
        else:
            scores = score_plans(metrics, direction).to_numpy()
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"DEA_Score": scores}).to_csv(path, index=False)

    results = df.copy()
    results["DEA_Score"] = scores
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

    Only the Pareto plans are scored. Every weight in the model is above
    0, so a beaten plan cannot score 1, and its constraint is implied by
    the plan that beats it. On the class data this takes one hull from
    about 100 s to 0.06 s, with the same result.
    """
    front = pareto_2d(results, x, y, direction)
    metrics = front[[x, y]].copy()
    metrics["_one"] = 1.0
    scores = score_plans(metrics, {x: direction[x], y: direction[y],
                                   "_one": "higher"})
    return front[scores >= 1 - TOL].sort_values([x, y])


def pareto_2d(results: pd.DataFrame, x: str, y: str,
              direction: dict[str, str]) -> pd.DataFrame:
    """Plans that no other plan beats on both x and y, sorted by x.

    This is the bivariate Pareto frontier (Gopalan et al. 2026, Figure 4).
    """
    # Negate "lower" metrics so that higher is always better: 3 splits
    # beats 10, and -3 > -10. The order of plans stays the same, so the
    # loop below needs just one rule.
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
    """Format a metric value: whole numbers with commas, else 4 digits."""
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

    # The near and Pareto plans get their own traces, so leave them out of
    # the gray ones. The red x plans stay in: the x is drawn on top.
    front_index = pareto.index if pareto is not None else pd.Index([])
    others = results.drop(index=near.index.union(front_index))
    nx, ny = names.get(x, x), names.get(y, y)

    hover = ("<b>%{customdata[0]}</b><br>" + nx + ": %{x}<br>" + ny + ": %{y}"
             "<br>DEA score: %{customdata[1]:.3f}<extra></extra>")

    # customdata carries the plan ID and score with each point, for the
    # hover text and for _select. astype(object) keeps each value's own
    # type: without it, numpy turns an int ID 17 into 17.0 to match the
    # float score, and "17.0" would not match the ID "17" on a click.
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

    fig.add_trace(scatter(eff, f"DEA frontier, all {len(direction)} metrics ({len(eff)})",
                          FRONTIER, 8, "x", 1, outline=0))

    # Ring and label the clicked plan. _select stores the ID as a string,
    # so compare against the IDs as strings.
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
    """Return the side panel for one plan, or a hint when none is picked."""
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

    # Metric rows: value and percentile bar. The percentile is the share of
    # plans this one ties or beats, so the best plan gets 100%.
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
@dataclass
class Dataset:
    """One set of scored plans the app can show."""

    results: pd.DataFrame        # add_dea_scores() output
    direction: dict[str, str]    # metric column -> "higher" or "lower"
    id_col: str                  # column that names each plan
    labels: dict[str, str] = field(default_factory=dict)  # column -> label


def create_app(
    datasets: dict[str, Dataset],
    n_default: int = 0,
    title: str = "DEA Frontier of Prospective Redistricting Plans",
) -> Dash:
    """Build the Dash app. A "Data" dropdown picks a dataset by name.

    The axes, the slider, the intro text, and the side panel all follow
    the chosen dataset, since datasets can have different metrics.
    """
    # Give each dataset a clean 0..n-1 index. build_figure finds plans by
    # index label (drop, union), so the labels must be unique.
    datasets = {name: Dataset(data.results.reset_index(drop=True), data.direction,
                              data.id_col, data.labels)
                for name, data in datasets.items()}
    first = next(iter(datasets))
    label = {"fontSize": "12px", "color": MUTED, "marginBottom": "4px", "display": "block"}

    app = Dash(__name__, title="DEA Frontier")
    app.layout = html.Div([
        html.H2(title, style={"margin": "0 0 4px"}),
        html.P(id="intro", style={"color": MUTED, "margin": "0 0 16px", "maxWidth": "900px"}),

        # Controls. The axis options and the slider range are filled in by
        # _choose_data, because they depend on the dataset.
        html.Div([
            html.Div([html.Label("Data", style=label),
                      dcc.Dropdown(id="data", options=list(datasets), value=first,
                                   clearable=False)],
                     style={"width": "200px"}),
            html.Div([html.Label("X axis", style=label),
                      dcc.Dropdown(id="x", clearable=False)],
                     style={"width": "200px"}),
            html.Div([html.Label("Y axis", style=label),
                      dcc.Dropdown(id="y", clearable=False)],
                     style={"width": "200px"}),
            html.Div([html.Label("Plans closest to frontier (n)", style=label),
                      dcc.Slider(id="n", min=0, max=1, step=1, value=0,
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

        # The ID of the clicked plan. The figure and the panel both read it.
        dcc.Store(id="selected"),
    ], style={"fontFamily": FONT, "color": INK, "padding": "20px 24px",
              "maxWidth": "1400px", "margin": "0 auto"})

    # Each callback runs when any of its Inputs changes, and its return
    # values go to its Outputs, in order.

    # A new dataset: reset the axes to its first two metrics, size the
    # slider to its plans off the frontier, and rewrite the intro.
    @app.callback(
        Output("x", "options"), Output("x", "value"),
        Output("y", "options"), Output("y", "value"),
        Output("n", "max"), Output("n", "marks"), Output("n", "value"),
        Output("intro", "children"),
        Input("data", "value"),
    )
    def _choose_data(name):
        data = datasets[name]
        cols = list(data.direction)
        options = [{"label": data.labels.get(c, c), "value": c} for c in cols]
        n_off = int((~data.results["On_Frontier"]).sum())
        # max(n_off, 1) keeps the slider's max above its min of 0.
        return (options, cols[0], options, cols[1] if len(cols) > 1 else cols[0],
                max(n_off, 1), {0: "0", n_off: str(n_off)}, min(n_default, n_off),
                _intro(data))

    @app.callback(
        Output("graph", "figure"),
        Input("data", "value"), Input("x", "value"), Input("y", "value"),
        Input("n", "value"), Input("toggles", "value"), Input("selected", "data"),
    )
    def _update_figure(name, x, y, n, toggles, selected):
        data = datasets[name]
        # while switching datasets, x and y can still name the old metrics
        if x not in data.direction or y not in data.direction:
            return no_update
        toggles = toggles or []

        # With x == y every plan sits on one diagonal line, so there is
        # no 2-metric frontier to draw.
        hull = pareto = None
        if "pareto" in toggles and x != y:
            pareto = pareto_2d(data.results, x, y, data.direction)
        if "hull" in toggles and x != y:
            hull = convex_hull_2d(data.results, x, y, data.direction)
        return build_figure(data.results, data.direction, data.labels, data.id_col,
                            x, y, n or 0, "others" in toggles, pareto, hull, selected)

    @app.callback(
        Output("selected", "data"),
        Input("graph", "clickData"), Input("data", "value"),
        prevent_initial_call=True,
    )
    def _select(click, _name):
        # a new dataset has different plans, so clear the selection
        if ctx.triggered_id == "data":
            return None
        if not click:
            return no_update
        point = click["points"][0]
        # The hull line and the selection ring have no customdata, so a
        # click on them keeps the current selection.
        if "customdata" not in point:
            return no_update
        return str(point["customdata"][0])

    @app.callback(Output("panel", "children"),
                  Input("selected", "data"), Input("data", "value"))
    def _update_panel(selected, name):
        data = datasets[name]
        return plan_panel(data.results, data.direction, data.labels, data.id_col,
                          selected)

    return app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _intro(data: Dataset) -> str:
    """Return the text above the chart for one dataset."""
    n_eff = int(data.results["On_Frontier"].sum())
    return ("Axes are flipped where needed so up and right are always better. "
            "Dark points on the dashed line: the Pareto frontier for the two "
            "metrics shown (no plan beats them on both). Solid line: its convex "
            f"hull. Red x: the {n_eff} of {len(data.results)} plans on the DEA "
            f"frontier, which uses all {len(data.direction)} metrics together. "
            "An x can sit far from the lines, because it is on the frontier "
            "through a metric not shown.")


def _cache_key(metrics: pd.DataFrame, direction: dict[str, str]) -> str:
    """Return a short hash of everything that changes the DEA scores.

    That is the metric values in row order, the column names, the
    directions, and the source of the DEA code, so editing any of them
    makes a new cache file instead of reading a stale one.
    """
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(metrics, index=False).to_numpy().tobytes())
    digest.update(json.dumps([list(metrics.columns), direction]).encode())
    digest.update(Path(slacks_based_measure.__file__).read_bytes())
    return digest.hexdigest()[:16]
