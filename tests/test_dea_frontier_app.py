"""
Tests for redistricting/dea_frontier_app.py.

Run from the top folder of the project (the one that holds both the
redistricting/ and tests/ folders):

    python -m pytest tests/test_dea_frontier_app.py -v

Scoring uses the real score_plans, so Pyomo and GLPK must be installed
(they are in environment.yml).

Most tests use plans A, B, C from the SBM tests, whose scores were worked
out by hand in tests/DEA/test_slacks_based_measure.py:

    plan  compactness (higher)  county splits (lower)  DEA score
    A     0.40                  10                     1 / 1.575 = 0.635
    B     0.50                   6                     1 (beats A and C)
    C     0.50                   8                     8 / 9 = 0.889
"""

from contextvars import copy_context

import pandas as pd
from dash import no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
from pytest import approx

from redistricting import dea_frontier_app
from redistricting.dea_frontier_app import (
    Dataset,
    _cache_key,
    _intro,
    add_dea_scores,
    axis_range,
    build_figure,
    convex_hull_2d,
    create_app,
    fmt,
    near_frontier,
    pareto_2d,
    plan_panel,
)

# Plans A, B, C from the SBM tests: B beats A and C.
plans = pd.DataFrame({
    "plan_id": ["A", "B", "C"],
    "compactness": [0.40, 0.50, 0.50],
    "county_splits": [10, 6, 8],
})

direction = {"compactness": "higher", "county_splits": "lower"}

labels = {"compactness": "Compactness (PP)", "county_splits": "County Splits"}


# --------------------------------------------------------------------------
# add_dea_scores
# --------------------------------------------------------------------------
def test_add_dea_scores_marks_frontier():
    results = add_dea_scores(plans, direction)

    assert list(results["On_Frontier"]) == [False, True, False]


def test_add_dea_scores_values():
    results = add_dea_scores(plans, direction)

    assert list(results["DEA_Score"]) == approx([1 / 1.575, 1, 8 / 9])


def test_add_dea_scores_does_not_change_input():
    before = plans.copy()

    add_dea_scores(plans, direction)

    pd.testing.assert_frame_equal(plans, before)


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


def test_add_dea_scores_makes_missing_cache_folder(tmp_path):
    folder = tmp_path / "new" / "folder"

    add_dea_scores(plans, direction, folder)

    assert len(list(folder.glob("dea_scores_*.csv"))) == 1


def test_add_dea_scores_ignores_id_column():
    results = add_dea_scores(plans, direction)

    assert list(results["plan_id"]) == ["A", "B", "C"]


# --------------------------------------------------------------------------
# _cache_key
# --------------------------------------------------------------------------
def test_cache_key_same_input_same_key():
    metrics = plans[list(direction)]

    assert _cache_key(metrics, direction) == _cache_key(metrics.copy(), direction)
    assert len(_cache_key(metrics, direction)) == 16


def test_cache_key_changes_with_direction():
    metrics = plans[list(direction)]
    flipped = {"compactness": "higher", "county_splits": "higher"}

    assert _cache_key(metrics, direction) != _cache_key(metrics, flipped)


def test_cache_key_changes_with_row_order():
    # the cache stores scores by row position, so a new order needs a new file
    metrics = plans[list(direction)]

    assert _cache_key(metrics, direction) != _cache_key(metrics.iloc[::-1], direction)


# --------------------------------------------------------------------------
# convex_hull_2d and pareto_2d
# --------------------------------------------------------------------------
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


def test_pareto_2d_drops_plan_tied_on_one_metric():
    # C ties B on compactness but has more county splits, so B beats it
    line = pareto_2d(plans, "county_splits", "compactness", direction)

    assert "C" not in list(line["plan_id"])


def test_pareto_2d_two_higher_metrics():
    # X is best on m1, Y is best on m2, Z is beaten by both
    df = pd.DataFrame({"plan_id": ["X", "Y", "Z"],
                       "m1": [3, 1, 1], "m2": [1, 3, 1]})

    line = pareto_2d(df, "m1", "m2", {"m1": "higher", "m2": "higher"})

    assert list(line["plan_id"]) == ["Y", "X"]  # sorted by m1


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


def test_hull_corners_are_on_pareto_frontier():
    pareto = pareto_2d(figure_2, "pop_dev", "splits", figure_2_direction)
    hull = convex_hull_2d(figure_2, "pop_dev", "splits", figure_2_direction)

    assert set(hull["plan_id"]) <= set(pareto["plan_id"])


# --------------------------------------------------------------------------
# Small helpers: near_frontier, fmt, _intro
# --------------------------------------------------------------------------
def test_near_frontier_picks_highest_scores_off_frontier():
    results = add_dea_scores(plans, direction)

    assert list(near_frontier(results, 1)["plan_id"]) == ["C"]
    assert list(near_frontier(results, 2)["plan_id"]) == ["C", "A"]


def test_near_frontier_never_includes_frontier_plans():
    results = add_dea_scores(plans, direction)

    # asking for more plans than there are just returns the ones off it
    assert "B" not in list(near_frontier(results, 10)["plan_id"])


def test_fmt():
    assert fmt(6) == "6"
    assert fmt(12345.0) == "12,345"  # whole numbers get commas
    assert fmt(0.5) == "0.5"
    assert fmt(0.123456) == "0.1235"  # 4 significant digits
    assert fmt(0) == "0"


def test_intro_counts_frontier_plans():
    data = Dataset(add_dea_scores(plans, direction), direction, "plan_id")

    text = _intro(data)

    assert "the 1 of 3 plans" in text
    assert "all 2 metrics" in text


# --------------------------------------------------------------------------
# build_figure
# --------------------------------------------------------------------------
def _figure(n=0, show_others=True, line=True, selected=None, x="compactness",
            y="county_splits", results=None):
    """Build the figure for plans A, B, C with the given settings."""
    results = add_dea_scores(plans, direction) if results is None else results
    pareto = pareto_2d(results, x, y, direction) if line else None
    hull = convex_hull_2d(results, x, y, direction) if line else None
    return build_figure(results, direction, labels, "plan_id", x, y, n,
                        show_others, pareto, hull, selected)


def _trace(fig, name_start):
    """The trace whose legend name starts with name_start, or None."""
    for trace in fig.data:
        if trace.name and trace.name.startswith(name_start):
            return trace
    return None


def test_figure_traces():
    fig = _figure()

    assert [t.name for t in fig.data] == [
        "Other plans (2)",
        "Convex hull, these 2 metrics (1)",
        "Pareto frontier, these 2 metrics (1)",
        "DEA frontier, all 2 metrics (1)",
    ]


def test_figure_red_x_marks_frontier_plans():
    red_x = _trace(_figure(), "DEA frontier")

    assert red_x.marker.symbol == "x"
    assert [row[0] for row in red_x.customdata] == ["B"]


def test_figure_near_frontier_plans():
    fig = _figure(n=1)

    near = _trace(fig, "1 closest")
    assert [row[0] for row in near.customdata] == ["C"]
    # C is drawn in blue, so it leaves the gray "other plans"
    others = _trace(fig, "Other plans")
    assert [row[0] for row in others.customdata] == ["A"]


def test_figure_hide_other_plans():
    fig = _figure(show_others=False)

    assert _trace(fig, "Other plans") is None


def test_figure_without_lines():
    fig = _figure(line=False)

    assert _trace(fig, "Convex hull") is None
    assert _trace(fig, "Pareto frontier") is None
    # with no Pareto line, B stays in the gray plans too
    assert len(_trace(fig, "Other plans").customdata) == 3


def test_figure_lower_is_better_axis_is_flipped():
    fig = _figure()

    x_start, x_end = fig.layout.xaxis.range   # compactness: higher is better
    y_start, y_end = fig.layout.yaxis.range   # county splits: lower is better
    assert x_start < x_end
    assert y_start > y_end


def test_figure_axes_cover_every_plan():
    fig = _figure()

    x_start, x_end = fig.layout.xaxis.range
    y_end, y_start = fig.layout.yaxis.range   # flipped, so high comes first
    assert x_start < 0.40 and x_end > 0.50
    assert y_start < 6 and y_end > 10


def test_figure_scale_stays_the_same_when_plans_are_hidden():
    # the professor's request: the chart keeps the scale it has when
    # "Show other plans" is ticked, whatever else is shown
    full = _figure()
    for fig in [_figure(show_others=False), _figure(show_others=False, line=False),
                _figure(n=2, show_others=False), _figure(selected="A")]:
        assert fig.layout.xaxis.range == full.layout.xaxis.range
        assert fig.layout.yaxis.range == full.layout.yaxis.range


def test_axis_range_pads_both_ends():
    values = pd.Series([0.0, 10.0])

    assert axis_range(values, "higher") == approx([-0.5, 10.5])
    assert axis_range(values, "lower") == approx([10.5, -0.5])
    assert axis_range(values, "higher", pad=0) == approx([0, 10])


def test_axis_range_with_one_value():
    # all plans tie, so there is no span to pad by; pad around the value
    start, end = axis_range(pd.Series([4.0, 4.0]), "higher")

    assert start < 4 < end


def test_axis_range_all_zero():
    start, end = axis_range(pd.Series([0.0, 0.0]), "higher")

    assert start < 0 < end


def test_figure_axis_titles_use_labels():
    fig = _figure()

    assert fig.layout.xaxis.title.text.startswith("Compactness (PP)")
    assert "better →" in fig.layout.xaxis.title.text
    assert fig.layout.yaxis.title.text.startswith("County Splits")


def test_figure_keeps_zoom_until_axes_change():
    assert _figure().layout.uirevision == "compactness|county_splits"


def test_figure_selected_plan_gets_a_ring():
    fig = _figure(selected="A")

    ring = _trace(fig, "Selected")
    assert list(ring.text) == ["A"]
    assert list(ring.x) == [0.40]


def test_figure_unknown_selection_is_ignored():
    fig = _figure(selected="not a plan")

    assert _trace(fig, "Selected") is None


def test_figure_number_ids_stay_whole():
    # A plan ID of 17 must come back as 17, not 17.0, or clicks won't match
    numbered = plans.assign(plan_id=[17, 18, 19])
    results = add_dea_scores(numbered, direction)

    fig = _figure(results=results, selected="17")

    red_x = _trace(fig, "DEA frontier")
    assert str(red_x.customdata[0][0]) == "18"
    assert _trace(fig, "Selected") is not None


# --------------------------------------------------------------------------
# plan_panel
# --------------------------------------------------------------------------
def _panel_text(component):
    """All the text in a Dash component and everything inside it."""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(_panel_text(c) for c in component)
    children = getattr(component, "children", None)
    return "" if children is None else _panel_text(children)


def _panel(plan_id):
    results = add_dea_scores(plans, direction)
    return _panel_text(plan_panel(results, direction, labels, "plan_id", plan_id))


def test_panel_with_nothing_selected():
    assert "Click a plan" in _panel(None)


def test_panel_unknown_plan():
    assert "Z not found" in _panel("Z")


def test_panel_frontier_plan():
    text = _panel("B")

    assert "on frontier" in text
    assert "DEA score 1.000" in text
    # B is best on both metrics
    assert text.count("As good as or better than 100% of plans") == 2


def test_panel_plan_off_frontier():
    text = _panel("A")

    assert "off frontier" in text
    assert "DEA score 0.635" in text
    # A is worst on both metrics: it only ties itself, 1 of 3 plans
    assert text.count("As good as or better than 33% of plans") == 2


def test_panel_shows_labels_and_values():
    text = _panel("C")

    assert "Compactness (PP) ↑" in text
    assert "County Splits ↓" in text
    assert "0.5" in text and "8" in text


# --------------------------------------------------------------------------
# create_app
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# The app's callbacks: what happens when someone uses the controls
# --------------------------------------------------------------------------
def _app(n_default=0, **extra):
    """An app with plans A, B, C as "Main", plus any extra datasets."""
    data = Dataset(add_dea_scores(plans, direction), direction, "plan_id", labels)
    return create_app({"Main": data, **extra}, n_default=n_default)


def _callback(app, output):
    """The plain function behind the callback that fills `output`.

    For example, "graph.figure" gives _update_figure.
    """
    for key, entry in app.callback_map.items():
        if output in key:
            return entry["callback"].__wrapped__
    raise KeyError(output)


def _run_triggered_by(prop_id, func, *args):
    """Call a callback as if the control `prop_id` had just changed.

    Dash normally sets this up when a callback runs in the browser, and
    _select reads it to tell a click from a dataset change.
    """
    def run():
        context_value.set(AttributeDict(triggered_inputs=[{"prop_id": prop_id}]))
        return func(*args)
    return copy_context().run(run)


def test_choose_data_fills_controls():
    choose = _callback(_app(), "x.options")

    x_opts, x, y_opts, y, n_max, marks, n, intro = choose("Main")

    assert x_opts == [{"label": "Compactness (PP)", "value": "compactness"},
                      {"label": "County Splits", "value": "county_splits"}]
    assert y_opts == x_opts
    assert (x, y) == ("compactness", "county_splits")
    assert n_max == 2                  # A and C are off the frontier
    assert marks == {0: "0", 2: "2"}
    assert n == 0
    assert "1 of 3 plans" in intro


def test_choose_data_slider_start_is_capped():
    choose = _callback(_app(n_default=50), "x.options")

    assert choose("Main")[6] == 2      # only 2 plans are off the frontier


def test_choose_data_all_plans_on_frontier():
    # X is best on m1 and Y on m2, so both are on the frontier
    all_eff = pd.DataFrame({"plan_id": ["X", "Y"], "m1": [1.0, 2.0], "m2": [2.0, 1.0]})
    both = {"m1": "higher", "m2": "higher"}
    extra = Dataset(add_dea_scores(all_eff, both), both, "plan_id")
    choose = _callback(_app(Both=extra), "x.options")

    n_max = choose("Both")[4]

    assert n_max == 1  # kept above the slider's min of 0


def test_choose_data_one_metric():
    one = pd.DataFrame({"plan_id": ["X", "Y"], "score": [0.3, 0.6]})
    one_dir = {"score": "higher"}
    extra = Dataset(add_dea_scores(one, one_dir), one_dir, "plan_id")
    choose = _callback(_app(One=extra), "x.options")

    _, x, _, y, *_ = choose("One")

    assert x == y == "score"


def test_update_figure_draws_chart():
    update = _callback(_app(), "graph.figure")

    fig = update("Main", "compactness", "county_splits", 1,
                 ["others", "hull", "pareto"], None)

    assert _trace(fig, "1 closest") is not None
    assert _trace(fig, "Convex hull") is not None


def test_update_figure_waits_while_dataset_switches():
    update = _callback(_app(), "graph.figure")

    # x and y still name metrics from a different dataset
    assert update("Main", "old_metric", "county_splits", 0, [], None) is no_update


def test_update_figure_same_metric_on_both_axes():
    update = _callback(_app(), "graph.figure")

    fig = update("Main", "compactness", "compactness", 0, ["hull", "pareto"], None)

    assert _trace(fig, "Convex hull") is None
    assert _trace(fig, "Pareto frontier") is None


def test_update_figure_all_toggles_off():
    update = _callback(_app(), "graph.figure")

    # Dash sends None when every box is unchecked
    fig = update("Main", "compactness", "county_splits", None, None, None)

    assert [t.name for t in fig.data] == ["DEA frontier, all 2 metrics (1)"]


def test_update_figure_with_repeated_index_labels():
    # create_app renumbers rows, so repeated index labels can't confuse
    # build_figure when it drops the near-frontier plans
    repeated = add_dea_scores(plans, direction).set_axis([5, 5, 5])
    app = create_app({"Main": Dataset(repeated, direction, "plan_id")})
    update = _callback(app, "graph.figure")

    fig = update("Main", "compactness", "county_splits", 1, ["others"], None)

    assert len(_trace(fig, "Other plans").customdata) == 2


def test_select_click_on_a_plan():
    select = _callback(_app(), "selected.data")
    click = {"points": [{"customdata": ["C", 0.889]}]}

    assert _run_triggered_by("graph.clickData", select, click, "Main") == "C"


def test_select_number_id_comes_back_as_text():
    select = _callback(_app(), "selected.data")
    click = {"points": [{"customdata": [17, 1.0]}]}

    assert _run_triggered_by("graph.clickData", select, click, "Main") == "17"


def test_select_click_on_a_line_keeps_selection():
    select = _callback(_app(), "selected.data")
    click = {"points": [{"x": 0.5, "y": 6}]}  # the hull line has no customdata

    assert _run_triggered_by("graph.clickData", select, click, "Main") is no_update


def test_select_no_click():
    select = _callback(_app(), "selected.data")

    assert _run_triggered_by("graph.clickData", select, None, "Main") is no_update


def test_select_cleared_by_new_dataset():
    select = _callback(_app(), "selected.data")
    click = {"points": [{"customdata": ["C", 0.889]}]}

    assert _run_triggered_by("data.value", select, click, "Main") is None


def test_update_panel():
    panel = _callback(_app(), "panel.children")

    assert "Click a plan" in _panel_text(panel(None, "Main"))
    assert "on frontier" in _panel_text(panel("B", "Main"))


def _fail_if_called(*args):
    raise AssertionError("score_plans ran instead of reading the cache")
