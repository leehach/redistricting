"""Score redistricting plans with the slacks-based measure (SBM) of DEA.

A slack is the gap between a plan and a better plan on one metric. It
says how much better the plan could be on that metric.

    ┌──────┬────────────────────┬─────────────────────┐
    │ Plan │ Compactness (good) │ County splits (bad) │
    ├──────┼────────────────────┼─────────────────────┤
    │ A    │ 0.40               │ 10                  │
    ├──────┼────────────────────┼─────────────────────┤
    │ B    │ 0.50               │ 6                   │
    └──────┴────────────────────┴─────────────────────┘

Measured against B, A has two slacks: 0.10 on compactness and 4 on
county splits.

SBM (Kaoru Tone, 2001) turns each slack into a share of the plan's own
value, then combines the shares into one score. Bigger gaps give a
lower score:

    compactness:    0.10 / 0.40 = 25%
    county splits:     4 / 10   = 40%
    score = 1 / (1 + (0.25 + 0.40) / 2) = 0.7547

This model also keeps the bar at 1 or more (see `baseline_constraint`),
which lowers A's score to 0.6349.

With two plans you can do this by hand. With many plans, you do not
know which plans (or mix of plans) to measure against. So this module
builds an optimization model with Pyomo and solves it with GLPK, once
per plan. Pyomo is an open-source modeling library: you describe the
problem in Python, and it sends the problem to an external solver.
"""

from pyomo.environ import (
    AbstractModel,
    Constraint,
    Objective,
    RangeSet,
    Var,
    check_optimal_termination,
    maximize,
    value,
)
from pyomo.opt import SolverFactory
import pandas as pd

DIRECTIONS = {"higher", "lower"}


def score_plans(metrics: pd.DataFrame, direction: dict[str, str]) -> pd.Series:
    """Return the SBM score of each plan (row) in `metrics`.

    `metrics` has one row per plan and one column per metric. `direction`
    maps every column to "higher" (higher is better) or "lower" (lower is
    better). The result has the same index as `metrics`.
    """
    missing = set(metrics.columns) - set(direction)
    if missing:
        raise ValueError(f"No direction given for metrics: {sorted(missing)}")

    unknown = set(direction) - set(metrics.columns)
    if unknown:
        raise ValueError(f"Direction given for unknown metrics: {sorted(unknown)}")

    invalid = set(direction.values()) - DIRECTIONS
    if invalid:
        raise ValueError(
            f"Direction must be 'higher' or 'lower', got: {sorted(invalid)}"
        )

    higher = [name for name in metrics.columns if direction[name] == "higher"]
    lower = [name for name in metrics.columns if direction[name] == "lower"]

    # with no "higher" metric, the model caps every score at 0
    if not higher:
        raise ValueError("At least one metric must have direction 'higher'")

    # `> 0` is False for NaN, so this also catches blank cells
    if not (metrics > 0).all().all():
        raise ValueError("All metric values must be greater than 0")

    higher_metrics = metrics[higher]
    lower_metrics = metrics[lower]

    scores = [
        calculate_efficiency(higher_metrics, lower_metrics, row)
        for row in range(1, len(metrics) + 1)
    ]

    return pd.Series(scores, index=metrics.index, name="sbm_score")


def calculate_efficiency(good_plans, bad_plans, current_row):
    """Return the SBM score of one plan.

    `good_plans` and `bad_plans` have one row per plan, in the same order.
    Good columns are higher-is-better; bad columns are lower-is-better.
    `current_row` is the plan to score, counted from 1 (`iHat` in the
    original code). score_plans calls this once per row.

    Raises ValueError for mismatched rows or values of 0 or less, and
    RuntimeError if the solver does not reach an optimal solution.
    """
    (good_plan_num_rows, good_plan_num_cols) = good_plans.shape
    (bad_plan_number_rows, bad_plan_number_cols) = bad_plans.shape

    if good_plan_num_rows != bad_plan_number_rows:
        raise ValueError("Number of rows in good and bad plans are not equal!")

    good_plans_dict = _make_dict(good_plans)
    bad_plans_dict = _make_dict(bad_plans)

    # a 0 turns a floor constraint into `score <= 0`, which breaks DEA
    if any(v <= 0 for v in good_plans_dict.values()) or any(
        v <= 0 for v in bad_plans_dict.values()
    ):
        raise ValueError("All metric values must be greater than 0")

    # An AbstractModel is a template: declare the parts now, and Pyomo
    # builds them later in create_instance.
    DEA_model = AbstractModel()

    # S in the SBM math: the number of good metrics plus bad metrics
    total_metrics = good_plan_num_cols + bad_plan_number_cols

    # Sets (index ranges), counted from 1. The attribute names are ours to
    # choose; Pyomo finds each part by its type (RangeSet, Var, ...).
    DEA_model.good_cols = RangeSet(1, good_plan_num_cols)  # good metrics
    DEA_model.bad_cols = RangeSet(1, bad_plan_number_cols)  # bad metrics

    # one index per plan (row) - good and bad have the same rows, checked above
    DEA_model.plans = RangeSet(1, good_plan_num_rows)

    # Variables are the numbers the solver picks to get the best objective
    # while keeping every constraint. Var(some_set) makes one per item in
    # the set (indexed); Var() makes just one.
    DEA_model.good_weights = Var(DEA_model.good_cols)  # one weight per metric
    DEA_model.bad_weights = Var(DEA_model.bad_cols)  # one weight per metric
    DEA_model.baseline = Var()  # v, the bar - see principal_constraint
    DEA_model.efficiency_score = Var()  # the SBM score

    # A "rule" is a function that returns an expression. Pyomo calls it to
    # build an Objective or a Constraint. This one says: maximize the score.
    def obj_rule(model):
        return model.efficiency_score

    DEA_model.obj = Objective(rule=obj_rule, sense=maximize)

    # For the scored plan only: score = 1 + G - B - v, where G and B are
    # the plan's metrics times their weights. The solver picks the weights.
    # A lower bar v gives a higher score.
    def efficiency_constraint_rule(model):
        return (
            model.efficiency_score
            + model.baseline
            - sum(
                good_plans_dict[current_row, i] * model.good_weights[i]
                for i in model.good_cols
            )
            + sum(
                bad_plans_dict[current_row, j] * model.bad_weights[j]
                for j in model.bad_cols
            )
            == 1.0
        )

    DEA_model.efficiency_constraint = Constraint(rule=efficiency_constraint_rule)

    # The core ("principal") constraint: with the same weights, no plan's
    # G - B can go above the bar v. Indexed by `plans`, so Pyomo makes one
    # constraint per plan. `i` and `j` walk across that plan's good and bad
    # columns. This is what caps every score at 1.
    def principal_constraint_rule(model, plan):
        return (
            sum(
                good_plans_dict[plan, i] * model.good_weights[i]
                for i in model.good_cols
            )
            - model.baseline
            - sum(
                bad_plans_dict[plan, j] * model.bad_weights[j] for j in model.bad_cols
            )
            <= 0.0
        )

    DEA_model.principal_constraint = Constraint(
        DEA_model.plans, rule=principal_constraint_rule
    )

    # The bar v cannot go below 1. A single constraint can take `expr=`
    # directly instead of a rule (efficiency_constraint could too).
    DEA_model.baseline_constraint = Constraint(expr=DEA_model.baseline >= 1.0)

    # Floors: S * value * weight >= score, so each metric adds at least
    # score / S. While the score is above 0, no weight can be 0, so the
    # solver cannot ignore a metric, and every slack counts.

    # `metric` is the number of one metric (column). Pyomo calls the rule once
    # for each value in good_cols, so there is one floor per good metric.
    def good_floor_constraint_rule(model, metric):
        return (
            total_metrics * good_plans_dict[current_row, metric]
        ) * model.good_weights[metric] - model.efficiency_score >= 0.0

    DEA_model.good_floor_constraint = Constraint(
        DEA_model.good_cols, rule=good_floor_constraint_rule
    )

    def bad_floor_constraint_rule(model, metric):
        return (
            total_metrics * bad_plans_dict[current_row, metric]
        ) * model.bad_weights[metric] - model.efficiency_score >= 0.0

    DEA_model.bad_floor_constraint = Constraint(
        DEA_model.bad_cols, rule=bad_floor_constraint_rule
    )

    # build a concrete, solvable model from the template
    model = DEA_model.create_instance()

    # GLPK (GNU Linear Programming Kit) is a free, open-source solver, so it
    # needs no license. It installs from conda-forge (see environment.yml).
    Opt = SolverFactory("glpk")

    # solve, and get a report back (status, termination condition, ...)
    Soln = Opt.solve(model)

    # a valid model always solves to optimal, so anything else means
    # the model or the solver is broken - stop instead of writing a fake score
    if not check_optimal_termination(Soln):
        raise RuntimeError(
            f"Plan {current_row} did not solve: {Soln.solver.termination_condition}"
        )

    return value(model.obj)


def _make_dict(df):
    """Turn a DataFrame into a dict that maps (row, col) to its value.

    Both numbers start at 1, to match the model's RangeSets:

        0.40,0.70        {(1, 1): 0.40, (1, 2): 0.70,   # plan 1
        0.50,0.60   ->    (2, 1): 0.50, (2, 2): 0.60,   # plan 2
        0.45,0.80         (3, 1): 0.45, (3, 2): 0.80}   # plan 3

    Pyomo can read some data straight from a DataFrame, so we may not
    need this.
    """
    # a plain NumPy array - reading one cell is much faster than df.iloc
    values = df.to_numpy()
    (M, N) = values.shape  # rows, columns

    return {
        (row + 1, col + 1): values[row, col]
        for row in range(M)
        for col in range(N)
    }
