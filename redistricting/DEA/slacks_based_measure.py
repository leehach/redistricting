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

# Slacks-based measure of DEA
# What is it?

# A slack is a gap between one plan and better plan(s).

# ┌──────┬────────────────────┬─────────────────────┐
# │ Plan │ Compactness (good) │ County splits (bad) │
# ├──────┼────────────────────┼─────────────────────┤
# │ A    │ 0.40               │ 10                  │
# ├──────┼────────────────────┼─────────────────────┤
# │ B    │ 0.50               │ 6                   │
# └──────┴────────────────────┴─────────────────────┘

# Measured against B, A has two slacks. A slack says how much better a plan/row could be for a given metric.

# And SBM then?
# Created by Kaoru Tone in 2001, it is one way to score DEA.
# Turn slacks into percetnages of the plan's own value:
# compactness - 0.10 (slack) / 0.40 (actual value) = 25%
# splits - 4 / 10 -> 40%
# combine into one score -> bigger gaps give a lower score

# The above uses two plans, but when you have many plans to compare to, that's when you now need this model + solver.

# Now.. something I haven't quite nailed down, but I will, is the fact that tehre are

# folder_path = "C:\w64"
# from folder_path import *

# formats each table into a 1-indexed dict - needed for pyomo
# What is pyomo? It's OSS Optimization Modeling, it models problems to be sent to an external solver.


# Turns a DataFrame into a dictionary
# chance we don't need to do this, as pyomo can take a DataFrame in some instances.
def make_dict(df):
    # a plain NumPy array - reading one cell is much faster than df.iloc
    values = df.to_numpy()
    (M, N) = values.shape

    # plus 1 because the model below starts at one (RangeSet)
    return {
        (row + 1, col + 1): values[row, col]
        for row in range(M)
        for col in range(N)
    }


# What's happening here we're turning the csv data from this:

# 0.40,0.70
# 0.50,0.60
# 0.45,0.80

# to

# a = {
#     (1, 1): 0.40,  (1, 2): 0.70,   # plan 1
#     (2, 1): 0.50,  (2, 2): 0.60,   # plan 2
#     (3, 1): 0.45,  (3, 2): 0.80,   # plan 3
# }

# M is rows and N is coluimns - the shape comes from df.shape, so make_dict
# returns only the dict


# this scores one **plan** at a time, so that's row by row
# This accepts:
# the "good" and "bad" plans, as DataFrames - the shapes come from them
# iHat is the row number


def calculate_efficiency(good_plans, bad_plans, current_row):
    (good_plan_num_rows, good_plan_num_cols) = good_plans.shape
    (bad_plan_number_rows, bad_plan_number_cols) = bad_plans.shape

    if good_plan_num_rows != bad_plan_number_rows:
        raise ValueError("Number of rows in good and bad plans are not equal!")

    good_plans_dict = make_dict(good_plans)
    bad_plans_dict = make_dict(bad_plans)

    # records with values of 0 will mess up DEA
    if any(v <= 0 for v in good_plans_dict.values()) or any(
        v <= 0 for v in bad_plans_dict.values()
    ):
        raise ValueError("All metric values must be greater than 0")

    # Instantiate an empty pyomo model
    DEA_model = AbstractModel()

    # total number of metrics (these are columns)
    total_metrics = good_plan_num_cols + bad_plan_number_cols

    # set good and bad metrics to properties on the model
    # interesting is that the attribute namne here doesn't matter - N_g for instance.
    # Pyomo will go by the types - the RangeSet or Set or Var
    # so this is all setup
    DEA_model.good_cols = RangeSet(1, good_plan_num_cols)  # range of columns in good
    DEA_model.bad_cols = RangeSet(1, bad_plan_number_cols)  # range of columns in bad

    # one index per plan (row) - good and bad have the same rows, checked above
    DEA_model.plans = RangeSet(1, good_plan_num_rows)

    # variables in pyomo are things are numbers that the solve will fill in - what it picks.
    # It'll take later constraints and our objective, and find the "best" value

    # Two types: indexed (thats the first two, creates as many as there are in the collection)
    # Singe: empty parameter - just creates one.
    DEA_model.good_weights = Var(
        DEA_model.good_cols
    )  # one for each good metric/col - a weight per metric
    DEA_model.bad_weights = Var(
        DEA_model.bad_cols
    )  # one for each bad metric/col - a weight per metric
    DEA_model.baseline = Var()  # a "baseline" - will define later.
    DEA_model.efficiency_score = Var()  # efficiency score

    # Define "rules"
    # First, one for the objective.
    # Taken together, this fn defines the attirbute we want to optimize for
    # and the Objective set below, is saying - hey maximize this

    # Always return an expression - anything that can be evaluated to produce a value
    def obj_rule(model):
        return model.efficiency_score

    DEA_model.obj = Objective(rule=obj_rule, sense=maximize)

    # This time we're pairing a rule with a Constraint rather than an Objective
    # This calculates, for each row, the sum of each metric times its weight
    # We don't know the weight! That's for the solver to figure out - we added it as a variable, remember.
    # Also, the baseline... just keep in mind we want this to be as low as possible to get the higest possible EPS
    def efficiency_constraint_rule(DEA_model):
        return (
            DEA_model.efficiency_score
            + DEA_model.baseline
            - sum(
                good_plans_dict[current_row, i] * DEA_model.good_weights[i]
                for i in DEA_model.good_cols
            )
            + sum(
                bad_plans_dict[current_row, j] * DEA_model.bad_weights[j]
                for j in DEA_model.bad_cols
            )
            == 1.0
        )

    DEA_model.efficency_constraint = Constraint(rule=efficiency_constraint_rule)

    # this is the core constraint, hence principal.
    # It makes one constraint per plan, because it is indexed by `plans`
    # So this is an "Indexed constraint" enforcing the good score of a plan minus the bad score of a plan is <= v - the baseline.
    # Now, the solver picks the baseline balancing the rules we're attaching.
    # i "walks across" the columns of a given plan's row and sums em, same for the bad and subtracts
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

    # now constaint that is just an expr - you can pass a short form.
    # we could have done it above
    # this is stating that the bar/baseline v cannot go below 1

    DEA_model.baseline_constraint = Constraint(expr=DEA_model.baseline >= 1.0)

    # These next rules ensure that we don't assign a weight of 0 to any given metrics,
    # and thus ignore a metric completely.
    # It assigns a floor, dynamically, depending on the eps it is also solving for and the row-metric value.
    # The floor is the minimum share. And for a given eps. We do it once for the bad and once for the good.

    def good_floor_constraint_rule(model, num_good):  # not sure what ng is here
        return (
            total_metrics * good_plans_dict[current_row, num_good]
        ) * model.good_weights[num_good] - model.efficiency_score >= 0.0

    DEA_model.good_floor_constraint = Constraint(
        DEA_model.good_cols, rule=good_floor_constraint_rule
    )

    def bad_floor_constraint_rule(model, num_bad):
        return (
            total_metrics * bad_plans_dict[current_row, num_bad]
        ) * model.bad_weights[num_bad] - model.efficiency_score >= 0.0

    DEA_model.bad_floor_constraint = Constraint(
        DEA_model.bad_cols, rule=bad_floor_constraint_rule
    )

    # Create the concrete model and solve
    model = DEA_model.create_instance()  # builds the model

    # assign a solve that actually does the math
    # solver set to gurobi for 9 metrics, glpk for 3 and 6 metrics? Check this..Yes Gurobi licence expired
    Opt = SolverFactory("glpk")
    # Opt = SolverFactory("gurobi")

    # sovle and get a report back
    Soln = Opt.solve(model)

    # a valid model always solves to optimal, so anything else means
    # the model or the solver is broken - stop instead of writing a fake score
    if not check_optimal_termination(Soln):
        raise RuntimeError(
            f"Plan {current_row} did not solve: {Soln.solver.termination_condition}"
        )

    return value(model.obj)


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
