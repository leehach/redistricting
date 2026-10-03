from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import os

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


# note: leaving this as "make dict" - might refactor we'll see
def make_dict(df):
    (M, N) = df.shape
    print(M, N)
    a = {}
    for row in range(M):
        for col in range(N):
            a[(row + 1, col + 1)] = df.iloc[
                row, col
            ]  # iloc = integer location (pandas) / # plus 1 because the model below starts at one (RangeSet)

    return (M, N, a)


# What's happening here we're turning the csv data from this:

# 0.40,0.70
# 0.50,0.60
# 0.45,0.80

# to

# M = 3
# N = 2
# a = {
#     (1, 1): 0.40,  (1, 2): 0.70,   # plan 1
#     (2, 1): 0.50,  (2, 2): 0.60,   # plan 2
#     (3, 1): 0.45,  (3, 2): 0.80,   # plan 3
# }

# M is rows and N is coluimns


# this scores one **plan** at a time, so that's row by row
# This accepts:
# the "good" and "bad" plans
# the shapes of each
# then `MM`` unused variable that sets the definitive shape above - we could use it and loop in the function or we'll see
# iHat is the row number


def calculate_efficiency(
    good_plans_dict,
    bad_plans_dict,
    good_plan_num_rows,
    good_plan_num_cols,
    bad_plan_number_rows,
    bad_plan_number_cols,
    current_row,
):
    if good_plan_num_rows != bad_plan_number_rows:
        raise ValueError("Number of rows in good and bad plans are not equal!")

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
    DEA_model.num_cols = Set(
        initialize=DEA_model.good_cols | DEA_model.bad_cols
    )  # Set of the union of column indicies -> so {1,2,3} - is dead code - unused

    # we do the same as above for the rows - set up RangeSets and a Set with the union of all row indicies
    DEA_model.good_rows = RangeSet(1, good_plan_num_rows)
    DEA_model.bad_rows = RangeSet(1, bad_plan_number_rows)
    DEA_model.num_rows = Set(initialize=DEA_model.good_rows | DEA_model.bad_rows)

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

    ### ignoring for now ###
    # def obj_rule (DEA_model):
    #  return sum(a_g[iHat,i]*DEA_model.g[i] for i in DEA_model.N_g)- sum(a_b[iHat,j]*DEA_model.b[j] for j in DEA_model.N_b) - DEA_model.v
    # DEA_model.obj = Objective(rule = obj_rule, sense = maximize)

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
    # It takes one constraint per plan, because we set it up that way about with the indexed variable
    # So this is an "Indexed constraint" enforcing the good score of a plan minus the bad score of a plan is <= v - the baseline.
    # Now, the solver picks the baseline balancing the rules we're attaching.
    # i "walks across" the columns of a given plan's row and sums em, same for the bad and subtracts
    def principal_constraint_rule(model, rows):
        return (
            sum(
                good_plans_dict[rows, i] * model.good_weights[i]
                for i in model.good_cols
            )
            - model.baseline
            - sum(
                bad_plans_dict[rows, j] * model.bad_weights[j] for j in model.bad_cols
            )
            <= 0.0
        )

    DEA_model.principal_constraint = Constraint(
        DEA_model.num_rows, rule=principal_constraint_rule
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


# and now the execution loop
# loop over each plan - 1 based to match the model

if __name__ == "__main__":
    # one dict per scored plan; becomes a DataFrame after the loop
    results = []

    # inputs

    # note no headers, so this is going off memory/investigation
    # Idea: we can keep a dict of measures who are good and bad and dynamically lookup to avoid this.
    good_plans = pd.read_csv(
        "data/Plans1_g.csv", header=None
    )  # metrics/critera for which "good is better" -> higher is better (i.e. compactness)
    bad_plans = pd.read_csv(
        "data/Plans2_b.csv", header=None
    )  # metrics/critera for whiich "bad is better" -> lower is better (i.e county splits)

    # (MM_g, NN_g, a_g) = make_dict(good_plans)
    # (MM_b, NN_b, a_b) = make_dict(bad_plans)
    (num_good_plan_rows, num_good_plan_cols, good_plan_dict) = make_dict(good_plans)
    (num_bad_plan_rows, num_bad_plan_cols, bad_plan_dict) = make_dict(bad_plans)

    count = -1
    for row_num in range(1, num_good_plan_rows + 1):
        efficiency = calculate_efficiency(
            good_plan_dict,
            bad_plan_dict,
            num_good_plan_rows,
            num_good_plan_cols,
            num_bad_plan_rows,
            num_bad_plan_cols,
            row_num,
        )

        count += 1
        print("count = ", count)
        print(f"DMU {row_num} has efficiency {efficiency}")

        results.append({"plan_id": count, "dea_efficiency": efficiency})

    dea_eff = pd.DataFrame(results)
    dea_eff.to_csv("df_eff_g_b_EPS_justinplussteve_nov142025.csv")
