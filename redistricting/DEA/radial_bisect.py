"""Score redistricting plans with a radial measure of DEA, found by bisection.

Radial DEA (Data Envelopment Analysis) scores how efficient each
redistricting plan is compared with the other plans. Bisection is how the
code finds that score.

Words used in this file:
    plan    one row of `metrics` (one way to draw the districts)
    metric  one column of `metrics`, such as compactness or county splits
    blend   a weighted mix of real plans; the weights are `plan_shares` (λ)
    θ       theta, the score: a number from 0 to 1, where 1 is efficient

The one function, calculate_radial_efficiency, builds the model for one
plan, then bisects θ to find its score. make_dict (in pyomo_data.py)
reshapes each table for Pyomo.

THE QUESTION: what is the smallest θ at which a blend of real plans beats
the plan on EVERY metric at the same time? (Smaller θ asks for a bigger
improvement.) The blend must meet every metric together, not just one of
them, and not some average of them.

A radial score asks this as one factor θ (between 0 and 1): can a blend
shrink every lower metric to θ times the plan's value and grow every
higher metric to the plan's value divided by θ, all at once? A higher
metric is better when it goes up. A lower metric is better when it goes
down.

    ┌──────┬──────────────────────┬───────────────────────┐
    │ Plan │ Compactness (higher) │ County splits (lower) │
    ├──────┼──────────────────────┼───────────────────────┤
    │ A    │ 0.40                 │ 10                    │
    ├──────┼──────────────────────┼───────────────────────┤
    │ B    │ 0.50                 │ 6                     │
    ├──────┼──────────────────────┼───────────────────────┤
    │ C    │ 0.50                 │ 8                     │
    └──────┴──────────────────────┴───────────────────────┘

To score A, try smaller and smaller θ. Each θ turns into a target:

    θ      compactness at least   county splits at most   a plan that meets it
    1.00   0.40 / 1.00 = 0.400    10 * 1.00 = 10          A itself
    0.90   0.40 / 0.90 = 0.444    10 * 0.90 = 9           B (0.50, 6)
    0.80   0.40 / 0.80 = 0.500    10 * 0.80 = 8           B (0.50, 6)
    0.75   0.40 / 0.75 = 0.533    10 * 0.75 = 7.5         none: 0.50 is the best

0.8 is the smallest θ that still works. The compactness target reaches the
best compactness of any plan (0.50) when θ = 0.40 / 0.50 = 0.8. County
splits do not limit it: B has 6 and the target is 8. So A scores 0.8.

C scores 1. It already has the best compactness, so no θ below 1 works,
even though B has fewer county splits. Radial scores cannot see that
leftover gap. The slacks-based measure (SBM) can: it scores C at 8/9.

Sometimes no single plan meets the target, but a blend does. Take
P1 = (1, 1), P2 = (4, 4) and D = (2, 4.5) as (compactness, splits). At
θ = 2/3, D's target is compactness 3 and splits 3. P1 has too little
compactness and P2 has too many splits. But 1/3 of P1 plus 2/3 of P2 gives
compactness 1/3 * 1 + 2/3 * 4 = 3 and splits 3. So D scores 2/3. The
shares (1/3 and 2/3) are 0 or more and add up to 1.

Why bisect: θ shrinks the lower metrics and 1 / θ grows the higher ones.
1 / θ is not linear, so one linear program cannot solve for θ itself. But
for one fixed θ, "does a blend exist?" is a plain yes/no linear program.
Pyomo builds it and GLPK solves it.

Bisection is a binary search on θ. Start with the range [0.001, 1]. Try
the middle. If a blend exists, the answer is at or below the middle, so
keep the lower half. If not, keep the upper half. Stop when the range is
narrower than TOLERANCE. Each try cuts the range in half, so each plan
takes about 14 solves. For plan A:

    try   θ        blend?   new range
    1     0.5005   no       0.5005 .. 1
    2     0.7502   no       0.7502 .. 1
    3     0.8751   yes      0.7502 .. 0.8751
    4     0.8127   yes      0.7502 .. 0.8127
    5     0.7815   no       0.7815 .. 0.8127
    ...                     closes in on 0.8

NOT MEASURED: a plan whose search falls to THETA_FLOOR scores NaN. A
score at the floor is not measured: the true θ may be below it, or not
exist at all. This happens when every θ works, which needs 0s. A 0 takes
θ out of that metric's constraint (θ * 0 = 0 and 0 / θ = 0). For
example:

    lower only, splits (0), (4)               ->  NaN, NaN
    (compactness, splits) (0, 0), (0.5, 4)    ->  NaN, 1
    higher only, compactness (0), (0.5)       ->  NaN, 1

The 1st plan should score 1 in the first two examples and about 0 in
the third. In the first, the 2nd plan should score about 0: a blend of
0 splits beats 4 splits at every θ. One θ cannot tell 1 from 0 here, so
the score is NaN, not a number that looks real when plans are ranked
or averaged.
"""

import math

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Var,
    check_optimal_termination,
)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from redistricting.DEA.pyomo_data import make_dict

# What the solver reports when no blend exists. GLPK finds the problem
# infeasible, but Pyomo's reader labels it "unbounded" (see PRESOLVE below).
# The shares are bounded (they add to 1) and the objective is 0, so the
# problem cannot truly be unbounded: "not optimal" can only mean "no blend".
# Any other report (other, time limit, error) is a real failure.
NO_BLEND = {TerminationCondition.infeasible, TerminationCondition.unbounded}

# The lowest θ the search tries. The search needs a range to start in. 1 is
# known to work (a plan is its own blend), and this is assumed to fail: it
# asks for 1,000 times the higher metrics and 1/1,000 of the lower ones. If
# a plan could go lower still, the search never finds the edge, so the
# plan scores NaN (see NOT MEASURED in the module docstring).
THETA_FLOOR = 0.001

# The search stops when the range of possible θ is this narrow. It returns
# the top of the range, so a score can be up to this much above the exact θ.
TOLERANCE = 0.0001


def calculate_radial_efficiency(higher_metrics, lower_metrics, current_row):
    """Return the radial score (θ) of one plan.

    `higher_metrics` and `lower_metrics` have one row per plan, in the same
    order. Their columns are the higher-is-better and lower-is-better
    metrics. `current_row` is the plan to score, counted from 1.
    dea.score calls this once per plan. Returns NaN when θ cannot be
    measured (see NOT MEASURED in the module docstring).

    Raises ValueError for mismatched rows, no metrics, or values below 0
    (or blank), and RuntimeError if the solver fails for a reason other
    than "no blend exists".
    """
    # STEP 1: get the sizes and turn the tables into dicts that Pyomo can
    # read.
    (plan_count, higher_col_count) = higher_metrics.shape
    (lower_plan_count, lower_col_count) = lower_metrics.shape

    if plan_count != lower_plan_count:
        raise ValueError("Number of rows in higher and lower metrics are not equal!")

    if higher_col_count + lower_col_count == 0:
        raise ValueError("Give at least one metric")

    higher_dict = make_dict(higher_metrics)
    lower_dict = make_dict(lower_metrics)

    # `>= 0` is False for NaN, so this also catches blank cells
    all_valid = all(v >= 0 for v in [*higher_dict.values(), *lower_dict.values()])
    if not all_valid:
        raise ValueError("All metric values must be 0 or greater")

    # STEP 2: build the model once. Only θ changes between solves.
    #
    # In math, for plan o (the one we score) and plans j = 1..n:
    #
    #     find   λ_j >= 0
    #     so     Σ λ_j = 1                              (shares add to 1)
    #            Σ λ_j * lower_j  <= θ * lower_o        (each lower metric)
    #            Σ λ_j * higher_j >= higher_o / θ       (each higher metric)
    model = ConcreteModel()

    # Sets (index ranges), counted from 1
    model.higher_cols = RangeSet(1, higher_col_count)
    model.lower_cols = RangeSet(1, lower_col_count)
    model.plans = RangeSet(1, plan_count)

    # How much of each plan goes into the blend (the λ in the DEA math).
    # These are the only variables: the solver picks them.
    model.plan_shares = Var(model.plans, within=NonNegativeReals)

    # θ. It is a Param, not a Var: the solver does not pick it, the
    # bisection loop does. It is mutable so the loop can change it between
    # solves without building the model again.
    model.theta = Param(initialize=1.0, mutable=True)

    # We only ask if a blend exists (yes or no), so there is nothing to
    # minimize. Any blend that meets the constraints is enough.
    model.obj = Objective(expr=0)

    # The shares add to 1, so the blend is a weighted average of plans.
    # Example: 1/3 of P1 = (1, 1) plus 2/3 of P2 = (4, 4) gives (3, 3).
    model.shares_sum_constraint = Constraint(
        expr=sum(model.plan_shares[plan] for plan in model.plans) == 1
    )

    # The blend's lower metric can be at most θ times the scored plan's.
    # Plan A at θ = 0.8: the blend can have at most 10 * 0.8 = 8 splits.
    # Pyomo calls this rule once per lower metric, so each one gets its own
    # constraint.
    def lower_shrink_constraint_rule(model, metric):
        return (
            sum(
                lower_dict[plan, metric] * model.plan_shares[plan]
                for plan in model.plans
            )
            <= model.theta * lower_dict[current_row, metric]
        )

    model.lower_shrink_constraint = Constraint(
        model.lower_cols, rule=lower_shrink_constraint_rule
    )

    # The blend's higher metric must be at least the scored plan's divided
    # by θ. Plan A at θ = 0.8: the blend needs at least 0.40 / 0.8 = 0.50
    # compactness. Pyomo calls this rule once per higher metric.
    def higher_grow_constraint_rule(model, metric):
        return (
            sum(
                higher_dict[plan, metric] * model.plan_shares[plan]
                for plan in model.plans
            )
            >= (1 / model.theta) * higher_dict[current_row, metric]
        )

    model.higher_grow_constraint = Constraint(
        model.higher_cols, rule=higher_grow_constraint_rule
    )

    # STEP 3: set up the solver.
    #
    # GLPK (GNU Linear Programming Kit) is a free, open-source solver, so it
    # needs no license. It installs from conda-forge (see environment.yml).
    solver = SolverFactory("glpk")

    # PRESOLVE: before it solves, GLPK shrinks the problem. When that leaves
    # a row with no columns, GLPK calls the row feasible if it is off by less
    # than 0.001. The tolerance is absolute, so it is wrong for small values:
    # an ask of 0.5005 when the best is 0.5 passes. Turn it off.
    # Source: GLPK 5.0 (our version), src/npp/npp3.c, npp_empty_row:
    #     double eps = 1e-3;   /* absolute tolerance for row value */
    # https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
    # The class data scores the same with presolve on or off, and it is not
    # slower. Without this line, GLPK reports "other" for no blend, so the
    # tests fail with RuntimeError. If "other" were accepted as "no blend"
    # instead, test_radial_allows_a_zero_metric would score 0.998, not 1.
    solver.options["nopresol"] = ""

    # STEP 4: bisect θ (see the trace for plan A in the module docstring).
    #
    # θ = 1 always works: the blend is the plan itself. So `high` is always
    # a θ with a blend, and `low` is a θ with none (or the floor). The
    # answer is always between them, and each loop halves the gap.
    low = THETA_FLOOR
    high = 1.0

    while high - low > TOLERANCE:
        # Try the middle of the range.
        theta = (low + high) / 2
        model.theta = theta

        # We read the solver's report, not the solution, so skip loading it.
        # We need to know IF a blend exists, not which blend it is.
        report = solver.solve(model, load_solutions=False)

        condition = report.solver.termination_condition

        # Three cases: yes (a blend exists), no (none exists), or broken.
        if check_optimal_termination(report):
            high = theta  # a blend exists, so try a smaller θ
        elif report.solver.status == SolverStatus.ok and condition in NO_BLEND:
            low = theta  # no blend exists, so θ must be larger
        else:
            # The model or the solver is broken. Stop instead of writing a
            # fake score.
            raise RuntimeError(
                f"Plan {current_row} did not solve at theta={theta}: {condition}"
            )

    # The search fell to the floor, so θ was not measured. Return NaN so
    # the plan cannot pass for a real score.
    if high <= THETA_FLOOR + TOLERANCE:
        return math.nan

    # `high` is the smallest θ we saw a blend for, so it is the score.
    return high
