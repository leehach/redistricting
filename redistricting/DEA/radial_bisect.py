"""Score redistricting plans with a radial measure of DEA, found by bisection.

Radial DEA determines how efficient a redistricting plan is relative to other plans.
Bisection is the method the code uses to calculate that efficiency score.
A blend is a weighted combination of one or more plans' metric values (`plan_shares`).

=> What's the smallest θ for which a feasible blend exists? <=
(Smaller θ means demanding bigger improvements to the measure)
To do this, we're essentially binary traversing (bisecting) theta:
1 first; then 0.5; then 0 or 0.75; etc, using the Solver
to attempt to find a blend w/ lowest theta. Only needs one, then
stops, hence the `Objective(expr=0)`

I think the math/constraints are intuitive. ONe thing ot mention is the solver
needs to satisfy both the compactness and county splits not some combo or one or the other.

A radial score asks one question: by what one factor θ (between 0 and 1)
can a plan shrink every lower metric and grow every higher metric (by
1 / θ), all at once, and still match a blend of real plans? A higher
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

Why bisect: θ shrinks the lower metrics and 1 / θ grows the higher ones,
so one linear program cannot solve for θ itself. But for one fixed θ,
"does a blend exist?" is a plain linear program. Pyomo builds it and GLPK
solves it. We stop when the range of θ is narrower than TOLERANCE.
"""

import pandas as pd
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

DIRECTIONS = {"higher", "lower"}

# What the solver reports when no blend exists. GLPK finds the problem
# infeasible, but Pyomo's reader labels it "unbounded" (see PRESOLVE below).
# The shares are bounded (they add to 1) and the objective is 0, so the
# problem cannot truly be unbounded: "not optimal" can only mean "no blend".
# Any other report (other, time limit, error) is a real failure.
NO_BLEND = {TerminationCondition.infeasible, TerminationCondition.unbounded}

# The lowest θ the search tries. The search needs a range to start in. 1 is
# known to work (a plan is its own blend), and this is assumed to fail: it
# asks for 1,000 times the higher metrics and 1/1,000 of the lower ones. If
# a plan could go lower still, the search never finds the edge, and the
# plan scores about this (its exact θ is below it).
THETA_FLOOR = 0.001

# The search stops when the range of possible θ is this narrow. It returns
# the top of the range, so a score can be up to this much above the exact θ.
TOLERANCE = 0.0001


def radial_scores(metrics: pd.DataFrame, direction: dict[str, str]) -> pd.Series:
    """Return the radial score of each plan (row) in `metrics`.

    `metrics` has one row per plan and one column per metric. `direction`
    maps every column to "higher" (higher is better) or "lower" (lower is
    better). The result has the same index as `metrics`. A score of 1 means
    the plan is efficient. Values must be 0 or greater.
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

    if metrics.shape[1] == 0:
        raise ValueError("Give at least one metric")

    # `>= 0` is False for NaN, so this also catches blank cells
    all_valid = (metrics >= 0).to_numpy().all()
    if not all_valid:
        raise ValueError("All metric values must be 0 or greater")

    higher = [name for name in metrics.columns if direction[name] == "higher"]
    lower = [name for name in metrics.columns if direction[name] == "lower"]

    scores = [
        calculate_radial_efficiency(metrics[higher], metrics[lower], row)
        for row in range(1, len(metrics) + 1)
    ]

    return pd.Series(scores, index=metrics.index, name="radial_score")


def calculate_radial_efficiency(higher_metrics, lower_metrics, current_row):
    """Return the radial score (θ) of one plan.

    `higher_metrics` and `lower_metrics` have one row per plan, in the same
    order. Their columns are the higher-is-better and lower-is-better
    metrics. `current_row` is the plan to score, counted from 1.
    radial_scores calls this once per row.

    Raises ValueError for mismatched rows or values below 0 (or blank), and
    RuntimeError if the solver fails for a reason other than "no blend
    exists".
    """
    (plan_count, higher_col_count) = higher_metrics.shape
    (lower_plan_count, lower_col_count) = lower_metrics.shape

    if plan_count != lower_plan_count:
        raise ValueError("Number of rows in higher and lower metrics are not equal!")

    higher_dict = _make_dict(higher_metrics)
    lower_dict = _make_dict(lower_metrics)

    # `>= 0` is False for NaN, so this also catches blank cells
    all_valid = all(v >= 0 for v in [*higher_dict.values(), *lower_dict.values()])
    if not all_valid:
        raise ValueError("All metric values must be 0 or greater")

    model = ConcreteModel()

    # Sets (index ranges), counted from 1
    model.higher_cols = RangeSet(1, higher_col_count)
    model.lower_cols = RangeSet(1, lower_col_count)
    model.plans = RangeSet(1, plan_count)

    # How much of each plan goes into the blend (the λ in the DEA math)
    model.plan_shares = Var(model.plans, within=NonNegativeReals)

    # θ. It is mutable so the search can change it between solves without
    # building the model again.
    model.theta = Param(initialize=1.0, mutable=True)

    # We only ask if a blend exists, so the objective does nothing.
    model.obj = Objective(expr=0)

    model.shares_sum_constraint = Constraint(
        expr=sum(model.plan_shares[plan] for plan in model.plans) == 1
    )

    # The blend's lower metric can be at most θ times the scored plan's.
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
    # by θ.
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
    # instead, test_radial_scores_allows_zero would score 0.998, not 1.
    solver.options["nopresol"] = ""

    # θ = 1 always works: the blend is the plan itself. So `high` is always
    # a θ with a blend, and `low` is a θ with none (or the floor).
    low = THETA_FLOOR
    high = 1.0

    while high - low > TOLERANCE:
        theta = (low + high) / 2
        model.theta = theta

        # We read the solver's report, not the solution, so skip loading it.
        report = solver.solve(model, load_solutions=False)

        condition = report.solver.termination_condition

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

    return high


def _make_dict(df):
    """Turn a DataFrame into a dict that maps (row, col) to its value.

    Both numbers start at 1, to match the model's RangeSets:

        0.40,0.70        {(1, 1): 0.40, (1, 2): 0.70,   # plan 1
        0.50,0.60   ->    (2, 1): 0.50, (2, 2): 0.60,   # plan 2
        0.45,0.80         (3, 1): 0.45, (3, 2): 0.80}   # plan 3
    """
    # a plain NumPy array - reading one cell is much faster than df.iloc
    values = df.to_numpy()
    (M, N) = values.shape  # rows, columns

    return {
        (row + 1, col + 1): values[row, col] for row in range(M) for col in range(N)
    }
