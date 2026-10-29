"""Show the DEA frontier app for the class plans.

The class data comes as two CSV files with no headers. Rows are plans,
and both files list the same plans in the same order:

- data/Plans1_g.csv: good metrics, where higher is better.
- data/Plans2_b.csv: bad metrics, where lower is better.

This script names the columns good_1, good_2, ... and bad_1, bad_2, ...
by the file they came from, numbers the plans from 0, and runs the app.

Run it from the repo root, then open http://127.0.0.1:8050:
    micromamba run -n redistricting python -m examples.frontier_app.class_plans_app
"""

import pandas as pd

from redistricting.dea_frontier_app import add_dea_scores, create_app


def read_metrics(path, prefix):
    """Read a headerless CSV and name its columns prefix_1, prefix_2, ..."""
    table = pd.read_csv(path, header=None)
    table.columns = [f"{prefix}_{i + 1}" for i in range(table.shape[1])]
    return table


good = read_metrics("data/Plans1_g.csv", "good")
bad = read_metrics("data/Plans2_b.csv", "bad")

plans = pd.concat([good, bad], axis=1)
plans.insert(0, "plan_id", range(len(plans)))

direction = {name: "higher" for name in good.columns}
direction |= {name: "lower" for name in bad.columns}

results = add_dea_scores(plans, direction)
app = create_app(results, direction, "plan_id")
app.run()
