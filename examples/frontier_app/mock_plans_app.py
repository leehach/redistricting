"""Show the SBM frontier app for 60 made-up plans.

mock_plans.csv has one row per plan and four metrics with trade-offs built
in (more compact plans tend to split more counties), so the frontier is
not trivial. All values are above 0, because SBM needs that.

Run it from the repo root, then open http://127.0.0.1:8050:
    micromamba run -n redistricting python -m examples.frontier_app.mock_plans_app
"""

from pathlib import Path

import pandas as pd

from redistricting.dea_frontier_app import add_sbm_scores, create_app

plans = pd.read_csv(Path(__file__).parent / "mock_plans.csv")

direction = {
    "compactness": "higher",
    "county_splits": "lower",
    "pop_dev_pct": "lower",
    "partisan_gap": "lower",
}

labels = {
    "compactness": "Compactness (PP)",
    "county_splits": "County Splits",
    "pop_dev_pct": "Pop. Deviation (%)",
    "partisan_gap": "Partisan Gap",
}

results = add_sbm_scores(plans, direction)
app = create_app(results, direction, "plan_id", labels)
app.run()
