"""Show the DEA frontier app, with a "Data" dropdown to switch datasets.

- Mock: 60 made-up plans in mock_plans.csv. Each plan puts its effort on
  some metrics at the cost of the others, so every pair of metrics has a
  trade-off. All values are above 0, because SBM needs that.
- Class: the 2,355 class plans, when data/Plans1_g.csv and
  data/Plans2_b.csv are present (data/ is not in git). The files have no
  headers, so the columns are named good_1, ... and bad_1, ... by the
  file they came from. Scoring them takes about 3.5 minutes, so the
  scores are cached in the user's temp folder; later runs start in
  seconds.

Run it from the repo root, then open http://127.0.0.1:8050:
    micromamba run -n redistricting python -m examples.frontier_app.app # TODO Fix - no micromanba for all.
"""

import tempfile
from pathlib import Path

import pandas as pd

from redistricting.dea_frontier_app import Dataset, add_dea_scores, create_app

CLASS_GOOD = Path("data/Plans1_g.csv")
CLASS_BAD = Path("data/Plans2_b.csv")
CACHE_DIR = Path(tempfile.gettempdir()) / "redistricting"


def main():
    # The keys are the names shown in the "Data" dropdown. The first one
    # is shown when the app opens.
    datasets = {"Mock (60 plans)": load_mock_plans()}
    if CLASS_GOOD.exists() and CLASS_BAD.exists():
        datasets["Class (2,355 plans)"] = load_class_plans()
    else:
        print(f"No {CLASS_GOOD} or {CLASS_BAD}, so only the mock data is shown.")

    create_app(datasets).run()


def load_mock_plans():
    """Score the mock plans. They are fast to score, so no cache."""
    plans = pd.read_csv(Path(__file__).parent / "mock_plans.csv")

    # Which way is better for each metric. Only these columns are scored.
    direction = {
        "compactness": "higher",
        "county_splits": "lower",
        "pop_dev_pct": "lower",
        "partisan_gap": "lower",
    }

    # Names for the axes and the side panel, in place of the column names.
    labels = {
        "compactness": "Compactness (PP)",
        "county_splits": "County Splits",
        "pop_dev_pct": "Pop. Deviation (%)",
        "partisan_gap": "Partisan Gap",
    }
    return Dataset(add_dea_scores(plans, direction), direction, "plan_id", labels)


def load_class_plans():
    """Score the class plans, with the scores cached in CACHE_DIR."""
    good = read_metrics(CLASS_GOOD, "good")
    bad = read_metrics(CLASS_BAD, "bad")

    # Row i of each file is the same plan, so concat joins them by row.
    plans = pd.concat([good, bad], axis=1)

    # The files have no ID column, and the app needs one to label and find
    # plans. Use the row number, put first to match mock_plans.csv.
    plans.insert(0, "plan_id", range(len(plans)))

    # Higher is better for the good_* metrics, lower for the bad_* ones.
    # `|=` merges the second dict into the first.
    direction = {name: "higher" for name in good.columns}
    direction |= {name: "lower" for name in bad.columns}

    # No labels, so the app shows the column names (good_1, bad_1, ...).
    results = add_dea_scores(plans, direction, CACHE_DIR)
    return Dataset(results, direction, "plan_id")


def read_metrics(path, prefix):
    """Read a headerless CSV and name its columns prefix_1, prefix_2, ..."""
    table = pd.read_csv(path, header=None)
    table.columns = [f"{prefix}_{i + 1}" for i in range(table.shape[1])]
    return table


if __name__ == "__main__":
    main()
