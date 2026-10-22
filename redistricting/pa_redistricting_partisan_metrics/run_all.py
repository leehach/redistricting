"""Make every result in one go: open this file in VS Code and press Run.

Change the two settings below, then run again. Results go to:

    results/
      overview.csv                  one row per plan, for comparing plans
      summaries/<plan>_summary.csv  population per Senate district
      maps/<plan>.png               map of each plan
      partisan/                     efficiency gap and other partisan metrics

Same as the terminal command:  python -m pa_redistricting run --all
"""

import logging
from pathlib import Path

from pa_redistricting import MissingAPIKeyError
from pa_redistricting.cli import run_everything

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PLANS = "all"                                  # or words from plan names: ["allentown", "rh"]
OUT = Path(__file__).parent / "results"        # output folder, next to this file

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # show download progress
    try:
        run_everything(PLANS, OUT)
    except MissingAPIKeyError as e:
        raise SystemExit(str(e)) from None
