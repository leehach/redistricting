"""PlanScore metrics for redistricting plans in any state.

The script asks for a state, then for a plan CSV (or a folder of them). Each
plan needs one row per voting district (Census block in CA, HI and OR) with a
GEOID20 column and a District column, the same format as the PA citizen plans.

Tip: in File Explorer, right-click the file or folder > "Copy as path", then
paste it at the prompt.

Each plan's metrics are saved next to it, as <plan>_metrics.csv:
election, Democratic vote share and seats, efficiency gap, partisan bias,
mean-median and declination, one row per election.

Full results also go to results/states/<state>/plans/:
    overview.csv                       one row per plan, for comparing plans
    partisan_metrics.csv               every metric, every election, every plan
    <plan>__<election>_districts.csv   votes and winner per district
    <plan>_population.csv              population and deviation per district
"""

import logging
from pathlib import Path

import pandas as pd

from pa_redistricting.partisan_metrics import METRICS, favors
from pa_redistricting.state_data import STATES, load_state, pick_state
from pa_redistricting.state_metrics import plan_name, read_plan, score_plan

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RESULTS_FOLDER = Path(__file__).parent / "results" / "states"   # next to this file
METRICS_SUFFIX = "_metrics"          # my_plan.csv -> my_plan_metrics.csv


# ---------------------------------------------------------------------------
# Step 1: ask which plan files to score
# ---------------------------------------------------------------------------
def ask_for_plans() -> list[Path]:
    """Ask for a plan CSV or a folder of them, until one is found."""
    while True:
        answer = input("\nPath to a plan CSV, or a folder of plan CSVs: ")
        path = Path(answer.strip().strip('"'))      # "Copy as path" adds quotes

        if path.is_file():
            return [path]

        if path.is_dir():
            # every CSV in the folder, except metrics files from earlier runs
            plans = [f for f in sorted(path.glob("*.csv"))
                     if not f.stem.endswith(METRICS_SUFFIX)]
            if plans:
                print(f"Found {len(plans)} plan files.")
                return plans
            print(f"No .csv files in {path}")
        else:
            print(f"Can't find {answer.strip()!r}. Try again.")


# ---------------------------------------------------------------------------
# Step 2: show one plan's results in the terminal
# ---------------------------------------------------------------------------
def print_report(name: str, score) -> None:
    population = score.population_stats

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Districts: {population['districts']}   "
          f"population range {population['overall_range']:.2f} points "
          f"({population['min_pct']:+.2f}% to {population['max_pct']:+.2f}%)")
    if population["unassigned_pop"]:
        print(f"!! {population['unassigned_pop']:,} people are in voting districts "
              f"the plan leaves out")

    for _, election in score.metrics.iterrows():
        print(f"[{election['election']}]  Dem vote {election['dem_vote_share']:.1%}  ->  "
              f"Dem seats {election['dem_seats']}/{election['n_districts']}")
        for metric in METRICS:
            value = election[metric]
            if pd.isna(value):
                text = "n/a (one party wins every seat)"
            else:
                text = f"{value:+.3f}"
            print(f"   {metric:<15} {text:>8}   favors {favors(value)}")


# ---------------------------------------------------------------------------
# Step 3: save one plan's results
# ---------------------------------------------------------------------------
def save_metrics_next_to_plan(plan_file: Path, score) -> None:
    """my_plan.csv -> my_plan_metrics.csv in the same folder, one row per election."""
    metrics_file = plan_file.with_name(plan_file.stem + METRICS_SUFFIX + ".csv")
    table = score.metrics.copy()
    table.insert(0, "plan", plan_file.stem)
    table.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")


def save_details(name: str, score, folder: Path) -> None:
    """Votes per district (one file per election) and population per district."""
    for election, districts in score.districts.items():
        districts.insert(0, "plan", name)
        districts.to_csv(folder / f"{name}__{election}_districts.csv", index=False)

    score.population.insert(0, "plan", name)
    score.population.to_csv(folder / f"{name}_population.csv", index=False)


def overview_row(name: str, score) -> dict:
    """One line of overview.csv, using the first election (composite)."""
    population = score.population_stats
    election = score.metrics.iloc[0]

    row = {
        "plan": name,
        "districts": population["districts"],
        "overall_range": round(population["overall_range"], 2),
        "unassigned_pop": population["unassigned_pop"],
        "majority_minority_vap": population["majority_minority_vap"],
        "election": election["election"],
        "dem_vote_share": round(election["dem_vote_share"], 4),
        "dem_seats": int(election["dem_seats"]),
    }
    for metric in METRICS:
        row[metric] = round(election[metric], 4)
    return row


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # show progress

    # 1. Choose a state and load its shapes, demographics and votes
    state = pick_state()
    state_data, _ = load_state(state)

    # 2. Choose the plan(s) to score
    plan_files = ask_for_plans()

    results = RESULTS_FOLDER / state.lower() / "plans"
    results.mkdir(parents=True, exist_ok=True)

    # 3. Score each plan and save its results
    overview, all_metrics = [], []
    for plan_file in plan_files:
        name = plan_name(plan_file)
        try:
            score = score_plan(read_plan(plan_file), state_data)
        except ValueError as problem:
            print(f"\nSkipping {plan_file.name}: {problem}")
            continue

        print_report(name, score)
        save_metrics_next_to_plan(plan_file, score)
        save_details(name, score, results)
        all_metrics.append(score.metrics.assign(plan=name))
        overview.append(overview_row(name, score))

    # 4. Save the tables that compare every plan
    if not overview:
        return
    pd.DataFrame(overview).to_csv(results / "overview.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(results / "partisan_metrics.csv",
                                                     index=False)
    print(f"\n{STATES[state][0]}: scored {len(overview)} plan(s). Saved to {results}")


if __name__ == "__main__":
    main()
