"""PlanScore metrics for redistricting plans in any state.

The script asks for a state, then for the plans: a CSV, a folder of CSVs, or a
GitHub folder URL. (Set PLANS_SOURCE in pa_redistricting/config.py to skip that
question.) Each CSV holds one plan (GEOID20, District) or several in long format
(GEOID20, plan, District), with one row per voting district (Census block in
CA, HI and OR) per plan.

Tip: in File Explorer, right-click the file or folder > "Copy as path", then
paste it at the prompt.

For local CSVs, metrics are saved next to each file, as <file>_metrics.csv:
plan, election, Democratic vote share and seats, efficiency gap, partisan bias,
mean-median and declination, one row per plan per election.

Full results also go to results/states/<state>/plans/:
    overview.csv                       one row per plan, for comparing plans
    partisan_metrics.csv               every metric, every election, every plan
    <plan>__<election>_districts.csv   votes and winner per district
    <plan>_population.csv              population and deviation per district
"""

import logging
from pathlib import Path

import pandas as pd

from pa_redistricting import config
from pa_redistricting.partisan_metrics import METRICS, favors
from pa_redistricting.plan_assignments import METRICS_SUFFIX, Plan, read_plans
from pa_redistricting.state_data import STATES, load_state, pick_state, pick_vest
from pa_redistricting.state_metrics import score_plan

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RESULTS_FOLDER = Path(__file__).parent / "results" / "states"   # next to this file


# ---------------------------------------------------------------------------
# Step 1: find the plans to score
# ---------------------------------------------------------------------------
def ask_for_plans() -> list[Plan]:
    """The plans at config.PLANS_SOURCE, or else ask for a CSV, folder or
    GitHub folder URL until plans are found."""
    if config.PLANS_SOURCE is not None:
        print(f"\nReading plans from {config.PLANS_SOURCE} (PLANS_SOURCE in config.py)")
        return read_plans(config.PLANS_SOURCE)

    while True:
        answer = input("\nPath to a plan CSV, a folder of plan CSVs, or a GitHub folder URL: ")
        try:
            plans = read_plans(answer)
        except (ValueError, FileNotFoundError) as problem:
            print(f"{problem}. Try again.")
            continue
        files = {p.source for p in plans}
        print(f"Found {len(plans)} plan(s) in {len(files)} file(s).")
        return plans


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
def save_metrics_next_to_plans(source: str, tables: list[pd.DataFrame]) -> None:
    """my_plans.csv -> my_plans_metrics.csv in the same folder: one row per plan
    per election. Only for local files (a GitHub folder can't be written to)."""
    plan_file = Path(source)
    if not plan_file.is_file():
        return
    metrics_file = plan_file.with_name(plan_file.stem + METRICS_SUFFIX + ".csv")
    pd.concat(tables, ignore_index=True).to_csv(metrics_file, index=False)
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
    state_data, _ = load_state(state, vest=pick_vest(state))   # your own file, or ALARM's

    # 2. Find the plan(s) to score
    plans = ask_for_plans()

    results = RESULTS_FOLDER / state.lower() / "plans"
    results.mkdir(parents=True, exist_ok=True)

    # 3. Score each plan and save its results
    overview, all_metrics = [], []
    by_source: dict[str, list[pd.DataFrame]] = {}
    for plan in plans:
        try:
            score = score_plan(plan.assignment, state_data)
        except ValueError as problem:
            print(f"\nSkipping {plan.name}: {problem}")
            continue

        print_report(plan.name, score)
        save_details(plan.name, score, results)
        metrics = score.metrics.copy()
        metrics.insert(0, "plan", plan.name)
        by_source.setdefault(plan.source, []).append(metrics)
        all_metrics.append(score.metrics.assign(plan=plan.name))
        overview.append(overview_row(plan.name, score))

    for source, tables in by_source.items():
        save_metrics_next_to_plans(source, tables)

    # 4. Save the tables that compare every plan
    if not overview:
        return
    pd.DataFrame(overview).to_csv(results / "overview.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(results / "partisan_metrics.csv",
                                                     index=False)
    print(f"\n{STATES[state][0]}: scored {len(overview)} plan(s). Saved to {results}")


if __name__ == "__main__":
    main()
