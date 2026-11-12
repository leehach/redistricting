"""PlanScore metrics for redistricting plans in any state.

Open this file in VS Code and press Run. It asks for a state, then for a plan
CSV (or a folder of them). Each plan needs one row per voting district (Census
block in CA, HI and OR) with a GEOID20 column and a District column, the same
format as the PA citizen plans.

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

OUT = Path(__file__).parent / "results" / "states"     # output folder, next to this file
METRICS_SUFFIX = "_metrics"                             # <plan>_metrics.csv, next to the plan


def ask_for_plans() -> list[Path]:
    """Ask for a plan CSV or a folder of them until one is found."""
    while True:
        answer = input("\nPath to a plan CSV, or a folder of plan CSVs: ").strip().strip('"')
        path = Path(answer)
        if path.is_dir():
            files = sorted(f for f in path.glob("*.csv")
                           if not f.stem.endswith(METRICS_SUFFIX))   # skip earlier results
            if files:
                print(f"Found {len(files)} plan files.")
                return files
            print(f"No .csv files in {path}")
        elif path.is_file():
            return [path]
        else:
            print(f"Can't find {answer!r}. Try again.")


def print_report(name: str, score) -> None:
    print("\n" + "=" * 72 + f"\n{name}\n" + "=" * 72)
    s = score.population_stats
    print(f"Districts: {s['districts']}   population range {s['overall_range']:.2f} points "
          f"({s['min_pct']:+.2f}% to {s['max_pct']:+.2f}%)")
    if s["unassigned_pop"]:
        print(f"!! {s['unassigned_pop']:,} people are in voting districts the plan leaves out")
    for _, m in score.metrics.iterrows():
        print(f"[{m['election']}]  Dem vote {m['dem_vote_share']:.1%}  ->  "
              f"Dem seats {m['dem_seats']}/{m['n_districts']}")
        for metric in METRICS:
            val = m[metric]
            txt = "n/a (one party wins every seat)" if pd.isna(val) else f"{val:+.3f}"
            print(f"   {metric:<15} {txt:>8}   favors {favors(val)}")


def overview_row(name: str, score) -> dict:
    s, m = score.population_stats, score.metrics.iloc[0]   # first election: composite
    return {"plan": name, "districts": s["districts"],
            "overall_range": round(s["overall_range"], 2),
            "unassigned_pop": s["unassigned_pop"],
            "majority_minority_vap": s["majority_minority_vap"],
            "election": m["election"], "dem_vote_share": round(m["dem_vote_share"], 4),
            "dem_seats": int(m["dem_seats"]),
            **{metric: round(m[metric], 4) for metric in METRICS}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # show progress
    abbr = pick_state()
    state, _ = load_state(abbr)
    plans = ask_for_plans()

    out = OUT / abbr.lower() / "plans"
    out.mkdir(parents=True, exist_ok=True)
    overview, all_metrics = [], []
    for path in plans:
        name = plan_name(path)
        try:
            score = score_plan(read_plan(path), state)
        except ValueError as e:
            print(f"\nSkipping {path.name}: {e}")
            continue
        print_report(name, score)

        beside_plan = path.with_name(f"{path.stem}{METRICS_SUFFIX}.csv")
        score.metrics.assign(plan=path.stem)[["plan", *score.metrics.columns]].to_csv(
            beside_plan, index=False)
        print(f"Metrics saved to {beside_plan}")

        for election, table in score.districts.items():
            table.insert(0, "plan", name)
            table.to_csv(out / f"{name}__{election}_districts.csv", index=False)
        score.population.insert(0, "plan", name)
        score.population.to_csv(out / f"{name}_population.csv", index=False)
        all_metrics.append(score.metrics.assign(plan=name))
        overview.append(overview_row(name, score))

    if overview:
        pd.DataFrame(overview).to_csv(out / "overview.csv", index=False)
        pd.concat(all_metrics, ignore_index=True).to_csv(out / "partisan_metrics.csv",
                                                         index=False)
        print(f"\n{STATES[abbr][0]}: scored {len(overview)} plan(s). Saved to {out}")
