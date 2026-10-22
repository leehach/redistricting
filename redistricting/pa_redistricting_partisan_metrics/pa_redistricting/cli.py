"""Command-line interface. The only module that prompts, prints reports or writes files.

    python -m pa_redistricting run --all              everything below -> results/
    python -m pa_redistricting list                   plan names and unassigned VTDs
    python -m pa_redistricting summary allentown      population summary -> summaries/
    python -m pa_redistricting maps --all             PNG maps -> maps/
    python -m pa_redistricting partisan --all         partisan metrics -> partisan/
    python -m pa_redistricting codebook               Census column labels

With no plan words and no --all, run, summary, maps and partisan show a numbered menu.
From Python or an IDE, run_everything() does the same as the run command.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .plan_assignments import fetch_plan_assignments
from .census import MissingAPIKeyError, fetch_codebook
from .config import N_DISTRICTS
from .election_results import available_elections, fetch_elections
from .voting_districts import load_vtds
from .maps import plot_plan
from .partisan_metrics import METRICS, favors, partisan_summary
from .plans import Plans
from .population import population_summary

MAP_DPI = 200


def choose_plans(all_plans: list[str], words: list[str], use_all: bool = False,
                 ask: Callable[[str], str] = input) -> list[str]:
    """Pick plans from --all, words matched against plan names, or a numbered menu."""
    if use_all:
        return all_plans
    if words:
        chosen = [p for p in all_plans if any(w.lower() in p.lower() for w in words)]
        if not chosen:
            print(f"No plan names contain: {', '.join(words)}")
        return chosen

    for i, p in enumerate(all_plans, 1):
        print(f"{i:>3}. {p}")
    answer = ask("\nType plan number(s) separated by commas (or 'all'): ").strip()
    if answer.lower() == "all":
        return all_plans
    picks = []
    for part in answer.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(all_plans):
            picks.append(all_plans[int(part) - 1])
        elif part:
            print(f"  Ignoring '{part}' (not a number from the list)")
    return picks


def print_population_report(plan: str, out: pd.DataFrame, s: dict) -> None:
    print("\n" + "=" * 72 + f"\n{plan}\n" + "=" * 72)
    print(f"Districts: {s['districts']}   (expected {N_DISTRICTS})")
    print(f"Ideal population per district: {s['ideal']:,.0f}")
    print(f"Largest:  District {int(s['largest']['district']):>2}  "
          f"{int(s['largest']['pop_total']):>9,}  ({s['max_pct']:+.2f}%)")
    print(f"Smallest: District {int(s['smallest']['district']):>2}  "
          f"{int(s['smallest']['pop_total']):>9,}  ({s['min_pct']:+.2f}%)")
    print(f"Overall range (largest % minus smallest %): {s['overall_range']:.2f} points")
    if s["unassigned_pop"]:
        print(f"!! {s['unassigned_pop']:,} people live in voting districts this plan left unassigned")
    print(f"Majority-minority districts (by voting-age pop): {s['majority_minority_vap']}")
    print(f"Majority-Black districts (by voting-age pop):    {s['majority_black_vap']}")

    print("\n district   population   deviation   % dev   VTDs  counties")
    for _, r in out.iterrows():
        print(f" {int(r['district']):>8}   {int(r['pop_total']):>10,}   {int(r['deviation']):>+9,}"
              f"   {r['pct_deviation']:>+5.2f}   {int(r['n_vtds']):>4}   {int(r['n_counties']):>7}")


def print_partisan_report(plan: str, summary: pd.DataFrame) -> None:
    print("\n" + "=" * 72 + f"\n{plan}\n" + "=" * 72)
    for _, s in summary.iterrows():
        if s["n_districts"] != N_DISTRICTS:
            print(f"   note: {s['n_districts']} districts (expected {N_DISTRICTS})")
        print(f"[{s['election']}]  Dem vote {s['dem_vote_share']:.1%}  ->  "
              f"Dem seats {s['dem_seats']}/{s['n_districts']}")
        for m in METRICS:
            val = s[m]
            txt = "n/a (one party wins every seat)" if pd.isna(val) else f"{val:+.3f}"
            print(f"   {m:<15} {txt:>8}   favors {favors(val)}")


def print_efficiency_gap_ranking(metrics: pd.DataFrame) -> None:
    election = metrics["election"].iloc[0]
    rank = (metrics[metrics["election"] == election]
            .assign(abs_eg=lambda d: d["efficiency_gap"].abs()).sort_values("abs_eg"))
    print(f"\nPlans ranked by efficiency gap ({election}), closest to zero first:")
    for _, r in rank.iterrows():
        print(f"   {r['efficiency_gap']:+.3f}  {r['plan']}")


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------
def write_summary(plans: Plans, plan: str, out_dir: Path) -> tuple[Path, pd.DataFrame, dict]:
    """Save one plan's population summary as <plan>_summary.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table, stats = population_summary(plans.get(plan, geometry=False))
    path = out_dir / f"{plan}_summary.csv"
    table.insert(0, "plan", plan)
    table.to_csv(path, index=False)
    return path, table, stats


def write_map(plans: Plans, plan: str, out_dir: Path) -> Path:
    """Save one plan's map as <plan>.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{plan}.png"
    plot_plan(plans.get(plan), plan).savefig(path, dpi=MAP_DPI, bbox_inches="tight",
                                             facecolor="white")
    return path


def write_partisan(assignment: pd.DataFrame, plan: str, votes: pd.DataFrame,
                   elections: dict[str, tuple[str, str]], out_dir: Path) -> pd.DataFrame:
    """Save one plan's votes per district as <plan>__<election>_districts.csv.
    Returns the plan's metrics, one row per election."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, districts = partisan_summary(assignment, votes, elections)
    for election, table in districts.items():
        table.insert(0, "plan", plan)
        table.to_csv(out_dir / f"{plan}__{election}_districts.csv", index=False)
    summary.insert(0, "plan", plan)
    return summary


def save_partisan_metrics(summaries: list[pd.DataFrame], out_dir: Path) -> Path:
    """Every plan's metrics in one table, partisan_metrics.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "partisan_metrics.csv"
    pd.concat(summaries, ignore_index=True).to_csv(path, index=False)
    return path


def overview_row(plan: str, s: dict, partisan: pd.DataFrame | None = None) -> dict:
    """One line of overview.csv: the headline numbers for comparing plans.
    Partisan columns come from the first election scored (normally "composite")."""
    row = {
        "plan": plan,
        "districts": s["districts"],
        "max_pct_deviation": round(s["max_pct"], 2),
        "min_pct_deviation": round(s["min_pct"], 2),
        "overall_range": round(s["overall_range"], 2),
        "unassigned_pop": s["unassigned_pop"],
        "majority_minority_vap": s["majority_minority_vap"],
        "majority_black_vap": s["majority_black_vap"],
    }
    if partisan is not None:
        p = partisan.iloc[0]
        row["election"] = p["election"]
        row["dem_vote_share"] = round(p["dem_vote_share"], 4)
        row["dem_seats"] = int(p["dem_seats"])
        row.update({m: round(p[m], 4) for m in METRICS})
    return row


def write_results(plans: Plans, chosen: list[str], out: Path, votes: pd.DataFrame) -> Path:
    """Summary, map and partisan metrics for each chosen plan, plus overview.csv.

    out/
      overview.csv
      summaries/<plan>_summary.csv
      maps/<plan>.png
      partisan/partisan_metrics.csv
      partisan/<plan>__<election>_districts.csv
    """
    elections = available_elections(votes)
    rows, partisan = [], []
    for i, plan in enumerate(chosen, 1):
        _, _, stats = write_summary(plans, plan, out / "summaries")
        write_map(plans, plan, out / "maps")
        p = write_partisan(plans.assignments[plan], plan, votes, elections, out / "partisan")
        partisan.append(p)
        rows.append(overview_row(plan, stats, p))
        print(f"[{i}/{len(chosen)}] {plan}")

    save_partisan_metrics(partisan, out / "partisan")
    path = out / "overview.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved results for {len(chosen)} plans in {out.resolve()}")
    return path


def run_everything(plans: str | list[str] = "all", out: str | Path = "results",
                   data: Plans | None = None, votes: pd.DataFrame | None = None) -> Path:
    """Everything the run command does, callable from a script, notebook or IDE.

    plans: "all", one word, or a list of words matched against plan names.
    out:   output folder (created if needed).
    data:  an already loaded Plans; downloaded with Plans.load() if left out.
    votes: already loaded election results; downloaded with fetch_elections() if left out.
    Returns the path of overview.csv.
    """
    words = [plans] if isinstance(plans, str) else list(plans)
    if not words:
        raise ValueError('plans is empty; use "all" or a list of words from plan names')

    if data is None:
        data = Plans.load()
    chosen = data.names if words == ["all"] else data.match(words)
    if not chosen:
        raise ValueError(f"No plan names contain: {', '.join(words)}")
    if votes is None:
        votes = fetch_elections()
    return write_results(data, chosen, Path(out), votes)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(args) -> None:
    plans = Plans.load()
    chosen = choose_plans(plans.names, args.plans, args.all)
    if chosen:
        write_results(plans, chosen, args.out, fetch_elections())


def cmd_list(args) -> None:
    vtds = load_vtds()
    for i, (name, assign) in enumerate(fetch_plan_assignments().items(), 1):
        missing = (~vtds["GEOID"].isin(assign["GEOID"])).sum()
        print(f"{i:>3}. {name}" + (f"  ({missing} voting districts unassigned)" if missing else ""))


def cmd_summary(args) -> None:
    plans = Plans.load()
    for plan in choose_plans(plans.names, args.plans, args.all):
        path, table, stats = write_summary(plans, plan, args.out)
        print_population_report(plan, table, stats)
        print(f"\nSaved {path}")


def cmd_maps(args) -> None:
    plans = Plans.load()
    chosen = choose_plans(plans.names, args.plans, args.all)
    for i, plan in enumerate(chosen, 1):
        path = write_map(plans, plan, args.out)
        print(f"[{i}/{len(chosen)}] saved {path}")


def cmd_partisan(args) -> None:
    # Plans and votes only: no shapes and no Census key needed
    assignments = fetch_plan_assignments()
    chosen = choose_plans(list(assignments), args.plans, args.all)
    if not chosen:
        return
    votes = fetch_elections()
    elections = available_elections(votes)

    summaries = []
    for plan in chosen:
        summary = write_partisan(assignments[plan], plan, votes, elections, args.out)
        print_partisan_report(plan, summary)
        summaries.append(summary)
    print(f"\nSaved {save_partisan_metrics(summaries, args.out)}")
    if len(chosen) > 1:
        print_efficiency_gap_ranking(pd.concat(summaries, ignore_index=True))


def cmd_codebook(args) -> None:
    print(fetch_codebook().to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pa_redistricting",
                                     description="Analyze PA Senate citizen redistricting plans.")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide progress messages")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list plans and unassigned voting districts").set_defaults(func=cmd_list)
    sub.add_parser("codebook", help="show Census column labels").set_defaults(func=cmd_codebook)

    for name, func, default_out, help_text in [
        ("run", cmd_run, "results", "summaries, maps and an overview, all in one folder"),
        ("summary", cmd_summary, "summaries", "population summary per Senate district"),
        ("maps", cmd_maps, "maps", "PNG map per plan"),
        ("partisan", cmd_partisan, "partisan", "efficiency gap and other partisan metrics"),
    ]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("plans", nargs="*", help="words to match in plan names")
        cmd.add_argument("--all", action="store_true", help="every plan")
        cmd.add_argument("--out", type=Path, default=Path(default_out), help="output folder")
        cmd.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")
    try:
        args.func(args)
    except MissingAPIKeyError as e:
        raise SystemExit(str(e)) from None
