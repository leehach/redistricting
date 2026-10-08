"""
STEP 5 (in-memory version of SenatePlanPopulation(Step5).py)

Senate district population summary for the plans you choose. Takes each plan
from the in-memory Plans object (step 3, no geometry needed) and adds the
voting districts up by Senate district.

How to run:
  python step5_plan_summary.py                     -> numbered list; type the plan number(s)
  python step5_plan_summary.py allentown           -> every plan whose name contains "allentown"
  python step5_plan_summary.py allentown ccfd      -> several plans at once
  python step5_plan_summary.py --all               -> every plan

For each plan it prints a short report and saves summaries/<plan>_summary.csv
(the final output) with one row per Senate district.

From a notebook:
  from step3_plans import Plans
  from step5_plan_summary import summarize
  plans = Plans.load()
  out, stats = summarize(plans.get("allentown_map_by_as_jan-4-2022", geometry=False))

Requirements:  pip install geopandas pandas pyogrio
"""

import sys
from pathlib import Path

import pandas as pd

from step3_plans import Plans, choose_plans

OUT_DIR = Path("summaries")
N_DISTRICTS = 50

# Counts that are added up per Senate district
SUM_FIELDS = [
    "pop_total", "pop_white", "pop_black", "pop_aian", "pop_asian", "pop_nhpi",
    "pop_other", "pop_two_plus", "pop_hispanic", "pop_nh_white", "pop_nh_black",
    "vap_total", "vap_white", "vap_black", "vap_hispanic", "vap_nh_white", "vap_nh_black",
    "gq_total", "housing_total", "housing_occupied", "housing_vacant",
]


# ---------------------------------------------------------------------------
# The summary itself 
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_districts: int = N_DISTRICTS) -> tuple[pd.DataFrame, dict]:
    """Add up voting districts by Senate district and compute deviation."""
    fields = [f for f in SUM_FIELDS if f in df.columns]
    df = df.copy()
    df[fields] = df[fields].apply(pd.to_numeric, errors="coerce").fillna(0)

    state_pop = df["pop_total"].sum()          # everyone, assigned or not
    ideal = state_pop / n_districts

    assigned = df[df["district"].notna()]
    unassigned_pop = int(df.loc[df["district"].isna(), "pop_total"].sum())

    g = assigned.groupby("district")
    out = g[fields].sum()
    out.insert(0, "n_vtds", g.size())
    out.insert(1, "n_counties", g["COUNTYFP"].nunique())

    # Population balance
    out["ideal_pop"] = round(ideal)
    out["deviation"] = out["pop_total"] - ideal
    out["pct_deviation"] = 100 * out["deviation"] / ideal

    # Shares, recomputed from the summed counts 
    def pct(num, den):
        return (100 * out[num] / out[den].where(out[den] > 0)).round(2)

    out["pct_hispanic"] = pct("pop_hispanic", "pop_total")
    out["pct_nh_white"] = pct("pop_nh_white", "pop_total")
    out["pct_nh_black"] = pct("pop_nh_black", "pop_total")
    out["pct_vap_nh_black"] = pct("vap_nh_black", "vap_total")
    out["pct_vap_hispanic"] = pct("vap_hispanic", "vap_total")
    out["pct_vap_minority"] = (100 - pct("vap_nh_white", "vap_total")).round(2)

    out["deviation"] = out["deviation"].round().astype(int)
    out["pct_deviation"] = out["pct_deviation"].round(2)
    out = out.reset_index()
    out["district"] = out["district"].astype(int)
    out = out.sort_values("district")

    stats = {
        "districts": len(out),
        "state_pop": int(state_pop),
        "ideal": ideal,
        "unassigned_pop": unassigned_pop,
        "largest": out.loc[out["pop_total"].idxmax()],
        "smallest": out.loc[out["pop_total"].idxmin()],
        "max_pct": out["pct_deviation"].max(),
        "min_pct": out["pct_deviation"].min(),
        "majority_minority_vap": int((out["pct_vap_minority"] > 50).sum()),
        "majority_black_vap": int((out["pct_vap_nh_black"] > 50).sum()),
    }
    stats["overall_range"] = stats["max_pct"] - stats["min_pct"]
    return out, stats


def report(plan: str, out: pd.DataFrame, s: dict) -> None:
    print("\n" + "=" * 72)
    print(plan)
    print("=" * 72)
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


# ---------------------------------------------------------------------------
def main() -> None:
    plans = Plans.load()
    chosen = choose_plans(plans.names, sys.argv[1:])
    if not chosen:
        print("Nothing to summarize.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    for plan in chosen:
        out, stats = summarize(plans.get(plan, geometry=False))
        report(plan, out, stats)
        path = OUT_DIR / f"{plan}_summary.csv"
        out.insert(0, "plan", plan)
        out.to_csv(path, index=False)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
