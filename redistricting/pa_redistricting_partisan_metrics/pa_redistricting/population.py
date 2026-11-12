"""Population summary of a plan by Senate district.

    from pa_redistricting import Plans, population_summary
    plans = Plans.load()
    table, stats = population_summary(plans.get("allentown_map_by_as_jan-4-2022",
                                                geometry=False))
"""

import pandas as pd

from .census import EXTRA_COLUMNS
from .config import N_DISTRICTS
from .demographics import COUNT_COLUMNS

# Counts added up per Senate district (whichever ones the plan's table has)
SUM_FIELDS = [*COUNT_COLUMNS, *EXTRA_COLUMNS]


def population_summary(df: pd.DataFrame,
                       n_districts: int = N_DISTRICTS) -> tuple[pd.DataFrame, dict]:
    """Add up voting districts by Senate district and compute population deviation.

    df: one plan from Plans.get(...). Returns (one row per district, overall stats).
    The ideal population uses everyone in the state, including people in
    voting districts the plan left unassigned.
    """
    fields = [f for f in SUM_FIELDS if f in df.columns]
    df = df.copy()
    df[fields] = df[fields].apply(pd.to_numeric, errors="coerce").fillna(0)

    state_pop = df["pop_total"].sum()
    ideal = state_pop / n_districts

    assigned = df[df["district"].notna()]
    unassigned_pop = int(df.loc[df["district"].isna(), "pop_total"].sum())

    g = assigned.groupby("district")
    out = g[fields].sum()
    out.insert(0, "n_vtds", g.size())
    out.insert(1, "n_counties", g["COUNTYFP"].nunique())

    out["ideal_pop"] = round(ideal)
    out["deviation"] = out["pop_total"] - ideal
    out["pct_deviation"] = 100 * out["deviation"] / ideal

    def pct(num, den):  # shares recomputed from summed counts, never averaged
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
    out = out.sort_values("district").reset_index(drop=True)

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
