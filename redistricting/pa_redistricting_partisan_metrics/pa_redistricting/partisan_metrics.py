"""Partisan asymmetry metrics for a plan, following PlanScore's definitions
(https://planscore.org/metrics/).

    from pa_redistricting import fetch_elections, partisan_summary
    summary, districts = partisan_summary(plan_df, fetch_elections())

plan_df is anything with GEOID and district columns: an entry of
Plans.assignments or Plans.get(name, geometry=False).

Sign convention for every metric: POSITIVE favors Republicans,
NEGATIVE favors Democrats.
"""

import logging
import math

import numpy as np
import pandas as pd

from .election_results import available_elections

log = logging.getLogger(__name__)

METRICS = ["efficiency_gap", "partisan_bias", "mean_median", "declination"]


def efficiency_gap(dem: np.ndarray, rep: np.ndarray) -> float:
    """(Dem wasted - Rep wasted) / total votes. Wasted votes are every vote for
    the loser plus the winner's votes above 50%. Ties count as Republican wins."""
    total = dem + rep
    half = total / 2
    dem_won = dem > rep
    wasted_dem = np.where(dem_won, dem - half, dem)
    wasted_rep = np.where(dem_won, rep, rep - half)
    return float((wasted_dem.sum() - wasted_rep.sum()) / total.sum())


def partisan_bias(dem_share: np.ndarray, statewide_dem_share: float) -> float:
    """Shift every district equally until the state is tied 50-50, then
    0.5 minus the Democratic seat share."""
    shifted = dem_share + (0.5 - statewide_dem_share)
    return float(0.5 - (shifted > 0.5).mean())


def mean_median(dem_share: np.ndarray) -> float:
    """Mean minus median of the district Democratic shares."""
    return float(dem_share.mean() - np.median(dem_share))


def declination(dem_share: np.ndarray) -> float:
    """Warrington's declination in fractions of 90 degrees. NaN when one party
    wins every seat."""
    n = len(dem_share)
    rep_won = dem_share[dem_share <= 0.5]
    dem_won = dem_share[dem_share > 0.5]
    k = len(rep_won)
    if k == 0 or k == n:
        return float("nan")
    theta_r = math.atan((0.5 - rep_won.mean()) / (k / (2 * n)))
    theta_d = math.atan((dem_won.mean() - 0.5) / ((n - k) / (2 * n)))
    return 2 * (theta_d - theta_r) / math.pi


def score_districts(districts: pd.DataFrame) -> dict:
    """Seats and all four metrics from one row per district with 'dem' and 'rep'."""
    dem = districts["dem"].to_numpy(float)
    rep = districts["rep"].to_numpy(float)
    share = dem / (dem + rep)
    statewide = dem.sum() / (dem.sum() + rep.sum())
    return {
        "n_districts": len(districts),
        "dem_vote_share": float(statewide),
        "dem_seats": int((share > 0.5).sum()),
        "rep_seats": int((share <= 0.5).sum()),
        "dem_seat_share": float((share > 0.5).mean()),
        "efficiency_gap": efficiency_gap(dem, rep),
        "partisan_bias": partisan_bias(share, statewide),
        "mean_median": mean_median(share),
        "declination": declination(share),
    }


def favors(x: float) -> str:
    """Which party a metric value favors under the sign convention above."""
    if x is None or np.isnan(x):
        return "n/a"
    if x == 0:
        return "balanced"
    return "Republicans" if x > 0 else "Democrats"


def partisan_summary(plan_df: pd.DataFrame, votes: pd.DataFrame,
                     elections: dict[str, tuple[str, str]] | None = None
                     ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score one plan under each election.

    Returns (one row per election, {election: one row per district}).
    Votes in voting districts the plan leaves unassigned are excluded.
    """
    elections = elections or available_elections(votes)
    vote_cols = sorted({c for pair in elections.values() for c in pair})

    plan_df = plan_df[["GEOID", "district"]].assign(
        GEOID=lambda d: d["GEOID"].astype(str).str.strip())
    v = votes[["GEOID"] + vote_cols]

    # Votes are the base table, so VTDs a plan leaves out show up as unassigned
    df = v.merge(plan_df, on="GEOID", how="left")
    df[vote_cols] = df[vote_cols].fillna(0)

    not_in_votes = (~plan_df["GEOID"].isin(v["GEOID"])).mean()
    if not_in_votes > 0.02:
        log.warning("%.0f%% of this plan's GEOIDs have no election data", 100 * not_in_votes)

    unassigned_votes = df.loc[df["district"].isna(), vote_cols].sum().sum()
    if unassigned_votes:
        log.warning("%s votes fall in unassigned voting districts and are excluded",
                    f"{unassigned_votes:,.0f}")
    df = df.dropna(subset=["district"]).astype({"district": int})

    rows, district_tables = [], {}
    for name, (dcol, rcol) in elections.items():
        dist = (df.groupby("district")[[dcol, rcol]].sum()
                .rename(columns={dcol: "dem", rcol: "rep"}))
        rows.append({"election": name, **score_districts(dist)})

        dist["dem_share"] = (dist["dem"] / (dist["dem"] + dist["rep"])).round(4)
        dist["winner"] = np.where(dist["dem_share"] > 0.5, "D", "R")
        district_tables[name] = dist.reset_index()

    summary = pd.DataFrame(rows)
    for m in METRICS:
        summary[f"{m}_favors"] = summary[m].apply(favors)
    return summary, district_tables
