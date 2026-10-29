"""Fields and metrics for the GUI, in the order they show up. add new ones at the bottom"""

import sys
from pathlib import Path

import pandas as pd

from pa_redistricting.partisan_metrics import favors

from registry import GEOMETRY, field, metric

#Repo root to import classmate modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from redistricting import compactness
except ImportError:  # no compactness.py on this branch
    compactness = None


# fields (the lists are what auto-match looks for)
field("dem_votes", "Democratic votes",
      ["dem_votes", "demvotes", "votes_dem", "dem", "democratic", "democrat", "ndv"], numeric=True)
field("rep_votes", "Republican votes",
      ["rep_votes", "repvotes", "votes_rep", "rep", "republican", "nrv"], numeric=True)
field("county", "County", ["county", "countyfp", "countyfp20", "cntyfp", "county_name", "cnty"])


# partisan. positive = favors R, negative = favors D
PARTISAN_NEEDS = ["dem_votes", "rep_votes"]


# votes, share and winner per district + seats for the whole plan
@metric("Partisan", "Partisan favor", needs=PARTISAN_NEEDS)
def partisan_favor(ctx):
    scores, districts = ctx.partisan
    return districts.assign(dem_vote_share=round(scores["dem_vote_share"], 4),
                            dem_seats=int(scores["dem_seats"]), rep_seats=int(scores["rep_seats"]))


# the other 4 are all one plan-wide number, so one function makes all of them
def _partisan_score(name):
    def score(ctx):
        value = ctx.partisan[0][name]
        return {name: round(value, 4), f"{name}_favors": favors(value)}
    return score


for _name, _label in [("efficiency_gap", "Efficiency gap"), ("mean_median", "Mean-median"),
                      ("declination", "Declination"), ("partisan_bias", "Partisan bias")]:
    metric("Partisan", _label, needs=PARTISAN_NEEDS)(_partisan_score(_name))


# compactness, uses the other group's compactness.py. 1 = most compact
def _compactness(name):
    def score(ctx):
        shapes = ctx.district_shapes.geometry
        return pd.Series(compactness.SHAPE_MEASURES[name](shapes), index=shapes.index,
                         name=name).round(4)
    return score


if compactness is not None:
    for _name, _label in [("polsby_popper", "Polsby-Popper"), ("schwartzberg", "Schwartzberg"),
                          ("reock", "Reock"), ("convex_hull", "Convex hull ratio")]:
        metric("Compactness", _label, needs=[GEOMETRY])(_compactness(_name))


# county splits: counties per district, and how many counties got split up
@metric("County splits", "County splits", needs=["county"])
def county_splits(ctx):
    county, district = ctx.column("county"), ctx.district
    per_district = county.groupby(district).nunique().rename("n_counties")
    districts_per_county = district.groupby(county).nunique()
    return pd.DataFrame(per_district).assign(
        counties_split_in_plan=int((districts_per_county > 1).sum()))
