"""Analyze Pennsylvania Senate citizen redistricting plans.

    from pa_redistricting import Plans, population_summary, plot_plan
    plans = Plans.load()
    table, stats = population_summary(plans.get(plans.names[0], geometry=False))

    from pa_redistricting import fetch_elections, partisan_summary
    summary, districts = partisan_summary(plans.assignments[plans.names[0]], fetch_elections())

Command line:  python -m pa_redistricting --help
"""

from .census import MissingAPIKeyError
from .election_results import fetch_elections
from .maps import plot_plan
from .partisan_metrics import partisan_summary
from .plans import Plans
from .population import population_summary

__all__ = ["MissingAPIKeyError", "Plans", "fetch_elections", "partisan_summary", "plot_plan",
           "population_summary"]
