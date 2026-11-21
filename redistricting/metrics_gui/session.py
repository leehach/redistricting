"""The session's in-memory plans object: plan name -> its district assignments. nothing is saved to disk

A plan is the minimum: one row per unit, GEOID + district. Votes, demographics and shapes are not part
of a plan, they're joined on when an analysis runs (see reconcile.py).

    plans = SessionPlans()
    plans.add(Plan("my_plan", assignment, kind="PAF"))
    plans.names              # ["my_plan"]
    plans.get("my_plan")     # DataFrame[GEOID, district]
    plans.long_table()       # every plan stacked: GEOID, plan, district
"""

from dataclasses import dataclass, field

import pandas as pd

COLUMNS = ["GEOID", "district"]
KINDS = {"PAF": "precinct", "BAF": "block"}  # precinct / block assignment file


@dataclass
class Plan:
    name: str
    assignment: pd.DataFrame  # GEOID (text), district (int)
    kind: str = "PAF"         # "PAF" (precincts) or "BAF" (Census blocks)
    source: str = ""          # the file it came from
    notes: list[str] = field(default_factory=list)  # what happened when it was read

    @property
    def n_units(self) -> int:
        return len(self.assignment)

    @property
    def n_districts(self) -> int:
        return int(self.assignment["district"].nunique())

    def summary(self) -> str:
        return f"{self.kind}: {self.n_units:,} {KINDS[self.kind]}s in {self.n_districts} districts"


class SessionPlans:
    def __init__(self):
        self._plans: dict[str, Plan] = {}

    @property
    def names(self) -> list[str]:
        return list(self._plans)

    def add(self, plan: Plan, replace: bool = False) -> Plan:
        plan.name = plan.name.strip()
        if not plan.name:
            raise ValueError("The plan needs a name")
        if plan.name in self._plans and not replace:
            raise KeyError(f"There is already a plan named {plan.name!r}")
        if list(plan.assignment.columns) != COLUMNS:
            raise ValueError(f"A plan's table has exactly the columns {COLUMNS}, "
                             f"not {list(plan.assignment.columns)}")
        if plan.kind not in KINDS:
            raise ValueError(f"kind is one of {list(KINDS)}, not {plan.kind!r}")
        self._plans[plan.name] = plan
        return plan

    def free_name(self, name: str) -> str:
        """name if it isn't taken, otherwise name_2, name_3 ..."""
        if name not in self._plans:
            return name
        n = 2
        while f"{name}_{n}" in self._plans:
            n += 1
        return f"{name}_{n}"

    def plan(self, name: str) -> Plan:
        if name not in self._plans:
            raise KeyError(f"No plan named {name!r}")
        return self._plans[name]

    def get(self, name: str) -> pd.DataFrame:
        return self.plan(name).assignment

    def remove(self, name: str) -> None:
        del self._plans[self.plan(name).name]

    def long_table(self) -> pd.DataFrame:
        """every plan in one table: GEOID, plan, district"""
        if not self._plans:
            return pd.DataFrame(columns=["GEOID", "plan", "district"])
        return pd.concat([p.assignment.assign(plan=p.name)[["GEOID", "plan", "district"]]
                          for p in self], ignore_index=True)

    def __contains__(self, name: str) -> bool:
        return name in self._plans

    def __iter__(self):
        return iter(self._plans.values())

    def __len__(self) -> int:
        return len(self._plans)

    def __repr__(self) -> str:
        return f"<SessionPlans: {len(self)} plans>"
