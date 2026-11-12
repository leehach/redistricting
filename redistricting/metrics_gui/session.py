"""The session's in-memory plans object: plan name -> its reconciled table. nothing is saved to disk

    plans = SessionPlans()
    plans.add("my_plan", gdf, notes=["voting data: 9,178 of 9,178 precincts matched"])
    plans.names              # ["my_plan"]
    plans.get("my_plan")     # the table, ready for MetricsDialog(data=..., name=...)
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Plan:
    name: str
    data: pd.DataFrame  # 1 row per precinct: id, district, votes, demographics (+ geometry)
    source: str = ""    # where it came from, ex. "disk" or "repo"
    notes: list[str] = field(default_factory=list)  # what reconcile said about the join

    @property
    def has_shapes(self) -> bool:
        return getattr(self.data, "active_geometry_name", None) is not None

    def summary(self) -> str:
        fields = len(self.data.columns) - self.has_shapes
        shapes = "with shapes" if self.has_shapes else "no shapes"
        return f"{len(self.data):,} precincts, {fields} fields, {shapes}"


class SessionPlans:
    def __init__(self):
        self._plans: dict[str, Plan] = {}

    @property
    def names(self) -> list[str]:
        return list(self._plans)

    def add(self, name: str, data: pd.DataFrame, source: str = "", notes: list[str] | None = None,
            replace: bool = False) -> Plan:
        name = name.strip()
        if not name:
            raise ValueError("The plan needs a name")
        if name in self._plans and not replace:
            raise KeyError(f"There is already a plan named {name!r}")
        self._plans[name] = Plan(name, data, source, list(notes or []))
        return self._plans[name]

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
        return self.plan(name).data

    def remove(self, name: str) -> None:
        del self._plans[self.plan(name).name]

    def __contains__(self, name: str) -> bool:
        return name in self._plans

    def __iter__(self):
        return iter(self._plans.values())

    def __len__(self) -> int:
        return len(self._plans)

    def __repr__(self) -> str:
        return f"<SessionPlans: {len(self)} plans>"
