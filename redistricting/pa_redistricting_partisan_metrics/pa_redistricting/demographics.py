"""Demographic counts per PA voting district, with one set of column names
whichever source they come from.

VEST (via the ALARM Project) and the Census API both come from the 2020
PL 94-171 release. Race columns are NON-HISPANIC unless the name says otherwise:

    pop_hispanic       Hispanic or Latino, any race
    pop_nh_white       White alone, not Hispanic      (VEST: pop_white)
    pop_nh_black       Black alone, not Hispanic      (VEST: pop_black)
    ...                aian, asian, nhpi, other, two_plus
    vap_*              the same, for the voting-age (18+) population

    from pa_redistricting import fetch_elections
    from pa_redistricting.demographics import vest_demographics, compare_sources
    demo = vest_demographics(fetch_elections())
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

RACES = ["white", "black", "aian", "asian", "nhpi", "other", "two_plus"]
VEST_RACES = ["white", "black", "aian", "asian", "nhpi", "other", "two"]

# Counts every demographics source provides, in this order
COUNT_COLUMNS = [
    "pop_total", "pop_hispanic", *[f"pop_nh_{r}" for r in RACES],
    "vap_total", "vap_hispanic", *[f"vap_nh_{r}" for r in RACES],
]

# VEST column -> package column
VEST_COLUMNS = {
    "pop": "pop_total", "pop_hisp": "pop_hispanic",
    "vap": "vap_total", "vap_hisp": "vap_hispanic",
    **{f"{p}_{v}": f"{p}_nh_{r}"
       for p in ("pop", "vap") for v, r in zip(VEST_RACES, RACES)},
}

# Percentage columns: name -> (numerator, denominator). Added only when both exist.
PERCENTAGES = {
    "pct_hispanic": ("pop_hispanic", "pop_total"),
    "pct_nh_white": ("pop_nh_white", "pop_total"),
    "pct_nh_black": ("pop_nh_black", "pop_total"),
    "pct_vap": ("vap_total", "pop_total"),
    "pct_vacant": ("housing_vacant", "housing_total"),
}


def add_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Add the PERCENTAGES columns that df has counts for; blank where the
    denominator is zero."""
    def pct(num: str, den: str) -> pd.Series:
        d = df[den].astype("float").where(df[den] > 0)
        return (100 * df[num].astype("float") / d).round(2)

    return df.assign(**{name: pct(num, den) for name, (num, den) in PERCENTAGES.items()
                        if num in df.columns and den in df.columns})


def vest_demographics(vest: pd.DataFrame) -> pd.DataFrame:
    """The demographic columns of the VEST file (fetch_elections()), renamed to
    COUNT_COLUMNS. One row per voting district, keyed by GEOID."""
    missing = [c for c in VEST_COLUMNS if c not in vest.columns]
    if missing:
        raise ValueError(f"VEST file is missing demographic columns: {', '.join(missing)}")
    df = vest[["GEOID", *VEST_COLUMNS]].rename(columns=VEST_COLUMNS)
    df[COUNT_COLUMNS] = df[COUNT_COLUMNS].astype("Int64")
    return add_percentages(df).sort_values("GEOID").reset_index(drop=True)


def compare_sources(vest: pd.DataFrame, census: pd.DataFrame
                    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Check VEST against the Census API, column by column. Meant as a one-off
    check: both come from the same PL 94-171 release, so they should agree.

        from pa_redistricting.census import fetch_census
        table, only = compare_sources(vest_demographics(fetch_elections()), fetch_census())

    Returns (one row per count column, {"only_vest": GEOIDs, "only_census": GEOIDs}).
    Counts are compared only for GEOIDs in both. Matching sources show
    vtds_differ == 0 and difference == 0 on every row.
    """
    v, c = vest.set_index("GEOID"), census.set_index("GEOID")
    only = {"only_vest": sorted(v.index.difference(c.index)),
            "only_census": sorted(c.index.difference(v.index))}
    for name, ids in only.items():
        if ids:
            log.warning("%d GEOIDs %s, e.g. %s", len(ids), name.replace("_", " in "), ids[0])

    both = v.index.intersection(c.index)
    rows = []
    for col in COUNT_COLUMNS:
        a, b = v.loc[both, col], c.loc[both, col]
        rows.append({"column": col,
                     "vtds_differ": int((a != b).fillna(True).sum()),
                     "vest_total": int(a.sum()),
                     "census_total": int(b.sum()),
                     "difference": int(a.sum() - b.sum())})
    return pd.DataFrame(rows), only
