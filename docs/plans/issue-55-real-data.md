# Plan: real plans in the Qt DEA explorer (issue #55)

Started 2026-11-12. Owner: Ben. POC branch: `bdp/qt-real-data-poc`, made
from `qt-frontier-explorer` (Charlie's Qt app).

## Goal

In the Qt app, a user loads real plans: one file per plan, a folder of
files, or one file with many plans. The app measures each plan, scores
the plans with SBM DEA, and shows the frontier.

## Shape

We follow R `redist`, which keeps the map apart from the plans. We use
the tables that `compactness.py` already uses:

- `geounits`: one row per voting district (VTD), indexed by geoid, with
  a projected shape, `population`, `county`, `dem` and `rep`.
- `plans`: one row per geounit, indexed by geoid, and one column per
  plan. Each cell is a district.

Every kind of upload becomes one `plans` table, so the rest of the code
has one input shape.

## Decisions (2026-11-12)

- **Geounits are VTDs.** Dave's Redistricting App (DRA) has two CSV
  exports, both `GEOID20,District`. The default one assigns census blocks
  (15-digit geoids). "Precinct Assignment" assigns VTDs (11 digits). The
  POC reads only VTD files and tells the user what to export if they load
  a block file. Block files are a later phase: map blocks to VTDs with the
  Census block assignment file (BAF).
- **Base data comes from TIGER/Line (shapes) and ALARM (population,
  county, votes).** No Census API key is needed. It stays in memory for
  one app run.
- **Metrics for the POC:** mean Polsby-Popper (higher), max population
  deviation in percent (lower), split counties (lower), and the absolute
  efficiency gap (lower). All are above 0 on the 50 citizen plans, as SBM
  needs.
- **DEA is SBM through `dea.score`.** The app no longer uses its own
  radial engine for real plans.

## Open

- **Peers and targets panel.** Charlie's side panel shows "frontier
  peers" and "frontier targets" from his own radial DEA. `dea.score` gives
  only scores, so the POC hides them for real plans. Choose: drop them, or
  add peers and targets to our SBM code as a later phase.
- **A value of 0.** SBM needs every value above 0. A plan with 0 split
  counties or a 0 efficiency gap would fail. We need a rule (for example,
  a small offset).
- **Speed.** Measuring 50 plans takes about 74 s. Scoring with SBM takes
  about 1 s. Which metric takes most of the time is not measured yet.

## Phases (one PR each)

0. Charlie's Qt PR merges into `dev`.
1. `plan_metrics.combine_plans` and checks, with tests (no downloads).
2. `plan_metrics.measure` and `load_pa_geounits`, with tests and an
   `examples/` script.
3. The Qt app uses `dea.score(..., "sbm")`. Remove `run_dea` and its
   scipy code, and settle the peers and targets panel.
4. The Qt load screen: files or a folder, a progress display, clear
   errors, and later a choice of metrics and directions.
5. Later: DRA block files, more metrics, and one file with many plans
   in long form (plan, geoid, district).
