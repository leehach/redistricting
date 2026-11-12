# pa_redistricting

Analyze Pennsylvania Senate citizen redistricting plans: join each plan to 2020
voting-district shapes and demographic data (VEST by default, or the Census API),
summarize population by district, draw maps, and score partisan asymmetry
(efficiency gap, partisan bias, mean-median difference, declination).

## Install

From this folder, with the `redistricting` environment active:

```powershell
pip install -e ".[test]"
```

A Census API key is only needed for `--demographics census` or `vest+census`
(group quarters, housing, race including Hispanics). The default VEST data needs no key.
To use the Census options, set your key once (free from https://api.census.gov/data/key_signup.html):

```powershell
conda env config vars set CENSUS_API_KEY=PASTE_KEY_HERE -n gus8066
conda deactivate
conda activate gus8066
```

## Run everything

Open `run_all.py` in VS Code and press Run. It downloads the data once and saves,
for every plan:

```
results/
  overview.csv                  one row per plan, for comparing plans
  summaries/<plan>_summary.csv  population per Senate district
  maps/<plan>.png               map of each plan
  partisan/partisan_metrics.csv every plan's partisan metrics, per election
  partisan/<plan>__<election>_districts.csv   votes and winner per district
```

`overview.csv` includes the partisan metrics for the first election scored
(normally `composite`, the average of every statewide race). Positive values favor
Republicans, negative values favor Democrats.

Edit `PLANS` and `OUT` at the top of the file to choose plans or the output folder.
The terminal equivalent is `python -m pa_redistricting run --all`.
`run_all.py` uses VEST demographics; for the Census options use the command line below.

If a Census option says no API key was found, close VS Code's terminals (or restart
VS Code) so a fresh one loads the key from `gus8066`, then try again.

## Command line

```powershell
python -m pa_redistricting run --all               # everything above -> results/
python -m pa_redistricting list                    # plan names, unassigned VTDs
python -m pa_redistricting summary allentown       # population summary -> summaries/
python -m pa_redistricting summary --all
python -m pa_redistricting summary --all --demographics vest+census   # adds group quarters, housing
python -m pa_redistricting maps --all              # PNG maps -> maps/
python -m pa_redistricting partisan --all          # partisan metrics -> partisan/
python -m pa_redistricting codebook                # Census column labels
```

With no plan words and no `--all`, `run`, `summary`, `maps` and `partisan` show a numbered menu.
No command needs a Census key by default. `run`, `summary` and `maps` take
`--demographics vest` (default), `census` or `vest+census`; the last two need a key.
Add `-q` to hide progress messages and `--out FOLDER` to change where files go.

## Python

```python
from pa_redistricting import Plans, population_summary, plot_plan

plans = Plans.load()                       # VEST demographics, no key needed
# plans = Plans.load("vest+census")        # plus group quarters, housing (needs a key)
gdf = plans.get(plans.names[0])            # one row per voting district
table, stats = population_summary(plans.get(plans.names[0], geometry=False))
fig = plot_plan(gdf, plans.names[0])
fig.savefig("plan.png", dpi=200)

from pa_redistricting import fetch_elections, partisan_summary
summary, districts = partisan_summary(plans.assignments[plans.names[0]], fetch_elections())
```

## Demographic columns

Both sources come from the 2020 Census redistricting data (PL 94-171) and use the
same column names. Race columns are **non-Hispanic** unless the name says otherwise:

| Columns | Meaning | Source |
|---|---|---|
| `pop_total`, `vap_total` | Total population / voting-age population (18+) | VEST or Census |
| `pop_hispanic`, `vap_hispanic` | Hispanic or Latino, any race | VEST or Census |
| `pop_nh_<race>`, `vap_nh_<race>` | Race alone, not Hispanic (`white`, `black`, `aian`, `asian`, `nhpi`, `other`, `two_plus`) | VEST or Census |
| `pop_<race>_incl_hisp`, `vap_<race>_incl_hisp` | Race alone, including Hispanics | Census only |
| `gq_total`, `gq_correctional` | Group quarters; adult correctional facilities | Census only |
| `housing_total`, `housing_occupied`, `housing_vacant` | Housing units | Census only |

Earlier versions had `pop_white`, `pop_black`, etc. that included Hispanics; those
columns are now `pop_white_incl_hisp`, etc., and only appear with the Census options.

VEST demographics were checked against the 2020 Census API on 2026-11-11 with
`demographics.compare_sources()`: identical for all 9,178 PA voting districts.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Shared constants and data-source URLs |
| `plan_assignments.py` | Download and parse plan CSVs (`GEOID20,District`) |
| `voting_districts.py` | 2020 Census voting-district shapes |
| `demographics.py` | Shared demographic columns, the VEST loader, `compare_sources()` |
| `census.py` | Census API counts: the same columns as VEST, plus group quarters, housing, race including Hispanics |
| `election_results.py` | Downloads the VEST file (votes and demographics) and the elections to score |
| `plans.py` | `Plans`: joins plans, shapes and demographic data |
| `population.py` | Population summary and deviation per Senate district |
| `maps.py` | Plan maps with consistent district colors |
| `partisan_metrics.py` | Efficiency gap, partisan bias, mean-median, declination |
| `cli.py` | Command-line commands, and `run_everything()` for scripts and IDEs |

## Tests

```powershell
pytest
```

The tests use small made-up datasets and never touch the network.

## Data sources

- Plans: [ccfdpa/pa-citizen-redistricting-plans-2020](https://github.com/ccfdpa/pa-citizen-redistricting-plans-2020) (`pa-senate`)
- Shapes: 2020 Census TIGER/Line voting districts for Pennsylvania
- Demographics and votes: [alarm-redist/census-2020](https://github.com/alarm-redist/census-2020) (`census-vest-2020/pa_2020_vtd.csv`): 2020 Census PL 94-171 counts joined with [VEST](https://dataverse.harvard.edu/dataverse/electionscience) election results
- Optional Census-only fields: 2020 Decennial Redistricting Data (PL 94-171), Census Data API
- Metric definitions: [PlanScore](https://planscore.org/metrics/)

The ALARM data is licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
please cite the Voting and Election Science Team and the U.S. Census Bureau.
