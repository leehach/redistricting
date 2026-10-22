# pa_redistricting

Analyze Pennsylvania Senate citizen redistricting plans: join each plan to 2020
voting-district shapes and Census data, summarize population by district,
draw maps, and score partisan asymmetry (efficiency gap, partisan bias,
mean-median difference, declination).

## Install

From this folder, with the `gus8066` environment active:

```powershell
pip install -e ".[test]"
```

Set your Census API key once (free from https://api.census.gov/data/key_signup.html):

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

If the Run button says no Census API key was found, close VS Code's terminals (or restart
VS Code) so a fresh one loads the key from `gus8066`, then press Run again.

## Command line

```powershell
python -m pa_redistricting run --all               # everything above -> results/
python -m pa_redistricting list                    # plan names, unassigned VTDs
python -m pa_redistricting summary allentown       # population summary -> summaries/
python -m pa_redistricting summary --all
python -m pa_redistricting maps --all              # PNG maps -> maps/
python -m pa_redistricting partisan --all          # partisan metrics -> partisan/
python -m pa_redistricting codebook                # Census column labels
```

With no plan words and no `--all`, `run`, `summary`, `maps` and `partisan` show a numbered menu.
`partisan` needs no Census key: it only downloads the plans and the election results.
Add `-q` to hide progress messages and `--out FOLDER` to change where files go.

## Python

```python
from pa_redistricting import Plans, population_summary, plot_plan

plans = Plans.load()                       # or Plans.load(census_api_key="...")
gdf = plans.get(plans.names[0])            # one row per voting district
table, stats = population_summary(plans.get(plans.names[0], geometry=False))
fig = plot_plan(gdf, plans.names[0])
fig.savefig("plan.png", dpi=200)

from pa_redistricting import fetch_elections, partisan_summary
summary, districts = partisan_summary(plans.assignments[plans.names[0]], fetch_elections())
```

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Shared constants and data-source URLs |
| `plan_assignments.py` | Download and parse plan CSVs (`GEOID20,District`) |
| `voting_districts.py` | 2020 Census voting-district shapes |
| `census.py` | 2020 Census PL 94-171 counts per voting district |
| `election_results.py` | Votes per voting district (ALARM Project) and the elections to score |
| `plans.py` | `Plans`: joins plans, shapes and Census data |
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
- Census data: 2020 Decennial Redistricting Data (PL 94-171), Census Data API
- Votes: [alarm-redist/census-2020](https://github.com/alarm-redist/census-2020) (`census-vest-2020/pa_2020_vtd.csv`)
- Metric definitions: [PlanScore](https://planscore.org/metrics/)
