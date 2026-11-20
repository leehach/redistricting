# Plan Metrics GUI

A dialog window for analysts to load their own file, tick metrics they would 
like to perform, match the file's fields, and save one CSV with a row per district

## Setup (once, from the repo root)

The class env has everything (pyqt5 and requests are in the yml now). the GUI finds our
pa_redistricting package from its spot in the repo, so nothing else to install
```powershell
conda env create -f environment.yml
conda activate redistricting
```
already made the env before pyqt was added? `conda env update -n redistricting -f environment.yml`

## Running it

```powershell
conda activate gus8066        # or redistricting
cd redistricting\metrics_gui
python main.py
```

run `main.py`

## Main window

`python app.py` opens the main window instead. plans you import live in memory (`window.plans`,
a `SessionPlans`) until you close it, and the table shows what's in there

**Plans** - `Import Plans...` (Ctrl+O) reads any number of plan files at once, `Import Folder...`
reads every .csv/.txt/.tsv in a folder (files ending in `_metrics` are skipped). you can also drag
files or folders onto the window. a plan is just the minimum:

- column 1 = GEOID, whatever the header says. 11 characters = precinct (PAF), 15 = block (BAF)
- column 2 = district, a whole number
- anything after that is ignored. no header row, or a pandas index column in front, is fine too.
  blank districts = unassigned, left out

files that don't fit are skipped and listed at the end, the rest still import. importing a name
that's already loaded asks once: replace, keep both, or skip

**Precinct data** - votes, demographics, shapes. not part of a plan, it's joined on by GEOID when
an analysis opens. `Download State Data...` (Ctrl+D) picks any state (+ DC) and any of:

- shapes - 2020 Census TIGER/Line, needed for compactness
- VEST - 2020 election results + demographics (ALARM Project), no API key
- Census - 2020 PL 94-171 demographics from the Census API, needs `CENSUS_API_KEY`. its race
  columns include Hispanics so they're named `pop_white_incl_hisp` etc (VEST's `pop_white` is
  non-Hispanic)

all by 2020 voting district, except CA, HI, ME and OR which are by Census block (ALARM only has
blocks for those, so plans for them need to be BAFs). downloads are saved in
`~/.redistricting_cache` so each one only downloads once. `Add Files...` adds your own tables (any
format, the GEOID column just has to be findable: GEOID20, GEOID, precinct_id ...)

**Metrics...** opens the dialog below with the selected plan joined to the precinct data for its
state (from the GEOIDs) plus any files you added. the status bar says how the join went
(ex. "PA VEST data: 9,178 of 9,178 precincts matched")

none of the data parts need the window:

```python
from plan_files import read_plan_files
from reconcile import reconcile
plans, problems = read_plan_files(["my_plans/"])
table = reconcile(plans[0].assignment, extras={"VEST": vest}).data
```

to add an analysis: put a button next to `Metrics...` in `main_window.ui`, run
`pyuic5 main_window.ui -o main_window_ui.py`, and connect it in `app.py` like `open_metrics`
(`self.current_plan()` gives you the plan, `self.analysis_table(plan)` the joined table)

## How to use

1. Input file - click `...` and pick a csv, shapefile, geopackage or geojson. one row per
   precinct (with a district column) or one row per district both work
2. Metrics - check what you want. "Compactness (All)" etc checks the whole group, the arrow
   opens it up so you can pick individual ones
3. Match fields - only shows the fields needed for what you checked. it guesses from the
   column names, if it misses one just pick it from the dropdown
4. Output CSV - defaults to next to the input file
5. Calculate - greyed out until everything is filled in, bottom of the window tells you what's missing

| Metric | Fields Needed |
|---|---|
| Partisan (favor, efficiency gap, mean-median, declination, bias) | District, Democratic votes, Republican votes (raw counts, not shares) |
| Compactness (Polsby-Popper, Schwartzberg, Reock, convex hull) | District, plus shapes (a spatial file) |
| County splits | District, County |

Output is one row per district. plan-wide stuff (efficiency gap, seats etc) just repeats on
every row. positive = favors R, negative = favors D. compactness 1 = most compact

## Files

- `app.py` - the main window (plan imports, precinct data, buttons that open analyses)
- `main_window.ui` / `main_window_ui.py` - its layout, same deal as the dialog's
- `plan_files.py` - reads PAFs/BAFs (GEOID, district) one at a time or in bulk
- `session.py` - the in-memory plans object, GEOID + district per plan
- `state_data.py` - shapes, VEST and Census downloads for any state, cached
- `reconcile.py` - joins a plan to precinct data on the GEOID when an analysis runs
- `metrics_dialog.ui` - the layout, edit in Qt Designer
- `metrics_dialog_ui.py` - generated from the .ui, don't edit by hand
- `main.py` - the window code (buttons, checkboxes, field dropdowns)
- `registry.py` - plug in stuff, shouldn't need to touch it
- `metrics.py` - all the metrics and fields, add new ones here

## Editing the layout in Qt Designer

1. open `metrics_dialog.ui` in designer
2. change whatever you want but DON'T rename the widgets (`inputLineEdit`, `metricTree`,
   `fieldForm`, `runButton` etc), main.py uses those names
3. leave the metric tree and field form empty, the code fills them in
4. save, then run `pyuic5 metrics_dialog.ui -o metrics_dialog_ui.py`
5. new buttons won't do anything until you connect them in main.py
   (`self.someButton.clicked.connect(...)`)

## Adding a metric

everything goes in `metrics.py`. add a field if your metric needs a column that isn't there
yet, then the metric. it shows up in the window automatically

```python
field("population", "Population", ["population", "pop", "totpop", "p0010001"], numeric=True)

@metric("Population", "Population deviation", needs=["population"])
def population_deviation(ctx):
    pop = ctx.sum_by_district("population")["population"]
    return ((pop - pop.mean()) / pop.mean() * 100).round(2).rename("pct_deviation")
```

- the list in `field` = column names the auto-matcher looks for (case doesn't matter)
- `needs` = fields the metric uses, add `"geometry"` if it needs shapes. district is always included
- `ctx.column("x")` gets the user's column, `ctx.sum_by_district(...)` adds things up by
  district, `ctx.district_shapes` is one shape per district
- return something indexed by district (one value per district) or a dict (one value for
  the whole plan)

order in metrics.py = order in the window. look at the compactness part for how to hook up
someone else's module

can also open it from code with data already loaded: `MetricsDialog(data=gdf, name="my_plan")`
