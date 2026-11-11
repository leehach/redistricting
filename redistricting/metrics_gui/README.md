# Plan Metrics GUI

A dialog window for analysts to load their own file, tick metrics they would 
like to perform, match the file's fields, and save one CSV with a row per district

## Setup (once, from the repo root)

- **gus8066 (now):** installs our package so python can import the partisan math, plus esda for compactness
  ```powershell
  pip install -e redistricting\pa_redistricting_partisan_metrics
  conda install -c conda-forge esda
  ```
- **environment.yml (later):** add `pyqt=5`, `requests`, `matplotlib` to the yml first, our package still needs the pip line
  ```powershell
  conda env create -f environment.yml
  conda activate redistricting
  pip install -e redistricting\pa_redistricting_partisan_metrics
  ```

## Running it

```powershell
conda activate gus8066        # or redistricting
cd redistricting\metrics_gui
python main.py
```

run `main.py`

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
