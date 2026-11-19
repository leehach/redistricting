import pandas as pd
import pytest

from pa_redistricting import cli
from pa_redistricting.cli import build_parser, choose_plans, run_everything
from pa_redistricting.demographics import COUNT_COLUMNS

PLANS = ["allentown_map", "ccfd_plan", "other_plan"]


def test_all_returns_every_plan():
    assert choose_plans(PLANS, [], use_all=True) == PLANS


def test_words_match_case_insensitively():
    assert choose_plans(PLANS, ["CCFD", "allen"]) == ["allentown_map", "ccfd_plan"]


def test_no_match_returns_empty():
    assert choose_plans(PLANS, ["zzz"]) == []


def test_menu_picks_numbers_and_ignores_junk():
    assert choose_plans(PLANS, [], ask=lambda prompt: "3, x, 1") == ["other_plan", "allentown_map"]


def test_menu_all():
    assert choose_plans(PLANS, [], ask=lambda prompt: "all") == PLANS


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_summary_options():
    args = build_parser().parse_args(["summary", "allentown", "--out", "results"])
    assert args.plans == ["allentown"]
    assert str(args.out) == "results"
    assert not args.all


# ---------------------------------------------------------------------------
# run / run_everything, on a tiny made-up state
# ---------------------------------------------------------------------------
def tiny_plans():
    shapely_geometry = pytest.importorskip("shapely.geometry")
    gpd = pytest.importorskip("geopandas")
    from pa_redistricting.plans import Plans

    geoids = ["42001000010", "42001000020", "4200300F437"]
    vtds = gpd.GeoDataFrame(
        {"GEOID": geoids, "VTD_NAME": ["A", "B", "C"], "COUNTYFP": ["001", "001", "003"],
         "ALAND": [1, 1, 1], "AWATER": [0, 0, 0]},
        geometry=[shapely_geometry.box(i, 0, i + 1, 1) for i in range(3)], crs="EPSG:4269")
    census = pd.DataFrame({
        "GEOID": geoids, "pop": [100, 200, 300], "pop_hisp": [10, 20, 30],
        "pop_white": [80, 100, 150], "pop_black": [5, 60, 100],
        "vap": [80, 160, 240], "vap_hisp": [8, 16, 24],
        "vap_white": [60, 80, 100], "vap_black": [4, 50, 90],
    })
    assignments = {
        "allentown_map": pd.DataFrame({"GEOID": geoids, "district": [1, 1, 2]}),
        "ccfd_plan": pd.DataFrame({"GEOID": geoids[:2], "district": [1, 2]}),
    }
    return Plans(vtds, census, assignments)


def tiny_votes():
    return pd.DataFrame({"GEOID": ["42001000010", "42001000020", "4200300F437"],
                         "ndv": [70, 30, 40], "nrv": [30, 70, 60]})


def test_run_everything_writes_summaries_maps_and_overview(tmp_path):
    path = run_everything("all", tmp_path, data=tiny_plans(), votes=tiny_votes())
    assert path == tmp_path / "overview.csv"
    for plan in ["allentown_map", "ccfd_plan"]:
        assert (tmp_path / "summaries" / f"{plan}_summary.csv").exists()
        assert (tmp_path / "maps" / f"{plan}.png").exists()
        assert (tmp_path / "partisan" / f"{plan}__composite_districts.csv").exists()

    overview = pd.read_csv(path)
    assert overview["plan"].tolist() == ["allentown_map", "ccfd_plan"]
    assert overview["unassigned_pop"].tolist() == [0, 300]
    assert overview["election"].tolist() == ["composite", "composite"]
    # allentown_map: district 1 = 100 D / 100 R (a tie, so R), district 2 = 40 D / 60 R
    assert overview["dem_seats"].tolist() == [0, 1]


def test_run_everything_writes_one_partisan_table_for_all_plans(tmp_path):
    run_everything("all", tmp_path, data=tiny_plans(), votes=tiny_votes())
    metrics = pd.read_csv(tmp_path / "partisan" / "partisan_metrics.csv")
    assert metrics["plan"].tolist() == ["allentown_map", "ccfd_plan"]
    assert "efficiency_gap_favors" in metrics.columns


def test_run_everything_matches_words(tmp_path):
    run_everything(["ALLEN"], tmp_path, data=tiny_plans(), votes=tiny_votes())
    assert [p.name for p in (tmp_path / "maps").iterdir()] == ["allentown_map.png"]


def test_run_everything_accepts_one_word(tmp_path):
    run_everything("ccfd", tmp_path, data=tiny_plans(), votes=tiny_votes())
    assert [p.name for p in (tmp_path / "maps").iterdir()] == ["ccfd_plan.png"]


@pytest.mark.parametrize("plans", [["zzz"], []])
def test_run_everything_rejects_no_plans(tmp_path, plans):
    with pytest.raises(ValueError):
        run_everything(plans, tmp_path, data=tiny_plans(), votes=tiny_votes())
    assert not any(tmp_path.iterdir())


def test_parser_run_defaults_to_results_folder():
    args = build_parser().parse_args(["run", "--all"])
    assert args.all
    assert str(args.out) == "results"


def test_parser_partisan_defaults_to_partisan_folder():
    args = build_parser().parse_args(["partisan", "rh"])
    assert args.plans == ["rh"]
    assert str(args.out) == "partisan"
    assert args.plans_source is None


@pytest.mark.parametrize("command", ["list", "run", "summary", "maps", "partisan"])
def test_parser_plans_source(command):
    args = build_parser().parse_args([command, "--plans-source", "D:/plans/oh.csv"])
    assert args.plans_source == "D:/plans/oh.csv"


@pytest.mark.parametrize("command", ["run", "summary", "maps", "partisan"])
def test_parser_vest(command):
    assert build_parser().parse_args([command]).vest is None
    assert build_parser().parse_args([command, "--vest", "my_pa.csv"]).vest == "my_pa.csv"


def test_load_votes_uses_own_file_or_alarm(tmp_path, monkeypatch):
    from pa_redistricting import config
    monkeypatch.setattr(config, "VEST_SOURCE", None)
    monkeypatch.setattr(cli, "fetch_elections", lambda: "alarm")
    assert cli.load_votes(None) == "alarm"
    assert cli.user_votes(None) is None

    path = tmp_path / "my_pa.csv"
    pd.DataFrame({"GEOID20": ["42001000010"], **{c: [1] for c in COUNT_COLUMNS},
                  "ndv": [1.0], "nrv": [2.0]}).to_csv(path, index=False)
    assert cli.load_votes(str(path))["GEOID"].tolist() == ["42001000010"]
