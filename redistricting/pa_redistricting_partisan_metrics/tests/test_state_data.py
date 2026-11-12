import io

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from pa_redistricting import state_data
from pa_redistricting.demographics import VEST_COLUMNS
from pa_redistricting.state_data import (STATES, add_vote_shares, find_state, geography,
                                         join_state, load_vest, pick_state, state_menu,
                                         tiger_url)

DEMOGRAPHICS = {c: [10, 20, 30] for c in VEST_COLUMNS}
VEST = pd.DataFrame({"GEOID": ["4200100001A", "42001000020", "42001000030"],
                     **DEMOGRAPHICS,
                     "ndv": [60.0, 0.0, 25.5], "nrv": [40.0, 0.0, 74.5],
                     "adv_20": [3.0, 0.0, 1.0], "arv_20": [1.0, 0.0, 1.0]})


def shapes(geoids):
    return gpd.GeoDataFrame({"GEOID": geoids, "VTD_NAME": geoids, "COUNTYFP": "001",
                             "ALAND": 1, "AWATER": 0},
                            geometry=[box(i, 0, i + 1, 1) for i in range(len(geoids))],
                            crs="EPSG:4269")


def test_all_states_and_dc():
    assert len(STATES) == 51
    assert len({fips for _, fips in STATES.values()}) == 51


@pytest.mark.parametrize("text", ["PA", "pa", " Pennsylvania ", "pennsylvania", "42"])
def test_find_state_accepts_abbreviation_name_or_fips(text):
    assert find_state(text) == "PA"


@pytest.mark.parametrize("text", ["", "Penn", "99", "XX"])
def test_find_state_unknown(text):
    assert find_state(text) is None


def test_block_states():
    assert geography("CA") == geography("or") == "block"
    assert geography("PA") == "vtd"


def test_menu_lists_every_state():
    assert all(name in state_menu() for name, _ in STATES.values())


def test_pick_state_asks_again_until_valid(capsys):
    answers = iter(["Narnia", "", "new york"])
    assert pick_state(ask=lambda prompt: next(answers)) == "NY"
    assert "No state matches 'Narnia'" in capsys.readouterr().out


def test_tiger_urls():
    assert tiger_url("PA") == ("https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/"
                               "42_PENNSYLVANIA/42/tl_2020_42_vtd20.zip")
    assert "11_DISTRICT_OF_COLUMBIA/11/tl_2020_11_vtd20.zip" in tiger_url("DC")
    assert tiger_url("CA").endswith("06_CALIFORNIA/06/tl_2020_06_tabblock20.zip")


class Response:
    def __init__(self, content=b"", status=200, text=""):
        self.content, self.status_code, self.text = content, status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("REDISTRICTING_CACHE", str(tmp_path))
    return tmp_path


def test_load_vest_downloads_once_and_keeps_geoids_as_text(monkeypatch, cache):
    urls = []
    csv = b"GEOID20,ndv\n 4200100001A,1\n01001000010,2\n"
    monkeypatch.setattr(state_data.requests, "get",
                        lambda url, **kw: urls.append(url) or Response(csv))
    first = load_vest("pa")
    second = load_vest("Pennsylvania")
    assert urls == ["https://raw.githubusercontent.com/alarm-redist/census-2020/"
                    "main/census-vest-2020/pa_2020_vtd.csv"]
    assert first["GEOID"].tolist() == second["GEOID"].tolist() == ["4200100001A", "01001000010"]
    assert not list(cache.glob("*.partial"))


def test_load_vest_refresh_downloads_again(monkeypatch, cache):
    urls = []
    monkeypatch.setattr(state_data.requests, "get",
                        lambda url, **kw: urls.append(url) or Response(b"GEOID20\n1\n"))
    load_vest("PA")
    load_vest("PA", refresh=True)
    assert len(urls) == 2


def test_failed_download_leaves_nothing_in_cache(monkeypatch, cache):
    import requests
    monkeypatch.setattr(state_data.requests, "get", lambda url, **kw: Response(status=500))
    with pytest.raises(requests.HTTPError):
        load_vest("PA")
    assert list(cache.iterdir()) == []


def test_load_shapes_looks_up_folder_when_usual_address_fails(monkeypatch, cache):
    urls = []

    def get(url, **kw):
        urls.append(url)
        if url == state_data.TIGER_STATE_ROOT:
            return Response(text='<a href="11_DC_RENAMED/">11_DC_RENAMED/</a>')
        if "11_DC_RENAMED" in url:
            return Response(b"zip bytes")
        return Response(status=404)

    monkeypatch.setattr(state_data.requests, "get", get)
    monkeypatch.setattr(state_data.gpd, "read_file", lambda path: shapes(["11001000001"])
                        .rename(columns={"GEOID": "GEOID20", "VTD_NAME": "NAME20",
                                         "COUNTYFP": "COUNTYFP20", "ALAND": "ALAND20",
                                         "AWATER": "AWATER20"}))
    gdf = state_data.load_shapes("DC")
    assert urls[-1].endswith("11_DC_RENAMED/11/tl_2020_11_vtd20.zip")
    assert list(gdf.columns) == ["GEOID", "VTD_NAME", "COUNTYFP", "ALAND", "AWATER", "geometry"]


def test_vote_shares():
    df = add_vote_shares(VEST)
    assert df["dem_share"].tolist()[0] == 0.6
    assert pd.isna(df["dem_share"].iloc[1])            # no votes: blank, not an error
    assert df["dem_share"].iloc[2] == 0.255
    assert df["dem_share_20"].tolist()[0] == 0.75


def test_join_keeps_every_shape_and_reports_mismatches():
    gdf, report = join_state(shapes(["4200100001A", "42001000020", "42001009999"]), VEST)
    assert isinstance(gdf, gpd.GeoDataFrame) and gdf.crs == "EPSG:4269"
    assert gdf.columns[-1] == "geometry"
    assert len(gdf) == 3
    assert report["matched"] == 2
    assert report["only_in_shapes"] == ["42001009999"]
    assert report["only_in_data"] == ["42001000030"]
    assert report["pop_without_shape"] == 30
    assert gdf.loc[gdf["GEOID"] == "42001009999", "pop_total"].isna().all()
    assert gdf.loc[gdf["GEOID"] == "4200100001A", "dem_share"].iloc[0] == 0.6
    assert "pop_nh_white" in gdf.columns


def test_join_without_election_data():
    gdf, report = join_state(shapes(["4200100001A"]),
                             VEST.drop(columns=["ndv", "nrv", "adv_20", "arv_20"]))
    assert report["election_columns"] == []
    assert "dem_share" not in gdf.columns


def test_join_drops_duplicate_geoids(caplog):
    gdf, _ = join_state(shapes(["4200100001A"]), pd.concat([VEST, VEST.iloc[[0]]]))
    assert len(gdf) == 1
    assert "duplicate GEOIDs" in caplog.text
