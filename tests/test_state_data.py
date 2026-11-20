"""
Tests for redistricting/metrics_gui/state_data.py. No network: downloads and the Census API are faked.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_state_data.py
"""

import pandas as pd
import pytest

from redistricting.metrics_gui import state_data as sd


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("REDISTRICTING_CACHE", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("text, abbr", [("pa", "PA"), ("Pennsylvania", "PA"), ("42", "PA"),
                                        ("6", "CA"), (" maine ", "ME")])
def test_state_abbr(text, abbr):
    assert sd.state_abbr(text) == abbr


def test_unknown_state_raises():
    with pytest.raises(ValueError, match="Unknown state"):
        sd.state_abbr("Atlantis")


@pytest.mark.parametrize("abbr", ["CA", "HI", "ME", "OR"])
def test_block_states(abbr):
    # ALARM only has block files for these, so shapes and Census must be blocks to match
    assert sd.geography(abbr) == "block" and sd.unit_name(abbr) == "Census blocks"


def test_every_other_state_is_voting_districts():
    assert {a for a in sd.STATES if sd.geography(a) == "vtd"} == set(sd.STATES) - sd.BLOCK_STATES


def test_tiger_url():
    assert sd.tiger_url("PA") == (sd.TIGER_ROOT + "42_PENNSYLVANIA/42/tl_2020_42_vtd20.zip")
    assert sd.tiger_url("NH").endswith("33_NEW_HAMPSHIRE/33/tl_2020_33_vtd20.zip")
    assert sd.tiger_url("ME").endswith("23_MAINE/23/tl_2020_23_tabblock20.zip")


def test_load_vest_downloads_once_then_uses_the_cache(cache, monkeypatch):
    calls = []

    def fake_download(url, path):
        calls.append(url)
        path.write_text("GEOID20,state,county,pop,ndv,nrv\n10001000010,10,001,5,3,2\n")

    monkeypatch.setattr(sd, "_download", fake_download)
    for _ in range(2):
        df = sd.load_vest("de")
    assert calls == [sd.ALARM_URL.format(abbr="de", level="vtd")]
    assert df["GEOID"].tolist() == ["10001000010"] and df["county"].tolist() == ["001"]
    assert (cache / "de_2020_vtd.csv").exists()


class FakeResponse:
    def __init__(self, rows=None, text=""):
        self.rows, self.text, self.status_code = rows, text, 200 if rows else 400

    def json(self):
        if self.rows is None:
            raise ValueError("not json")
        return self.rows


def census_row(*geo):
    return ["name", *["1"] * len(sd.VARIABLES), *geo]


def test_fetch_census_vtd_state_is_one_request(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "k")
    urls = []

    def fake_get(url, timeout):
        urls.append(url)
        header = ["NAME", *sd.VARIABLES, "state", "county", "voting district"]
        return FakeResponse([header, census_row("10", "001", "000010"),
                             census_row("10", "003", "000020")])

    monkeypatch.setattr(sd.requests, "get", fake_get)
    df = sd.fetch_census("DE")
    assert len(urls) == 1 and "voting%20district:*" in urls[0] and "county:*" in urls[0]
    assert df["GEOID"].tolist() == ["10001000010", "10003000020"]
    assert df.columns[:3].tolist() == ["GEOID", "NAME_CENSUS", "COUNTYFP"]
    assert "voting district" not in df.columns and str(df[list(sd.VARIABLES.values())[0]].dtype) == "Int64"


def test_census_race_columns_say_they_include_hispanics(monkeypatch):
    # VEST's pop_white is non-Hispanic; the Census P1 one isn't, so it must not share the name
    monkeypatch.setenv("CENSUS_API_KEY", "k")
    header = ["NAME", *sd.VARIABLES, "state", "county", "voting district"]
    monkeypatch.setattr(sd.requests, "get", lambda url, timeout: FakeResponse(
        [header, census_row("10", "001", "000010")]))
    df = sd.fetch_census("DE")
    assert "pop_white_incl_hisp" in df.columns and "vap_black_incl_hisp" in df.columns
    assert "pop_white" not in df.columns and "vap_white" not in df.columns
    total = sd.VARIABLES["P1_001N"]
    assert total in df.columns and f"{total}_incl_hisp" not in df.columns  # totals keep their name


def test_fetch_census_block_state_goes_county_by_county(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "k")
    urls, progress = [], []

    def fake_get(url, timeout):
        urls.append(url)
        if "for=county:*" in url:
            return FakeResponse([["NAME", *sd.VARIABLES, "state", "county"],
                                 census_row("15", "003"), census_row("15", "001")])
        county = url.split("in=county:")[1][:3]
        header = ["NAME", *sd.VARIABLES, "state", "county", "tract", "block"]
        return FakeResponse([header, census_row("15", county, "000100", "1000")])

    monkeypatch.setattr(sd.requests, "get", fake_get)
    df = sd.fetch_census("HI", say=progress.append)
    assert df["GEOID"].tolist() == ["150010001001000", "150030001001000"]
    assert len(urls) == 3 and len(progress) == 2 and "county 2 of 2" in progress[-1]


def test_census_error_page_is_reported(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "k")
    monkeypatch.setattr(sd.requests, "get", lambda url, timeout: FakeResponse(text="Invalid Key"))
    with pytest.raises(RuntimeError, match="Invalid Key"):
        sd.fetch_census("DE")


def test_census_without_a_key_raises(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    assert not sd.has_census_key()
    with pytest.raises(sd.MissingAPIKeyError):
        sd.fetch_census("DE")


def test_load_census_is_cached_with_text_ids(cache, monkeypatch):
    calls = []

    def fake_fetch(state, api_key=None, say=None):
        calls.append(state)
        return pd.DataFrame({"GEOID": ["01001000010"], "COUNTYFP": ["001"], "pop_total": [5]})

    monkeypatch.setattr(sd, "fetch_census", fake_fetch)
    first, second = sd.load_census("AL"), sd.load_census("AL")
    assert calls == ["AL"]
    assert second["GEOID"].tolist() == ["01001000010"] and second["COUNTYFP"].tolist() == ["001"]
