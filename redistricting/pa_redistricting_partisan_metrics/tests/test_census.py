from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from pa_redistricting import census
from pa_redistricting.census import MissingAPIKeyError, get_api_key


class FakeResponse:
    def __init__(self, rows, status_code=200):
        self.rows, self.status_code, self.text = rows, status_code, "not json"

    def json(self):
        if self.rows is None:
            raise ValueError("not json")
        return self.rows


def fake_census_api(urls_seen):
    """Stand-in for requests.get: one voting district per county, every count = 10."""
    def get(url, timeout):
        urls_seen.append(url)
        county = parse_qs(urlparse(url).query)["in"][1].split(":")[1]
        header = ["NAME", *census.VARIABLES, "state", "county", "voting district"]
        row = [f"VTD {county}", *["10"] * len(census.VARIABLES), "42", county, "000010"]
        return FakeResponse([header, row])
    return get


def test_passed_key_wins_over_environment(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "from-env")
    assert get_api_key("  passed  ") == "passed"


def test_environment_key_used_when_none_passed(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "from-env")
    assert get_api_key() == "from-env"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="CENSUS_API_KEY"):
        get_api_key()


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("made a request without a key")

    monkeypatch.setattr(census.requests, "get", no_network)
    with pytest.raises(MissingAPIKeyError):
        census.fetch_census()


def test_fetch_census_builds_geoids_and_counts(monkeypatch):
    urls = []
    monkeypatch.setattr(census.requests, "get", fake_census_api(urls))
    df = census.fetch_census(api_key="abc")

    assert len(urls) == len(census.PA_COUNTIES) == 67
    assert all(url.endswith("&key=abc") for url in urls)
    assert df["GEOID"].iloc[0] == "42001000010"
    assert df["GEOID"].str.len().eq(11).all()
    assert str(df["pop_total"].dtype) == "Int64"
    assert df["pct_hispanic"].iloc[0] == 100.0


def test_api_error_names_the_county(monkeypatch):
    monkeypatch.setattr(census.requests, "get", lambda url, timeout: FakeResponse(None, 400))
    with pytest.raises(RuntimeError, match="County 001"):
        census.fetch_census(api_key="abc")


def test_percentages_blank_when_denominator_is_zero():
    df = pd.DataFrame({c: pd.array([0], dtype="Int64") for c in census.VARIABLES.values()})
    assert pd.isna(census.add_percentages(df)["pct_hispanic"].iloc[0])
