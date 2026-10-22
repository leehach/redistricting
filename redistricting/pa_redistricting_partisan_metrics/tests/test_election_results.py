import pandas as pd
import pytest

from pa_redistricting import election_results
from pa_redistricting.election_results import available_elections, fetch_elections

VOTES = pd.DataFrame({"GEOID": ["42001000010"], "ndv": [10], "nrv": [5],
                      "pre_20_dem_bid": [9], "pre_20_rep_tru": [6], "pop": [100]})


def test_available_keeps_elections_with_both_columns():
    assert available_elections(VOTES) == {"composite": ("ndv", "nrv"),
                                          "pres_2020": ("pre_20_dem_bid", "pre_20_rep_tru")}


def test_missing_elections_are_skipped_with_a_warning(caplog):
    available_elections(VOTES)
    assert "pres_2016" in caplog.text
    assert "pre_20_dem_bid" in caplog.text      # lists the columns that do exist


def test_no_elections_raises():
    with pytest.raises(ValueError, match="None of the requested elections"):
        available_elections(VOTES[["GEOID", "pop"]])


def test_fetch_renames_geoid_and_keeps_it_as_text(monkeypatch):
    class Response:
        content = b"GEOID20,ndv,nrv\n 4200300F437,10,5\n42001000010,1,2\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(election_results.requests, "get", lambda url, timeout: Response())
    df = fetch_elections()
    assert df["GEOID"].tolist() == ["4200300F437", "42001000010"]
    assert "GEOID20" not in df.columns
