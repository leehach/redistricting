import pytest

from pa_redistricting import plan_assignments
from pa_redistricting.plan_assignments import plan_name, read_assignment

GOOD = "GEOID20,District\n42001000010,5\n42001000020,12\n42001000030,\n"


def test_reads_columns_by_name():
    df = read_assignment(GOOD)
    assert list(df.columns) == ["GEOID", "district"]
    assert df["GEOID"].tolist() == ["42001000010", "42001000020"]
    assert df["district"].tolist() == [5, 12]


def test_blank_district_is_dropped_as_unassigned():
    assert "42001000030" not in read_assignment(GOOD)["GEOID"].tolist()


def test_space_after_comma_in_header_is_allowed():
    df = read_assignment("GEOID20, District\n42001000010, 3\n")
    assert df["district"].tolist() == [3]


@pytest.mark.parametrize("header", ["GEOID,District", "geoid20,district", "GEOID20,District,Notes"])
def test_other_headers_raise(header):
    with pytest.raises(ValueError, match="expected header"):
        read_assignment(f"{header}\n42001000010,1\n")


@pytest.mark.parametrize("geoid", ["420010000101000", "4200100001", "42001000010.0", "4200100001a"])
def test_non_vtd_geoids_raise(geoid):
    with pytest.raises(ValueError, match="11-character VTD codes"):
        read_assignment(f"GEOID20,District\n{geoid},1\n")


def test_vtd_codes_may_contain_letters():
    assert read_assignment("GEOID20,District\n4200300F437,1\n")["GEOID"].tolist() == ["4200300F437"]


def test_non_numeric_district_raises():
    with pytest.raises(ValueError):
        read_assignment("GEOID20,District\n42001000010,ZZ\n")


def test_plan_name_replaces_symbols():
    assert plan_name("pa-senate/My Plan (v2).csv") == "My_Plan__v2_"


def test_fetch_returns_every_plan(monkeypatch):
    files = {"pa-senate/a.csv": GOOD, "pa-senate/b.csv": GOOD}
    monkeypatch.setattr(plan_assignments, "list_plan_files", lambda: sorted(files))
    monkeypatch.setattr(plan_assignments, "download_text", files.__getitem__)
    assert list(plan_assignments.fetch_plan_assignments()) == ["a", "b"]


def test_fetch_names_the_bad_file(monkeypatch):
    files = {"pa-senate/good.csv": GOOD, "pa-senate/bad.csv": "GEOID,District\n42001000010,1\n"}
    monkeypatch.setattr(plan_assignments, "list_plan_files", lambda: sorted(files))
    monkeypatch.setattr(plan_assignments, "download_text", files.__getitem__)
    with pytest.raises(ValueError) as exc:
        plan_assignments.fetch_plan_assignments()
    assert "while reading pa-senate/bad.csv" in exc.value.__notes__
