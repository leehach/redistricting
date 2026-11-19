import pytest

from pa_redistricting import config, plan_assignments
from pa_redistricting.plan_assignments import (fetch_plan_assignments, plan_name, read_plan,
                                               read_plans)

GOOD = "GEOID20,District\n42001000010,5\n42001000020,12\n42001000030,\n"
LONG = ("GEOID20,plan,District\n"
        "39001000010,a,1\n39001000020,a,2\n39001000010,b,2\n39001000020,b,2\n")


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# One CSV
# ---------------------------------------------------------------------------
def test_single_plan_file(tmp_path):
    [plan] = read_plans(write(tmp_path, "my plan.csv", GOOD))
    assert plan.name == "my_plan"
    assert plan.assignment.columns.tolist() == ["GEOID", "district"]
    assert plan.assignment["GEOID"].tolist() == ["42001000010", "42001000020"]
    assert plan.assignment["district"].tolist() == [5, 12]      # blank district dropped


def test_long_format_file(tmp_path):
    plans = read_plans(write(tmp_path, "all.csv", LONG))
    assert [p.name for p in plans] == ["a", "b"]
    assert plans[1].assignment["district"].tolist() == [2, 2]


@pytest.mark.parametrize("header", ["GEOID20,District", "geoid,district", "GEOID20, District",
                                    "GEOID20,District,Notes"])
def test_flexible_headers(tmp_path, header):
    [plan] = read_plans(write(tmp_path, "p.csv", f"{header}\n42001000010,3\n"))
    assert plan.assignment["district"].tolist() == [3]


def test_byte_order_mark(tmp_path):
    [plan] = read_plans(write(tmp_path, "p.csv", "﻿GEOID20,District\n42001000010,3\n"))
    assert plan.assignment["GEOID"].tolist() == ["42001000010"]


def test_vtd_codes_may_contain_letters(tmp_path):
    [plan] = read_plans(write(tmp_path, "p.csv", "GEOID20,District\n4200300F437,1\n"))
    assert plan.assignment["GEOID"].tolist() == ["4200300F437"]


def test_pads_geoids_and_accepts_decimal_districts(tmp_path):
    al = read_plan(write(tmp_path, "al.csv", "GEOID20,District\n1001000100,1.0\n"))
    assert al["GEOID"].tolist() == ["01001000100"]
    assert al["district"].tolist() == [1]
    block = read_plan(write(tmp_path, "ca.csv", "GEOID20,District\n60014001001000,4\n"))
    assert block["GEOID"].tolist() == ["060014001001000"]


@pytest.mark.parametrize("text, message", [
    ("GEOID20,District\n42001000010,ZZ\n", "whole numbers"),
    ("GEOID20,District\n42001000010,1.5\n", "whole numbers"),
    ("GEOID20,District\n42001000010,\n", "no voting district has a district number"),
    ("GEOID20,District\n,1\n", "no GEOID"),
    ("name,value\nfoo,1\n", "no GEOID column"),
    ("GEOID20,plan,District\n42001000010,,1\n", "no plan name"),
])
def test_bad_files_raise(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_plans(write(tmp_path, "bad.csv", text))


def test_geoid_repeated_across_plans_is_fine_but_not_within_one(tmp_path):
    ok = "GEOID20,plan,District\n39001000010,a,1\n39001000010,b,2\n"
    assert len(read_plans(write(tmp_path, "ok.csv", ok))) == 2
    dup = ok + "39001000010,a,3\n"
    with pytest.raises(ValueError, match=r"dup.csv \[a\]: 1 GEOIDs appear more than once"):
        read_plans(write(tmp_path, "dup.csv", dup))


def test_read_plan_returns_one_and_rejects_several(tmp_path):
    assert read_plan(write(tmp_path, "one.csv", GOOD))["district"].tolist() == [5, 12]
    with pytest.raises(ValueError, match="holds 2 plans"):
        read_plan(write(tmp_path, "many.csv", LONG))


def test_plan_name_replaces_symbols():
    assert plan_name("pa-senate/My Plan (v2).csv") == "My_Plan__v2_"


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
def test_folder_mixes_single_and_long_and_skips_metrics(tmp_path):
    write(tmp_path, "one.csv", "GEOID20,District\n39001000010,1\n")
    write(tmp_path, "many.csv", "GEOID20,plan,District\n39001000010,x,1\n39001000010,y,2\n")
    write(tmp_path, "one_metrics.csv", "plan,election\none,composite\n")
    assert sorted(p.name for p in read_plans(tmp_path)) == ["one", "x", "y"]


def test_bad_file_in_folder_is_skipped(tmp_path, caplog):
    write(tmp_path, "good.csv", "GEOID20,District\n39001000010,1\n")
    write(tmp_path, "bad.csv", "name,value\nfoo,1\n")
    assert [p.name for p in read_plans(tmp_path)] == ["good"]
    assert "Skipping bad.csv" in caplog.text


def test_folder_with_no_plans_raises(tmp_path):
    with pytest.raises(ValueError, match="No plan CSVs"):
        read_plans(tmp_path)
    write(tmp_path, "bad.csv", "name,value\nfoo,1\n")
    with pytest.raises(ValueError, match="could be read as plans"):
        read_plans(tmp_path)


def test_same_plan_name_in_two_files_gets_file_prefix(tmp_path):
    write(tmp_path, "f1.csv", "GEOID20,plan,District\n39001000010,a,1\n")
    write(tmp_path, "f2.csv", "GEOID20,plan,District\n39001000010,a,2\n")
    assert sorted(p.name for p in read_plans(tmp_path)) == ["f1__a", "f2__a"]


# ---------------------------------------------------------------------------
# Choosing the source
# ---------------------------------------------------------------------------
def test_quoted_path_from_copy_as_path(tmp_path):
    path = write(tmp_path, "p.csv", GOOD)
    assert len(read_plans(f'  "{path}"  ')) == 1


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_plans(tmp_path / "nope.csv")


def test_no_source_raises(monkeypatch):
    monkeypatch.setattr(config, "PLANS_SOURCE", None)
    with pytest.raises(ValueError, match="No plans source"):
        read_plans()


def test_default_source_comes_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PLANS_SOURCE", str(write(tmp_path, "p.csv", GOOD)))
    assert [p.name for p in read_plans()] == ["p"]


def test_other_urls_rejected():
    with pytest.raises(ValueError, match="Unsupported URL"):
        read_plans("https://example.com/plans.csv")


# ---------------------------------------------------------------------------
# GitHub folders
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, json=None, text=""):
        self._json, self.content = json, text.encode("utf-8")

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def fake_github(files: dict[str, str], urls_seen: list[str]):
    """Stand-in for requests.get: a repo tree listing, then raw file contents."""
    def get(url, timeout):
        urls_seen.append(url)
        if "api.github.com" in url:
            return FakeResponse(json={"tree": [{"type": "blob", "path": p} for p in files]
                                      + [{"type": "blob", "path": "README.md"}]})
        for path, text in files.items():
            if url.endswith(path.replace(" ", "%20")):
                return FakeResponse(text=text)
        raise AssertionError(f"unexpected URL {url}")
    return get


def test_github_folder(monkeypatch):
    urls = []
    files = {"plans/a.csv": GOOD, "plans/b c.csv": LONG, "other/x.csv": GOOD}
    monkeypatch.setattr(plan_assignments.requests, "get", fake_github(files, urls))
    plans = read_plans("https://github.com/owner/repo/tree/main/plans")
    # "a" is both a.csv's plan and a plan inside "b c.csv", so both get their file name
    assert sorted(p.name for p in plans) == ["a__a", "b", "b_c__a"]
    assert urls[0] == "https://api.github.com/repos/owner/repo/git/trees/main?recursive=1"
    assert not any("other/x.csv" in u for u in urls)


def test_fetch_plan_assignments_defaults_to_example_plans(monkeypatch):
    urls = []
    files = {"pa-senate/a.csv": GOOD, "pa-senate/b.csv": GOOD}
    monkeypatch.setattr(config, "PLANS_SOURCE", None)
    monkeypatch.setattr(plan_assignments.requests, "get", fake_github(files, urls))
    assert list(fetch_plan_assignments()) == ["a", "b"]
    assert "ccfdpa/pa-citizen-redistricting-plans-2020" in urls[0]
