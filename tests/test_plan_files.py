"""
Tests for redistricting/metrics_gui/plan_files.py.

Run from the folder that holds ``redistricting/``:

    python -m pytest tests/test_plan_files.py

The convention: column 1 is the GEOID and column 2 the district, whatever the headers say.
"""

import pandas as pd
import pytest

from redistricting.metrics_gui.plan_files import (PlanFileError, clean_assignment, looks_like_geoid,
                                                  plan_files_in, read_plan_file, read_plan_files)

PAF = "GEOID20,District\n42001000010,1\n42001000020,1\n42001000030,2\n"


def write(folder, name, text):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def raw(rows, columns=("a", "b")):
    return pd.DataFrame(rows, columns=list(columns), dtype=str)


# --- reading one file ---

def test_reads_a_paf_named_after_its_file(tmp_path):
    plan = read_plan_file(write(tmp_path, "My Plan.csv", PAF))
    assert plan.name == "My Plan" and plan.kind == "PAF"
    assert list(plan.assignment.columns) == ["GEOID", "district"]
    assert plan.assignment["GEOID"].tolist() == ["42001000010", "42001000020", "42001000030"]
    assert plan.assignment["district"].tolist() == [1, 1, 2]
    assert plan.notes == []


@pytest.mark.parametrize("header", ["GEOID20,District", "id,assigned", "foo,bar", "District,GEOID20"])
def test_columns_are_read_by_position_not_name(tmp_path, header):
    text = PAF.replace("GEOID20,District", header)
    plan = read_plan_file(write(tmp_path, "p.csv", text))
    assert plan.assignment["GEOID"].iloc[0] == "42001000010"


def test_file_with_no_header_row(tmp_path):
    plan = read_plan_file(write(tmp_path, "p.csv", "42001000010,1\n42001000020,2\n"))
    assert plan.assignment["GEOID"].tolist() == ["42001000010", "42001000020"]


def test_tabs_and_semicolons(tmp_path):
    for i, sep in enumerate(["\t", ";"]):
        plan = read_plan_file(write(tmp_path, f"p{i}.txt", PAF.replace(",", sep)))
        assert plan.assignment["district"].tolist() == [1, 1, 2]


def test_extra_columns_are_ignored_with_a_note(tmp_path):
    text = "GEOID20,District,name,pop\n42001000010,1,x,5\n42001000020,2,y,6\n"
    plan = read_plan_file(write(tmp_path, "p.csv", text))
    assert list(plan.assignment.columns) == ["GEOID", "district"]
    assert any("ignored 2 column" in n for n in plan.notes)


def test_pandas_index_column_in_front_is_skipped(tmp_path):
    text = ",GEOID20,District\n0,42001000010,1\n1,42001000020,2\n2,42001000030,2\n"
    plan = read_plan_file(write(tmp_path, "p.csv", text))
    assert plan.assignment["GEOID"].tolist()[0] == "42001000010"
    assert any("index column" in n for n in plan.notes)


def test_bom_and_quoted_path(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(PAF, encoding="utf-8-sig")
    assert read_plan_file(f'"{path}"').assignment["GEOID"].iloc[0] == "42001000010"


def test_missing_file_raises(tmp_path):
    with pytest.raises(PlanFileError, match="no such file"):
        read_plan_file(tmp_path / "nope.csv")


@pytest.mark.parametrize("text", ["", "\n  \n"])
def test_empty_file_raises(tmp_path, text):
    with pytest.raises(PlanFileError, match="empty"):
        read_plan_file(write(tmp_path, "p.csv", text))


def test_single_column_file_raises(tmp_path):
    with pytest.raises(PlanFileError, match="at least 2 columns"):
        read_plan_file(write(tmp_path, "p.csv", "GEOID20\n42001000010\n42001000020\n"))


# --- the GEOID column ---

def test_block_geoids_make_it_a_baf():
    out, kind, _ = clean_assignment(raw([["420010301001000", "1"], ["420010301001001", "2"]]))
    assert kind == "BAF" and out["GEOID"].str.len().eq(15).all()


@pytest.mark.parametrize("stripped, kind, full", [
    ("1001000010", "PAF", "01001000010"),         # Alabama VTD, 10 characters after Excel
    ("10010201001000", "BAF", "010010201001000"),  # Alabama block, 14 characters after Excel
])
def test_leading_zero_a_spreadsheet_dropped_is_put_back(stripped, kind, full):
    out, found, _ = clean_assignment(raw([[stripped, "1"]]))
    assert found == kind and out["GEOID"].tolist() == [full]


def test_geoids_read_as_numbers_are_fixed():
    out, _, _ = clean_assignment(raw([["42001000010.0", "1"]]))
    assert out["GEOID"].tolist() == ["42001000010"]


@pytest.mark.parametrize("geoid", [
    "4200300A000",   # PA, letters
    "10001001-28",   # DE, hyphen
    "5400100005a",   # WV, lowercase
    "460030VTD-1",   # SD
    "050050001/1",   # AR-style slash
    "3000900A1.0",   # MT-style dot: not a number read as a float, so kept
])
def test_real_vtd_codes_from_other_states_are_kept_exactly(geoid):
    out, kind, _ = clean_assignment(raw([[geoid, "1"]]))
    assert kind == "PAF" and out["GEOID"].tolist() == [geoid]


def test_case_is_kept_because_it_matters():
    # Mississippi really has both of these voting districts
    out, _, _ = clean_assignment(raw([["28059505-1B", "1"], ["28059505-1b", "2"]]))
    assert out["GEOID"].tolist() == ["28059505-1B", "28059505-1b"]


def test_stripped_zero_put_back_on_a_vtd_code_with_a_hyphen():
    out, _, _ = clean_assignment(raw([["201337-700", "1"]]))  # Alaska, after Excel
    assert out["GEOID"].tolist() == ["0201337-700"]


def test_swapped_columns_are_caught():
    with pytest.raises(PlanFileError, match="column 1 should be GEOIDs"):
        clean_assignment(raw([["1", "42001000010"], ["2", "42001000020"]]))


def test_mixed_precinct_and_block_geoids_raise():
    with pytest.raises(PlanFileError, match="mixes"):
        clean_assignment(raw([["42001000010", "1"], ["420010301001000", "2"]]))


def test_duplicate_geoids_raise():
    with pytest.raises(PlanFileError, match="more than once"):
        clean_assignment(raw([["42001000010", "1"], ["42001000010", "2"]]))


# --- the district column ---

def test_blank_districts_are_unassigned_and_left_out():
    out, _, notes = clean_assignment(raw([["42001000010", "1"], ["42001000020", ""]]))
    assert out["GEOID"].tolist() == ["42001000010"]
    assert any("1 GEOIDs have no district" in n for n in notes)


def test_districts_like_3_point_0_are_whole_numbers():
    out, _, _ = clean_assignment(raw([["42001000010", "3.0"], ["42001000020", " 04 "]]))
    assert out["district"].tolist() == [3, 4]


@pytest.mark.parametrize("bad", ["ZZ", "1.5"])
def test_non_whole_district_raises(bad):
    with pytest.raises(PlanFileError, match="column 2 should be whole district numbers"):
        clean_assignment(raw([["42001000010", bad]]))


def test_district_without_geoid_raises():
    with pytest.raises(PlanFileError, match="no GEOID"):
        clean_assignment(raw([["42001000010", "1"], [None, "2"]]))


def test_one_column_raises():
    with pytest.raises(PlanFileError, match="at least 2 columns"):
        clean_assignment(pd.DataFrame({"GEOID": ["42001000010"]}))


def test_blank_rows_are_ignored():
    out, _, notes = clean_assignment(raw([["42001000010", "1"], [None, None], ["", ""]]))
    assert len(out) == 1 and notes == []


# --- many files ---

def test_folder_reads_every_plan_file_but_metrics_outputs(tmp_path):
    for name in ["b.csv", "a.txt", "c.tsv", "a_metrics.csv", "notes.md"]:
        write(tmp_path, name, PAF)
    assert [f.name for f in plan_files_in(tmp_path)] == ["a.txt", "b.csv", "c.tsv"]
    plans, problems = read_plan_files([tmp_path])
    assert [p.name for p in plans] == ["a", "b", "c"] and problems == []


def test_bad_files_are_skipped_and_reported(tmp_path):
    good = write(tmp_path, "good.csv", PAF)
    bad = write(tmp_path, "bad.csv", "x,y\n1,2\n")
    plans, problems = read_plan_files([good, bad, tmp_path / "missing.csv"])
    assert [p.name for p in plans] == ["good"]
    assert [p for p, _ in problems] == [str(bad), str(tmp_path / "missing.csv")]


def test_empty_folder_is_reported(tmp_path):
    plans, problems = read_plan_files([tmp_path])
    assert plans == [] and "no .csv" in problems[0][1]


@pytest.mark.parametrize("text, expected", [
    ("42001000010", True), ("1001000010", True), ("420010301001000", True),
    ("10001001-28", True), ("GEOID20", False), ("District", False), ("District_ID", False),
    ("1", False), ("4200100001X0", False),
])
def test_looks_like_geoid(text, expected):
    assert looks_like_geoid(text) is expected
