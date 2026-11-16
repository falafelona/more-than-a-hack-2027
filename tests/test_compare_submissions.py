# ws:eval: tests for scripts/compare_submissions.py. Made-up label files only: nothing here reads
# match data.
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_submissions",
                                               ROOT / "scripts/compare_submissions.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)
pm = cmp.pm


def codes(text):
    """'HHAD' -> label codes, one per letter (H 0, A 1, D 2, X 3)."""
    return np.array(["HADX".index(c) for c in text], dtype=np.int8)


def write(path, text):
    path.write_text("frame,label\n" + "".join(f"{i},{c}\n" for i, c in enumerate(text)))
    return str(path)


def test_identical_files_change_nothing():
    r = cmp.compare(codes("HHHAAADDD"), codes("HHHAAADDD"))
    assert r["changed"] == 0 and r["moves"] == {} and r["stretches"] == 0
    assert r["possessions"][0] == r["possessions"][1]


def test_a_switch_moved_later_is_one_stretch_of_changed_frames():
    r = cmp.compare(codes("H" * 10 + "A" * 10), codes("H" * 15 + "A" * 5))
    assert r["changed"] == 5
    assert r["moves"] == {("A", "H"): 5}
    assert r["stretches"] == 1 and r["stretch_max"] == 5 and r["stretch_mean"] == 5.0
    assert r["switches"] == [1, 1]
    assert [p[0] for p in r["possessions"]] == [2, 2]          # same number of possessions


def test_changes_to_and_from_dead_are_counted_by_kind():
    r = cmp.compare(codes("HHDDAA"), codes("HDDDDA"))
    assert r["moves"] == {("H", "D"): 1, ("A", "D"): 1}
    assert r["stretches"] == 2


def test_possessions_are_runs_of_h_or_a_like_validate_submission():
    c = codes("HHHDDAAAHHD")
    n, mean_s = cmp.possessions(c)
    assert n == 3                                                # HHH, AAA, HH: D is a stoppage
    assert mean_s == pytest.approx(8 / 25 / 3)
    assert n == len(pm.possession_spells(c, pm.valid_runs(c)))


def test_a_switch_through_dead_is_not_a_team_switch():
    assert cmp.team_switches(codes("HHDAA")) == 0
    assert cmp.team_switches(codes("HHAAHH")) == 2


def test_the_report_has_the_changed_frames_and_the_possession_difference():
    r = cmp.compare(codes("HHAAAA"), codes("HHHHAA"))
    text = "\n".join(cmp.report("a.csv", "b.csv", r))
    assert "frames with a different label: 2" in text
    assert "  ref A -> new H: 2" in text
    assert "new minus ref = +0" in text


def test_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="length mismatch"):
        cmp.compare(codes("HHA"), codes("HH"))


def test_main_prints_aggregates_and_no_frame_numbers(tmp_path, capsys):
    n = 5000
    a = "D" * 777 + "H" * 1234 + "A" * (n - 777 - 1234)         # the switch is at frame 2011
    b = "D" * 777 + "H" * 1771 + "A" * (n - 777 - 1771)         # moved to frame 2548
    assert cmp.main([write(tmp_path / "a.csv", a), write(tmp_path / "b.csv", b)]) == 0
    out = capsys.readouterr().out
    assert "frames with a different label: 537" in out
    assert "  ref A -> new H: 537" in out
    assert "possessions: ref 2" in out and "new minus ref = +0" in out
    for frame_number in ("2011", "2548", "2010", "2547"):
        assert frame_number not in out                         # nothing per frame
    assert len(out.splitlines()) < 15


def test_main_refuses_a_bad_file(tmp_path, capsys):
    good = write(tmp_path / "a.csv", "HHAA")
    bad = tmp_path / "b.csv"
    bad.write_text("frame,label\n0,H\n1,Q\n2,A\n3,A\n")
    assert cmp.main([good, str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_refuses_files_of_different_length(tmp_path, capsys):
    assert cmp.main([write(tmp_path / "a.csv", "HHAA"), write(tmp_path / "b.csv", "HHA")]) == 1
    assert "FAIL" in capsys.readouterr().out
