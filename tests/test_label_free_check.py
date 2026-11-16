# ws:int: tests for scripts/label_free_check.py (task M3). Made-up matches only.

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import two_halves as th

from falafelona import io, pipeline, vendor
from falafelona.config import load_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "label_free_check.py"
spec = importlib.util.spec_from_file_location("label_free_check", SCRIPT)
lfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lfc)

FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")


def run(monkeypatch, tmp_path, **options):
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match(**options))
    return pipeline.run(FINAL, tmp_path / "out.csv", n_frames=th.N)


# --- possessions ---------------------------------------------------------------------------------


def test_a_possession_is_a_maximal_run_of_one_team():
    assert lfc.possessions(np.array(list("HHADDAAH"))) == [2, 1, 2, 1]
    assert lfc.possessions(np.array(list("DDXD"))) == []


def test_graded_possessions_skip_ungraded_frames_and_end_at_the_half():
    labels = np.array(list("HHHHAA"))
    period = np.array([0, 1, 1, 2, 2, 2])
    assert lfc.possessions(labels, period) == [2, 1, 2]


# --- kickoff -------------------------------------------------------------------------------------


def test_the_kickoff_report_finds_both_tags_and_a_dead_second_half_wait(monkeypatch, tmp_path):
    result = run(monkeypatch, tmp_path)
    report = lfc.kickoff_report(result["cleaned"], result["labels"], result["dead_score"])

    second = report[2]
    assert second["tag found"] is True
    assert second["graded seconds before kickoff"] == (th.KICKOFF_2 - th.HALF_2[0]) / th.FPS
    assert second["% labelled D"] == 100.0
    assert second["% dead score >= 0.5"] == 100.0
    assert report[1]["graded seconds before kickoff"] == (th.KICKOFF_1 - th.HALF_1[0]) / th.FPS


def test_the_kickoff_report_says_when_the_second_half_tag_is_missing(monkeypatch, tmp_path):
    with pytest.warns(RuntimeWarning, match="no second-half Kick Off"):
        result = run(monkeypatch, tmp_path, with_second_kickoff=False)
    report = lfc.kickoff_report(result["cleaned"], result["labels"], result["dead_score"])
    assert report[2] == {"tag found": False,
                         "graded frames in this half": th.HALF_2[1] - th.HALF_2[0]}


# --- frozen ball ---------------------------------------------------------------------------------


def frozen_match(xy, period):
    n = len(xy)
    return io.Match(n_frames=n, fps=25.0, period=np.asarray(period, dtype=np.int8),
                    ball_xy=np.asarray(xy, dtype=float), ball_speed=np.full(n, np.nan),
                    players_xy=np.full((n, 46, 2), np.nan), players_speed=np.full((n, 46), np.nan),
                    slot_team=np.array(["H"] * 23 + ["A"] * 23), pitch=(105.0, 68.0))


def test_the_frozen_ball_report_counts_exact_repeats_in_graded_frames():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 60, size=(200, 2))           # moving: no exact repeats
    xy[50:110] = [30.0, 30.0]                         # frozen for 60 frames: 59 repeats
    xy[150:160] = np.nan                              # a gap: NaN never repeats
    period = np.ones(200)
    period[180:] = 0                                  # ungraded frames do not count
    xy[180:] = [1.0, 1.0]

    report = lfc.frozen_report(frozen_match(xy, period))
    assert report["frames repeating the previous position"] == (59, round(100 * 59 / 170, 1))
    assert report["longest run, seconds"] == round(59 / 25, 1)
    assert report["runs of 2 s or more"] == 1


# --- the whole report ----------------------------------------------------------------------------


def scalars_only(value):
    if isinstance(value, dict):
        return all(scalars_only(v) for v in value.values())
    if isinstance(value, tuple):
        return len(value) <= 2 and all(scalars_only(v) for v in value)
    return isinstance(value, bool | int | float | str)


def test_the_report_holds_aggregates_only(monkeypatch, tmp_path):
    """No arrays, no lists: the script may print only counts, shares and lengths."""
    result = run(monkeypatch, tmp_path)
    report = lfc.check(result, FINAL)
    assert scalars_only(report)
    assert report["file"]["rows"] == th.N
    assert report["still ball"]["still_ball_at_spot"][0] > 0
    assert report["tags"]["throw-in"] == 1 and report["tags"]["kick off (start)"] == 2


def test_main_builds_into_a_temporary_folder_and_prints_the_report(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    monkeypatch.setattr(pipeline, "N_ROWS", th.N)
    report = lfc.main(["--config", str(vendor.REPO_ROOT / "configs" / "final.yaml")])
    printed = capsys.readouterr().out
    assert "== kickoff" in printed and "== frozen ball (raw tracking)" in printed
    assert report["file"]["sha256"] in printed
