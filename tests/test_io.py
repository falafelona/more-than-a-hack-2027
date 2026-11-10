# io.py loads the synthetic bundle exactly, in metres, and refuses unsafe data locations.
import subprocess
import sys

import numpy as np
import pytest

from falafelona import io, vendor


def test_loads_synthetic_bundle_exactly(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    assert match.ball_xy.shape == (500, 2)
    assert match.players_xy.shape == (500, 46, 2)
    np.testing.assert_allclose(match.ball_xy, bundle["ball_xy"], atol=1e-3, equal_nan=True)
    np.testing.assert_allclose(match.players_xy, bundle["players_xy"], atol=1e-3, equal_nan=True)
    assert list(match.slot_team) == list(bundle["slot_team"])
    assert match.fps == 25.0


def test_alignment_frame_id_minus_one(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    # frame_id 1 in the file is submission row 0
    np.testing.assert_allclose(match.ball_xy[0], bundle["ball_xy"][0], atol=1e-3)


def test_ball_gap_and_out_of_bounds_kept(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    lo, hi = bundle["ball_gap"]
    assert np.isnan(match.ball_xy[lo:hi]).all()
    lo, hi = bundle["ball_out"]
    length = vendor.games.GAMES[vendor.games.GAME]["pitch"][0]
    assert (match.ball_xy[lo:hi, 0] > length).all()


def test_cache_gives_same_arrays(bundle):
    first = io.load_match(n_frames=bundle["n_frames"])
    assert list((bundle["root"] / "cache").iterdir())
    second = io.load_match(n_frames=bundle["n_frames"])
    np.testing.assert_array_equal(first.players_xy, second.players_xy)
    np.testing.assert_array_equal(first.ball_xy, second.ball_xy)


def test_summary_has_no_values(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    summary = io.summary(match)
    assert summary["ball_xy"] == (500, 2)
    assert summary["slots_H"] == 23 and summary["slots_A"] == 23


def test_period_uses_official_half_blocks():
    period = io.period_of_frames(199_319)
    assert int(np.sum(period == 1)) == 90_651 - 11_185 + 1
    assert int(np.sum(period == 2)) == 190_197 - 110_808 + 1
    assert period[11_184] == 0 and period[11_185] == 1


def test_requires_mtah_data_dir(monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        io.data_dir()


def test_refuses_data_dir_inside_repo(monkeypatch):
    monkeypatch.setenv("MTAH_DATA_DIR", str(vendor.REPO_ROOT / "vendor" / "challenge"))
    with pytest.raises(RuntimeError, match="inside the repo"):
        io.data_dir()


def test_import_reads_no_data():
    # Importing io with no MTAH_DATA_DIR must not fail and must not load anything (PR #4 did).
    env = {"PATH": "", "PYTHONPATH": str(vendor.REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", "import falafelona.io"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


# --- Smart Tagging and homography (task 1.7) ---

def test_tags_in_submission_rows(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    got = [(e.start, e.end, e.code, e.type, e.team, e.half) for e in match.tags]
    assert got == [
        (0, 50, "SET PIECES", "Kick Off (Start)", "H", 1),
        (50, 160, "BUILD UP", None, "H", 1),
        (200, 225, "SET PIECES", "Throw-in", None, 1),    # Team N/A -> None
        (250, 350, "Progression", None, "A", 1),           # "ATLETIC CLUB" matched
        (450, 500, "SET PIECES", "Kick Off (Start)", "A", 2),  # clipped to n_frames
    ]                                                      # the tag after the end is dropped


def test_select_tags_is_case_insensitive(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    kickoffs = io.select_tags(match.tags, code="set pieces", type="KICK OFF (START)")
    assert [e.half for e in kickoffs] == [1, 2]
    assert [e.code for e in io.select_tags(match.tags, team="A")] == ["Progression", "SET PIECES"]


def test_unknown_club_names_raise(bundle, monkeypatch):
    monkeypatch.setitem(io.TAG_TEAMS, vendor.games.GAME, {"real madrid": "H"})
    with pytest.raises(ValueError, match="maps to H or A"):
        io.load_match(n_frames=bundle["n_frames"], use_cache=False)


def test_homography_row_i_is_frame_i(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    assert match.homography.shape == (500, 3, 3)
    np.testing.assert_allclose(match.homography, bundle["homography"], equal_nan=True)
    np.testing.assert_allclose(match.homography_error, bundle["homography_error"])
    np.testing.assert_array_equal(match.homography_valid, bundle["homography_valid"])
    assert np.isnan(match.homography[~match.homography_valid]).all()


def test_homography_shorter_than_n_frames_raises(bundle):
    with pytest.raises(ValueError, match="expected frames"):
        io.load_match(n_frames=bundle["n_frames"] + 1, use_cache=False)


def test_summary_counts_tags_and_homography(bundle):
    summary = io.summary(io.load_match(n_frames=bundle["n_frames"], use_cache=False))
    assert (summary["tags"], summary["tags_H"], summary["tags_A"], summary["tags_no_team"]) \
        == (5, 2, 2, 1)
    assert summary["homography_valid_pct"] == 98.0
