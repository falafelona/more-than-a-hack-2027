"""Tests for the ball stage, method `none` (task 2.2, team-board issue #9). Synthetic data only.

Issue numbers (#9, #19, #30, #35) are issues on falafelona/team-board, not this repo's PRs.

What `ball: method: none` means: do not touch the ball track. The baseline config uses it, so
the baseline can reproduce the official score (spellF1 0.2433).

PART 1 checks exactly that. If one of these tests fails, the ball stage has started changing
the data, and the baseline would no longer match the official one.

PART 2 is groundwork that 2.2 does not strictly need: it checks the plumbing that the later
methods (#19 reject false detections, #30 fill short gaps, #35 Kalman) will plug into.

The fake match comes from the `bundle` fixture (tests/conftest.py, tests/synthetic.py):
500 frames, with the ball missing on frames 100-139 and the ball outside the pitch on
frames 250-264.
"""

import numpy as np
import pytest

from falafelona import ball, io, vendor
from falafelona.ball import (
    INTERPOLATED,
    METHODS,
    OBSERVED,
    REJECTED,
    UNKNOWN,
    BallTrack,
    DataCleaner,
    clean,
)
from falafelona.config import load_config

CONFIGS = vendor.REPO_ROOT / "configs"
NONE_CFG = {"ball": {"method": "none"}}


@pytest.fixture
def match(bundle):
    """The fake match, loaded the same way the real pipeline loads the real one."""
    return io.load_match(n_frames=bundle["n_frames"], use_cache=False)


# ---------------------------------------------------------------------------------------------
# PART 1: method `none` leaves the ball track exactly as it was loaded
# ---------------------------------------------------------------------------------------------


def test_none_returns_the_raw_track(match):
    """Every ball position and speed that goes in comes out identical, missing ones included."""
    track = clean(match, NONE_CFG)

    assert isinstance(track, BallTrack)
    # assert_array_equal treats NaN (ball missing) as equal to NaN, which plain == does not
    np.testing.assert_array_equal(track.xy, match.ball_xy)
    np.testing.assert_array_equal(track.speed, match.ball_speed)


def test_none_returns_copies_not_the_raw_arrays(match):
    """The output is a copy. A later method that edits it cannot change the raw data by accident."""
    track = clean(match, NONE_CFG)

    assert not np.shares_memory(track.xy, match.ball_xy)
    assert not np.shares_memory(track.speed, match.ball_speed)


def test_none_marks_which_frames_have_a_ball(bundle, match):
    """Frames with a ball are OBSERVED, frames without one are UNKNOWN. Nothing else is used."""
    track = clean(match, NONE_CFG)
    gap_start, gap_stop = bundle["ball_gap"]

    assert track.source.dtype == np.int8
    assert (track.source[gap_start:gap_stop] == UNKNOWN).all()  # the ball-missing frames
    assert (track.source[:gap_start] == OBSERVED).all()  # everything before the gap
    assert (track.source[gap_stop:] == OBSERVED).all()  # everything after the gap
    # `none` never fills a gap (INTERPOLATED) or throws a ball away (REJECTED)
    assert not np.isin(track.source, (INTERPOLATED, REJECTED)).any()


def test_none_keeps_balls_outside_the_pitch(bundle, match):
    """A ball outside the pitch is kept as it is. Removing it is #19's job, not the baseline's."""
    out_start, out_stop = bundle["ball_out"]
    track = clean(match, NONE_CFG)

    np.testing.assert_array_equal(track.xy[out_start:out_stop], match.ball_xy[out_start:out_stop])
    assert (track.source[out_start:out_stop] == OBSERVED).all()


def test_none_gives_the_same_answer_every_time(match):
    """Running it twice on the same match gives the same result (runs must be repeatable)."""
    first = clean(match, NONE_CFG)
    second = clean(match, NONE_CFG)

    np.testing.assert_array_equal(first.xy, second.xy)
    np.testing.assert_array_equal(first.source, second.source)


def test_datacleaner_without_config_hands_back_the_same_match(match):
    """pipeline.py builds DataCleaner(match) with no config today. It must get the same Match back.

    "The same Match" means the very same object, so nothing is copied and nothing can differ.
    """
    cleaner = DataCleaner(match)

    assert cleaner.return_cleaned_data() is match
    assert isinstance(cleaner.track, BallTrack)


def test_baseline_config_runs_through_ball_clean(match):
    """Issue #9 "done when": the baseline config runs through ball.clean, ball left untouched."""
    cfg = load_config(CONFIGS / "baseline.yaml")  # the real file, not a made-up dict
    track = clean(match, cfg)

    assert isinstance(track, BallTrack)
    np.testing.assert_array_equal(track.xy, match.ball_xy)
    np.testing.assert_array_equal(track.speed, match.ball_speed)


def test_baseline_config_hands_back_the_same_match(match):
    """With configs/baseline.yaml (ball: method: none) the stage also returns the same Match."""
    cfg = load_config(CONFIGS / "baseline.yaml")

    assert DataCleaner(match, cfg).return_cleaned_data() is match


@pytest.mark.parametrize("name", ["baseline.yaml", "final.yaml"])
def test_shipped_configs_name_a_method_that_exists(name):
    """baseline.yaml and final.yaml both say `ball: method: none`. That name must be registered."""
    cfg = load_config(CONFIGS / name)

    assert cfg["ball"]["method"] in METHODS


def test_unknown_method_gives_a_clear_error(match):
    """A typo in the config fails loudly and lists the valid names, instead of doing nothing."""
    with pytest.raises(ValueError, match="none"):  # the message lists the valid method "none"
        clean(match, {"ball": {"method": "no_such_method"}})
    with pytest.raises(ValueError, match="no_such_method"):  # and repeats the bad name
        DataCleaner(match, {"ball": {"method": "no_such_method"}})


def test_datacleaner_keeps_its_type_annotation():
    """DataCleaner's `raw_data: Match` annotation must stay a real class, not a string.

    tests/test_pipeline.py checks this too. It breaks if someone adds
    `from __future__ import annotations` at the top of ball.py.
    """
    assert DataCleaner.__init__.__annotations__["raw_data"] is io.Match


# ---------------------------------------------------------------------------------------------
# PART 2: groundwork for the later methods (not required by 2.2 itself)
# ---------------------------------------------------------------------------------------------


def test_a_method_that_changes_the_ball_gives_a_new_match(match, monkeypatch):
    """When a method does change the ball, the next stage gets a new Match; the raw one is intact.

    There is no real cleaning method yet, so this registers a fake one called "reject_one".
    It throws away the ball on frame 10 (a frame where the fake match does have a ball).
    """
    frame = 10
    assert not np.isnan(match.ball_xy[frame]).any()  # the fake match has a ball on frame 10
    raw_xy = match.ball_xy.copy()
    raw_speed = match.ball_speed.copy()

    def reject_one(m, cfg):
        track = ball._none(m, cfg)
        track.xy[frame] = np.nan
        track.speed[frame] = np.nan
        track.source[frame] = REJECTED
        return track

    monkeypatch.setitem(METHODS, "reject_one", reject_one)  # only for the length of this test
    out = DataCleaner(match, {"ball": {"method": "reject_one"}}).return_cleaned_data()

    assert out is not match  # a change means a new Match
    assert np.isnan(out.ball_xy[frame]).all()  # the ball on frame 10 is gone
    assert np.isnan(out.ball_speed[frame])
    # every other frame is the same as the raw track
    other_frames = np.arange(match.n_frames) != frame
    np.testing.assert_array_equal(out.ball_xy[other_frames], raw_xy[other_frames])
    # the raw Match was not modified
    np.testing.assert_array_equal(match.ball_xy, raw_xy)
    np.testing.assert_array_equal(match.ball_speed, raw_speed)
    # the player data is shared with the raw Match, not copied: this stage does not touch it
    assert out.players_xy is match.players_xy
    assert out.n_frames == match.n_frames