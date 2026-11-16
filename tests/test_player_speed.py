# ws:ball: tests for player_speed in src/falafelona/velocity.py. Synthetic data only.
#
# Every track here is made up inside the test: no match data. Safe for anyone, for CI and for
# AI tools (CLAUDE.md).

import warnings

import numpy as np
import pytest

from falafelona.velocity import _pieces, ball_velocity, player_speed, player_speed_settings

FPS = 25.0
N = 100
GRADED = np.ones(N, dtype=np.int8)
# 2 frames on each side at 25 fps; a seen position needed before and after the frame
SETTINGS = {"half_window_s": 0.08, "min_points": 3, "min_each_side": 1, "max_speed": 12.0}


def straight_line(n, vx, vy, start=(10.0, 20.0)):
    """A player moving at a constant (vx, vy) m/s."""
    t = np.arange(n) / FPS
    return np.column_stack([start[0] + vx * t, start[1] + vy * t])


def empty_slots(n, n_slots):
    return np.full((n, n_slots, 2), np.nan)


# Each slot gets its own speed, and an empty slot stays empty. 3-4-5 and 0-1.5.
def test_each_slot_has_its_own_speed():
    xy = empty_slots(N, 3)
    xy[:, 0] = straight_line(N, 3.0, 4.0)
    xy[:, 2] = straight_line(N, 0.0, 1.5)
    speed = player_speed(xy, GRADED, FPS, SETTINGS)
    assert speed.shape == (N, 3)
    assert np.allclose(speed[1:-1, 0], 5.0)
    assert np.allclose(speed[1:-1, 2], 1.5)
    assert np.isnan(speed[:, 1]).all()               # nobody in slot 1


# It is the same line fit as the ball's, so on the same track the two must agree exactly.
def test_same_fit_as_the_ball():
    rng = np.random.default_rng(0)
    track = straight_line(N, 2.0, -1.0) + rng.normal(0, 0.05, (N, 2))
    track[40:43] = np.nan
    xy = track[:, None, :]
    players = player_speed(xy, GRADED, FPS, SETTINGS)[:, 0]
    _, ball = ball_velocity(track, GRADED, FPS, SETTINGS)
    assert np.array_equal(players, ball, equal_nan=True)


# A slot that is empty the whole match must not raise a warning (an all-NaN slot is normal:
# 46 slots for 22 players).
def test_empty_slots_raise_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        speed = player_speed(empty_slots(N, 4), GRADED, FPS, SETTINGS)
    assert np.isnan(speed).all()


# Another player takes over a slot, so the position jumps 30 m between two frames. The track is
# cut there: both sides keep their true speed, and only the frame on each side of the cut (its
# window sees one side only) gets none. Without the cut, a fit across the jump reads 200 m/s.
def test_a_big_jump_cuts_the_track():
    xy = empty_slots(N, 1)
    xy[:, 0] = straight_line(N, 3.0, 0.0)
    xy[50:, 0, 0] += 30.0                            # slot continues somewhere else
    loose = player_speed(xy, GRADED, FPS, {**SETTINGS, "max_speed": 1000.0})   # no cut, no cap
    speed = player_speed(xy, GRADED, FPS, SETTINGS)
    assert loose[49, 0] > 100.0
    assert np.isnan(speed[[49, 50], 0]).all()
    assert np.allclose(speed[1:49, 0], 3.0)
    assert np.allclose(speed[51:-1, 0], 3.0)


# The case the cap alone misses: a standing player's slot is taken over by someone 3 m away.
# A line fit across that step reads about 10 m/s for several frames, under max_speed. One such
# slot per team in a stoppage is enough to blur it, so the jump must cut the track.
def test_a_small_jump_under_max_speed_is_cut_too():
    n = 100
    xy = np.full((n, 1, 2), [50.0, 30.0])
    xy[50:, 0, 0] += 3.0
    section = {"half_window_s": 0.2, "min_points": 5, "min_each_side": 2, "max_speed": 12.0}
    loose = player_speed(xy, np.ones(n, dtype=np.int8), FPS, {**section, "max_speed": 1000.0})
    speed = player_speed(xy, np.ones(n, dtype=np.int8), FPS, section)
    assert 5.0 < np.nanmax(loose) < 12.0             # no cut: a run, and the cap would keep it
    assert np.nanmax(speed) < 1e-9                   # cut: standing still on both sides


# A slot can vanish for a while. It is only cut if it reappears further away than a player
# could have run in that time.
def test_a_slot_that_reappears_within_reach_is_not_cut():
    track = np.full((N, 2), np.nan)
    track[:40] = [10.0, 10.0]
    track[65:] = [18.0, 10.0]                        # 8 m in 25 frames (1 s): 8 m/s, possible
    assert _pieces(track, 12.0 / FPS) == [(0, N)]
    track[65:] = [30.0, 10.0]                        # 20 m in 1 s: another player
    assert _pieces(track, 12.0 / FPS) == [(0, 65), (65, N)]


# Frames outside the graded halves get no speed, and a window never mixes the two halves.
def test_halves_and_ungraded_frames():
    xy = empty_slots(N, 1)
    xy[:, 0] = straight_line(N, 3.0, 4.0)
    xy[50:, 0] += 40.0                               # second half starts elsewhere
    period = np.array([1] * 50 + [2] * 50, dtype=np.int8)
    one_sided = {**SETTINGS, "min_each_side": 0}
    assert np.allclose(player_speed(xy, period, FPS, one_sided), 5.0)
    assert np.isnan(player_speed(xy, np.zeros(N, dtype=np.int8), FPS, SETTINGS)).all()


# What the speed is for: 20 of 22 players standing about while the rest jog. The average over
# the slots that have a value must read well under the "still" level (1 m/s) during the
# stoppage and well over it while the game runs, in metres per second, with a bit of jitter.
def test_average_tells_a_stoppage_from_play():
    rng = np.random.default_rng(1)
    n, n_slots = 1000, 22
    xy = np.empty((n, n_slots, 2))
    for slot in range(n_slots):
        start = rng.uniform(10, 90, size=2)
        velocity = rng.normal(0, 2.0, size=2)
        velocity *= 3.0 / max(np.hypot(*velocity), 1e-9)       # jogging at 3 m/s
        walk = start + np.arange(n)[:, None] / FPS * velocity
        walk[400:600] = walk[400]                              # stoppage: stand still
        xy[:, slot] = walk + rng.normal(0, 0.03, size=(n, 2))  # 3 cm of tracking jitter
    section = {"half_window_s": 0.2, "min_points": 5, "min_each_side": 2, "max_speed": 12.0}
    speeds = player_speed(xy, np.ones(n, dtype=np.int8), FPS, section)
    with warnings.catch_warnings():                  # first and last frame: no slot has a value
        warnings.simplefilter("ignore", RuntimeWarning)
        average = np.nanmean(speeds, axis=1)
    assert np.nanmax(average[420:580]) < 0.5                   # standing: far below 1 m/s
    assert np.nanmin(average[20:380]) > 2.0                    # jogging: far above it
    assert np.nanmin(average[620:980]) > 2.0


def test_settings_in_frames_and_cap():
    out = player_speed_settings({**SETTINGS, "half_window_s": 0.2}, 25.0)
    assert out == {"half_window": 5, "min_points": 3, "min_each_side": 1, "max_speed": 12.0}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"max_speed": None}, "max_speed must be a positive number"),
        ({"max_speed": 0}, "max_speed must be a positive number"),
        ({"max_speed": -3.0}, "max_speed must be a positive number"),
        ({"max_speed": True}, "max_speed must be a positive number"),
        ({"half_window_s": 0}, "player speed block: .*half_window_s"),   # shared keys: checked
        ({"min_points": 99}, "player speed block: .*min_points"),        # and named as ours
    ],
)
def test_bad_settings(change, message):
    with pytest.raises(ValueError, match=message):
        player_speed(empty_slots(10, 2), np.ones(10, dtype=np.int8), FPS, {**SETTINGS, **change})


def test_missing_max_speed_is_named():
    section = {key: value for key, value in SETTINGS.items() if key != "max_speed"}
    with pytest.raises(ValueError, match="max_speed"):
        player_speed_settings(section, FPS)


def test_missing_shared_key_is_named():
    section = {key: value for key, value in SETTINGS.items() if key != "min_each_side"}
    with pytest.raises(ValueError, match="player speed block: .*min_each_side"):
        player_speed_settings(section, FPS)


def test_settings_must_be_a_mapping():
    with pytest.raises(TypeError, match="player speed block"):
        player_speed_settings([1, 2, 3], FPS)
