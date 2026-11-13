# ws:ball: tests for player_speed in src/falafelona/velocity.py. Synthetic data only.
#
# Every track here is made up inside the test: no match data. Safe for anyone, for CI and for
# AI tools (CLAUDE.md).

import warnings

import numpy as np
import pytest

from falafelona.velocity import ball_velocity, player_speed, player_speed_settings

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


# Another player takes over a slot, so the position jumps 30 m between two frames. A line fit
# across the jump reads far above any sprint; max_speed drops those frames and only those.
def test_jump_is_dropped_by_max_speed():
    xy = empty_slots(N, 1)
    xy[:, 0] = straight_line(N, 3.0, 0.0)
    xy[50:, 0, 0] += 30.0                            # slot continues somewhere else
    loose = player_speed(xy, GRADED, FPS, {**SETTINGS, "max_speed": 1000.0})
    capped = player_speed(xy, GRADED, FPS, SETTINGS)
    assert loose[49, 0] > 100.0                      # without the cap the jump looks like 200 m/s
    assert np.isnan(capped[48:52, 0]).all()          # every window that contains the jump
    assert np.allclose(capped[1:48, 0], 3.0)
    assert np.allclose(capped[52:-1, 0], 3.0)


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