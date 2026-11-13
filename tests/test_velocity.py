# ws:ball: tests for src/falafelona/velocity.py. Synthetic data only.
#
# Every track here is made up inside the test: no match data. Safe for anyone, for CI and for
# AI tools (CLAUDE.md).

import numpy as np
import pytest

from falafelona.velocity import ball_velocity, velocity_settings

FPS = 25.0
GRADED = np.ones(100, dtype=np.int8)
# 2 frames on each side at 25 fps; a seen position needed before and after the frame
SETTINGS = {"half_window_s": 0.08, "min_points": 3, "min_each_side": 1}
ONE_SIDED = {**SETTINGS, "min_each_side": 0}


def straight_line(n, vx, vy):
    """A ball moving at a constant (vx, vy) m/s, starting at (10, 20)."""
    t = np.arange(n) / FPS
    return np.column_stack([10 + vx * t, 20 + vy * t])


# A ball at a steady 5 m/s must come out as exactly (3, 4) m/s on every frame that has a seen
# position on both sides. The first and last frame only have one side.
def test_constant_speed():
    vxy, speed = ball_velocity(straight_line(100, 3.0, 4.0), GRADED, FPS, SETTINGS)
    assert np.allclose(vxy[1:-1], [3.0, 4.0])
    assert np.allclose(speed[1:-1], 5.0)
    assert np.isnan(speed[[0, -1]]).all()


# With one-sided fits allowed, the edges get a value too (exact on a straight line).
def test_constant_speed_one_sided_edges():
    _, speed = ball_velocity(straight_line(100, 3.0, 4.0), GRADED, FPS, ONE_SIDED)
    assert np.allclose(speed, 5.0)


# A missing frame gets no velocity itself, but its neighbours still do.
def test_one_missing_frame():
    xy = straight_line(100, 3.0, 4.0)
    xy[50] = np.nan
    _, speed = ball_velocity(xy, GRADED, FPS, SETTINGS)
    assert np.isnan(speed[50])
    assert np.allclose(speed[[48, 49, 51, 52]], 5.0)


# Too few seen positions in the window: no velocity.
def test_too_few_points():
    xy = np.full((100, 2), np.nan)
    xy[50:52] = [[10.0, 20.0], [10.2, 20.0]]  # only 2 seen frames, min_points is 3
    _, speed = ball_velocity(xy, GRADED, FPS, ONE_SIDED)
    assert np.isnan(speed).all()


# Next to a gap the window sees one side only. On a ball that speeds up, that one-sided fit
# lags behind; by default it gives no value, and it is biased when allowed.
def test_gap_edge_needs_both_sides():
    n = 200
    t = np.arange(n) / FPS
    xy = np.column_stack([10 + 3 * t + 2 * t ** 2, np.full(n, 20.0)])  # accelerating at 4 m/s2
    xy[100:150] = np.nan
    true_vx = 3 + 4 * t
    settings = {"half_window_s": 0.24, "min_points": 3, "min_each_side": 1}   # 6 frames

    vxy, _ = ball_velocity(xy, np.ones(n, dtype=np.int8), FPS, settings)
    one_sided, _ = ball_velocity(xy, np.ones(n, dtype=np.int8), FPS,
                                 {**settings, "min_each_side": 0})

    assert np.isnan(vxy[[99, 150], 0]).all()                  # last frame before, first after
    assert abs(vxy[80, 0] - true_vx[80]) < 1e-6               # both sides: exact mid-run
    assert abs(one_sided[99, 0] - true_vx[99]) > 0.3          # one-sided: lags behind


# The ball jumps to another place at half-time. Windows must not mix the two halves.
def test_halves_are_not_mixed():
    xy = straight_line(100, 3.0, 4.0)
    xy[50:] += 40.0  # second half starts somewhere else
    period = np.array([1] * 50 + [2] * 50, dtype=np.int8)
    _, speed = ball_velocity(xy, period, FPS, ONE_SIDED)
    assert np.allclose(speed, 5.0)


# Frames outside the graded halves get no velocity.
def test_ungraded_frames_are_nan():
    _, speed = ball_velocity(straight_line(100, 3.0, 4.0), np.zeros(100, dtype=np.int8), FPS,
                             SETTINGS)
    assert np.isnan(speed).all()


# With tracking jitter, the line fit is closer to the truth than "after minus before".
def test_less_noisy_than_two_point_difference():
    rng = np.random.default_rng(0)
    n, k = 5000, 3
    xy = straight_line(n, 3.0, 4.0) + rng.normal(0, 0.1, (n, 2))
    settings = {"half_window_s": k / FPS, "min_points": 3, "min_each_side": 1}
    vxy, _ = ball_velocity(xy, np.ones(n, dtype=np.int8), FPS, settings)
    two_point = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    fit_error = np.abs(vxy[k:-k] - [3.0, 4.0]).mean()
    two_point_error = np.abs(two_point - [3.0, 4.0]).mean()
    assert fit_error < two_point_error


# The window is set in seconds, so it means the same time at any frame rate.
def test_window_is_in_seconds():
    section = {"half_window_s": 0.2, "min_points": 3, "min_each_side": 1}
    assert velocity_settings(section, 25.0)["half_window"] == 5
    assert velocity_settings(section, 50.0)["half_window"] == 10


@pytest.mark.parametrize(
    "change, message",
    [
        ({"half_window_s": None}, "half_window_s must be a positive number"),
        ({"half_window_s": 0}, "half_window_s must be a positive number"),
        ({"half_window_s": True}, "half_window_s must be a positive number"),
        ({"half_window_s": 0.01}, "shorter than one frame"),
        ({"min_points": 1}, "min_points"),
        ({"min_points": 6}, "min_points"),               # window is 5 frames
        ({"min_points": 3.0}, "min_points"),
        ({"min_points": True}, "min_points"),
        ({"min_each_side": 3}, "min_each_side"),         # half window is 2 frames
        ({"min_each_side": -1}, "min_each_side"),
    ],
)
def test_bad_settings(change, message):
    with pytest.raises(ValueError, match=message):
        ball_velocity(np.zeros((10, 2)), np.ones(10, dtype=np.int8), FPS, {**SETTINGS, **change})


def test_missing_settings_are_named():
    with pytest.raises(ValueError, match="min_each_side"):
        velocity_settings({"half_window_s": 0.08, "min_points": 3}, FPS)
