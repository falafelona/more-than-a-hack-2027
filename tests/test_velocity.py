# ws:ball: tests for src/falafelona/velocity.py. Synthetic data only.
#
# Every track here is made up inside the test: no match data. Safe for anyone, for CI and for
# AI tools (CLAUDE.md).

import numpy as np
import pytest

from falafelona.velocity import ball_velocity

FPS = 25.0
GRADED = np.ones(100, dtype=np.int8)


def straight_line(n, vx, vy):
    """A ball moving at a constant (vx, vy) m/s, starting at (10, 20)."""
    t = np.arange(n) / FPS
    return np.column_stack([10 + vx * t, 20 + vy * t])


# A ball at a steady 5 m/s must come out as exactly (3, 4) m/s on every frame, edges included.
def test_constant_speed():
    vxy, speed = ball_velocity(straight_line(100, 3.0, 4.0), GRADED, FPS, half_window=2)
    assert np.allclose(vxy, [3.0, 4.0])
    assert np.allclose(speed, 5.0)


# A missing frame gets no velocity itself, but its neighbours still do.
def test_one_missing_frame():
    xy = straight_line(100, 3.0, 4.0)
    xy[50] = np.nan
    _, speed = ball_velocity(xy, GRADED, FPS, half_window=2)
    assert np.isnan(speed[50])
    assert np.allclose(speed[[48, 49, 51, 52]], 5.0)


# Too few seen positions in the window: no velocity.
def test_too_few_points():
    xy = np.full((100, 2), np.nan)
    xy[50:52] = [[10.0, 20.0], [10.2, 20.0]]  # only 2 seen frames, min_points is 3
    _, speed = ball_velocity(xy, GRADED, FPS, half_window=2)
    assert np.isnan(speed).all()


# The ball jumps to another place at half-time. Windows must not mix the two halves.
def test_halves_are_not_mixed():
    xy = straight_line(100, 3.0, 4.0)
    xy[50:] += 40.0  # second half starts somewhere else
    period = np.array([1] * 50 + [2] * 50, dtype=np.int8)
    _, speed = ball_velocity(xy, period, FPS, half_window=2)
    assert np.allclose(speed, 5.0)


# Frames outside the graded halves get no velocity.
def test_ungraded_frames_are_nan():
    _, speed = ball_velocity(straight_line(100, 3.0, 4.0), np.zeros(100, dtype=np.int8), FPS, 2)
    assert np.isnan(speed).all()


# With tracking jitter, the line fit is closer to the truth than "after minus before".
def test_less_noisy_than_two_point_difference():
    rng = np.random.default_rng(0)
    n, k = 5000, 3
    xy = straight_line(n, 3.0, 4.0) + rng.normal(0, 0.1, (n, 2))
    vxy, _ = ball_velocity(xy, np.ones(n, dtype=np.int8), FPS, half_window=k)
    two_point = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    fit_error = np.abs(vxy[k:-k] - [3.0, 4.0]).mean()
    two_point_error = np.abs(two_point - [3.0, 4.0]).mean()
    assert fit_error < two_point_error


def test_bad_settings():
    xy = np.zeros((10, 2))
    period = np.ones(10, dtype=np.int8)
    with pytest.raises(ValueError):
        ball_velocity(xy, period, FPS, half_window=0)
    with pytest.raises(ValueError):
        ball_velocity(xy, period, FPS, half_window=2, min_points=1)