# ws:ball: ball velocity per frame, a feature for the possession HMM.
#
# WHAT: from the cleaned ball track (ball.clean: none / reject / interpolate / kalman), how fast
# and in which direction the ball moves on every frame, in metres per second.
# Possession uses it to tell a pass or a fast ball flying past a player (no switch) from a ball
# a player actually controls (slow, moving with the player).
#
# HOW: for each frame we take the ball positions in a small window around it and fit a straight
# line through them (least squares). The slope of that line is the velocity. Compared with
# "position after minus position before", the fit averages out tracking jitter and still works
# when a frame inside the window is missing.
#
#   track = ball.clean(match, cfg)
#   vxy, speed = velocity.ball_velocity(track.xy, match.period, match.fps,
#                                       half_window=..., min_points=...)
#
# half_window and min_points are windows, so they live in configs/*.yaml (CLAUDE.md), not here.
# No data is read here. The tests use made-up tracks.

import numpy as np


# Sum of `values` over the window around each frame, each neighbour weighted by `weights`
# (one weight per offset -k..k). Frames outside the array count as zero.
def _window_sum(values, weights):
    return np.convolve(values, weights[::-1], mode="same")


# The slope (metres per frame) of the straight line fitted through the seen ball positions in
# the window around each frame, for one unbroken stretch of frames (one half).
# Least squares with time t = offset from the frame (-k..k), using only the seen frames:
#   slope = (N * sum(t*x) - sum(t) * sum(x)) / (N * sum(t*t) - sum(t)^2)
def _slopes(xy, half_window, min_points):
    n = len(xy)
    slopes = np.full((n, 2), np.nan)
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    if n < len(offsets):
        return slopes  # stretch shorter than one window

    seen = ~np.isnan(xy).any(axis=1)
    weight = seen.astype(float)
    ones = np.ones_like(offsets)
    count = _window_sum(weight, ones)            # N: seen frames in the window
    sum_t = _window_sum(weight, offsets)
    sum_tt = _window_sum(weight, offsets ** 2)
    spread = count * sum_tt - sum_t ** 2         # 0 when all seen frames share one time

    usable = seen & (count >= min_points) & (spread > 0)
    for axis in (0, 1):
        x = np.where(seen, xy[:, axis], 0.0)     # missing frames add nothing to the sums
        sum_x = _window_sum(x, ones)
        sum_tx = _window_sum(x, offsets)
        slopes[usable, axis] = ((count * sum_tx - sum_t * sum_x)[usable] / spread[usable])
    return slopes


# The function possession calls. Each half is handled on its own, so a window never mixes
# first-half and second-half positions. Velocity is NaN where the ball on that frame is
# unknown, where fewer than min_points positions are seen in the window, and outside the
# graded halves.
def ball_velocity(xy, period, fps, half_window, min_points=3):
    """xy: float[n, 2] metres (NaN = unknown). Returns vxy float[n, 2] and speed float[n], m/s."""
    half_window = int(half_window)
    if half_window < 1:
        raise ValueError(f"half_window must be 1 or more, got {half_window!r}")
    if not 2 <= min_points <= 2 * half_window + 1:
        raise ValueError(f"min_points must be between 2 and the window size, got {min_points!r}")

    vxy = np.full((len(xy), 2), np.nan)
    for half in (1, 2):
        frames = np.flatnonzero(period == half)
        if frames.size:
            part = slice(frames[0], frames[-1] + 1)
            vxy[part] = _slopes(xy[part], half_window, min_points) * fps  # per frame -> per s
    speed = np.hypot(vxy[:, 0], vxy[:, 1])
    return vxy, speed