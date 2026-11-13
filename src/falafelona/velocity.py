"""ws:ball: ball velocity per frame, a feature for the possession HMM.

WHAT: from the cleaned ball track (ball.clean: none / reject / interpolate / kalman), how fast
and in which direction the ball moves on every frame, in metres per second.
Possession uses it to tell a pass or a fast ball flying past a player (no switch) from a ball
a player actually controls (slow, moving with the player). The ATD's own speed field is not
enough for that: it has no direction, and its unit is still being checked.

HOW: for each frame we take the ball positions in a small window around it and fit a straight
line through them (least squares). The slope of that line is the velocity. Compared with
"position after minus position before", the fit averages out tracking jitter and still works
when a frame inside the window is missing.

    track = ball.clean(match, cfg)
    vxy, speed = velocity.ball_velocity(track.xy, match.period, match.fps,
                                        cfg["possession"]["velocity"])

SETTINGS: a `velocity:` block inside the `possession:` section of the config (CLAUDE.md: no
thresholds or windows in the code), checked by velocity_settings():
    half_window_s   seconds on each side of the frame (rounded to whole frames)
    min_points      seen positions needed in the window, the frame itself included
    min_each_side   seen positions needed BEFORE and AFTER the frame. Next to a gap the window
                    only sees one side, and a one-sided fit lags behind a ball that speeds up or
                    slows down, which is where takeovers are decided. 0 allows one-sided fits.
To be tuned on the dev set (D15).

No data is read here. The tests use made-up tracks.
"""

import numpy as np

SETTING_KEYS = ("half_window_s", "min_points", "min_each_side")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value):
    return isinstance(value, int) and not isinstance(value, bool)


def velocity_settings(section, fps):
    """Check a `velocity:` config block and turn it into frames.

    Returns {"half_window": frames, "min_points": n, "min_each_side": n}. Raises ValueError
    with the name of the bad key (TypeError if it is not a mapping), so a typo in the config
    fails loudly instead of silently.
    """
    if not isinstance(section, dict):
        raise TypeError("velocity settings must be a mapping")
    missing = [key for key in SETTING_KEYS if key not in section]
    if missing:
        raise ValueError(f"velocity settings: missing {missing}")
    window_s = section["half_window_s"]
    if not _is_number(window_s) or window_s <= 0:
        raise ValueError(f"velocity: half_window_s must be a positive number, got {window_s!r}")
    half_window = int(np.floor(window_s * fps + 0.5))       # nearest whole frame
    if half_window < 1:
        raise ValueError(f"velocity: half_window_s {window_s} is shorter than one frame")
    min_points, each_side = section["min_points"], section["min_each_side"]
    if not _is_count(min_points) or not 2 <= min_points <= 2 * half_window + 1:
        raise ValueError(f"velocity: min_points must be a whole number from 2 to the window "
                         f"size ({2 * half_window + 1} frames), got {min_points!r}")
    if not _is_count(each_side) or not 0 <= each_side <= half_window:
        raise ValueError(f"velocity: min_each_side must be a whole number from 0 to "
                         f"{half_window}, got {each_side!r}")
    return {"half_window": half_window, "min_points": min_points, "min_each_side": each_side}


def _window_sum(values, weights):
    """Sum of `values` over the window around each frame, each neighbour weighted by `weights`.

    One weight per offset -k..k. Frames outside the array count as zero.
    """
    return np.convolve(values, weights[::-1], mode="same")


def _slopes(xy, half_window, min_points, min_each_side):
    """Slope (metres per frame) of the line fitted through the seen positions around each frame.

    For one unbroken stretch of frames (one half). Least squares with time t = offset from the
    frame (-k..k), using only the seen frames:
        slope = (N * sum(t*x) - sum(t) * sum(x)) / (N * sum(t*t) - sum(t)^2)
    """
    n = len(xy)
    slopes = np.full((n, 2), np.nan)
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    if n < len(offsets):
        return slopes  # stretch shorter than one window

    seen = ~np.isnan(xy).any(axis=1)
    weight = seen.astype(float)
    ones = np.ones_like(offsets)
    count = _window_sum(weight, ones)            # N: seen frames in the window
    before = _window_sum(weight, (offsets < 0).astype(float))
    after = _window_sum(weight, (offsets > 0).astype(float))
    sum_t = _window_sum(weight, offsets)
    sum_tt = _window_sum(weight, offsets ** 2)
    spread = count * sum_tt - sum_t ** 2         # 0 when all seen frames share one time

    usable = (seen & (count >= min_points) & (spread > 0)
              & (before >= min_each_side) & (after >= min_each_side))
    for axis in (0, 1):
        x = np.where(seen, xy[:, axis], 0.0)     # missing frames add nothing to the sums
        sum_x = _window_sum(x, ones)
        sum_tx = _window_sum(x, offsets)
        slopes[usable, axis] = (count * sum_tx - sum_t * sum_x)[usable] / spread[usable]
    return slopes


def ball_velocity(xy, period, fps, section):
    """Velocity of the ball on every frame. The function possession calls.

    xy: float[n, 2] metres (NaN = unknown). period: int8[n] (1, 2, or 0 = ungraded).
    section: the `velocity:` config block (see velocity_settings).
    Returns vxy float[n, 2] and speed float[n], in m/s.

    Each half is handled on its own, so a window never mixes first-half and second-half
    positions. Velocity is NaN where the ball on that frame is unknown, where the window has
    fewer than min_points seen positions or fewer than min_each_side on either side, and
    outside the graded halves.
    """
    settings = velocity_settings(section, fps)
    vxy = np.full((len(xy), 2), np.nan)
    for half in (1, 2):
        frames = np.flatnonzero(period == half)
        if frames.size:
            part = slice(frames[0], frames[-1] + 1)
            vxy[part] = _slopes(xy[part], **settings) * fps  # per frame -> per second
    speed = np.hypot(vxy[:, 0], vxy[:, 1])
    return vxy, speed
