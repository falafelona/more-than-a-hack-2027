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

PLAYERS: player_speed() does the same line fit for every player slot, from the slot's own
positions, in metres per second. The ATD's players_speed field is not used for this: it may be
mostly empty and its unit is still being checked (the same doubt as for the ball's).

    speed = velocity.player_speed(match.players_xy, match.period, match.fps,
                                  cfg["deadball"]["speed"])      # float[n, n_slots]

Its settings are the three keys above plus one more, checked by player_speed_settings():
    max_speed       m/s, set above the fastest sprint (about 12 m/s). The 46 slots are
                    tracklets: when another player takes over a slot, its position jumps. Used
                    twice:
                    1. Jumps break the track. Where a slot moves further between two seen
                       positions than max_speed allows (max_speed / fps metres per frame), the
                       track is cut there and each piece is fitted on its own. Without the cut, a
                       line fit across a jump of a few metres reads as a sprint for several
                       frames, still under max_speed, so the cap alone would not catch it.
                    2. Any speed still above it is dropped (NaN), as a last guard.
deadball.players_still uses it, with the `speed:` block of the `deadball:` section.

No data is read here. The tests use made-up tracks.
"""

import itertools

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


def player_speed_settings(section, fps):
    """Check a player speed block (the velocity keys plus max_speed) and turn it into frames.

    Returns {"half_window": frames, "min_points": n, "min_each_side": n, "max_speed": m/s}.
    Errors name the bad key and say they come from the player speed block, not from the
    ball's `velocity:` block.
    """
    try:
        settings = velocity_settings(section, fps)
    except (TypeError, ValueError) as error:
        raise type(error)(f"player speed block: {error}") from error
    if "max_speed" not in section:
        raise ValueError("player speed block: missing ['max_speed']")
    cap = section["max_speed"]
    if not _is_number(cap) or cap <= 0:
        raise ValueError(f"player speed block: max_speed must be a positive number, got {cap!r}")
    settings["max_speed"] = float(cap)
    return settings


def player_speed(xy, period, fps, section):
    """Speed of every player slot on every frame, from the slot's positions, in m/s.

    xy: float[n, n_slots, 2] metres (NaN = slot empty), as Match.players_xy.
    period: int8[n] (1, 2, or 0 = ungraded). section: the player speed block, see
    player_speed_settings. Returns float[n, n_slots].

    Each slot is fitted on its own with the same line fit as the ball (ball_velocity), and each
    half is handled on its own. Where the slot jumps further than max_speed allows, its track is
    cut and each piece is fitted on its own (see _pieces). A value is NaN where the slot is empty
    on that frame, where the window has too few seen positions (also next to a cut: the window
    then only sees one side), outside the graded halves, and where the speed is still above
    max_speed.
    """
    settings = player_speed_settings(section, fps)
    cap = settings.pop("max_speed")
    speed = np.full(xy.shape[:2], np.nan)
    for half in (1, 2):
        frames = np.flatnonzero(period == half)
        if not frames.size:
            continue
        part = slice(frames[0], frames[-1] + 1)
        for slot in range(xy.shape[1]):
            track = xy[part, slot]
            if np.isnan(track).all():                    # slot never used in this half
                continue
            slopes = np.full_like(track, np.nan)
            for first, stop in _pieces(track, cap / fps):
                slopes[first:stop] = _slopes(track[first:stop], **settings) * fps  # m/s
            speed[part, slot] = np.hypot(slopes[:, 0], slopes[:, 1])
    return np.where(speed > cap, np.nan, speed)


def _pieces(track, max_step):
    """Cut one slot's track where it jumps. Returns (first, stop) frame ranges covering it.

    A jump: two seen positions further apart than max_step metres per frame between them (a
    missing frame in between allows one more step). The new piece starts at the frame after
    the jump. A piece shorter than the fit window simply gets no speed.
    """
    seen = np.flatnonzero(~np.isnan(track).any(axis=1))
    if seen.size < 2:
        return [(0, len(track))]
    step = np.hypot(*np.diff(track[seen], axis=0).T)        # metres between seen positions
    allowed = max_step * np.diff(seen)                      # metres possible in that time
    cuts = seen[1:][step > allowed]
    edges = [0, *cuts.tolist(), len(track)]
    return list(itertools.pairwise(edges))
