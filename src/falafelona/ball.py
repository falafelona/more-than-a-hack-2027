# ws:ball: clean the ball track.
#
# Input: raw ball positions (NaN where missing: ~41% of graded frames).
# Output: cleaned ball track, same shape, NaN where still unknown.
#
# Methods are selected and tuned via the `ball:` section of the config.
#
# Issue numbers like #9, #19, #30, #35 in this file are issues on falafelona/team-board,
# not PRs of this repo ("PR #9" below means this repo's PR).
#
# Two ways in, same result:
#   clean(match, cfg)       -> BallTrack(xy, source, speed)   (the interface in architecture.md)
#   DataCleaner(match, cfg) -> .return_cleaned_data() is a Match (what pipeline.py passes on)
#
# Every method is a function (match, cfg) -> BallTrack registered in METHODS. Methods so far:
#   none    the raw track, unchanged (the baseline, task 2.2)
#   reject  throws away false detections: impossible jumps and a ball far from every player
#           (#19, task 2.3)
#   interpolate  fills short gaps with a straight line between the last and the next seen
#           ball (#30, task 2.4). Can run `reject` first.
#   kalman  fills gaps with a Kalman filter and RTS smoother (constant-velocity model), and
#           only where the model is still sure enough (#35, task 2.5). Can run `reject` first.
# To add another, write the function and add one line to METHODS.
#
# NOTE: no `from __future__ import annotations` here. tests/test_pipeline.py checks that
# DataCleaner.__init__.__annotations__["raw_data"] is Match, and that needs a real object,
# not a string.

from dataclasses import dataclass, replace

import numpy as np

from falafelona.io import Match

# Where each frame's ball position came from. Downstream stages (deadball, possession) can
# trust OBSERVED more than INTERPOLATED, and know that REJECTED was seen but thrown away.
OBSERVED = 0  # the tracker saw the ball; value untouched
INTERPOLATED = 1  # we filled a gap (#30, #35)
REJECTED = 2  # the tracker saw something, we judged it a false detection (#19); xy is NaN
UNKNOWN = 3  # no ball, no estimate; xy is NaN

# Config used when DataCleaner is built without one (the baseline: pass the raw track through).
DEFAULT_CFG = {"ball": {"method": "none"}}


@dataclass(frozen=True)
class BallTrack:
    """The cleaned ball, one row per submission frame, positions in metres."""

    xy: np.ndarray  # float[n, 2], NaN where unknown
    source: np.ndarray  # int8[n], one of OBSERVED / INTERPOLATED / REJECTED / UNKNOWN
    speed: np.ndarray  # float[n], same convention as Match.ball_speed (NaN where no ball)


def _none(match, cfg):
    """Method `none` (baseline): the raw track, unchanged. No gap filling, no rejection.

    Frames with a position are OBSERVED, the rest are UNKNOWN. Speed is the ATD speed field
    as given, so this stage adds nothing the loader did not already read.
    """
    xy = match.ball_xy.copy()
    missing = np.isnan(xy).any(axis=1)
    source = np.where(missing, UNKNOWN, OBSERVED).astype(np.int8)
    return BallTrack(xy=xy, source=source, speed=match.ball_speed.copy())


# ---------------------------------------------------------------------------------------------
# Method `reject` (#19): throw away false detections
#
# Two checks. Each one is switched on by its keys in the `ball:` section of the config, and
# stays off when its keys are absent. No threshold is written in this file.
#
#   speed check    max_speed (m/s), speed_slack (m), max_false_run (s)
#       A ball cannot move faster than max_speed. If two consecutive seen frames are further
#       apart than max_speed * (seconds between them) + speed_slack, the track has jumped.
#       The jumps and the gaps cut the track into stretches. We keep the chain of stretches that
#       connects (the move from the end of one to the start of the next is possible) and holds
#       the most frames. A short stretch that does not fit the chain is a false detection, and
#       is rejected. A stretch longer than max_false_run seconds is never rejected: too long to
#       be a blip. Time between frames counts, so a ball that comes back far away after a gap is
#       fine.
#
#   player check   max_player_distance (m), and optionally min_players (default 1)
#       A ball inside the pitch that is further than max_player_distance from every tracked
#       player is rejected. Frames where fewer than min_players players are tracked are not
#       judged. A ball outside the pitch is never judged here: it is the evidence that the
#       dead-ball stage (deadball.py, out_of_bounds) works with.
#
# A rejected frame gets xy = NaN and source = REJECTED. Later stages that read match.ball_xy
# see it as a missing ball.
# ---------------------------------------------------------------------------------------------

_CHUNK_FRAMES = 20_000  # frames per block in the player check; only keeps memory use small


def _number(section, key, *, allow_zero=False, method="reject"):
    """One number from the config, checked to be finite and positive (or zero if allowed)."""
    if key not in section:
        raise ValueError(f"ball method {method!r} needs {key!r} in the ball section of the config")
    value = float(section[key])
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        kind = "zero or more" if allow_zero else "more than zero"
        raise ValueError(f"ball.{key} must be a number {kind}, got {section[key]!r}")
    return value


def _reject_settings(section):
    """The thresholds of method `reject`, read from cfg["ball"] and checked.

    Returns (speed, player). Each is a dict of settings, or None when that check is off.
    """
    speed = player = None
    if "max_speed" in section:
        speed = {
            "max_speed": _number(section, "max_speed"),
            "slack": _number(section, "speed_slack", allow_zero=True),
            "max_false_run": _number(section, "max_false_run"),
        }
    if "max_player_distance" in section:
        min_players = section.get("min_players", 1)
        if isinstance(min_players, bool) or not isinstance(min_players, int) or min_players < 1:
            raise ValueError(
                f"ball.min_players must be a whole number of 1 or more, got {min_players!r}"
            )
        player = {
            "max_distance": _number(section, "max_player_distance"),
            "min_players": min_players,
        }
    if speed is None and player is None:
        raise ValueError(
            "ball method 'reject' does nothing without 'max_speed' and/or 'max_player_distance' "
            "in the ball section of the config (use method 'none' to leave the track alone)"
        )
    return speed, player


def _far_from_players(match, max_distance, min_players):
    """bool[n]: the ball is inside the pitch, enough players are tracked, and the nearest one is
    further than max_distance."""
    far = np.zeros(match.n_frames, dtype=bool)
    if match.players_xy.shape[1] == 0:
        return far
    length, width = match.pitch
    ball = match.ball_xy
    inside = (
        (ball[:, 0] >= 0) & (ball[:, 0] <= length) & (ball[:, 1] >= 0) & (ball[:, 1] <= width)
    )  # False where the ball is missing (NaN compares False)
    candidates = np.flatnonzero(inside)
    for lo in range(0, len(candidates), _CHUNK_FRAMES):
        rows = candidates[lo : lo + _CHUNK_FRAMES]
        offset = match.players_xy[rows] - ball[rows, None, :]  # [rows, slots, 2]
        dist = np.hypot(offset[..., 0], offset[..., 1])  # NaN where a slot is empty
        tracked = ~np.isnan(dist)
        nearest = np.where(tracked, dist, np.inf).min(axis=1)
        far[rows] = (tracked.sum(axis=1) >= min_players) & (nearest > max_distance)
    return far


def _stretches(frames, pos, fps, max_speed, slack):
    """Cut the seen frames into stretches: runs of frames in a row, with no jump inside.

    frames: the seen frame numbers, increasing. pos: float[m, 2], their positions in metres.
    A stretch ends where a frame is missing, or where the ball jumps: a move between two
    consecutive seen frames longer than max_speed * seconds + slack.
    Stretch k is frames[starts[k]:ends[k]].

    Cutting at missing frames matters. After a gap the ball could be anywhere, so a fake first
    frame after a gap would otherwise be glued to the real ball from before the gap.
    """
    moved = np.hypot(*np.diff(pos, axis=0).T)
    spacing = np.diff(frames)
    allowed = max_speed * spacing / fps + slack
    breaks = np.flatnonzero((moved > allowed) | (spacing > 1)) + 1
    return np.concatenate(([0], breaks)), np.concatenate((breaks, [len(frames)]))


def _best_chain(
    first_frame, last_frame, first_pos, last_pos, count, protected, fps, max_speed, slack
):
    """Which stretches to keep: the most seen frames in one chain that never jumps.

    Stretch k runs from first_frame[k] (at first_pos[k]) to last_frame[k] (at last_pos[k]) and has
    count[k] seen frames. Two kept stretches that follow each other must connect: the move from
    the end of one to the start of the next is no longer than max_speed * seconds + slack.
    Stretches marked `protected` are always kept (two protected stretches that follow each other
    need not connect). Everything else is dropped if that lets the chain keep more frames.
    Returns the indices of the kept stretches, increasing.
    """
    n_stretches = len(count)
    best = np.full(n_stretches, -np.inf)  # most frames in a chain that ends at stretch k
    parent = np.full(n_stretches, -1)  # the stretch before k in that chain; -1 = chain starts at k
    last_protected = -1
    for k in range(n_stretches):
        # a protected stretch before k cannot be skipped, so the chain must come through it
        earlier = np.arange(max(last_protected, 0), k)
        value_if_start = 0.0 if last_protected < 0 else -np.inf
        best_before, before = -np.inf, -1
        if earlier.size:
            seconds = (first_frame[k] - last_frame[earlier]) / fps
            moved = np.hypot(*(first_pos[k] - last_pos[earlier]).T)
            connects = moved <= max_speed * seconds + slack
            if protected[k]:
                connects |= protected[earlier]
            options = np.where(connects, best[earlier], -np.inf)
            pick = int(np.argmax(options))
            best_before, before = options[pick], int(earlier[pick])
        if best_before > value_if_start:
            best[k], parent[k] = count[k] + best_before, before
        elif value_if_start > -np.inf:
            best[k], parent[k] = count[k] + value_if_start, -1
        if protected[k]:
            last_protected = k
    # the chain cannot end before a protected stretch
    ends = np.where(np.arange(n_stretches) >= last_protected, best, -np.inf)
    kept, k = [], int(np.argmax(ends))
    while k >= 0:
        kept.append(k)
        k = int(parent[k])
    return np.array(kept[::-1], dtype=int)


def _speed_check(xy, frames, fps, max_speed, slack, max_false_run):
    """The frame numbers, among the seen `frames`, that the speed check rejects."""
    if len(frames) < 2:
        return frames[:0]
    pos = xy[frames]
    starts, ends = _stretches(frames, pos, fps, max_speed, slack)
    first, last = frames[starts], frames[ends - 1]
    protected = (last - first + 1) > max_false_run * fps
    kept = _best_chain(
        first, last, pos[starts], pos[ends - 1], ends - starts, protected, fps, max_speed, slack
    )
    dropped = np.ones(len(starts), dtype=bool)
    dropped[kept] = False
    return frames[np.repeat(dropped, ends - starts)]


def _reject(match, cfg):
    """Method `reject`: false detections become NaN and are marked REJECTED."""
    speed_cfg, player_cfg = _reject_settings(cfg["ball"])
    xy = match.ball_xy.copy()
    seen = ~np.isnan(xy).any(axis=1)
    rejected = np.zeros(len(xy), dtype=bool)
    if player_cfg is not None:
        rejected |= _far_from_players(match, **player_cfg)
    if speed_cfg is not None:
        frames = np.flatnonzero(seen & ~rejected)  # the player check already removed some
        rejected[_speed_check(xy, frames, match.fps, **speed_cfg)] = True
    xy[rejected] = np.nan
    speed = match.ball_speed.copy()
    speed[rejected] = np.nan
    source = np.where(seen, np.where(rejected, REJECTED, OBSERVED), UNKNOWN).astype(np.int8)
    return BallTrack(xy=xy, source=source, speed=speed)


# ---------------------------------------------------------------------------------------------
# Method `interpolate` (#30): fill short gaps
#
# A gap is a run of frames with no ball, with a seen ball on both sides. If it is short, the
# ball is put on the straight line between the last seen position before the gap and the first
# one after it, at equal steps in time. Filled frames get source = INTERPOLATED. A long gap is
# left alone (NaN), so the dead-ball stage still sees it as a missing ball.
#
#   max_gap        longest gap to fill, in seconds. Required.
#   max_speed      optional, in m/s. A gap is only filled if the ball could really have gone
#   speed_slack    from one end to the other: distance <= max_speed * seconds + speed_slack.
#                  Otherwise a straight line would invent a path the ball cannot have taken.
#                  speed_slack (m, zero allowed) is required when max_speed is given.
#   reject_first   optional, true or false (default false). Run method `reject` first, with
#                  its own keys (see above), so that rejected false detections become gaps
#                  that can be filled. Needs those keys, including max_false_run when
#                  max_speed is given.
#
# A gap is never filled across the edge of a half, or from the start or end of the data (the
# ball is only known on one side there). The ball speed field is filled the same way, as a
# straight line between the two ends (NaN if either end has none).
# ---------------------------------------------------------------------------------------------


def _interpolate_settings(section):
    """The settings of method `interpolate`, read from cfg["ball"] and checked."""
    settings = {"max_gap": _number(section, "max_gap", method="interpolate")}
    settings["max_speed"] = settings["slack"] = None
    if "max_speed" in section:
        settings["max_speed"] = _number(section, "max_speed", method="interpolate")
        settings["slack"] = _number(section, "speed_slack", allow_zero=True, method="interpolate")
    reject_first = section.get("reject_first", False)
    if reject_first is not True and reject_first is not False:
        raise ValueError(f"ball.reject_first must be true or false, got {reject_first!r}")
    settings["reject_first"] = reject_first
    return settings


def _gaps(xy, period):
    """The gaps that have a seen ball on both sides and do not cross the edge of a half.

    Returns (before, after): for each gap, the frame number of the last seen frame before it
    and of the first seen frame after it. The gap itself is the frames in between.
    """
    missing = np.isnan(xy).any(axis=1)
    padded = np.concatenate(([False], missing, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]  # a gap is frames starts[k]:ends[k]
    inside = (starts > 0) & (ends < len(xy))  # a seen frame exists on both sides
    starts, ends = starts[inside], ends[inside]
    before, after = starts - 1, ends
    changes = np.concatenate(([0], np.cumsum(np.diff(period) != 0)))  # how often period changed
    one_half = (changes[after] == changes[before]) & (period[before] != 0)
    return before[one_half], after[one_half]


def _fill(track, period, fps, settings):
    """A new BallTrack with the short gaps of `track` filled in. `track` is not changed."""
    before, after = _gaps(track.xy, period)
    length = after - before - 1  # frames in each gap
    keep = length <= np.floor(settings["max_gap"] * fps + 1e-9)
    if settings["max_speed"] is not None:
        distance = np.hypot(*(track.xy[after] - track.xy[before]).T)
        seconds = (after - before) / fps
        keep &= distance <= settings["max_speed"] * seconds + settings["slack"]
    before, after, length = before[keep], after[keep], length[keep]

    xy, speed, source = track.xy.copy(), track.speed.copy(), track.source.copy()
    if len(before) == 0:
        return BallTrack(xy=xy, source=source, speed=speed)
    gap_of = np.repeat(np.arange(len(before)), length)  # which gap each filled frame is in
    first = np.cumsum(length) - length
    step = np.arange(length.sum()) - first[gap_of] + 1  # 1, 2, ... inside each gap
    frames = before[gap_of] + step
    t = (step / (after - before)[gap_of])[:, None]  # how far along the line, 0..1
    xy[frames] = track.xy[before[gap_of]] * (1 - t) + track.xy[after[gap_of]] * t
    speed[frames] = (
        track.speed[before[gap_of]] * (1 - t[:, 0]) + track.speed[after[gap_of]] * t[:, 0]
    )  # NaN if either end has no speed
    source[frames] = INTERPOLATED
    return BallTrack(xy=xy, source=source, speed=speed)


def _interpolate(match, cfg):
    """Method `interpolate`: short gaps get a straight line, long gaps stay NaN."""
    settings = _interpolate_settings(cfg["ball"])
    track = _reject(match, cfg) if settings["reject_first"] else _none(match, cfg)
    return _fill(track, match.period, match.fps, settings)


# ---------------------------------------------------------------------------------------------
# Method `kalman` (#35): fill gaps with a Kalman filter and RTS smoother
#
# The ball is modelled as moving at a steady velocity that is pushed around by random
# accelerations (a kick is an acceleration). The filter walks forward through each half, and
# the RTS smoother walks back, so the estimate inside a gap uses the balls seen on BOTH sides,
# and the speed the ball had when it left and when it came back: the path can curve, where
# `interpolate` draws a straight line. The smoother also says how sure it is. Its standard
# deviation grows towards the middle of a gap. A gap is filled only if the model is still
# sure enough everywhere inside it (sigma <= max_sigma): the model, not a fixed time, decides
# how long a gap may be. Observed frames are never changed. Filled frames get
# source = INTERPOLATED; a gap that is not filled stays NaN, so the dead-ball stage still sees
# a missing ball. Never filled across the edge of a half, or from the start or end of the data.
#
#   process_noise       how fast the ball's velocity wanders, in m/s per sqrt(s): in t seconds
#                       the velocity changes by about process_noise * sqrt(t) (white acceleration
#                       noise, continuous time). Required. Larger: the path may bend more, and
#                       the model becomes unsure sooner. Because it is per second, not per frame,
#                       the same value gives the same uncertainty at any frame rate.
#   measurement_noise   std of the tracked position, in m. Required.
#   initial_speed_std   std of the speed of the ball before its first sighting of a half, in
#                       m/s. Required.
#   max_sigma           a gap is filled only where the smoothed position std is at most this, in
#                       m. Required. Note the whole gap must pass: no half-filled gaps.
#   max_speed           optional, as in `interpolate`: do not fill a gap if the ball could not
#   speed_slack         have crossed it this fast. speed_slack is required with max_speed.
#   reject_first        optional, true or false, as in `interpolate`: run `reject` first.
#
# The ball speed field is filled with a straight line between the two ends, as `interpolate`
# does, so it stays in the unit of the field. x and y are filtered independently with the same
# model, so the covariance does not depend on the values: it is computed once per frame.
#
# Two things to know when choosing the settings:
#   - max_sigma is, in practice, a limit on the gap length: sigma depends on how long the gap
#     is and on the settings, not on where the ball was (a little on how many sightings
#     surround the gap: fewer sightings, larger sigma). configs/exp/ball-kalman.yaml says which
#     length its max_sigma allows. A gap that is filled counts as a ball in play for the
#     dead-ball stage, so choose it by the dead-ball result, not only by the ball error.
#   - A filled frame next to an observed one comes from the smoothed path, not from the
#     observation, so the track can step by up to about measurement_noise there.
# ---------------------------------------------------------------------------------------------


def _kalman_settings(section):
    """The settings of method `kalman`, read from cfg[\"ball\"] and checked."""
    settings = {
        "q": _number(section, "process_noise", method="kalman"),
        "r": _number(section, "measurement_noise", method="kalman"),
        "v0": _number(section, "initial_speed_std", method="kalman"),
        "max_sigma": _number(section, "max_sigma", method="kalman"),
        "max_speed": None,
        "slack": None,
    }
    if "max_speed" in section:
        settings["max_speed"] = _number(section, "max_speed", method="kalman")
        settings["slack"] = _number(section, "speed_slack", allow_zero=True, method="kalman")
    reject_first = section.get("reject_first", False)
    if reject_first is not True and reject_first is not False:
        raise ValueError(f"ball.reject_first must be true or false, got {reject_first!r}")
    settings["reject_first"] = reject_first
    return settings


def _smooth_run(z, observed, dt, q, r, v0):
    """Kalman filter and RTS smoother over one run of frames.

    z: float[m, 2] positions, only read where `observed` (bool[m]) is True. The run starts and
    ends on an observed frame. State per axis: position and velocity, constant velocity model
    with continuous-time white acceleration noise (velocity wanders by q * sqrt(seconds)),
    position noise of std r, and a start velocity of std v0. dt is the time between frames in
    seconds. Returns (xy, sigma): the smoothed positions float[m, 2] and the smoothed position
    std float[m] (the same for x and y).

    Written out as plain 2x2 algebra, one frame at a time. tests/test_ball_kalman.py checks it
    against the same filter built from matrices.
    """
    m = len(z)
    r2, q2 = r * r, q * q
    qpp, qpv, qvv = q2 * dt**3 / 3.0, q2 * dt**2 / 2.0, q2 * dt  # noise added in one step
    zx, zy = z[:, 0].tolist(), z[:, 1].tolist()
    obs = observed.tolist()

    # filtered state and covariance of every frame, and the predictions they came from
    fx, fvx, fy, fvy = [0.0] * m, [0.0] * m, [0.0] * m, [0.0] * m
    fa, fb, fc = [0.0] * m, [0.0] * m, [0.0] * m
    px_, pvx_, py_, pvy_ = [0.0] * m, [0.0] * m, [0.0] * m, [0.0] * m
    pa_, pb_, pc_ = [0.0] * m, [0.0] * m, [0.0] * m

    x, vx, y, vy = zx[0], 0.0, zy[0], 0.0
    a, b, c = r2, 0.0, v0 * v0
    fx[0], fvx[0], fy[0], fvy[0], fa[0], fb[0], fc[0] = x, vx, y, vy, a, b, c
    for t in range(1, m):
        x_p, y_p = x + dt * vx, y + dt * vy  # velocity stays what it was
        a_p = a + 2.0 * dt * b + dt * dt * c + qpp
        b_p = b + dt * c + qpv
        c_p = c + qvv
        px_[t], pvx_[t], py_[t], pvy_[t] = x_p, vx, y_p, vy
        pa_[t], pb_[t], pc_[t] = a_p, b_p, c_p
        if obs[t]:
            s = a_p + r2
            k0, k1 = a_p / s, b_p / s
            ix, iy = zx[t] - x_p, zy[t] - y_p
            x, y = x_p + k0 * ix, y_p + k0 * iy
            vx, vy = vx + k1 * ix, vy + k1 * iy
            a, b, c = a_p - k0 * a_p, b_p - k0 * b_p, c_p - k1 * b_p
        else:
            x, y, a, b, c = x_p, y_p, a_p, b_p, c_p
        fx[t], fvx[t], fy[t], fvy[t], fa[t], fb[t], fc[t] = x, vx, y, vy, a, b, c

    # backward pass (Rauch-Tung-Striebel): the last frame is already smoothed
    sx, svx, sy, svy = fx[:], fvx[:], fy[:], fvy[:]
    sa, sb, sc = fa[:], fb[:], fc[:]
    for t in range(m - 2, -1, -1):
        a_t, b_t, c_t = fa[t], fb[t], fc[t]
        a_n, b_n, c_n = pa_[t + 1], pb_[t + 1], pc_[t + 1]
        det = a_n * c_n - b_n * b_n
        m00, m01, m10, m11 = a_t + dt * b_t, b_t, b_t + dt * c_t, c_t  # P_t F^T
        g00 = (m00 * c_n - m01 * b_n) / det  # smoother gain: P_t F^T inverse(P_next)
        g01 = (m01 * a_n - m00 * b_n) / det
        g10 = (m10 * c_n - m11 * b_n) / det
        g11 = (m11 * a_n - m10 * b_n) / det
        ex, evx = sx[t + 1] - px_[t + 1], svx[t + 1] - pvx_[t + 1]
        ey, evy = sy[t + 1] - py_[t + 1], svy[t + 1] - pvy_[t + 1]
        sx[t], svx[t] = fx[t] + g00 * ex + g01 * evx, fvx[t] + g10 * ex + g11 * evx
        sy[t], svy[t] = fy[t] + g00 * ey + g01 * evy, fvy[t] + g10 * ey + g11 * evy
        d0, d1, d2 = sa[t + 1] - a_n, sb[t + 1] - b_n, sc[t + 1] - c_n
        e00, e01 = g00 * d0 + g01 * d1, g00 * d1 + g01 * d2
        e10, e11 = g10 * d0 + g11 * d1, g10 * d1 + g11 * d2
        sa[t] = a_t + e00 * g00 + e01 * g01
        sb[t] = b_t + e00 * g10 + e01 * g11
        sc[t] = c_t + e10 * g10 + e11 * g11
    return np.column_stack([sx, sy]), np.sqrt(np.maximum(sa, 0.0))


def _period_runs(period):
    """[(start, stop)] of each stretch of equal period that is graded (period 1 or 2)."""
    changes = np.flatnonzero(np.diff(period) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(period)]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if period[a] != 0]


def _kalman(match, cfg):
    """Method `kalman`: gaps the smoother is sure about are filled, the rest stay NaN."""
    settings = _kalman_settings(cfg["ball"])
    track = _reject(match, cfg) if settings["reject_first"] else _none(match, cfg)
    xy, speed, source = track.xy.copy(), track.speed.copy(), track.source.copy()
    before, after = _gaps(track.xy, match.period)
    if len(before) == 0:
        return BallTrack(xy=xy, source=source, speed=speed)

    seen = ~np.isnan(track.xy).any(axis=1)
    estimate = np.full_like(track.xy, np.nan)
    sigma = np.full(len(xy), np.nan)
    for start, stop in _period_runs(match.period):
        frames = np.flatnonzero(seen[start:stop]) + start
        if len(frames) < 2:
            continue
        lo, hi = int(frames[0]), int(frames[-1]) + 1  # first to last seen frame of the stretch
        estimate[lo:hi], sigma[lo:hi] = _smooth_run(
            track.xy[lo:hi], seen[lo:hi], 1.0 / match.fps,
            settings["q"], settings["r"], settings["v0"],
        )

    keep = np.array(
        [np.all(sigma[b + 1 : a] <= settings["max_sigma"]) for b, a in zip(before, after)],
        dtype=bool,
    )
    if settings["max_speed"] is not None:
        distance = np.hypot(*(track.xy[after] - track.xy[before]).T)
        seconds = (after - before) / match.fps
        keep &= distance <= settings["max_speed"] * seconds + settings["slack"]
    before, after = before[keep], after[keep]
    for b, a in zip(before, after):
        frames = np.arange(b + 1, a)
        t = (frames - b) / (a - b)
        xy[frames] = estimate[frames]
        speed[frames] = track.speed[b] * (1 - t) + track.speed[a] * t  # NaN if an end has none
        source[frames] = INTERPOLATED
    return BallTrack(xy=xy, source=source, speed=speed)


# method name in configs/*.yaml -> function(match, cfg) -> BallTrack
METHODS = {
    "none": _none,
    "reject": _reject,
    "interpolate": _interpolate,
    "kalman": _kalman,
}


def clean(match, cfg):
    """Run the ball method named in cfg["ball"]["method"] and return a BallTrack."""
    name = cfg["ball"]["method"]
    if name not in METHODS:
        available = ", ".join(sorted(METHODS))
        raise ValueError(f"unknown ball method {name!r}; available: {available}")
    return METHODS[name](match, cfg)


def _with_track(match, track):
    """The Match the later stages read: `match` itself if nothing changed, else a copy.

    Only the ball fields differ in the copy; players and tags are shared with the raw Match
    (never modified), so the raw Match stays untouched.
    """
    changed = np.isin(track.source, (INTERPOLATED, REJECTED)).any()
    if not changed:
        return match
    return replace(match, ball_xy=track.xy, ball_speed=track.speed)


class DataCleaner:
    """Cleans the ball track of a Match (PR #9 structure).

    cfg is the loaded config dict; without it the baseline method `none` is used, which
    returns the raw Match object itself, unchanged (task 2.2).
    After construction: `.track` is the BallTrack, `.clean_data` the Match for later stages.
    """

    def __init__(self, raw_data: Match, cfg=None):
        self.raw_data = raw_data
        self.cfg = DEFAULT_CFG if cfg is None else cfg
        self.track = clean(raw_data, self.cfg)
        self.clean_data = _with_track(raw_data, self.track)

    def return_cleaned_data(self):
        return self.clean_data
