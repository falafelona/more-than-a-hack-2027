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
# To add another (#35 kalman), write the function and add one line to METHODS.
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


# method name in configs/*.yaml -> function(match, cfg) -> BallTrack
METHODS = {
    "none": _none,
    "reject": _reject,
    "interpolate": _interpolate,
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