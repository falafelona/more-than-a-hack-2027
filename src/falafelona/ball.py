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
# Every method is a function (match, cfg) -> BallTrack registered in METHODS. To add one
# (#19 reject, #30 interpolate, #35 kalman), write the function and add one line to METHODS.
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


# method name in configs/*.yaml -> function(match, cfg) -> BallTrack
METHODS = {
    "none": _none,
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