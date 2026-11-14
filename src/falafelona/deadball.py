# ws:dead: score how likely each frame is dead ball.
#
# Output: score in 0..1 per frame, fed to the decoder in possession.py (docs/decisions.md D13).
#
# Clues (each is a method; `combined` adds them up with weights from the config):
# - ball_missing: official baseline, dead wherever the ball is not observed
# - out_of_bounds: ball clearly past a line of the match's own pitch
# - players_still: players' typical (median) speed low for a while, speed computed from their
#   positions (velocity.player_speed, the `speed:` block), not from the ATD's speed field
# - gap_chance: inside each ball gap, the chance that the ball is dead (0-1), one value for the
#   whole gap, from its length, where the ball was last seen and where it comes back, and
#   whether the players stand still meanwhile (the `gap:` block). 0 where the ball is seen.
#   The HMM reads a dead score above 0.5 as "leans dead", so this lets a gap with no sign of a
#   stoppage keep its possession, where ball_missing makes every gap dead.
# - before_kickoff: graded second-half frames before its kickoff, found from the Kick Off (Start)
#   Smart Tag, never a hard-coded frame
# - before_restart: the seconds before each Smart Tagging restart (evidence, not truth;
#   throw-ins have team N/A, and team is not used here)
#
# Methods are selected via the `deadball:` section of the config, which is checked up front.
import warnings

import numpy as np

from falafelona import velocity
from falafelona.io import select_tags

METHODS = ("ball_missing", "out_of_bounds", "players_still", "gap_chance", "before_kickoff",
           "before_restart", "combined")
# The Smart Tagging Type values that are restarts (DATA.md). Kick Off (Start) has its own method.
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)")
# The settings each method reads from the config.
SETTINGS = {
    "out_of_bounds": ("out_margin",),
    "players_still": ("still_speed", "still_seconds"),
}
# players_still also needs the player speed block `speed:` (velocity.player_speed_settings).
SPEED_KEYS = (*velocity.SETTING_KEYS, "max_speed")
# gap_chance reads the `gap:` block: these settings, and one weight per clue in `weights:`.
GAP_SETTINGS = ("line_distance", "spot_distance", "tag_seconds", "length_weight")
GAP_CLUES = ("last_seen_near_line", "reappears_near_line", "reappears_at_centre",
             "players_still", "restart_tag_after")


class DeadBall:
    """Scores how likely each frame is dead ball (0 = alive, 1 = dead)."""

    def __init__(self, match, ball, cfg):
        self.data = match              # the Match from io.load_match() (after ball cleaning)
        self.ball = ball               # the BallTrack from ball.py, not used by any method yet
        self.cfg = cfg["deadball"]     # only our section of the config
        check_config(self.cfg, match.fps)   # a typo in the config fails here, with a clear message

    def score(self):
        """Run the method named in the config."""
        return self.run(self.cfg["method"])

    def run(self, method):
        """Run one method by its name."""
        if method == "out_of_bounds":
            return self.out_of_bounds()
        elif method == "ball_missing":
            return self.ball_missing()
        elif method == "players_still":
            return self.players_still()
        elif method == "gap_chance":
            return self.gap_chance()
        elif method == "before_kickoff":
            return self.before_kickoff()
        elif method == "before_restart":
            return self.before_restart()
        elif method == "combined":
            return self.combined()
        raise ValueError(f"unknown deadball method : {method}")

    def combined(self):
        """Each clue adds its weight where it fires. The total is capped at 1."""
        total = np.zeros(self.data.n_frames)            # start every frame at 0 points
        for method, weight in self.cfg["weights"].items():
            total = total + weight * self.run(method)   # a clue adds its weight where it is 1
        total = np.minimum(total, 1.0)                  # never above 1
        return total

    def ball_missing(self):
        """Official baseline: dead (1.0) wherever the ball is not seen, alive (0.0) otherwise."""
        missing = self.data.ball_xy[:,0]
        missing = np.isnan(missing)
        missing = missing.astype(float)
        return missing

    def out_of_bounds(self):
        """1.0 where the ball is clearly past a line (by more than out_margin), else 0.0."""
        length, width = self.data.pitch                 # this match's pitch (dev-set ones differ)
        margin = self.cfg["out_margin"]
        ball_x = self.data.ball_xy[:,0] # ball position along the length
        ball_y = self.data.ball_xy[:,1] # ball position along the width
        past_far_goal_line = ball_x > length + margin
        past_near_goal_line = ball_x < 0 - margin
        past_bottom_touchline = ball_y < 0 - margin
        past_top_touchline = ball_y > width + margin
        out = past_bottom_touchline | past_far_goal_line | past_near_goal_line | past_top_touchline
        out = out.astype(float)
        return out

    def players_still(self):
        """1.0 where the players' median speed stays below still_speed for still_seconds, else 0.0.

        Speeds come from the players' positions (velocity.player_speed with the `speed:` block),
        in m/s on both our match and the dev set. The ATD's own speed field is not used: its unit
        is not checked and it may be mostly empty. The median, not the mean, so a few players
        jogging back into position during a stoppage do not hide it. A frame where no player has
        a speed gives no median, so it is never counted as still.
        """
        min_frames = round(self.cfg["still_seconds"] * self.data.fps)   # seconds -> frames
        still = self._players_slow()
        still = keep_long_stretches(still, min_frames) # a short slow moment is not a stoppage
        still = still.astype(float)
        return still

    def _players_slow(self):
        """True on frames where the players' median speed is below still_speed (any length)."""
        threshold = self.cfg["still_speed"]
        speeds = velocity.player_speed(self.data.players_xy, self.data.period, self.data.fps,
                                       self.cfg["speed"])   # one row per frame, one column per slot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # numpy warns on frames with no speed
            typical = np.nanmedian(speeds, axis=1)     # one median per frame, empty slots skipped
        return typical < threshold                     # no median (NaN) gives False: not still

    def gap_chance(self):
        """Chance the ball is dead inside each ball gap, one value per gap; 0.0 where it is seen.

        A gap is a run of frames with no ball, cut where the half changes. Its chance is a
        logistic score (log-odds added up, then squashed to 0-1), from the `gap:` block:
            intercept + length_weight * log2(gap length in seconds)
            + each clue's weight where the clue holds:
              last_seen_near_line   the last ball before the gap was within line_distance of a
                                    touchline or goal line (or past it)
              reappears_near_line   the first ball after the gap is near a line: a throw-in,
                                    corner or goal kick being taken
              reappears_at_centre   the first ball after the gap is within spot_distance of the
                                    centre spot: a kickoff after a goal
              players_still         weight x the share of the gap's frames where the players'
                                    median speed is below still_speed
              restart_tag_after     a Smart Tagging restart starts inside the gap or within
                                    tag_seconds after it (0 on the dev set: it has no tags)
        A gap at the start or end of a half has no ball before or after it: those clues are 0.
        """
        gap = self.cfg["gap"]
        weights = gap["weights"]
        fps, n = self.data.fps, self.data.n_frames
        xy, period = self.data.ball_xy, self.data.period
        chance = np.zeros(n)
        runs = gap_runs(np.isnan(xy[:, 0]), period)
        if not runs:
            return chance
        slow = self._players_slow() if weights["players_still"] else np.zeros(n, dtype=bool)
        tag_frames = round(gap["tag_seconds"] * fps)
        restarts = [tag.start for tag in self.data.tags if (tag.type or "").lower() in RESTART_TYPES]

        for first, stop in runs:
            before = xy[first - 1] if first > 0 and period[first - 1] == period[first] else None
            after = xy[stop] if stop < n and period[stop] == period[first] else None
            clues = {
                "last_seen_near_line": _near_line(before, self.data.pitch, gap["line_distance"]),
                "reappears_near_line": _near_line(after, self.data.pitch, gap["line_distance"]),
                "reappears_at_centre": _near_centre(after, self.data.pitch, gap["spot_distance"]),
                "players_still": float(slow[first:stop].mean()),
                "restart_tag_after": any(first <= s <= stop + tag_frames for s in restarts),
            }
            log_odds = gap["intercept"] + gap["length_weight"] * np.log2((stop - first) / fps)
            log_odds += sum(weights[clue] * float(value) for clue, value in clues.items())
            chance[first:stop] = 1.0 / (1.0 + np.exp(-log_odds))
        return chance

    def before_kickoff(self):
        """1.0 on graded second-half frames before the second-half kickoff, else 0.0."""
        kickoffs = select_tags(self.data.tags, code="set pieces", type="kick off (start)")
        second_half = [tag for tag in kickoffs if tag.half == 2]
        if not second_half:                            # no kickoff tag: no evidence either way
            return np.zeros(self.data.n_frames)
        kickoff = second_half[0].start                 # the frame the second half kicks off
        frames = np.arange(self.data.n_frames)         # 0, 1, 2, ... one number per frame
        before = (self.data.period == 2) & (frames < kickoff)
        before = before.astype(float)
        return before

    def before_restart(self):
        """1.0 in the seconds before each restart tag (throw-in, corner, ...), else 0.0."""
        dead = np.zeros(self.data.n_frames)
        for restart, seconds in self.cfg["restart_seconds"].items():
            frames = round(seconds * self.data.fps)             # seconds -> frames
            for tag in select_tags(self.data.tags, type=restart):
                first = max(tag.start - frames, 0)              # not before the first frame
                dead[first:tag.start] = 1.0                     # dead until the restart is taken
        return dead


def check_config(section, fps=None):
    """Stop early, with a clear message, if the `deadball:` section has a mistake.

    fps: the match's frame rate. With it, the player speed block is checked in full (its window
    is set in seconds and checked in frames); without it, only that its keys are there.
    DeadBall always passes it.
    """
    method = section.get("method")
    if method not in METHODS:
        raise ValueError(f"unknown deadball method : {method}; available: {', '.join(METHODS)}")

    used = [method]                                    # the methods that will actually run
    if method == "combined":
        weights = section.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("deadball method combined needs 'weights': a clue name -> a number")
        for clue in weights:
            if clue == "combined" or clue not in METHODS:   # combined would call itself forever
                raise ValueError(f"deadball.weights: {clue!r} is not a clue; "
                                 f"available: {', '.join(m for m in METHODS if m != 'combined')}")
            _number(weights, clue, f"deadball.weights.{clue}")
        used = list(weights)

    for clue in used:
        for key in SETTINGS.get(clue, ()):
            _number(section, key, f"deadball.{key}", needed_by=clue)

    if "players_still" in used:
        _check_speed_block(section.get("speed"), fps)

    if "gap_chance" in used:
        _check_gap_block(section, fps)

    if "before_restart" in used:
        restarts = section.get("restart_seconds")
        if not isinstance(restarts, dict) or not restarts:
            raise ValueError("deadball method before_restart needs 'restart_seconds': "
                             "a restart type -> seconds")
        for restart in restarts:
            if restart.lower() not in RESTART_TYPES:   # a typo would silently match no tag
                raise ValueError(f"deadball.restart_seconds: {restart!r} is not a restart type; "
                                 f"available: {', '.join(RESTART_TYPES)}")
            _number(restarts, restart, f"deadball.restart_seconds.{restart}")


def _check_speed_block(block, fps):
    """The `speed:` block players_still hands to velocity.player_speed."""
    if not isinstance(block, dict) or not block:
        raise ValueError(f"deadball method players_still needs a 'speed' block with "
                         f"{', '.join(SPEED_KEYS)}")
    if fps is None:
        missing = [key for key in SPEED_KEYS if key not in block]
        if missing:
            raise ValueError(f"deadball.speed: missing {missing}")
        return
    try:
        velocity.player_speed_settings(block, fps)
    except (TypeError, ValueError) as error:
        raise ValueError(f"deadball.speed: {error}") from error


def _check_gap_block(section, fps):
    """The `gap:` block gap_chance reads, and the players' settings if it uses them."""
    gap = section.get("gap")
    if not isinstance(gap, dict) or not gap:
        raise ValueError(f"deadball method gap_chance needs a 'gap' block with intercept, "
                         f"{', '.join(GAP_SETTINGS)} and weights")
    intercept = gap.get("intercept")
    if isinstance(intercept, bool) or not isinstance(intercept, int | float) \
            or not np.isfinite(intercept):
        raise ValueError(f"deadball.gap.intercept must be a number, got {intercept!r}")
    for key in GAP_SETTINGS:
        _number(gap, key, f"deadball.gap.{key}", needed_by="gap_chance")
    weights = gap.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"deadball.gap.weights must give a number for each of "
                         f"{', '.join(GAP_CLUES)}")
    for clue in weights:
        if clue not in GAP_CLUES:                      # a typo would silently weigh nothing
            raise ValueError(f"deadball.gap.weights: {clue!r} is not a gap clue; "
                             f"available: {', '.join(GAP_CLUES)}")
    for clue in GAP_CLUES:                             # every clue named, 0 to switch one off
        _number(weights, clue, f"deadball.gap.weights.{clue}", needed_by="gap_chance")
    if weights["players_still"]:
        _number(section, "still_speed", "deadball.still_speed", needed_by="gap_chance")
        _check_speed_block(section.get("speed"), fps)


def gap_runs(missing, period):
    """(first, stop) of every run of missing ball, cut where the period changes. stop is exclusive."""
    change = (missing[1:] != missing[:-1]) | (period[1:] != period[:-1])
    edges = np.flatnonzero(change) + 1
    starts = np.concatenate(([0], edges)).astype(int)
    stops = np.concatenate((edges, [len(missing)])).astype(int)
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True) if missing[a]]


def _near_line(point, pitch, distance):
    """True if a ball position is within `distance` of a touchline or goal line, or past one."""
    if point is None or np.isnan(point).any():
        return False
    length, width = pitch
    x, y = point
    return min(x, length - x, y, width - y) <= distance


def _near_centre(point, pitch, distance):
    """True if a ball position is within `distance` of the centre spot."""
    if point is None or np.isnan(point).any():
        return False
    length, width = pitch
    return bool(np.hypot(point[0] - length / 2, point[1] - width / 2) <= distance)


def _number(section, key, name, needed_by=None):
    """A config value that must be a number >= 0."""
    if key not in section:
        raise ValueError(f"deadball method {needed_by} needs {key!r} in the config")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value) \
            or value < 0:
        raise ValueError(f"{name} must be a number >= 0, got {value!r}")
    return value


def keep_long_stretches(flags, min_frames):
    """Keep only the stretches of True that last at least min_frames frames in a row."""
    kept = np.zeros(len(flags), dtype=bool)
    start = None                                       # where the current stretch of True began
    for frame, flag in enumerate(flags):
        if flag and start is None:                     # a stretch begins
            start = frame
        elif not flag and start is not None:           # a stretch ends: keep it if long enough
            if frame - start >= min_frames:
                kept[start:frame] = True
            start = None
    if start is not None and len(flags) - start >= min_frames:   # a stretch still open at the end
        kept[start:] = True
    return kept


def detect(match, ball, cfg):
    """What pipeline.py calls (docs/architecture.md)."""
    return DeadBall(match, ball, cfg).score()
