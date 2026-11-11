# ws:dead: score how likely each frame is dead ball.
#
# Output: score in 0..1 per frame, fed to the decoder in possession.py (docs/decisions.md D13).
#
# Ideas:
# - ball out of bounds (pitch is 105.3 x 68 m)
# - players static / slow for a while
# - Smart Tagging set pieces and restarts (evidence, not truth; throw-ins have team N/A)
# - official baseline: dead wherever the ball is not observed. Start here.
# - second half: graded before its kickoff, and those frames are dead. Detect the kickoff from
#   the Kick Off (Start) Smart Tag, never a hard-coded frame.
#
# Methods are selected via the `deadball:` section of the config.
import numpy as np

from falafelona.io import select_tags
from falafelona.vendor import games


class DeadBall:
    """Scores how likely each frame is dead ball (0 = alive, 1 = dead)."""

    def __init__(self, match, ball, cfg):
        self.data = match              # the Match from io.load_match()
        self.ball = ball               # cleaned ball from ball.py (not built yet, so None for now)
        self.cfg = cfg["deadball"]     # only our section of the config

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
            if method == "combined":                    # would call itself forever
                raise ValueError("combined cannot be one of its own weights")
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
        length, width = games.GAMES[games.GAME]["pitch"]
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
        """1.0 where the players' average speed stays below still_speed for still_seconds, else 0.0."""
        threshold = self.cfg["still_speed"]
        min_frames = round(self.cfg["still_seconds"] * self.data.fps)   # seconds -> frames
        speeds = self.data.players_speed               # one row per frame, one column per player slot
        average = np.nanmean(speeds, axis=1)           # one average per frame, empty slots skipped
        still = average < threshold
        still = keep_long_stretches(still, min_frames) # a short slow moment is not a stoppage
        still = still.astype(float)
        return still

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