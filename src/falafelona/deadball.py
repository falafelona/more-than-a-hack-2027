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


class DeadBall:
    """Scores how likely each frame is dead ball (0 = alive, 1 = dead)."""

    def __init__(self, match, ball, cfg):
        self.data = match              # the Match from io.load_match()
        self.ball = ball               # cleaned ball from ball.py (not built yet, so None for now)
        self.cfg = cfg["deadball"]     # only our section of the config

    def score(self):
        """Run the method named in the config."""
        method = self.cfg["method"]
        if method == "ball_missing":
            return self.ball_missing()
        raise ValueError(f"unknown deadball method: {method}")

    def ball_missing(self):
        """Official baseline: dead (1.0) wherever the ball is not seen, alive (0.0) otherwise."""
        missing = self.data.ball_xy[:,0]
        missing = np.isnan(missing)
        missing = missing.astype(float)
        return missing



def detect(match, ball, cfg):
    """What pipeline.py calls (docs/architecture.md)."""
    return DeadBall(match, ball, cfg).score()