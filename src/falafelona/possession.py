# ws:poss: turn per-frame evidence into H / A / D possessions.
#
# spellF1 punishes flicker, so turning frames into clean possessions matters more than the
# per-frame guess.
#
# Ideas:
# - nearest player's team (official baseline), team-control features
# - a rule for a "controlled" possession vs a loose ball / challenge
# - H/A/D decoder (HMM / Viterbi) with a switch cost: the core of the system (docs/decisions.md D13)
# - modest minimum durations; very short possessions may count in the ground truth
# - official baseline smoothing: centred majority vote over +/- 50 frames (spellF1 0.2433)
#
# Methods are selected via the `possession:` section of the config.

from falafelona.io import Match


class Possession:
    """Turns the cleaned data into H / A / D per frame (PR #9 structure). Task 4.1, Dudu."""

    def __init__(self, data: Match):
        self.data = data
