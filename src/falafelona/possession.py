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
# Methods are selected via the `possession:` section of the config. So far:
#   nearest_majority  the official baseline (task 4.1)
#   switch_cost       Viterbi decoder with a switch cost instead of a vote (task 4.3)

import numpy as np

from falafelona.io import Match

# The four published baseline variants: (name, half-width of the vote window in frames).
VARIANTS = (
    ("baseline_raw", 0),
    ("baseline_smooth25", 25),
    ("baseline_smooth50", 50),
    ("baseline_smooth125", 125),
)
LABELS = ("H", "A", "D")


class Possession:
    """Turns the cleaned data into H / A / D per frame (PR #9 structure). Task 4.1, Dudu."""

    HOME, AWAY, NOBODY = 0, 1, -1

    def __init__(self, data: Match):
        self.data = data

    def nearest_player_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame team of the player closest to the ball: HOME / AWAY / NOBODY.

        Also returns the distance in metres to that player (NaN where NOBODY).
        """
        players = self.data.players_xy
        ball = self.data.ball_xy
        slot = np.asarray(self.data.slot_team)
        team_of_track = np.select([slot == "H", slot == "A"], [self.HOME, self.AWAY], self.NOBODY)

        diff = players - ball[:, None, :]  # (frames, tracks, 2)
        dist2 = np.sum(diff * diff, axis=2)  # squared distance, NaN if missing

        dist2 = np.where(np.isnan(dist2), np.inf, dist2)  # missing -> "infinitely far"
        nearest = np.argmin(dist2, axis=1)  # index of closest track
        best = dist2[np.arange(len(dist2)), nearest]

        found = np.isfinite(best)  # inf means nobody / no ball
        codes = np.where(found, team_of_track[nearest], self.NOBODY).astype(np.int8)
        distance = np.where(found, np.sqrt(best), np.nan).astype(np.float32)
        return codes, distance

    def smooth(self, codes: np.ndarray, window: int) -> np.ndarray:
        """Majority vote over a centred window of +/- `window` frames. D where nobody has the ball.

        Computed with cumulative sums, so the cost does not grow with the window -- a 125-frame
        window over 199319 frames is the same work as a 1-frame one.

        Ties go to the home side. With an odd-sized window over two classes a tie needs the two
        counts to be exactly equal, which is rare and never worth a coin flip.
        """
        if window <= 0:
            return np.where(codes == self.HOME, "H", np.where(codes == self.AWAY, "A", "D"))

        home_cumulative = np.concatenate(([0], np.cumsum(codes == self.HOME)))
        away_cumulative = np.concatenate(([0], np.cumsum(codes == self.AWAY)))

        index = np.arange(len(codes))
        low = np.maximum(0, index - window)
        high = np.minimum(len(codes), index + window + 1)
        home = home_cumulative[high] - home_cumulative[low]
        away = away_cumulative[high] - away_cumulative[low]

        labels = np.full(len(codes), "D", dtype="<U1")
        seen = (home + away) > 0
        labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
        return labels

    def nearest_majority(self, window: int = 50) -> np.ndarray:
        """Official baseline: nearest player, then a +/- `window` majority vote."""
        codes, _ = self.nearest_player_labels()
        return self.smooth(codes, window)


# ---------------------------------------------------------------------------------------------
# Method `switch_cost` (task 4.3): a decoder instead of a vote
#
# The team label is the cheapest path through the whole match (Viterbi), where:
#   - each frame costs 1 for every piece of evidence against the label (here: the nearest
#     player belongs to the other team), 0 where there is no evidence, and
#   - changing team costs `switch_cost`.
# So a switch only happens when at least `switch_cost` frames of evidence (net) point to the
# other team. A short blip of the other team's evidence cannot flip the label, which is exactly
# the flicker spellF1 punishes. Ball dead (dead_score >= dead_threshold) gives no evidence.
#
# D is placed exactly like the baseline: a frame is D when no frame within +/- dead_margin has
# team evidence (for dead_margin = window this is the D of nearest_majority). That keeps this
# method's only change the team decoding, so its effect can be measured on its own. A joint
# H/A/D decoder that treats D as a state is task 4.6, and can reuse viterbi() as it is.
# ---------------------------------------------------------------------------------------------


def viterbi(costs: np.ndarray, switch_cost: float) -> np.ndarray:
    """The cheapest state sequence: int[n] of state indices, given costs float[n, k].

    Total cost = sum of costs[t, state_t] + switch_cost for every t where the state changes.
    Deterministic: on ties, staying beats switching and a lower state index beats a higher one.
    """
    costs = np.asarray(costs, dtype=np.float64)
    n, k = costs.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    transition = np.full((k, k), float(switch_cost))
    np.fill_diagonal(transition, 0.0)
    # Visit "stay" first so argmin's first-index rule prefers staying on a tie.
    order = np.array([[s, *[p for p in range(k) if p != s]] for s in range(k)])

    back = np.zeros((n, k), dtype=np.int8)
    total = costs[0].copy()
    for t in range(1, n):
        candidates = total[order] + transition[order, np.arange(k)[:, None]]  # [state, prev]
        pick = np.argmin(candidates, axis=1)
        back[t] = order[np.arange(k), pick]
        total = candidates[np.arange(k), pick] + costs[t]

    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def no_evidence_within(codes: np.ndarray, margin: int) -> np.ndarray:
    """bool[n]: no frame within +/- margin has a team (HOME or AWAY). The baseline's D."""
    has_team = codes != Possession.NOBODY
    cumulative = np.concatenate(([0], np.cumsum(has_team)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - margin)
    high = np.minimum(len(codes), index + margin + 1)
    return (cumulative[high] - cumulative[low]) == 0


def decode_switch_cost(codes: np.ndarray, switch_cost: float, dead_margin: int) -> np.ndarray:
    """H / A / D per frame from team evidence codes (HOME / AWAY / NOBODY)."""
    costs = np.zeros((len(codes), 2))
    costs[codes == Possession.AWAY, 0] = 1.0   # evidence against H
    costs[codes == Possession.HOME, 1] = 1.0   # evidence against A
    team = np.array(["H", "A"])[viterbi(costs, switch_cost)]
    return np.where(no_evidence_within(codes, dead_margin), "D", team).astype("<U1")


def _number(section, key, *, whole=False):
    """A config value that must be a number >= 0 (and a whole number if `whole`)."""
    if key not in section:
        raise ValueError(f"possession method {section['method']!r} needs {key!r} in the config")
    value = section[key]
    ok = isinstance(value, int) if whole else isinstance(value, int | float)
    if isinstance(value, bool) or not ok or not np.isfinite(value) or value < 0:
        kind = "a whole number" if whole else "a number"
        raise ValueError(f"possession.{key} must be {kind} >= 0, got {value!r}")
    return value


def _team_codes(match, dead_score, section):
    """Nearest-player team per frame, NOBODY where the dead score is >= dead_threshold."""
    codes, _ = Possession(match).nearest_player_labels()
    dead = np.asarray(dead_score) >= section["dead_threshold"]
    return np.where(dead, Possession.NOBODY, codes).astype(np.int8)


def _nearest_majority(match, dead_score, section):
    codes = _team_codes(match, dead_score, section)
    return Possession(match).smooth(codes, section["window"])


def _switch_cost(match, dead_score, section):
    switch_cost = _number(section, "switch_cost")
    dead_margin = _number(section, "dead_margin", whole=True)
    _number(section, "dead_threshold")
    return decode_switch_cost(_team_codes(match, dead_score, section), switch_cost, dead_margin)


# method name in configs/*.yaml -> function(match, dead_score, cfg["possession"]) -> labels
METHODS = {
    "nearest_majority": _nearest_majority,
    "switch_cost": _switch_cost,
}


def predict(match: Match, ball, dead_score: np.ndarray, cfg: dict) -> np.ndarray:
    """What pipeline.py calls (docs/architecture.md): H / A / D per frame.

    `match` is the cleaned Match from ball.py, so its ball_xy already holds the cleaned track.
    `ball` (the BallTrack) is accepted for the interface but not used yet.
    The method and its numbers come from the `possession:` config section:
      nearest_majority  dead_threshold, window (the official baseline)
      switch_cost       dead_threshold, switch_cost, dead_margin (task 4.3, see above)
    """
    section = cfg["possession"]
    method = section["method"]
    if method not in METHODS:
        available = ", ".join(sorted(METHODS))
        raise ValueError(f"unknown possession method {method!r}; available: {available}")
    return METHODS[method](match, dead_score, section)
