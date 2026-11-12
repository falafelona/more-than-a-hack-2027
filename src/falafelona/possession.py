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


def viterbi(costs: np.ndarray, switch_cost) -> np.ndarray:
    """The cheapest state sequence: int[n] of state indices, given costs float[n, k].

    switch_cost: one number (every change costs the same) or a [k, k] table where
    switch_cost[a, b] is the cost of going from state a to state b (diagonal = 0).
    Total cost = sum of costs[t, state_t] + the switch cost of every change.
    Deterministic: on ties, staying beats switching and a lower state index beats a higher one.
    """
    costs = np.asarray(costs, dtype=np.float64)
    n, k = costs.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if np.ndim(switch_cost) == 0:
        transition = np.full((k, k), float(switch_cost))
        np.fill_diagonal(transition, 0.0)
    else:
        transition = np.asarray(switch_cost, dtype=np.float64)
        if transition.shape != (k, k):
            raise ValueError(f"switch_cost table must be {k}x{k}, got {transition.shape}")

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


STATES = ("H", "A", "D")
H_STATE, A_STATE, D_STATE = 0, 1, 2


def nearest_team_costs(codes: np.ndarray, team_cost: float) -> np.ndarray:
    """float[n, 2]: cost of H and A from the nearest player's team (the old 0/1 evidence)."""
    costs = np.zeros((len(codes), 2))
    costs[codes == Possession.AWAY, H_STATE] = team_cost     # away nearest: evidence against H
    costs[codes == Possession.HOME, A_STATE] = team_cost     # home nearest: evidence against A
    return costs


def team_distances(players_xy: np.ndarray, ball_xy: np.ndarray,
                   slot_team: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from the ball to the nearest home and the nearest away player per frame.

    inf where that team has no tracked player or the ball is missing.
    """
    diff = players_xy - ball_xy[:, None, :]                   # (frames, slots, 2)
    dist = np.sqrt(np.sum(diff * diff, axis=2))               # NaN if player or ball missing
    dist = np.where(np.isnan(dist), np.inf, dist)
    slot = np.asarray(slot_team)
    return dist[:, slot == "H"].min(axis=1), dist[:, slot == "A"].min(axis=1)


def margin_team_costs(d_home: np.ndarray, d_away: np.ndarray, team_cost: float,
                      margin_scale: float, near_distance: float) -> np.ndarray:
    """float[n, 2]: cost of H and A from how much closer one team is to the ball.

    p_home = sigmoid((d_away - d_home) / margin_scale): 0.5 when both are equally close.
    The evidence fades to 0 as the nearest player gets near_distance away (a loose ball or a
    ball in the air), and is 0 when the ball is missing, so the current owner holds.
    """
    seen = np.isfinite(d_home) | np.isfinite(d_away)
    far = 1e3                                                 # a team with nobody tracked
    home = np.where(np.isfinite(d_home), d_home, far)
    away = np.where(np.isfinite(d_away), d_away, far)

    margin = np.clip(away - home, -30.0, 30.0)
    p_home = np.clip(1.0 / (1.0 + np.exp(-margin / margin_scale)), 1e-6, 1 - 1e-6)
    closeness = np.clip(1.0 - np.minimum(home, away) / near_distance, 0.0, 1.0)
    weight = np.where(seen, team_cost * closeness, 0.0)

    costs = np.zeros((len(d_home), 2))
    costs[:, H_STATE] = weight * -np.log(p_home)
    costs[:, A_STATE] = weight * -np.log(1.0 - p_home)
    return costs


def add_dead_costs(team_costs: np.ndarray, dead_score: np.ndarray,
                   dead_weight: float) -> np.ndarray:
    """float[n, 3]: the H and A costs plus the D column from dead_score p in 0..1.

    D costs dead_weight * (1 - p); H and A each get dead_weight * p on top.
    """
    p = np.clip(np.nan_to_num(np.asarray(dead_score, dtype=np.float64)), 0.0, 1.0)
    costs = np.zeros((len(team_costs), 3))
    costs[:, :2] = team_costs + dead_weight * p[:, None]
    costs[:, D_STATE] = dead_weight * (1.0 - p)
    return costs


def hmm_costs(codes: np.ndarray, dead_score: np.ndarray, team_cost: float,
              dead_weight: float) -> np.ndarray:
    """float[n, 3] with the nearest-player evidence (team_evidence: nearest)."""
    return add_dead_costs(nearest_team_costs(codes, team_cost), dead_score, dead_weight)


def switch_table(switch_cost: float, to_dead_cost: float, from_dead_cost: float) -> np.ndarray:
    """[3, 3] switch costs between H, A and D (rows = from, columns = to)."""
    return np.array([
        [0.0, switch_cost, to_dead_cost],          # from H
        [switch_cost, 0.0, to_dead_cost],          # from A
        [from_dead_cost, from_dead_cost, 0.0],     # from D
    ])


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


TEAM_EVIDENCE = ("nearest", "margin")


def _hmm(match, dead_score, section):
    """3-state HMM: H, A and D decoded together with Viterbi.

    team_evidence: "nearest" (0/1 nearest player, the default) or "margin" (distance margin).
    """
    evidence = section.get("team_evidence", "nearest")
    if evidence not in TEAM_EVIDENCE:
        raise ValueError(f"possession.team_evidence must be one of {TEAM_EVIDENCE}, "
                         f"got {evidence!r}")
    team_cost = _number(section, "team_cost")
    dead_weight = _number(section, "dead_weight")
    table = switch_table(_number(section, "switch_cost"),
                         _number(section, "to_dead_cost"),
                         _number(section, "from_dead_cost"))

    if evidence == "nearest":
        codes, _ = Possession(match).nearest_player_labels()
        team = nearest_team_costs(codes, team_cost)
    else:
        d_home, d_away = team_distances(match.players_xy, match.ball_xy, match.slot_team)
        team = margin_team_costs(d_home, d_away, team_cost,
                                 _number(section, "margin_scale"),
                                 _number(section, "near_distance"))

    costs = add_dead_costs(team, dead_score, dead_weight)
    return np.array(STATES)[viterbi(costs, table)]


def _nearest_majority(match, dead_score, section):
    """Official baseline: nearest player, then a centred majority vote of +/- window frames."""
    window = _number(section, "window", whole=True)
    _number(section, "dead_threshold")
    return Possession(match).smooth(_team_codes(match, dead_score, section), window)


def _switch_cost(match, dead_score, section):
    switch_cost = _number(section, "switch_cost")
    dead_margin = _number(section, "dead_margin", whole=True)
    _number(section, "dead_threshold")
    return decode_switch_cost(_team_codes(match, dead_score, section), switch_cost, dead_margin)


# method name in configs/*.yaml -> function(match, dead_score, cfg["possession"]) -> labels
METHODS = {
    "nearest_majority": _nearest_majority,
    "switch_cost": _switch_cost,
    "hmm": _hmm,
}


def predict(match: Match, dead_score: np.ndarray, cfg: dict) -> np.ndarray:
    """What pipeline.py calls (docs/architecture.md): H / A / D per frame.

    `match` is the cleaned Match from ball.py, so its ball_xy already holds the cleaned track.
    `dead_score` comes from deadball.detect. The `possession.method` key of the config picks
    the method from METHODS; each method reads its own settings from the same section.
    """
    section = cfg["possession"]
    method = section["method"]
    if method not in METHODS:
        available = ", ".join(sorted(METHODS))
        raise ValueError(f"unknown possession method {method!r}; available: {available}")
    return METHODS[method](match, dead_score, section)
