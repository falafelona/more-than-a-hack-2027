# ws:poss: which team has the ball after a touch, learned from other matches.
#
# WHY: the dev-set touch oracle (devset.ORACLES "touch_team") gives the touches we ALREADY detect
# their true team and gains +0.094 spellF1 (7/7 matches). Our rule for a touch's team is simple:
# the nearest player within possession.touch.distance, if the other team is margin further. It
# ignores what happens next: who stays with the ball, who it goes to, which way it travels.
#
# WHAT: for every detected touch (possession.ball_touches with the possession.touch settings) a
# ridge logistic model (learned_dead.fit_logistic, no new dependency) gives p = chance that HOME
# has the ball just after it. Features: the distance margin before, at and after the touch and
# 1 s later (the receiver), players close to the ball, which nearest player moves WITH the ball
# afterwards, the ball's direction along each team's attack, the rule's team for this touch and
# its neighbours, ball speeds, and the time to the neighbouring touches. Home / away symmetric
# (SWAP, like learned_team), so it cannot learn a home lean.
#
# HOW IT IS USED: a `touch_learned:` block in the possession section, next to `touch:`:
#     possession:
#       touch: {...}                       # still gives the candidates and hold_s
#       touch_learned: {model: models/touch_all.json, cost: 1.0}
# then the touch evidence is: for hold_s frames from each touch, cost x |log-odds| on the team
# the model thinks does NOT have the ball (instead of the rule's flat cost on the other team).
# scripts/fit_touch.py writes the files (humans run it: it reads IDSSE). On the dev set the path
# may hold {match_id} (leave-one-match-out).
#
# Tests use made-up numbers only (tests/test_learned_touch.py).

import copy
import json
from pathlib import Path

import numpy as np

from falafelona import learned_dead as ld

FORMAT = "falafelona-touch-logreg"
VERSION = 1                  # bump when a feature changes, so an old model file is refused

LABEL_AFTER_S = 0.3          # the truth this long after a touch is the team that has it
NEAR_S = 0.2                 # margins just before / after the touch
AFTER2_S = 0.4
RECEIVER_S = 1.0             # where the ball is 1 s later: who received it
VEL_HALF_S = 0.12            # players' velocity: positions this far either side
NEIGHBOUR_S = 3.0            # a neighbouring touch further away than this counts as none
MARGIN_CAP_M = 15.0
FOLLOW_CAP = 15.0            # m/s
SPEED_CAP = 30.0
CLOSE_M = (2.0, 5.0)

FEATURES = (
    "margin", "margin_before", "margin_after", "margin_after2", "margin_receiver",
    "close2_diff", "close5_diff", "follow_diff", "dir_after", "dir_before",
    "rule_team", "prev_rule", "next_rule",
    "log_speed_before", "log_speed_after", "log_change", "log_near", "log_to_next",
    "log_since_prev", "seen_after",
)
# -1 = flips sign when home and away swap (says which team), +1 = stays.
SWAP = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1)
CONFIG_KEYS = ("model", "cost")


# --- config ------------------------------------------------------------------------------------


def check_settings(section):
    block = section.get("touch_learned")
    block = block if isinstance(block, dict) else {}
    unknown = [key for key in block if key not in CONFIG_KEYS]
    missing = [key for key in CONFIG_KEYS if key not in block]
    if unknown or missing:
        raise ValueError(f"possession.touch_learned: unknown {unknown}, missing {missing}; "
                         f"needs {', '.join(CONFIG_KEYS)}")
    if not isinstance(block["model"], str) or not block["model"]:
        raise ValueError(f"possession.touch_learned.model must be a file path, "
                         f"got {block['model']!r}")
    cost = block["cost"]
    if isinstance(cost, bool) or not isinstance(cost, int | float) or not np.isfinite(cost) \
            or cost < 0:
        raise ValueError(f"possession.touch_learned.cost must be a number >= 0, got {cost!r}")
    if "touch" not in section:
        raise ValueError("possession.touch_learned needs the possession.touch block "
                         "(it gives the touches and hold_s)")


def for_match(cfg, match_id):
    """cfg with {match_id} in the touch model path filled in (leave-one-match-out)."""
    block = (cfg.get("possession") or {}).get("touch_learned")
    if not isinstance(block, dict) or "{match_id}" not in str(block.get("model", "")):
        return cfg
    filled = copy.deepcopy(cfg)
    filled["possession"]["touch_learned"]["model"] = block["model"].replace("{match_id}",
                                                                           match_id)
    return filled


def model_path(section, repo_root=None):
    raw = section["touch_learned"]["model"]
    if "{match_id}" in raw:
        raise ValueError(f"possession.touch_learned.model {raw!r} holds {{match_id}}: only the "
                         "dev set fills that in. Give a real file for a run.")
    path = Path(raw)
    if path.is_absolute():
        return path
    if repo_root is None:
        from falafelona import vendor
        repo_root = vendor.REPO_ROOT
    return Path(repo_root) / path


# --- the features ------------------------------------------------------------------------------


def _margin_at(players_xy, slot_team, ball_xy, frames):
    """float[k] capped d_away - d_home at those frames (ball NaN -> 0), and d_near."""
    from falafelona import possession

    xy = ball_xy[frames]
    d_home, d_away = possession.team_distances(players_xy[frames], xy, slot_team)
    with np.errstate(invalid="ignore"):
        margin = np.clip(d_away - d_home, -MARGIN_CAP_M, MARGIN_CAP_M)
    margin = np.where(np.isfinite(d_home) & ~np.isfinite(d_away), MARGIN_CAP_M, margin)
    margin = np.where(~np.isfinite(d_home) & np.isfinite(d_away), -MARGIN_CAP_M, margin)
    margin = np.where(np.isfinite(d_home) | np.isfinite(d_away), margin, 0.0)
    return np.nan_to_num(margin), np.minimum(d_home, d_away)


def _close_counts(players_xy, slot_team, ball_xy, frames, radius):
    slot = np.asarray(slot_team)
    diff = players_xy[frames] - ball_xy[frames][:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    with np.errstate(invalid="ignore"):
        near = dist < radius
    return (near[:, slot == "H"].sum(axis=1) - near[:, slot == "A"].sum(axis=1)).astype(float)


def _player_velocity(players_xy, frames, k, fps):
    """float[k_frames, slots, 2] finite-difference velocity at those frames (NaN at the edges)."""
    n = len(players_xy)
    lo, hi = np.maximum(frames - k, 0), np.minimum(frames + k, n - 1)
    span = (hi - lo) / fps
    with np.errstate(invalid="ignore", divide="ignore"):
        return (players_xy[hi] - players_xy[lo]) / span[:, None, None]


def _follow(players_xy, slot_team, ball_xy, v_after, frames, fps):
    """float[k]: |v nearest away - v ball| - |v nearest home - v ball| (capped): positive =
    the nearest home player moves with the ball afterwards better than the nearest away one."""
    slot = np.asarray(slot_team)
    k = max(1, round(VEL_HALF_S * fps))
    after = np.minimum(frames + k, len(players_xy) - 1)
    v_players = _player_velocity(players_xy, after, k, fps)
    diff = players_xy[frames] - ball_xy[frames][:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    out = np.zeros(len(frames))
    gaps = []
    for team in ("H", "A"):
        cols = np.flatnonzero(slot == team)
        if not cols.size:
            gaps.append(np.full(len(frames), FOLLOW_CAP))
            continue
        nearest = cols[np.argmin(dist[:, cols], axis=1)]
        v = v_players[np.arange(len(frames)), nearest]
        g = np.hypot(*(v - v_after).T)
        gaps.append(np.where(np.isfinite(g), np.minimum(g, FOLLOW_CAP), FOLLOW_CAP))
    out = gaps[1] - gaps[0]
    return out


def touch_features(match, section, touches=None):
    """(touches int[k], x float[k, len(FEATURES)]) for the detected touches of a cleaned Match.

    section: the possession section (its `touch:` and `velocity:` blocks give the touches).
    """
    from falafelona import learned_team, possession, velocity

    if touches is None:
        events = possession.touch_events(match, section)
        if events is None:
            raise ValueError("learned touches need the possession.touch block")
        touches, rule = events
    else:
        touches = np.asarray(touches, dtype=np.int64)
        d_home, d_away = possession.team_distances(match.players_xy, match.ball_xy,
                                                   match.slot_team)
        block = section["touch"]
        rule = possession.touch_teams(touches, d_home, d_away, float(block["distance"]),
                                      float(block["margin"]))
    fps = float(match.fps)
    n = len(match.period)
    period = np.asarray(match.period)
    ball = np.asarray(match.ball_xy, dtype=float)
    players = np.asarray(match.players_xy, dtype=float)
    slots = np.asarray(match.slot_team)
    k = len(touches)
    x = np.zeros((k, len(FEATURES)))
    if k == 0:
        return touches, x

    offset = max(1, int(np.floor(float(section["touch"]["offset_s"]) * fps + 0.5)))
    vxy, speed = velocity.ball_velocity(ball, period, fps, section["velocity"])

    def at(seconds):
        idx = np.clip(touches + round(seconds * fps), 0, n - 1)
        same = period[idx] == period[touches]
        return np.where(same, idx, touches)

    def margin(frames):
        m, _ = _margin_at(players, slots, ball, frames)
        seen = np.isfinite(ball[frames, 0])
        return np.where(seen, m, 0.0)

    m0, near = _margin_at(players, slots, ball, touches)
    x[:, 0] = m0
    x[:, 1] = margin(at(-NEAR_S))
    x[:, 2] = margin(at(NEAR_S))
    x[:, 3] = margin(at(AFTER2_S))
    x[:, 4] = margin(at(RECEIVER_S))
    x[:, 5] = _close_counts(players, slots, ball, touches, CLOSE_M[0])
    x[:, 6] = _close_counts(players, slots, ball, touches, CLOSE_M[1])
    before = np.clip(touches - offset, 0, n - 1)
    after = np.clip(touches + offset, 0, n - 1)
    v_after = np.where(np.isfinite(vxy[after]), vxy[after], 0.0)
    v_before = np.where(np.isfinite(vxy[before]), vxy[before], 0.0)
    x[:, 7] = _follow(players, slots, ball, v_after, touches, fps)
    direction, _ = learned_team._home_direction(players, slots, period)
    d = direction[touches]
    s_after = np.maximum(np.hypot(*v_after.T), 1.0)
    s_before = np.maximum(np.hypot(*v_before.T), 1.0)
    x[:, 8] = d * v_after[:, 0] / s_after
    x[:, 9] = d * v_before[:, 0] / s_before

    sign = np.where(rule == possession.H_STATE, 1.0, np.where(rule == possession.A_STATE, -1.0,
                                                               0.0))
    x[:, 10] = sign
    gap_prev = np.diff(touches, prepend=-10 ** 9) / fps
    gap_next = np.diff(touches, append=10 ** 9) / fps
    same_prev = np.concatenate(([False], period[touches[1:]] == period[touches[:-1]]))
    same_next = np.concatenate((period[touches[:-1]] == period[touches[1:]], [False]))
    prev = np.concatenate(([0.0], sign[:-1]))
    nxt = np.concatenate((sign[1:], [0.0]))
    x[:, 11] = np.where(same_prev & (gap_prev <= NEIGHBOUR_S), prev, 0.0)
    x[:, 12] = np.where(same_next & (gap_next <= NEIGHBOUR_S), nxt, 0.0)

    def log_speed(frames):
        s = speed[frames]
        return np.where(np.isfinite(s), np.log1p(np.minimum(np.nan_to_num(s), SPEED_CAP)), 0.0)

    x[:, 13] = log_speed(before)
    x[:, 14] = log_speed(after)
    x[:, 15] = np.log1p(np.minimum(np.hypot(*(v_after - v_before).T), SPEED_CAP))
    x[:, 16] = np.log1p(np.minimum(np.where(np.isfinite(near), near, MARGIN_CAP_M),
                                   MARGIN_CAP_M))
    x[:, 17] = np.log1p(np.where(same_next, np.minimum(gap_next, NEIGHBOUR_S), NEIGHBOUR_S))
    x[:, 18] = np.log1p(np.where(same_prev, np.minimum(gap_prev, NEIGHBOUR_S), NEIGHBOUR_S))
    x[:, 19] = np.isfinite(ball[at(AFTER2_S), 0])
    return touches, np.nan_to_num(x)


# --- fitting -----------------------------------------------------------------------------------


def labels_after(truth, touches, fps):
    """str[k]: the truth LABEL_AFTER_S after each touch."""
    truth = np.asarray(truth)
    idx = np.minimum(np.asarray(touches) + round(LABEL_AFTER_S * fps), len(truth) - 1)
    return truth[idx]


def swap(x):
    x = np.asarray(x, dtype=float)
    return x * np.asarray(SWAP[:x.shape[1]], dtype=float)


def training_rows(match, truth, section):
    """(x, y, rule) from one cleaned match: touches whose truth just after is H or A, each twice
    (as is and swapped). y = 1 for H. rule = the rule's sign for the as-is rows (+1 / -1 / 0)."""
    touches, x = touch_features(match, section)
    label = labels_after(truth, touches, float(match.fps))
    keep = np.isin(label, ["H", "A"]) & (np.asarray(match.period)[touches] > 0)
    x, y = x[keep], (label[keep] == "H").astype(float)
    rule = x[:, FEATURES.index("rule_team")].copy()
    return (np.vstack([x, swap(x)]), np.concatenate([y, 1.0 - y]),
            np.concatenate([rule, -rule]))


def symmetric_chance(fit, x):
    return 0.5 * (ld.predict_proba(fit, x) + 1.0 - ld.predict_proba(fit, swap(x)))


def _fit(x, y, l2):
    if len(y) and y.min() != y.max():
        return ld.fit_logistic(x, y, l2)
    k = x.shape[1]
    return {"mean": [0.0] * k, "scale": [1.0] * k, "coef": [0.0] * k, "intercept": 0.0}


def fit_folds(rows_by_match, l2=10.0):
    """Leave-one-match-out fits plus "all". Held out: (log loss, accuracy, rule accuracy where
    the rule names a team, rule coverage, rows). Aggregates only."""
    ids = sorted(rows_by_match)
    if len(ids) < 2:
        raise ValueError("fit_folds needs at least two matches (one to hold out)")
    folds = {}
    for held in [*ids, None]:
        names = [i for i in ids if i != held]
        x = np.vstack([rows_by_match[i][0] for i in names])
        y = np.concatenate([rows_by_match[i][1] for i in names])
        fit = _fit(x, y, l2)
        fold = {"fit": fit, "trained_on": names, "rows": len(y)}
        if held is not None:
            hx, hy, rule = rows_by_match[held]
            chance = symmetric_chance(fit, hx) if len(hy) else np.zeros(0)
            named = rule != 0
            rule_acc = float(np.mean((rule[named] > 0) == (hy[named] >= 0.5))) if named.any() \
                else 0.0
            fold["holdout"] = (ld.log_loss(chance, hy) if len(hy) else 0.0,
                               float(np.mean((chance >= 0.5) == (hy >= 0.5))) if len(hy) else 0.0,
                               rule_acc, float(np.mean(named)) if len(hy) else 0.0, len(hy))
        folds["all" if held is None else held] = fold
    return folds


# --- the model file ----------------------------------------------------------------------------


def save_model(path, fold, l2):
    fit = fold["fit"]
    doc = {"format": FORMAT, "version": VERSION, "trained_on": sorted(fold["trained_on"]),
           "l2": l2, "rows": fold["rows"], "features": list(FEATURES),
           "mean": ld._round(fit["mean"]), "scale": ld._round(fit["scale"]),
           "coef": ld._round(fit["coef"]), "intercept": ld._round(fit["intercept"])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"touch model not found: {path} (make it with "
                                "scripts/fit_touch.py)")
    doc = json.loads(path.read_text())
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise ValueError(f"{path}: not a version {VERSION} {FORMAT} file")
    if doc.get("features") != list(FEATURES):
        raise ValueError(f"{path}: made for other features, refit it")
    for key in ("mean", "scale", "coef"):
        values = doc.get(key)
        if not isinstance(values, list) or len(values) != len(FEATURES) \
                or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {key} must be {len(FEATURES)} finite numbers")
    if np.any(np.asarray(doc["scale"]) <= 0) or not np.isfinite(doc.get("intercept", np.nan)):
        raise ValueError(f"{path}: bad scale or intercept")
    return doc


# --- the evidence ------------------------------------------------------------------------------


def touch_costs(match, section, repo_root=None, touches=None):
    """float[n, 2] H / A costs from the learned touches: for hold_s frames from each touch,
    cost x |log-odds| on the team the model thinks does NOT have the ball."""
    from falafelona import possession

    check_settings(section)
    model = load_model(model_path(section, repo_root))
    touches, x = touch_features(match, section, touches)
    n = len(match.period)
    costs = np.zeros((n, 2))
    if not len(touches):
        return costs
    p = np.clip(symmetric_chance(model, x), 1e-4, 1 - 1e-4)
    logit = np.log(p) - np.log1p(-p)
    hold = max(1, int(np.floor(float(section["touch"]["hold_s"]) * match.fps + 0.5)))
    cost = float(section["touch_learned"]["cost"])
    for frame, z in zip(touches, logit, strict=True):    # a later touch replaces an earlier one
        row = np.zeros(2)
        row[possession.A_STATE if z > 0 else possession.H_STATE] = cost * abs(z)
        costs[frame:frame + hold] = row
    return costs
