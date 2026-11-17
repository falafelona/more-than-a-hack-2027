# ws:poss: which team has the ball, learned from OTHER matches (idea #1, step 1).
#
# WHY: the H / A evidence is hand-built (distance margin, touches, restart and gap rules) and each
# piece was tuned on its own. Here one model learns how much each clue is worth, from the seven
# IDSSE matches (D15) with the ball cut like ours, and it also sees context the hand-built
# evidence ignores: the margin over the next and previous second, the players around the ball,
# the teams' shapes, and inside a ball gap who had the ball before it and who has it after it.
#
# WHAT: two ridge logistic models (learned_dead.fit_logistic, no new dependency), one for frames
# where the ball is seen and one for frames inside a ball gap. Each gives p = chance that HOME
# has the ball, given that the ball is alive. Fitted only on frames whose truth is H or A.
#
# HOW IT IS USED: possession `team_evidence: learned` with a `learned:` block:
#     possession:
#       method: hmm
#       team_evidence: learned
#       learned: {model: models/team_all.json}
# The HMM is unchanged: the model only replaces the per-frame H / A costs. The more likely team
# costs 0 and the other team_cost * |log-odds|, so a frame the model is unsure about (p near
# 0.5, e.g. deep in a gap) costs nothing for either team and does not push towards D.
# On the dev set the model path holds {match_id}; devset.py fills in the held-out match
# (for_match), so every dev match is scored by a model that never saw it. scripts/fit_team.py
# writes the files. Humans run it: it reads IDSSE.
#
# Tests use made-up numbers only (tests/test_learned_team.py).

import copy
import json
import warnings
from pathlib import Path

import numpy as np

from falafelona import learned_dead as ld

FORMAT = "falafelona-team-logreg"
VERSION = 6                  # bump when a feature changes, so an old model file is refused
#                              v2: trained home/away-symmetric (SWAP), no home lean
#                              v3: gap features interp_margin, push
#                              v4: gap features push_short, push_long, push_turn, field_pos
#                              v5: gap features width_diff, length_diff, line_diff, spread at the gap edges
#                              v6: 10 more gap features: changes since the gap started and until it
#                                  ends, and team clues x gap position / length (non-linear step A).
#                                  v5 files still load: they use the first 21 gap columns (READABLE)

# Constants of the features. They are part of the model: change one, bump VERSION.
MARGIN_CAP_M = 15.0          # |distance margin| above this counts as this
NEAR_CAP_M = 30.0
CLOSE_M = 5.0                # "around the ball": players within this distance
CONTEXT_S = (0.5, 2.0)       # centred windows of the margin, s on each side
SHAPE_WINDOW_S = 1.0
TOUCH_OFFSET_S = 0.12        # the same touch rule as possession.touch in final.yaml
TOUCH_MIN_CHANGE = 3.0
TOUCH_DISTANCE_M = 2.0
TOUCH_MARGIN_M = 0.1
TOUCH_DECAY_S = 1.0          # a touch's signal fades like exp(-seconds since / this)
TOUCH_LOOK_S = 2.0           # inside a gap: look this far past its end for the next touch
SPEED_CAP = 30.0
PUSH_WINDOW_S = 1.0          # players' movement toward home's attacking goal, smoothed +/- this
PUSH_SHORT_S = 0.25
PUSH_LONG_S = 3.0
VELOCITY = {"half_window_s": 0.12, "min_points": 4, "min_each_side": 1}

SEEN_FEATURES = (
    "margin", "margin_w0.5", "margin_w2", "log_near", "close_diff", "touch_recent",
    "log_ball_speed", "ball_speed_unknown", "spread_diff",
)
GAP_FEATURES_V5 = (
    "log2_gap_s", "frac", "last_margin", "next_margin", "last_margin_x_left",
    "next_margin_x_done", "touch_before", "touch_after_x_done", "spread_diff", "open_ended",
    "interp_margin", "push", "push_short", "push_long", "push_turn", "field_pos",
    "width_diff", "length_diff", "line_diff", "spread_before", "spread_after",
)
GAP_FEATURES = GAP_FEATURES_V5 + (
    "field_turn", "line_turn", "width_turn", "length_turn", "field_to_end", "push_to_end",
    "push_x_frac", "push_long_x_len", "field_x_frac", "line_x_frac",
)
PARTS = {"seen": SEEN_FEATURES, "gap": GAP_FEATURES}
READABLE = {5: {"seen": SEEN_FEATURES, "gap": GAP_FEATURES_V5}, 6: PARTS}
CONFIG_KEYS = ("model", "add_cost", "where")

# What each feature does when home and away swap roles: -1 = flips sign (it says "home" or
# "away"), +1 = stays (it says nothing about which team). Training sees every frame twice, as is
# and swapped (with the label flipped), and the chance is averaged with the swapped chance, so the
# model cannot learn "home has the ball more often" from the IDSSE matches: that share changes
# from match to match (0.43 to 0.62 on IDSSE) and says nothing about ours.
SWAP = {
    "seen": (-1, -1, -1, 1, -1, -1, 1, 1, -1),
    "gap": (1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
}


# --- config ------------------------------------------------------------------------------------


def check_settings(section):
    """Stop early, with a clear message, if the possession `learned:` block is wrong."""
    block = section.get("learned")
    if not isinstance(block, dict) or not block:
        raise ValueError("possession team_evidence learned needs a 'learned' block with a 'model'")
    unknown = [key for key in block if key not in CONFIG_KEYS]
    if unknown:
        raise ValueError(f"possession.learned: unknown setting(s) {unknown}; available: "
                         f"{', '.join(CONFIG_KEYS)}")
    model = block.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"possession.learned.model must be a file path, got {model!r}")


def for_match(cfg, match_id):
    """cfg with {match_id} in the possession model path filled in (leave-one-match-out)."""
    block = (cfg.get("possession") or {}).get("learned")
    if not isinstance(block, dict) or "{match_id}" not in str(block.get("model", "")):
        return cfg
    filled = copy.deepcopy(cfg)
    filled["possession"]["learned"]["model"] = block["model"].replace("{match_id}", match_id)
    return filled


def model_path(section, repo_root=None):
    """The model file: relative paths count from the repo root."""
    raw = section["learned"]["model"]
    if "{match_id}" in raw:
        raise ValueError(f"possession.learned.model {raw!r} holds {{match_id}}: only the dev set "
                         "(scripts/devset_eval.py) fills that in. Give a real file for a run.")
    path = Path(raw)
    if path.is_absolute():
        return path
    if repo_root is None:
        from falafelona import vendor
        repo_root = vendor.REPO_ROOT
    return Path(repo_root) / path


# --- the inputs --------------------------------------------------------------------------------


def _distances(players_xy, ball_xy, slot_team):
    """(d_home, d_away, close_home, close_away) per frame; distances inf where unknown."""
    diff = players_xy - ball_xy[:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist = np.where(np.isnan(dist), np.inf, dist)
    slot = np.asarray(slot_team)
    home, away = dist[:, slot == "H"], dist[:, slot == "A"]
    d_home = home.min(axis=1) if home.shape[1] else np.full(len(dist), np.inf)
    d_away = away.min(axis=1) if away.shape[1] else np.full(len(dist), np.inf)
    return d_home, d_away, (home < CLOSE_M).sum(axis=1), (away < CLOSE_M).sum(axis=1)


def _margin(d_home, d_away):
    """d_away - d_home, capped: positive = home closer. 0 where neither team is known."""
    cap = MARGIN_CAP_M
    home = np.where(np.isfinite(d_home), d_home, np.inf)
    away = np.where(np.isfinite(d_away), d_away, np.inf)
    with np.errstate(invalid="ignore"):
        margin = np.clip(away - home, -cap, cap)
    margin = np.where(np.isfinite(home) & ~np.isfinite(away), cap, margin)
    margin = np.where(~np.isfinite(home) & np.isfinite(away), -cap, margin)
    return np.where(np.isfinite(home) | np.isfinite(away), margin, 0.0)


def _spread(players_xy, slot_team, team):
    """Typical distance (m) of a team's players from their centre; NaN where fewer than 3."""
    xy = players_xy[:, np.asarray(slot_team) == team]
    seen = np.isfinite(xy[..., 0])
    count = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.nansum(np.where(seen[..., None], xy, 0.0), axis=1) / count[:, None]
        sq = np.where(seen, np.sum((xy - centre[:, None, :]) ** 2, axis=2), 0.0)
        spread = np.sqrt(sq.sum(axis=1) / count)
    return np.where(count >= 3, spread, np.nan)


def _touch_signal(n, period, fps, touches, teams):
    """float[n]: +1 for a home touch, -1 away, fading with time since it; 0 before any."""
    out = np.zeros(n)
    sign = np.zeros(n)
    when = np.full(n, -1)
    for frame, team in zip(np.asarray(touches), np.asarray(teams), strict=True):
        if team >= 0:
            sign[frame] = 1.0 if team == 0 else -1.0
            when[frame] = frame
    last_sign, last_frame = 0.0, -1
    for i in range(n):
        if i > 0 and period[i] != period[i - 1]:
            last_sign, last_frame = 0.0, -1               # a new half forgets the old touch
        if when[i] >= 0:
            last_sign, last_frame = sign[i], i
        if last_frame >= 0 and period[i] > 0:
            out[i] = last_sign * np.exp(-(i - last_frame) / (TOUCH_DECAY_S * fps))
    return out, sign


def _team_std(players_xy, slot_team, team, axis):
    """Spread (m) of a team's players along one axis (0 = x, length; 1 = y, width); NaN if < 3."""
    v = np.asarray(players_xy, dtype=float)[:, np.asarray(slot_team) == team, axis]
    count = np.isfinite(v).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(v, axis=1)
    return np.where(count >= 3, spread, np.nan)


def _line_height(players_xy, slot_team, team, direction, middle):
    """m: how far up the pitch a team's last outfield line stands, along its own attacking
    direction and from the half's middle: its second-deepest player (the deepest is the
    goalkeeper). A team with the ball pushes its back line up. NaN outside the graded halves."""
    x = np.asarray(players_xy, dtype=float)[:, np.asarray(slot_team) == team, 0]
    sign = direction if team == "H" else -direction
    up = (x - middle[:, None]) * sign[:, None]
    up = np.where(np.isfinite(up), up, np.inf)
    count = np.isfinite(up).sum(axis=1)
    if up.shape[1] < 2:
        return np.full(len(x), np.nan)
    second = np.sort(up, axis=1)[:, 1]
    return np.where((count >= 3) & (sign != 0), second, np.nan)


def _home_direction(players_xy, slot_team, period):
    """(direction, middle) float[n]: +1 / -1 = the x direction home attacks in that half, and
    the players' middle x in that half. Home attacks away from its goalkeeper: the home player
    whose mean x is furthest from the middle. 0 / 0 outside the graded halves."""
    xy = np.asarray(players_xy, dtype=float)
    period = np.asarray(period)
    home = np.asarray(slot_team) == "H"
    direction, middle = np.zeros(len(period)), np.zeros(len(period))
    for p in np.unique(period[period > 0]):
        rows = period == p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_x = np.nanmean(xy[rows][..., 0], axis=0)
        home_x = mean_x[home]
        if not np.isfinite(home_x).any():
            continue
        mid = np.nanmedian(mean_x)
        keeper = home_x[np.nanargmax(np.abs(home_x - mid))]
        direction[rows] = -np.sign(keeper - mid)
        middle[rows] = mid
    return direction, middle


def _push_parts(players_xy, slot_team, period, fps):
    """(speed, position) float[n], both along home's attacking direction: the players' mean
    x speed (m/s) and their mean x (m from the half's middle). 0 outside the graded halves.

    When a team has the ball both teams move toward the goal it attacks, and play is mostly in
    the other team's half.
    """
    xy = np.asarray(players_xy, dtype=float)
    period = np.asarray(period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre_x = np.nanmean(xy[..., 0], axis=1)
    vx = np.zeros(len(period))
    vx[1:] = (centre_x[1:] - centre_x[:-1]) * fps
    vx[1:][period[1:] != period[:-1]] = 0.0
    vx = np.where(np.isfinite(vx), vx, 0.0)
    direction, middle = _home_direction(xy, slot_team, period)
    pos = np.where(np.isfinite(centre_x), centre_x - middle, 0.0)
    return direction * vx, direction * pos


def build_inputs(data):
    """Plain arrays for frame_features, from a Match after ball cleaning."""
    from falafelona import deadball, possession, velocity

    fps = float(data.fps)
    d_home, d_away, close_home, close_away = _distances(data.players_xy, data.ball_xy,
                                                        data.slot_team)
    vxy, speed = velocity.ball_velocity(data.ball_xy, data.period, fps, VELOCITY)
    offset = max(1, int(np.floor(TOUCH_OFFSET_S * fps + 0.5)))
    touches = possession.ball_touches(vxy, offset, TOUCH_MIN_CHANGE)
    teams = possession.touch_teams(touches, d_home, d_away, TOUCH_DISTANCE_M, TOUCH_MARGIN_M)
    touch, touch_sign = _touch_signal(len(data.period), data.period, fps, touches, teams)
    missing = np.isnan(data.ball_xy[:, 0])
    push_v, field = _push_parts(data.players_xy, data.slot_team, data.period, fps)
    direction, middle = _home_direction(data.players_xy, data.slot_team, data.period)
    xy, slots = data.players_xy, data.slot_team
    return {
        "period": np.asarray(data.period),
        "fps": fps,
        "missing": missing,
        "margin": np.where(missing, np.nan, _margin(d_home, d_away)),
        "near": np.minimum(d_home, d_away),
        "close_diff": (close_home - close_away).astype(float),
        "touch": touch,
        "touch_sign": touch_sign,
        "ball_speed": speed,
        "spread_diff": _spread(data.players_xy, data.slot_team, "H")
        - _spread(data.players_xy, data.slot_team, "A"),
        "gap_runs": [(a, b) for a, b in deadball.gap_runs(missing, data.period)
                     if data.period[a] > 0],
        "players_xy": np.asarray(data.players_xy, dtype=float),
        "ball_xy": np.asarray(data.ball_xy, dtype=float),
        "slot_team": np.asarray(data.slot_team),
        "push": ld.centred_mean(push_v, data.period, round(PUSH_WINDOW_S * fps)),
        "push_short": ld.centred_mean(push_v, data.period, round(PUSH_SHORT_S * fps)),
        "push_long": ld.centred_mean(push_v, data.period, round(PUSH_LONG_S * fps)),
        "field_pos": ld.centred_mean(field, data.period, round(PUSH_WINDOW_S * fps)),
        "width_diff": _team_std(xy, slots, "H", 1) - _team_std(xy, slots, "A", 1),
        "length_diff": _team_std(xy, slots, "H", 0) - _team_std(xy, slots, "A", 0),
        "line_diff": _line_height(xy, slots, "H", direction, middle)
        - _line_height(xy, slots, "A", direction, middle),
    }


# --- the features ------------------------------------------------------------------------------


def frame_features(inp):
    """(in_gap bool[n], seen_x float[n, 9], gap_x float[n, 31]). Rows of the other part are 0."""
    period, fps = inp["period"], inp["fps"]
    n = len(period)
    in_gap = np.zeros(n, dtype=bool)
    margin = inp["margin"]
    spread = ld.centred_mean(inp["spread_diff"], period, round(SHAPE_WINDOW_S * fps))
    shape_w = round(SHAPE_WINDOW_S * fps)
    shape_width = ld.centred_mean(inp["width_diff"], period, shape_w)
    shape_length = ld.centred_mean(inp["length_diff"], period, shape_w)
    shape_line = ld.centred_mean(inp["line_diff"], period, shape_w)

    seen = np.zeros((n, len(SEEN_FEATURES)))
    seen[:, 0] = np.nan_to_num(margin)
    for k, w in enumerate(CONTEXT_S):
        seen[:, 1 + k] = ld.centred_mean(margin, period, round(w * fps))
    near = inp["near"]
    seen[:, 3] = np.where(np.isfinite(near), np.log1p(np.minimum(near, NEAR_CAP_M)),
                          np.log1p(NEAR_CAP_M))
    seen[:, 4] = inp["close_diff"]
    seen[:, 5] = inp["touch"]
    speed = inp["ball_speed"]
    known = np.isfinite(speed)
    seen[:, 6] = np.where(known, np.log1p(np.minimum(np.where(known, speed, 0.0), SPEED_CAP)),
                          0.0)
    seen[:, 7] = ~known
    seen[:, 8] = spread

    gap = np.zeros((n, len(GAP_FEATURES)))
    look = round(TOUCH_LOOK_S * fps)
    signs = inp["touch_sign"]
    for first, stop in inp["gap_runs"]:
        in_gap[first:stop] = True
        length = stop - first
        before = first > 0 and period[first - 1] == period[first]
        after = stop < n and period[stop] == period[first]
        frac = (np.arange(length) + 0.5) / length
        last = margin[first - 1] if before and np.isfinite(margin[first - 1]) else 0.0
        nxt = margin[stop] if after and np.isfinite(margin[stop]) else 0.0
        touch_after = 0.0
        if after:
            window = signs[stop:min(n, stop + look)]
            hits = np.flatnonzero(window != 0)
            touch_after = float(window[hits[0]]) if hits.size else 0.0
        gap[first:stop, 0] = np.log2(length / fps)
        gap[first:stop, 1] = frac
        gap[first:stop, 2] = last
        gap[first:stop, 3] = nxt
        gap[first:stop, 4] = last * (1.0 - frac)
        gap[first:stop, 5] = nxt * frac
        gap[first:stop, 6] = inp["touch"][first - 1] if before else 0.0
        gap[first:stop, 7] = touch_after * frac
        gap[first:stop, 8] = spread[first:stop]
        gap[first:stop, 9] = not (before and after)
        if before and after:
            a, b = inp["ball_xy"][first - 1], inp["ball_xy"][stop]
            path = a + frac[:, None] * (b - a)
            dh, da, _, _ = _distances(inp["players_xy"][first:stop], path, inp["slot_team"])
            gap[first:stop, 10] = _margin(dh, da)
        gap[first:stop, 11] = inp["push"][first:stop]
        gap[first:stop, 12] = inp["push_short"][first:stop]
        gap[first:stop, 13] = inp["push_long"][first:stop]
        gap[first:stop, 14] = inp["push"][first:stop] - inp["push"][first]
        gap[first:stop, 15] = inp["field_pos"][first:stop]
        gap[first:stop, 16] = shape_width[first:stop]
        gap[first:stop, 17] = shape_length[first:stop]
        gap[first:stop, 18] = shape_line[first:stop]
        gap[first:stop, 19] = spread[first - 1] if before else 0.0
        gap[first:stop, 20] = spread[stop] if after else 0.0
        field, push = inp["field_pos"], inp["push"]
        gap[first:stop, 21] = field[first:stop] - field[first]
        gap[first:stop, 22] = shape_line[first:stop] - shape_line[first]
        gap[first:stop, 23] = shape_width[first:stop] - shape_width[first]
        gap[first:stop, 24] = shape_length[first:stop] - shape_length[first]
        gap[first:stop, 25] = field[stop - 1] - field[first:stop]
        gap[first:stop, 26] = push[stop - 1] - push[first:stop]
        gap[first:stop, 27] = push[first:stop] * frac
        gap[first:stop, 28] = inp["push_long"][first:stop] * np.log2(length / fps)
        gap[first:stop, 29] = field[first:stop] * frac
        gap[first:stop, 30] = shape_line[first:stop] * frac
    seen[in_gap] = 0.0
    gap[~in_gap] = 0.0
    return in_gap, seen, gap


# --- fitting -----------------------------------------------------------------------------------


def training_rows(match, truth, stride=5):
    """Rows to fit on, from one (cleaned, degraded) match and its truth.

    Only graded frames whose truth is H or A (the model is "which team, given the ball is
    alive"), every `stride`-th frame, each twice: as is and with home and away swapped (SWAP).
    Returns {"seen": (x, y), "gap": (x, y)}, y = 1 for H.
    """
    in_gap, seen_x, gap_x = frame_features(build_inputs(match))
    truth = np.asarray(truth)
    keep = (np.arange(len(truth)) % stride == 0) & (match.period > 0) & np.isin(truth, ["H", "A"])
    y = (truth == "H").astype(float)
    rows = {"seen": (seen_x[keep & ~in_gap], y[keep & ~in_gap]),
            "gap": (gap_x[keep & in_gap], y[keep & in_gap])}
    return {part: (np.vstack([x, swap(part, x)]), np.concatenate([y, 1.0 - y]))
            for part, (x, y) in rows.items()}


def swap(part, x):
    """The rows of x with home and away swapped (see SWAP). x may hold only the first columns
    (a model file of an older version)."""
    x = np.asarray(x, dtype=float)
    return x * np.asarray(SWAP[part][:x.shape[1]], dtype=float)


def symmetric_chance(fit, part, x):
    """Chance home has the ball, averaged with 1 - the chance after swapping the teams.

    Exactly symmetric: swapping home and away gives 1 - the chance, so no home lean is left.
    """
    return 0.5 * (ld.predict_proba(fit, x) + 1.0 - ld.predict_proba(fit, swap(part, x)))


def _fit(x, y, l2):
    """fit_logistic, or a flat model (every frame at the base rate) if there is nothing to fit."""
    if len(y) and y.min() != y.max():
        return ld.fit_logistic(x, y, l2)
    base = float(np.clip(y.mean(), 1e-3, 1 - 1e-3)) if len(y) else 0.5
    k = x.shape[1]
    return {"mean": [0.0] * k, "scale": [1.0] * k, "coef": [0.0] * k,
            "intercept": float(np.log(base / (1 - base)))}


def _accuracy(chance, y):
    return float(np.mean((np.asarray(chance) >= 0.5) == (np.asarray(y) >= 0.5))) if len(y) else 0.0


def fit_folds(rows_by_match, l2=10.0):
    """Leave-one-match-out fits plus one fit on every match ("all").

    Each fold: "fits", "trained_on", "rows" and, for a held-out match, "holdout": per part
    (log loss, log loss of the base rate, accuracy, rows). Aggregates only.
    """
    ids = sorted(rows_by_match)
    if len(ids) < 2:
        raise ValueError("fit_folds needs at least two matches (one to hold out)")

    def stack(names, part):
        x = np.vstack([rows_by_match[i][part][0] for i in names])
        y = np.concatenate([rows_by_match[i][part][1] for i in names])
        return x, y

    folds = {}
    for held_out in [*ids, None]:
        names = [i for i in ids if i != held_out]
        fits, rows, holdout = {}, {}, {}
        for part in PARTS:
            x, y = stack(names, part)
            fits[part] = _fit(x, y, l2)
            rows[part] = len(y)
            if held_out is not None:
                hx, hy = rows_by_match[held_out][part]
                chance = symmetric_chance(fits[part], part, hx) if len(hy) else np.zeros(0)
                holdout[part] = (ld.log_loss(chance, hy) if len(hy) else 0.0,
                                 ld.log_loss(np.full(len(hy), y.mean()), hy) if len(hy) else 0.0,
                                 _accuracy(chance, hy), len(hy))
        folds["all" if held_out is None else held_out] = {
            "fits": fits, "trained_on": names, "rows": rows, "holdout": holdout}
    return folds


# --- the model file ----------------------------------------------------------------------------


def save_model(path, fits, trained_on, l2, rows):
    doc = {"format": FORMAT, "version": VERSION, "trained_on": sorted(trained_on), "l2": l2,
           "rows": rows}
    for name, features in PARTS.items():
        fit = fits[name]
        doc[name] = {"features": list(features), "mean": ld._round(fit["mean"]),
                     "scale": ld._round(fit["scale"]), "coef": ld._round(fit["coef"]),
                     "intercept": ld._round(fit["intercept"])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"learned team model not found: {path} "
                                "(make it with scripts/fit_team.py)")
    doc = json.loads(path.read_text())
    if doc.get("format") != FORMAT or doc.get("version") not in READABLE:
        raise ValueError(f"{path}: not a version {'/'.join(map(str, READABLE))} {FORMAT} file")
    for name, features in READABLE[doc["version"]].items():
        part = doc.get(name)
        if not isinstance(part, dict) or part.get("features") != list(features):
            raise ValueError(f"{path}: the {name} part was made for other features, refit it")
        for key in ("mean", "scale", "coef"):
            values = part.get(key)
            if not isinstance(values, list) or len(values) != len(features) \
                    or not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: {name}.{key} must be {len(features)} finite numbers")
        if not np.isfinite(part.get("intercept", np.nan)) or np.any(np.asarray(part["scale"]) <= 0):
            raise ValueError(f"{path}: {name} has a bad intercept or scale")
    return doc


# --- the evidence ------------------------------------------------------------------------------


def home_chance(data, section, repo_root=None):
    """float[n]: chance that HOME has the ball (given it is alive); 0.5 outside the graded halves."""
    model = load_model(model_path(section, repo_root))
    in_gap, seen_x, gap_x = frame_features(build_inputs(data))
    graded = np.asarray(data.period) > 0
    out = np.full(len(in_gap), 0.5)
    for part, mask, x in (("seen", graded & ~in_gap, seen_x), ("gap", graded & in_gap, gap_x)):
        if mask.any():
            k = len(model[part]["features"])          # an older file uses the first columns
            out[mask] = symmetric_chance(model[part], part, x[mask][:, :k])
    return out


def team_costs(chance, team_cost):
    """float[n, 2] H and A costs: the likelier team 0, the other team_cost * |log-odds|."""
    p = np.clip(np.asarray(chance, dtype=float), 1e-4, 1 - 1e-4)
    logit = np.log(p) - np.log1p(-p)
    costs = np.zeros((len(p), 2))
    costs[:, 0] = team_cost * np.maximum(0.0, -logit)     # H costs where home is unlikely
    costs[:, 1] = team_cost * np.maximum(0.0, logit)      # A costs where home is likely
    return costs