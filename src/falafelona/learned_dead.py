# ws:dead: a dead-ball score learned from OTHER matches (plan B, Mon 16 Nov).
#
# WHY: the hand-built score gives each ball gap ONE value for its whole length (gap_chance), so a
# 20 s gap comes out all dead or all alive even when it holds some of each, and its weights were
# set by hand. Here every frame gets its own chance of being dead ball, from a logistic model
# fitted on the seven IDSSE matches (D15), with the ball cut like ours (devset.py).
#
# WHAT: two small logistic models, one for frames inside a ball gap and one for frames where the
# ball is seen. Every feature is physical and cheap (no tags, no truth, nothing that depends on
# how the dev set places its gaps):
#   gap frames:     gap length; where the ball was last seen and where it comes back (near a
#                   line, at the centre, how fast it moved); the players' speed around THIS frame
#                   (windows of 1, 4 and 12 s) and the share of players standing still
#   seen frames:    ball speed, ball standing still (and at a restart spot), distance to a line,
#                   out of bounds, distance to the centre spot, how long it has been slow, and
#                   the same player-speed windows
# Fitting is plain ridge logistic regression in numpy (Newton steps), so there is no new
# dependency and the same data always gives the same file. The fitted numbers live in a small
# JSON file under models/ (rounded to 6 significant digits). The pipeline only APPLIES them.
#
# HOW IT IS USED: deadball method `learned`, on its own or as one clue of `combined`:
#     deadball:
#       method: combined
#       weights: {before_kickoff: 1.0, out_of_bounds: 1.0, before_restart: 0.6, learned: 1.0}
#       learned: {model: models/deadball_all.json}
# On the dev set the model path holds {match_id}, which devset.py fills in with the held-out
# match (for_match), so every dev match is scored by a model that never saw it (leave-one-match-
# out). scripts/fit_deadball.py writes those files. Humans run it: it reads IDSSE.
#
# Assumption to remember: the models are trained on a 50/50 mix of random and stoppage gaps, so
# their idea of "how often is a gap dead ball" is that mix. Our real mix is unknown (D15).
#
# Tests use made-up numbers only (tests/test_learned_dead.py).

import copy
import itertools
import json
import warnings
from pathlib import Path

import numpy as np

FORMAT = "falafelona-deadball-logreg"
VERSION = 2                  # bump when a feature changes, so an old model file is refused

# Windows of the player-speed features, in seconds on each side of the frame, and the other
# constants of the features. They are part of the model: change one, bump VERSION.
WINDOWS_S = (1.0, 4.0, 12.0)
SLOW_WINDOW_S = 4.0
LINE_SCALE_M = 3.0           # exp(-distance / scale): 1 on a line, 0.37 at 3 m
CENTRE_SCALE_M = 5.0
SPEED_CAP = 30.0             # m/s, ball speeds above this count as this
SLOW_RUN_CAP_S = 30.0
SPEED_LOOK_S = 0.5           # s: how far from a gap's edge to look for a known ball speed

# The settings the features read from the `deadball:` section (numbers >= 0, checked by
# deadball.check_config). The speed block `speed:` and the ball block `ball_speed:` are read too.
SETTINGS = ("still_speed", "still_ball_speed", "still_ball_seconds", "spot_line_m",
            "spot_centre_m", "goal_area_depth_m", "goal_area_half_width_m", "out_margin")
CONFIG_KEYS = ("model", "visible")     # the settings of the `learned:` block

GAP_FEATURES = (
    "log2_gap_s", "last_near_line", "next_near_line", "next_at_centre", "open_ended",
    "last_ball_speed", "next_ball_speed", "gap_players_slow",
    "players_speed_w1", "players_speed_w4", "players_speed_w12", "players_slow_w4",
)
VISIBLE_FEATURES = (
    "log_ball_speed", "ball_speed_unknown", "ball_still", "ball_still_at_spot", "at_spot",
    "near_line", "out_of_bounds", "near_centre", "log_slow_run",
    "players_speed_w1", "players_speed_w4", "players_speed_w12", "players_slow_w4",
)
PARTS = {"gap": GAP_FEATURES, "visible": VISIBLE_FEATURES}


# --- config ------------------------------------------------------------------------------------


def check_settings(section):
    """Stop early, with a clear message, if the `learned:` block of `deadball:` is wrong."""
    block = section.get("learned")
    if not isinstance(block, dict) or not block:
        raise ValueError("deadball method learned needs a 'learned' block with a 'model' path")
    unknown = [key for key in block if key not in CONFIG_KEYS]
    if unknown:
        raise ValueError(f"deadball.learned: unknown setting(s) {unknown}; available: "
                         f"{', '.join(CONFIG_KEYS)}")
    model = block.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"deadball.learned.model must be a file path, got {model!r}")
    if not isinstance(block.get("visible", True), bool):
        raise TypeError("deadball.learned.visible must be true or false, got "
                         f"{block['visible']!r}")


def for_match(cfg, match_id):
    """cfg with {match_id} in the model path filled in (the dev set's leave-one-match-out).

    Returns cfg itself when the config has no learned model or no placeholder.
    """
    block = (cfg.get("deadball") or {}).get("learned")
    if not isinstance(block, dict) or "{match_id}" not in str(block.get("model", "")):
        return cfg
    filled = copy.deepcopy(cfg)
    filled["deadball"]["learned"]["model"] = block["model"].replace("{match_id}", match_id)
    return filled


def model_path(section, repo_root=None):
    """The model file of a config: relative paths count from the repo root (vendor.REPO_ROOT)."""
    raw = section["learned"]["model"]
    if "{match_id}" in raw:
        raise ValueError(f"deadball.learned.model {raw!r} holds {{match_id}}: only the dev set "
                         "(scripts/devset_eval.py) fills that in. Give a real file for a run.")
    path = Path(raw)
    if path.is_absolute():
        return path
    if repo_root is None:
        from falafelona import vendor
        repo_root = vendor.REPO_ROOT
    return Path(repo_root) / path


# --- the inputs the features are built from ----------------------------------------------------


def player_inputs(data, section):
    """What the player tracks give: median speed per frame, and where the players stand still.

    Does not depend on the ball, so the fitting script computes it once per match.
    """
    from falafelona import velocity  # here: deadball imports this module at its top

    speeds = velocity.player_speed(data.players_xy, data.period, data.fps, section["speed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # numpy warns on frames with no speed
        typical = np.nanmedian(speeds, axis=1)
    return {"player_speed": typical, "slow_players": typical < section["still_speed"]}


def build_inputs(data, section, players=None):
    """Plain arrays for frame_features, from a Match (after ball cleaning) and `deadball:`.

    players: the output of player_inputs for this match, to skip recomputing it.
    """
    from falafelona import deadball, velocity

    players = players or player_inputs(data, section)
    fps = data.fps
    _, ball_speed = velocity.ball_velocity(data.ball_xy, data.period, fps, section["ball_speed"])
    slow_ball = np.isfinite(ball_speed) & (ball_speed < section["still_ball_speed"])
    still_ball = deadball.keep_long_stretches(slow_ball, round(section["still_ball_seconds"] * fps))
    return {
        "ball_xy": data.ball_xy,
        "period": data.period,
        "fps": float(fps),
        "pitch": data.pitch,
        "ball_speed": ball_speed,
        "slow_ball": slow_ball,
        "still_ball": still_ball,
        "at_spot": deadball.restart_spot(data.ball_xy, data.pitch, section),
        "out_margin": float(section["out_margin"]),
        "gap_runs": deadball.gap_runs(np.isnan(data.ball_xy[:, 0]), data.period),
        **players,
    }


# --- the features ------------------------------------------------------------------------------


def _segments(period):
    """(first, stop) of every unbroken stretch of one period value."""
    change = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = [0, *change.tolist(), len(period)]
    return list(itertools.pairwise(edges))


def centred_mean(values, period, half_window):
    """Mean of the finite `values` within +/- half_window frames, inside one period; 0 if none.

    NaN values are skipped, so a stretch where nobody has a speed gives 0.
    """
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    for first, stop in _segments(period):
        part = values[first:stop]
        finite = np.isfinite(part)
        sums = np.concatenate(([0.0], np.cumsum(np.where(finite, part, 0.0))))
        counts = np.concatenate(([0.0], np.cumsum(finite.astype(float))))
        index = np.arange(stop - first)
        low = np.maximum(index - half_window, 0)
        high = np.minimum(index + half_window + 1, stop - first)
        total, count = sums[high] - sums[low], counts[high] - counts[low]
        out[first:stop] = np.where(count > 0, total / np.maximum(count, 1.0), 0.0)
    return out


def _run_lengths(flags):
    """For each True frame, the length (in frames) of the run of True it belongs to; 0 elsewhere."""
    lengths = np.zeros(len(flags))
    padded = np.concatenate(([False], np.asarray(flags, dtype=bool), [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    for start, stop in zip(edges[0::2], edges[1::2], strict=True):
        lengths[start:stop] = stop - start
    return lengths


def _line_distance(points, pitch):
    """Metres to the nearest touchline or goal line (negative past it); NaN where unseen."""
    length, width = pitch
    x, y = points[:, 0], points[:, 1]
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))


def _near(distance, scale):
    """exp(-distance / scale) for distance >= 0 (1 at 0 m); 0 where the distance is unknown."""
    with np.errstate(invalid="ignore"):
        near = np.exp(-np.maximum(distance, 0.0) / scale)
    return np.where(np.isfinite(distance), near, 0.0)


def _speed_near(speed, start, step, frames, period):
    """The first known ball speed from frame `start` going `step` (+1 or -1) frames at a time.

    Looks at most `frames` frames and stays inside the half of `start`. NaN if none is known.
    The speed is unknown on the frames next to a gap (it needs seen positions on BOTH sides), so
    the speed at a gap's edge has to be read a few frames in.
    """
    for k in range(frames + 1):
        i = start + step * k
        if i < 0 or i >= len(speed) or period[i] != period[start]:
            break
        if np.isfinite(speed[i]):
            return float(speed[i])
    return float("nan")


def frame_features(inp):
    """Features of every frame. Returns (in_gap bool[n], gap_x float[n, 12], vis_x float[n, 13]).

    gap_x is meaningful on gap frames, vis_x on frames where the ball is seen; the other rows
    are zeros. Column names: GAP_FEATURES and VISIBLE_FEATURES.
    """
    xy, period, fps = inp["ball_xy"], inp["period"], inp["fps"]
    n = len(xy)
    pitch = inp["pitch"]
    in_gap = np.isnan(xy[:, 0])

    speed_windows = [np.log1p(centred_mean(inp["player_speed"], period, round(w * fps)))
                     for w in WINDOWS_S]
    slow_share = centred_mean(inp["slow_players"].astype(float), period,
                              round(SLOW_WINDOW_S * fps))

    # seen ball
    vis = np.zeros((n, len(VISIBLE_FEATURES)))
    ball_speed = inp["ball_speed"]
    known = np.isfinite(ball_speed)
    line = _line_distance(xy, pitch)
    centre = np.hypot(xy[:, 0] - pitch[0] / 2, xy[:, 1] - pitch[1] / 2)
    with np.errstate(invalid="ignore"):
        out = line < -inp["out_margin"]
    vis[:, 0] = np.where(known, np.log1p(np.minimum(np.where(known, ball_speed, 0.0), SPEED_CAP)),
                         0.0)
    vis[:, 1] = ~known
    vis[:, 2] = inp["still_ball"]
    vis[:, 3] = inp["still_ball"] * inp["at_spot"]
    vis[:, 4] = inp["at_spot"]
    vis[:, 5] = _near(line, LINE_SCALE_M)
    vis[:, 6] = out
    vis[:, 7] = _near(centre, CENTRE_SCALE_M)
    vis[:, 8] = np.log1p(np.minimum(_run_lengths(inp["slow_ball"]) / fps, SLOW_RUN_CAP_S))
    vis[:, 9:12] = np.column_stack(speed_windows)
    vis[:, 12] = slow_share
    vis[in_gap] = 0.0

    # ball gaps
    gap = np.zeros((n, len(GAP_FEATURES)))
    gap[:, 8:11] = np.column_stack(speed_windows)
    gap[:, 11] = slow_share
    look = round(SPEED_LOOK_S * fps)
    for first, stop in inp["gap_runs"]:
        before = first > 0 and period[first - 1] == period[first]
        after = stop < n and period[stop] == period[first]
        gap[first:stop, 0] = np.log2((stop - first) / fps)
        if before:
            gap[first:stop, 1] = _near(line[first - 1:first], LINE_SCALE_M)[0]
            last = _speed_near(ball_speed, first - 1, -1, look, period)
            if np.isfinite(last):
                gap[first:stop, 5] = np.log1p(min(last, SPEED_CAP))
        if after:
            gap[first:stop, 2] = _near(line[stop:stop + 1], LINE_SCALE_M)[0]
            gap[first:stop, 3] = _near(centre[stop:stop + 1], CENTRE_SCALE_M)[0]
            nxt = _speed_near(ball_speed, stop, 1, look, period)
            if np.isfinite(nxt):
                gap[first:stop, 6] = np.log1p(min(nxt, SPEED_CAP))
        gap[first:stop, 4] = not (before and after)
        gap[first:stop, 7] = inp["slow_players"][first:stop].mean()
    gap[~in_gap] = 0.0
    return in_gap, gap, vis


# --- fitting -----------------------------------------------------------------------------------


def _softplus(eta):
    return np.logaddexp(0.0, eta)


def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))


def fit_logistic(x, y, l2=10.0, max_iter=60, tol=1e-9):
    """Ridge logistic regression by Newton steps with step halving. No randomness.

    x: float[n, k], y: 0/1 float[n]. Every column is standardised first; the fitted numbers are
    for the standardised columns: chance = sigmoid(((x - mean) / scale) @ coef + intercept).
    l2 is the ridge on the coefficients (not on the intercept), as a total, not a mean.
    Returns {"mean", "scale", "coef", "intercept"}.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
        raise ValueError("fit_logistic: x must be [n, k] and y [n], with at least one row")
    if y.min() == y.max():
        raise ValueError("fit_logistic: all rows have the same label, nothing to fit")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0                              # a constant column fits to 0
    z = np.hstack([(x - mean) / scale, np.ones((len(x), 1))])
    ridge = np.full(z.shape[1], float(l2))
    ridge[-1] = 0.0
    base = y.mean()
    w = np.zeros(z.shape[1])
    w[-1] = np.log(base / (1.0 - base))

    def loss(weights):
        eta = z @ weights
        return float(np.sum(_softplus(eta) - y * eta) + 0.5 * np.sum(ridge * weights ** 2))

    current = loss(w)
    for _ in range(max_iter):
        p = _sigmoid(z @ w)
        grad = z.T @ (p - y) + ridge * w
        hessian = (z * (p * (1.0 - p))[:, None]).T @ z + np.diag(ridge) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hessian, grad)
        factor = 1.0
        for _ in range(12):                                # halve the step until the loss falls
            trial = w - factor * step
            value = loss(trial)
            if value <= current:
                break
            factor /= 2.0
        else:
            break
        w, current = trial, value
        if np.max(np.abs(factor * step)) < tol:
            break
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": w[:-1].tolist(),
            "intercept": float(w[-1])}


def predict_proba(part, x):
    """Chance of dead ball for the rows of x, from one fitted part (a fit_logistic result)."""
    z = (np.asarray(x, dtype=float) - np.asarray(part["mean"])) / np.asarray(part["scale"])
    return _sigmoid(z @ np.asarray(part["coef"]) + part["intercept"])


def log_loss(chance, y):
    """Mean log loss of chances against 0/1 labels."""
    chance = np.clip(np.asarray(chance, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(chance) + (1 - y) * np.log(1 - chance)))


def training_rows(match, truth, section, stride=5, players=None):
    """Rows to fit on, from one (cleaned, degraded) match and its truth.

    Only graded frames whose truth is H, A or D, every `stride`-th frame (neighbouring frames
    say the same thing, so this loses little and keeps the fit small). Returns
    {"gap": (x, y), "visible": (x, y)}, y = 1 where the truth is D.
    """
    inp = build_inputs(match, section, players)
    in_gap, gap_x, vis_x = frame_features(inp)
    keep = (np.arange(len(truth)) % stride == 0) & (match.period > 0) & (truth != "X")
    y = (truth == "D").astype(float)
    return {"gap": (gap_x[keep & in_gap], y[keep & in_gap]),
            "visible": (vis_x[keep & ~in_gap], y[keep & ~in_gap])}


def fit_folds(rows_by_match, l2=10.0):
    """Leave-one-match-out fits, and one fit on every match.

    rows_by_match: {match_id: training_rows result}. Returns {name: fold} where name is each
    match id (a model fitted on all the OTHER matches) and "all" (fitted on every match). A fold
    holds "fits", "trained_on", "rows" (row counts per part) and, for a held-out match,
    "holdout": per part (log loss of the fit, log loss of the training base rate, rows) on
    the rows of the match that fit never saw. Counts and losses only.
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
            fits[part] = fit_logistic(x, y, l2)
            rows[part] = len(y)
            if held_out is not None:
                hx, hy = rows_by_match[held_out][part]
                base = np.full(len(hy), y.mean())
                holdout[part] = (log_loss(predict_proba(fits[part], hx), hy),
                                 log_loss(base, hy), len(hy))
        folds["all" if held_out is None else held_out] = {
            "fits": fits, "trained_on": names, "rows": rows, "holdout": holdout}
    return folds


# --- the model file ----------------------------------------------------------------------------


def _round(values):
    """6 significant digits: the same data gives the same file on every machine."""
    if isinstance(values, list):
        return [_round(v) for v in values]
    return float(f"{values:.6g}")


def save_model(path, fits, trained_on, l2, rows):
    """Write the two fitted parts to a JSON file. fits: {"gap": fit, "visible": fit}."""
    doc = {"format": FORMAT, "version": VERSION, "trained_on": sorted(trained_on), "l2": l2,
           "rows": rows, "windows_s": list(WINDOWS_S)}
    for name, features in PARTS.items():
        fit = fits[name]
        doc[name] = {"features": list(features), "mean": _round(fit["mean"]),
                     "scale": _round(fit["scale"]), "coef": _round(fit["coef"]),
                     "intercept": _round(fit["intercept"])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_model(path):
    """Read and check a model file. Refuses a file made for other features."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"learned dead-ball model not found: {path} "
                                "(make it with scripts/fit_deadball.py)")
    doc = json.loads(path.read_text())
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise ValueError(f"{path}: not a version {VERSION} {FORMAT} file")
    if doc.get("windows_s") != list(WINDOWS_S):
        raise ValueError(f"{path}: made with other player-speed windows, refit it")
    for name, features in PARTS.items():
        part = doc.get(name)
        if not isinstance(part, dict) or part.get("features") != list(features):
            raise ValueError(f"{path}: the {name} part was made for other features, refit it")
        for key in ("mean", "scale", "coef"):
            values = part.get(key)
            if not isinstance(values, list) or len(values) != len(features) \
                    or not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: {name}.{key} must be {len(features)} finite numbers")
        if not np.isfinite(part.get("intercept", np.nan)):
            raise ValueError(f"{path}: {name}.intercept must be a finite number")
        if np.any(np.asarray(part["scale"]) <= 0):
            raise ValueError(f"{path}: {name}.scale must be above 0")
    return doc


# --- the score ---------------------------------------------------------------------------------


def score(data, section, repo_root=None):
    """Chance of dead ball on every frame (0 = alive, 1 = dead). What deadball.learned returns.

    data: the Match after ball cleaning. section: the `deadball:` section of the config.
    """
    model = load_model(model_path(section, repo_root))
    in_gap, gap_x, vis_x = frame_features(build_inputs(data, section))
    out = np.zeros(len(in_gap))
    out[in_gap] = predict_proba(model["gap"], gap_x[in_gap]) if in_gap.any() else 0.0
    seen = ~in_gap
    if section["learned"].get("visible", True) and seen.any():
        out[seen] = predict_proba(model["visible"], vis_x[seen])   # visible: false leaves 0 here
    return out
