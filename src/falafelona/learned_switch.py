# ws:poss: where possession changes INSIDE a ball gap, learned from other matches.
#
# WHY: the dev set shows we under-switch in gaps: a gap lasts ~19 s and holds 2-3 true
# possessions, but the decoder mostly gives it one team (ours ~290 possessions vs truth ~450 per
# match). The switch cost (18) is the same everywhere, so nothing tells the decoder where in a
# gap a change is likely.
#
# WHAT: one ridge logistic model (learned_dead.fit_logistic, no new dependency) on gap frames:
# q = chance that possession changes within +/- SWITCH_WINDOW_S of this frame. Its features say
# nothing about WHICH team (they are the same when home and away swap), only that "something
# turned": both teams' movement reverses, the back lines and the block move, the shapes change,
# and the two gap edges disagree on the team.
#
# HOW IT IS USED: a `hazard:` block in the possession section:
#     possession:
#       hazard: {model: models/switch_all.json, cost: 4.0, min_cost: 4.0}
# Inside gaps the H <-> A switch cost at a frame becomes
#     switch_cost - cost * (logit(q) - logit(base rate)),   clipped to [min_cost, switch_cost]
# so it only gets CHEAPER where the model sees a likelier-than-usual change. The team evidence
# still decides which team; this only lets the decoder change team where play turns.
# scripts/fit_switch.py writes the files (humans run it: it reads IDSSE). On the dev set the
# model path may hold {match_id} (leave-one-match-out), like learned_team.
#
# Tests use made-up numbers only (tests/test_learned_switch.py).

import copy
import json
from pathlib import Path

import numpy as np

from falafelona import learned_dead as ld
from falafelona import learned_team as lt

FORMAT = "falafelona-switch-logreg"
VERSION = 1                  # bump when a feature changes, so an old model file is refused

SWITCH_WINDOW_S = 0.5        # target: a change of the last team within +/- this
CHANGE_S = 1.0               # features compare t + this with t - this
CHANGE_SHORT_S = 0.5

FEATURES = (
    "log2_gap_s", "edge_s", "frac", "edges_disagree", "open_ended",
    "push_change", "push_flip", "abs_push", "push_short_change",
    "field_change", "line_change", "length_change", "width_change", "spread_change",
)
CONFIG_KEYS = ("model", "cost", "min_cost")


# --- config ------------------------------------------------------------------------------------


def check_settings(section):
    """Stop early, with a clear message, if the possession `hazard:` block is wrong."""
    block = section.get("hazard")
    block = block if isinstance(block, dict) else {}
    unknown = [key for key in block if key not in CONFIG_KEYS]
    missing = [key for key in CONFIG_KEYS if key not in block]
    if unknown or missing:
        raise ValueError(f"possession.hazard: unknown {unknown}, missing {missing}; "
                         f"needs {', '.join(CONFIG_KEYS)}")
    if not isinstance(block["model"], str) or not block["model"]:
        raise ValueError(f"possession.hazard.model must be a file path, got {block['model']!r}")
    for key in ("cost", "min_cost"):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int | float) \
                or not np.isfinite(value) or value < 0:
            raise ValueError(f"possession.hazard.{key} must be a number >= 0, got {value!r}")


def for_match(cfg, match_id):
    """cfg with {match_id} in the hazard model path filled in (leave-one-match-out)."""
    block = (cfg.get("possession") or {}).get("hazard")
    if not isinstance(block, dict) or "{match_id}" not in str(block.get("model", "")):
        return cfg
    filled = copy.deepcopy(cfg)
    filled["possession"]["hazard"]["model"] = block["model"].replace("{match_id}", match_id)
    return filled


def model_path(section, repo_root=None):
    raw = section["hazard"]["model"]
    if "{match_id}" in raw:
        raise ValueError(f"possession.hazard.model {raw!r} holds {{match_id}}: only the dev set "
                         "fills that in. Give a real file for a run.")
    path = Path(raw)
    if path.is_absolute():
        return path
    if repo_root is None:
        from falafelona import vendor
        repo_root = vendor.REPO_ROOT
    return Path(repo_root) / path


# --- the features ------------------------------------------------------------------------------


def _shifted(values, first, stop, step):
    """values[t + step] for t in [first, stop), the index clipped to [first - 1, stop] so it
    stays next to the gap (the gap edges are in the same period: gap_runs cuts at halves)."""
    n = len(values)
    index = np.clip(np.arange(first, stop) + step, max(first - 1, 0), min(stop, n - 1))
    return values[index]


def _change(values, first, stop, step):
    return np.abs(_shifted(values, first, stop, step) - _shifted(values, first, stop, -step))


def _flip(values, first, stop, step):
    return np.maximum(0.0, -_shifted(values, first, stop, step)
                      * _shifted(values, first, stop, -step))


def _edge_sign(margin, touch, frame):
    """+1 / -1 = the team the edge frame points to (touch first, then the margin), 0 unknown."""
    if touch[frame] != 0:
        return float(np.sign(touch[frame]))
    if np.isfinite(margin[frame]) and margin[frame] != 0:
        return float(np.sign(margin[frame]))
    return 0.0


def frame_features(inp):
    """(in_gap bool[n], x float[n, len(FEATURES)]); rows outside gaps are 0."""
    period, fps = inp["period"], inp["fps"]
    n = len(period)
    w = round(lt.SHAPE_WINDOW_S * fps)
    signals = {
        "field": inp["field_pos"],
        "line": ld.centred_mean(inp["line_diff"], period, w),
        "length": ld.centred_mean(inp["length_diff"], period, w),
        "width": ld.centred_mean(inp["width_diff"], period, w),
        "spread": ld.centred_mean(inp["spread_diff"], period, w),
    }
    step, short = max(1, round(CHANGE_S * fps)), max(1, round(CHANGE_SHORT_S * fps))
    push, push_short = inp["push"], inp["push_short"]
    margin, touch = inp["margin"], inp["touch"]

    in_gap = np.zeros(n, dtype=bool)
    x = np.zeros((n, len(FEATURES)))
    for first, stop in inp["gap_runs"]:
        in_gap[first:stop] = True
        length = stop - first
        before = first > 0 and period[first - 1] == period[first]
        after = stop < n and period[stop] == period[first]
        k = np.arange(length)
        x[first:stop, 0] = np.log2(length / fps)
        x[first:stop, 1] = np.log1p(np.minimum(k + 1, length - k) / fps)
        x[first:stop, 2] = (k + 0.5) / length
        if before and after:
            x[first:stop, 3] = -_edge_sign(margin, touch, first - 1) * _edge_sign(margin, touch,
                                                                                   stop)
        x[first:stop, 4] = not (before and after)
        x[first:stop, 5] = _change(push, first, stop, step)
        x[first:stop, 6] = _flip(push, first, stop, step)
        x[first:stop, 7] = np.abs(push[first:stop])
        x[first:stop, 8] = _change(push_short, first, stop, short)
        for column, name in enumerate(("field", "line", "length", "width", "spread"), start=9):
            x[first:stop, column] = _change(signals[name], first, stop, step)
    return in_gap, np.nan_to_num(x)


def switch_target(truth, period, fps):
    """bool[n]: the last team (H / A, carried over D and X inside a half) changes within
    +/- SWITCH_WINDOW_S of the frame."""
    truth = np.asarray(truth)
    period = np.asarray(period)
    n = len(truth)
    change = np.zeros(n, dtype=bool)
    last, last_period = "", None
    for i in range(n):
        if period[i] != last_period:
            last, last_period = "", period[i]
        if truth[i] in ("H", "A"):
            if last and truth[i] != last:
                change[i] = True
            last = truth[i]
    half = round(SWITCH_WINDOW_S * fps)
    near = np.zeros(n + 1)
    hits = np.flatnonzero(change)
    for h in hits:
        near[max(0, h - half)] += 1
        near[min(n, h + half + 1)] -= 1
    return np.cumsum(near)[:n] > 0


# --- fitting -----------------------------------------------------------------------------------


def training_rows(match, truth, stride=5):
    """(x, y) from one (cleaned, degraded) match: graded gap frames whose truth is H or A,
    every stride-th, y = 1 where possession changes within +/- SWITCH_WINDOW_S."""
    in_gap, x = frame_features(lt.build_inputs(match))
    truth = np.asarray(truth)
    period = np.asarray(match.period)
    y = switch_target(truth, period, float(match.fps)).astype(float)
    keep = (np.arange(len(truth)) % stride == 0) & (period > 0) & in_gap \
        & np.isin(truth, ["H", "A"])
    return x[keep], y[keep]


def auc(chance, y):
    """Area under the ROC curve (rank form): 0.5 = no better than chance. Aggregate only."""
    chance, y = np.asarray(chance, dtype=float), np.asarray(y) >= 0.5
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(chance, kind="mergesort")
    ranks = np.empty(len(chance))
    sorted_c = chance[order]
    i = 0
    while i < len(sorted_c):                          # average ranks over ties
        j = i
        while j + 1 < len(sorted_c) and sorted_c[j + 1] == sorted_c[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _fit(x, y, l2):
    if len(y) and y.min() != y.max():
        return ld.fit_logistic(x, y, l2)
    base = float(np.clip(y.mean(), 1e-3, 1 - 1e-3)) if len(y) else 0.5
    k = x.shape[1]
    return {"mean": [0.0] * k, "scale": [1.0] * k, "coef": [0.0] * k,
            "intercept": float(np.log(base / (1 - base)))}


def fit_folds(rows_by_match, l2=10.0):
    """Leave-one-match-out fits plus "all". Each fold: fit, trained_on, rows, base and, held
    out, (log loss, base log loss, auc, rows). Aggregates only."""
    ids = sorted(rows_by_match)
    if len(ids) < 2:
        raise ValueError("fit_folds needs at least two matches (one to hold out)")
    folds = {}
    for held_out in [*ids, None]:
        names = [i for i in ids if i != held_out]
        x = np.vstack([rows_by_match[i][0] for i in names])
        y = np.concatenate([rows_by_match[i][1] for i in names])
        fit = _fit(x, y, l2)
        base = float(np.clip(y.mean(), 1e-3, 1 - 1e-3)) if len(y) else 0.5
        fold = {"fit": fit, "trained_on": names, "rows": len(y), "base": base}
        if held_out is not None:
            hx, hy = rows_by_match[held_out]
            chance = ld.predict_proba(fit, hx) if len(hy) else np.zeros(0)
            fold["holdout"] = (ld.log_loss(chance, hy) if len(hy) else 0.0,
                               ld.log_loss(np.full(len(hy), base), hy) if len(hy) else 0.0,
                               auc(chance, hy), len(hy))
        folds["all" if held_out is None else held_out] = fold
    return folds


# --- the model file ----------------------------------------------------------------------------


def save_model(path, fold, l2):
    fit = fold["fit"]
    doc = {"format": FORMAT, "version": VERSION, "trained_on": sorted(fold["trained_on"]),
           "l2": l2, "rows": fold["rows"], "base": ld._round(fold["base"]),
           "features": list(FEATURES), "mean": ld._round(fit["mean"]),
           "scale": ld._round(fit["scale"]), "coef": ld._round(fit["coef"]),
           "intercept": ld._round(fit["intercept"])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"switch model not found: {path} (make it with "
                                "scripts/fit_switch.py)")
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
    if np.any(np.asarray(doc["scale"]) <= 0) or not 0 < doc.get("base", 0) < 1:
        raise ValueError(f"{path}: bad scale or base rate")
    return doc


# --- the switch costs --------------------------------------------------------------------------


def switch_chance(data, section, repo_root=None):
    """(in_gap bool[n], q float[n], base): chance of a change near each gap frame; base outside."""
    model = load_model(model_path(section, repo_root))
    in_gap, x = frame_features(lt.build_inputs(data))
    in_gap = in_gap & (np.asarray(data.period) > 0)
    q = np.full(len(in_gap), model["base"])
    if in_gap.any():
        q[in_gap] = ld.predict_proba(model, x[in_gap])
    return in_gap, q, model["base"]


def team_switch_costs(in_gap, q, base, switch_cost, cost, min_cost):
    """float[n]: the H <-> A switch cost of each frame. switch_cost outside gaps; inside,
    lowered by cost per unit of log-odds above the base rate, never below min_cost."""
    def logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p) - np.log1p(-p)

    lowered = switch_cost - cost * (logit(q) - logit(base))
    lowered = np.clip(lowered, min(min_cost, switch_cost), switch_cost)
    return np.where(in_gap, lowered, float(switch_cost))
