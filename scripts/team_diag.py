# ws:poss: label-free look at the learned team model, on our match or on IDSSE. Aggregates only.
#
# WHY: the learned team evidence won on the dev set, but on our match it moved the other way
#   (fewer, longer possessions instead of more). This prints the same numbers for both, so the
#   feature or the tracking difference that reads differently on our match shows up.
#
# WHAT (per part, seen-ball frames and ball-gap frames, graded halves only):
#   - frames, and the mean and spread of every model feature
#   - the model's chance that home has the ball: mean, share above 0.5, share unsure (0.35-0.65)
#   - seen frames: share where the nearest player is home (margin > 0), the plain baseline
#   - tracking: tracked home / away / unknown-team players per frame
# No truth is read, so the numbers on our match are fair to look at (D9 is about scores).
#
# HOW: humans only, in your own terminal with NO AI tool open:
#   .venv/bin/python scripts/team_diag.py --source real                 # MTAH_DATA_DIR, final.yaml
#   .venv/bin/python scripts/team_diag.py --source idsse                # IDSSE_DATA_DIR, gaps cut
#   .venv/bin/python scripts/team_diag.py --source idsse --gaps stoppage
# On IDSSE each match uses its leave-one-match-out model (models/team_lomo_<id>.json).
# Prints means, spreads and shares only: no frame numbers, positions or rows.

import argparse
from pathlib import Path

import numpy as np
import yaml

from falafelona import learned_team as lt
from falafelona.ball import DataCleaner
from falafelona.config import load_config
from falafelona.devset import smooth_ball
from falafelona.vendor import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final.yaml"
DEFAULT_DEVSET = REPO_ROOT / "configs" / "devset.yaml"
UNSURE = (0.35, 0.65)


def tracking(match):
    """Mean tracked players per graded frame: home, away, unknown team."""
    graded = np.asarray(match.period) > 0
    seen = np.isfinite(np.asarray(match.players_xy)[graded][..., 0])
    slot = np.asarray(match.slot_team)
    return {team: float(seen[:, slot == team].sum(axis=1).mean()) if graded.any() else 0.0
            for team in ("H", "A", "?")}

def ball_stats(match):
    """Why the ball can look faster: speed percentiles, jitter, still share, touch funnel.

    jitter: median |x[t+1] - 2 x[t] + x[t-1]| over three seen frames in a row, in cm. A smooth
    ball (even a fast one) has almost none; tracking noise shows up here directly.
    Touch funnel: velocity-change peaks per minute of SEEN ball (any team), then the share of
    them with a clear toucher, then the nearest player's distance at the peaks.
    """
    from falafelona import possession, velocity

    fps = float(match.fps)
    period = np.asarray(match.period)
    xy = np.asarray(match.ball_xy, dtype=float)
    graded = period > 0
    seen = graded & np.isfinite(xy[:, 0])
    vxy, speed = velocity.ball_velocity(xy, period, fps, lt.VELOCITY)
    known = seen & np.isfinite(speed)
    out = {"seen_share": float(seen.sum() / max(graded.sum(), 1))}
    sp = speed[known]
    out["speed_pct"] = ([float(v) for v in np.percentile(sp, (10, 25, 50, 75, 90))]
                        if sp.size else [0.0] * 5)
    out["still_share"] = float((sp < 0.5).mean()) if sp.size else 0.0
    out["fast_share"] = float((sp > 12.0).mean()) if sp.size else 0.0
    triple = seen[1:-1] & seen[:-2] & seen[2:] & (period[1:-1] == period[:-2]) \
        & (period[1:-1] == period[2:])
    second = np.hypot(*(xy[2:] - 2 * xy[1:-1] + xy[:-2]).T)[triple]
    out["jitter_cm"] = float(np.median(second) * 100) if second.size else 0.0
    step = np.hypot(*(xy[1:] - xy[:-1]).T)[seen[1:] & seen[:-1] & (period[1:] == period[:-1])]
    out["raw_speed_median"] = float(np.median(step) * fps) if step.size else 0.0
    offset = max(1, int(np.floor(lt.TOUCH_OFFSET_S * fps + 0.5)))
    peaks = possession.ball_touches(vxy, offset, lt.TOUCH_MIN_CHANGE)
    peaks = peaks[graded[peaks]] if len(peaks) else peaks
    d_home, d_away, _, _ = lt._distances(match.players_xy, xy, match.slot_team)
    teams = possession.touch_teams(peaks, d_home, d_away, lt.TOUCH_DISTANCE_M, lt.TOUCH_MARGIN_M)
    seen_min = seen.sum() / fps / 60.0
    out["peaks_per_seen_min"] = float(len(peaks) / seen_min) if seen_min else 0.0
    out["clear_share"] = float((teams >= 0).mean()) if len(peaks) else 0.0
    near = np.minimum(d_home, d_away)[peaks] if len(peaks) else np.zeros(0)
    near = near[np.isfinite(near)]
    out["peak_near_m"] = float(np.median(near)) if near.size else 0.0
    out["touches_per_seen_min"] = float((teams >= 0).sum() / seen_min) if seen_min else 0.0
    return out

LAGS = tuple(range(-12, 13, 2))     # frames; players taken this many frames AFTER the ball


def lag_scan(match, lags=LAGS):
    """Median distance (m) from the ball to the nearest player at the velocity-change peaks,
    with the players taken `lag` frames later than the ball, and the share of peaks with a
    clear toucher at that lag. If ball and players are in sync, the distance is lowest at 0;
    a low point elsewhere means one track runs ahead of the other.
    """
    from falafelona import possession, velocity

    fps = float(match.fps)
    period = np.asarray(match.period)
    xy = np.asarray(match.ball_xy, dtype=float)
    players = np.asarray(match.players_xy, dtype=float)
    slot = np.asarray(match.slot_team)
    vxy, _ = velocity.ball_velocity(xy, period, fps, lt.VELOCITY)
    offset = max(1, int(np.floor(lt.TOUCH_OFFSET_S * fps + 0.5)))
    peaks = possession.ball_touches(vxy, offset, lt.TOUCH_MIN_CHANGE)
    peaks = peaks[period[peaks] > 0] if len(peaks) else peaks
    n = len(period)
    out = {}
    for lag in lags:
        at = peaks + lag
        ok = (at >= 0) & (at < n)
        ok[ok] &= period[at[ok]] == period[peaks[ok]]
        if not ok.any():
            out[lag] = (0.0, 0.0)
            continue
        ball = xy[peaks[ok]]
        diff = players[at[ok]] - ball[:, None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        dist = np.where(np.isnan(dist), np.inf, dist)
        home = dist[:, slot == "H"].min(axis=1) if (slot == "H").any() else np.full(len(ball), np.inf)
        away = dist[:, slot == "A"].min(axis=1) if (slot == "A").any() else np.full(len(ball), np.inf)
        near = np.minimum(home, away)
        near = near[np.isfinite(near)]
        teams = possession.touch_teams(np.arange(len(ball)), home, away,
                                       lt.TOUCH_DISTANCE_M, lt.TOUCH_MARGIN_M)
        out[lag] = (float(np.median(near)) if near.size else 0.0, float((teams >= 0).mean()))
    return out


def format_lags(scan):
    best = min(scan, key=lambda lag: scan[lag][0])
    cells = ", ".join(f"{lag:+d}: {d:.2f} m ({c:.2f})" for lag, (d, c) in scan.items())
    return (f"  ball vs players time shift (players N frames later): nearest player at a peak "
            f"(clear share): {cells}\n  lowest at {best:+d} frames ({best / 25 * 1000:+.0f} ms "
            f"at 25 fps)")

def format_ball(b):
    p = b["speed_pct"]
    speed = (f"  ball speed (fit) p10/p25/p50/p75/p90: {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f} / "
             f"{p[3]:.2f} / {p[4]:.2f} m/s; still (<0.5) {b['still_share']:.3f}, "
             f"fast (>12) {b['fast_share']:.3f}")
    jitter = (f"  jitter {b['jitter_cm']:.1f} cm per frame; raw frame-to-frame speed median "
              f"{b['raw_speed_median']:.2f} m/s")
    funnel = (f"  touch funnel: {b['peaks_per_seen_min']:.1f} velocity-change peaks per seen "
              f"minute, {b['clear_share']:.3f} with a clear toucher = "
              f"{b['touches_per_seen_min']:.1f} clear touches per seen minute; nearest player "
              f"at a peak {b['peak_near_m']:.2f} m (median)")
    return "\n".join([f"  ball seen on {b['seen_share']:.3f} of graded frames", speed, jitter,
                      funnel])



def summarise(match, model):
    """Aggregates of the features and the model's chance on the graded frames of one match."""
    inp = lt.build_inputs(match)
    in_gap, seen_x, gap_x = lt.frame_features(inp)
    graded = np.asarray(match.period) > 0
    out = {"tracking": tracking(match), "ball": ball_stats(match), "lags": lag_scan(match)}
    for part, mask, x in (("seen", graded & ~in_gap, seen_x), ("gap", graded & in_gap, gap_x)):
        rows = x[mask]
        if not len(rows):
            out[part] = {"frames": 0}
            continue
        p = lt.symmetric_chance(model[part], part, rows)
        out[part] = {
            "frames": len(rows),
            "mean": rows.mean(axis=0).tolist(),
            "std": rows.std(axis=0).tolist(),
            "p_mean": float(p.mean()),
            "p_home": float((p > 0.5).mean()),
            "p_unsure": float(((p > UNSURE[0]) & (p < UNSURE[1])).mean()),
        }
    minutes = graded.sum() / match.fps / 60.0
    touches = np.asarray(inp["touch_sign"])[graded]
    out["touches_per_min"] = float((touches != 0).sum() / minutes) if minutes else 0.0
    speed = np.asarray(inp["ball_speed"])[graded & ~in_gap]
    speed = speed[np.isfinite(speed)]
    out["ball_speed_median"] = float(np.median(speed)) if speed.size else 0.0
    margin = inp["margin"][graded & ~in_gap]
    known = np.isfinite(margin) & (margin != 0)
    out["nearest_home"] = float((margin[known] > 0).mean()) if known.any() else 0.0
    return out


def average(summaries):
    """Frame-weighted mean of several summaries (for the IDSSE matches together)."""
    out = {"tracking": {t: float(np.mean([s["tracking"][t] for s in summaries]))
                        for t in ("H", "A", "?")},
           "nearest_home": float(np.mean([s["nearest_home"] for s in summaries])),
           "touches_per_min": float(np.mean([s["touches_per_min"] for s in summaries])),
           "ball_speed_median": float(np.mean([s["ball_speed_median"] for s in summaries])),
           "lags": {lag: tuple(float(np.mean([s["lags"][lag][i] for s in summaries]))
                               for i in (0, 1)) for lag in summaries[0]["lags"]},
           "ball": {key: (np.mean([s["ball"][key] for s in summaries], axis=0).tolist()
                          if key == "speed_pct"
                          else float(np.mean([s["ball"][key] for s in summaries])))
                    for key in summaries[0]["ball"]}}
    for part in lt.PARTS:
        parts = [s[part] for s in summaries if s[part]["frames"]]
        if not parts:
            out[part] = {"frames": 0}
            continue
        w = np.array([p["frames"] for p in parts], dtype=float)
        out[part] = {"frames": int(w.sum())}
        for key in ("mean", "std"):
            out[part][key] = (np.array([p[key] for p in parts]) * w[:, None]).sum(0) / w.sum()
        for key in ("p_mean", "p_home", "p_unsure"):
            out[part][key] = float((np.array([p[key] for p in parts]) * w).sum() / w.sum())
    return out


def format_summary(name, s):
    t = s["tracking"]
    lines = [f"{name}",
             f"  tracked per frame: home {t['H']:.1f}, away {t['A']:.1f}, unknown team {t['?']:.1f}",
                          f"  seen frames with the nearest player home: {s['nearest_home']:.3f}",
             (f"  clear touches per minute: {s['touches_per_min']:.1f}; "
                f"median ball speed (seen) {s['ball_speed_median']:.2f} m/s"),
             format_ball(s["ball"]), format_lags(s["lags"])]
    for part, features in lt.PARTS.items():
        d = s[part]
        if not d["frames"]:
            lines.append(f"  {part}: no frames")
            continue
        lines.append(f"  {part}: {d['frames']} frames; chance home mean {d['p_mean']:.3f}, "
                     f"above 0.5 {d['p_home']:.3f}, unsure {d['p_unsure']:.3f}")
        lines.append("    " + ", ".join(f"{f} {m:+.2f} ({sd:.2f})" for f, m, sd
                                       in zip(features, d["mean"], d["std"], strict=True)))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Label-free look at the learned team model.")
    parser.add_argument("--source", choices=["real", "idsse"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", default=None, help="model file (real match; default: config)")
    parser.add_argument("--devset", default=str(DEFAULT_DEVSET))
    parser.add_argument("--matches", nargs="+", default=None)
    parser.add_argument("--gaps", choices=["random", "stoppage"], default="random")
    parser.add_argument("--smooth-ball", nargs="+", type=float, default=None, metavar="SECONDS",
                        help="IDSSE only: smooth the ball (Gaussian sigma in s) before cutting "
                             "gaps and print the ball numbers for each, e.g. 0 0.08 0.16 0.24")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.source == "real":
        from falafelona.io import load_match
        path = Path(args.model) if args.model else lt.model_path(cfg["possession"])
        cleaned = DataCleaner(load_match(), cfg).return_cleaned_data()
        print(format_summary(f"our match (model {path.name})", summarise(cleaned, lt.load_model(path))))
        return 0

    from dataclasses import replace

    from falafelona import devset
    dev_cfg = yaml.safe_load(Path(args.devset).read_text())
    degrade = {**dev_cfg["degrade"], "placement": args.gaps}
    root = devset.idsse_dir()

    if args.smooth_ball is not None:
        devs = [devset.load_idsse(m, root=root) for m in args.matches or dev_cfg["matches"]]
        for sigma_s in args.smooth_ball:
            balls, lags = [], []
            for dev in devs:
                sigma = sigma_s * dev.match.fps
                smooth = replace(dev.match, ball_xy=smooth_ball(dev.match.ball_xy,
                                                                dev.match.period, sigma))
                cut, _ = devset.degrade_ball(smooth, degrade,
                                             devset.match_seed(dev_cfg["seed"], dev.match_id),
                                             dev.truth)
                cleaned = DataCleaner(cut, cfg).return_cleaned_data()
                balls.append(ball_stats(cleaned))
                lags.append(lag_scan(cleaned))
            ball = {key: (np.mean([b[key] for b in balls], axis=0).tolist() if key == "speed_pct"
                          else float(np.mean([b[key] for b in balls]))) for key in balls[0]}
            lag = {k: tuple(float(np.mean([s[k][i] for s in lags])) for i in (0, 1))
                   for k in lags[0]}
            print(f"IDSSE ball smoothed, sigma {sigma_s:g} s ({len(devs)} matches, {args.gaps} gaps)")
            print(format_ball(ball))
            print(format_lags(lag))
        return 0
    summaries = []
    for match_id in args.matches or dev_cfg["matches"]:
        dev = devset.load_idsse(match_id, root=root)
        cut, _ = devset.degrade_ball(dev.match, degrade,
                                     devset.match_seed(dev_cfg["seed"], match_id), dev.truth)
        cleaned = DataCleaner(cut, cfg).return_cleaned_data()
        model = lt.load_model(REPO_ROOT / "models" / f"team_lomo_{match_id}.json")
        summaries.append(summarise(cleaned, model))
        s = summaries[-1]
        best = min(s["lags"], key=lambda lag: s["lags"][lag][0])
        print(f"{match_id}: chance home mean seen {s['seen']['p_mean']:.3f} / gap "
              f"{s['gap']['p_mean']:.3f}; nearest home {s['nearest_home']:.3f}; "
              f"ball vs players lowest at {best:+d} frames")
    print(format_summary(f"IDSSE, {len(summaries)} matches, {args.gaps} gaps (frame-weighted)",
                         average(summaries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())