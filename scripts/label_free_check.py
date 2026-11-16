# ws:int: label-free checks on our match (task M3). No answers needed, aggregates only.
#
# WHAT: builds the submission from a config (to a temporary folder, never outputs/) and prints
# counts and shares that can show a silent failure without looking at any ground truth:
#   1. the file: rows, label shares, sha256
#   2. possessions: how many, and how long on average (all frames, and the graded halves only)
#   3. kickoff: for each half, whether its Kick Off (Start) tag was found, how long the graded
#      stretch before it is, and how much of it comes out dead. If the second-half tag is not
#      found, before_kickoff silently marks nothing dead there (about 2,900 frames on our match).
#   4. still ball: how many graded frames the still-ball clues fire on
#   5. frozen ball: graded frames whose ball position repeats the previous one EXACTLY. A tracker
#      that freezes the ball would look like a still ball and fool the still-ball clue.
#   6. tags: how many Smart Tagging restarts of each type
#
# HOW: humans only, in your own terminal with NO AI tool open (CLAUDE.md):
#   export MTAH_DATA_DIR=~/mtah-data
#   .venv/bin/python scripts/label_free_check.py                  # configs/final.yaml
#   .venv/bin/python scripts/label_free_check.py --config configs/exp/x.yaml
# Prints counts, shares and lengths only: no frame numbers, no positions, no rows.
# Paste the whole output back. The tests run it on made-up matches.

import argparse
import itertools
import tempfile
from pathlib import Path

import numpy as np

from falafelona import deadball, pipeline
from falafelona.config import load_config
from falafelona.io import select_tags

TEAMS = ("H", "A")
STILL_CLUES = ("still_ball", "still_ball_at_spot")
LONG_FROZEN_S = 2.0       # a frozen-ball run at least this long is counted on its own


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def possessions(labels, period=None):
    """Lengths in frames of the possessions: maximal runs of one team (H or A). D and X end a
    possession. With `period`, only graded frames count and a run also ends where the half
    changes."""
    labels = np.asarray(labels)
    key = labels if period is None else np.char.add(labels.astype(str), np.asarray(period).astype(str))
    keep = np.isin(labels, TEAMS) if period is None else np.isin(labels, TEAMS) & (np.asarray(period) > 0)
    lengths = []
    for (_, kept), run in itertools.groupby(zip(key, keep, strict=True)):
        if kept:
            lengths.append(sum(1 for _ in run))
    return lengths


def label_report(labels, period, fps):
    n = len(labels)
    report = {"rows": n}
    for label in ("H", "A", "D", "X"):
        count = int(np.sum(labels == label))
        report[label] = (count, pct(count, n))
    graded = np.asarray(period) > 0             # the frames the metric scores
    for label in ("H", "A", "D", "X"):
        count = int(np.sum((labels == label) & graded))
        report[f"{label}, graded halves"] = (count, pct(count, int(graded.sum())))
    for name, lengths in (("all frames", possessions(labels)),
                          ("graded halves", possessions(labels, period))):
        mean = round(float(np.mean(lengths)) / fps, 1) if lengths else 0.0
        report[f"possessions, {name}"] = (len(lengths), mean)
    return report


def kickoff_report(match, labels, dead_score):
    """Per half: tag found, graded seconds before the kickoff, % labelled D, % dead score >= 0.5."""
    kickoffs = select_tags(match.tags, code="set pieces", type="kick off (start)")
    frames = np.arange(match.n_frames)
    report = {}
    for half in (1, 2):
        tags = [tag for tag in kickoffs if tag.half == half]
        if not tags:
            graded = int(np.sum(match.period == half))
            report[half] = {"tag found": False, "graded frames in this half": graded}
            continue
        before = (match.period == half) & (frames < tags[0].start)
        count = int(before.sum())
        report[half] = {
            "tag found": True,
            "graded seconds before kickoff": round(count / match.fps, 1),
            "% labelled D": pct(int(np.sum(labels[before] == "D")), count),
            "% dead score >= 0.5": pct(int(np.sum(dead_score[before] >= 0.5)), count),
        }
    return report


def still_report(cleaned, cfg):
    """Graded frames each still-ball clue fires on, and as % of the graded frames with a ball."""
    graded = cleaned.period > 0
    seen = graded & ~np.isnan(cleaned.ball_xy[:, 0])
    dead = deadball.DeadBall(cleaned, None, cfg)
    report = {}
    for clue in STILL_CLUES:
        fires = dead.run(clue) > 0
        count = int(np.sum(fires & graded))
        report[clue] = (count, pct(count, int(seen.sum())))
    return report


def frozen_report(match):
    """Graded frames whose ball position equals the previous frame's exactly (both seen)."""
    xy, period = match.ball_xy, match.period
    same = np.zeros(match.n_frames, dtype=bool)
    same[1:] = (np.all(xy[1:] == xy[:-1], axis=1) & (period[1:] == period[:-1]) & (period[1:] > 0))
    seen = int(np.sum((period > 0) & ~np.isnan(xy[:, 0])))
    runs = [sum(1 for _ in run) for flag, run in itertools.groupby(same) if flag]
    long_frames = round(LONG_FROZEN_S * match.fps)
    return {
        "frames repeating the previous position": (int(same.sum()), pct(int(same.sum()), seen)),
        "longest run, seconds": round(max(runs, default=0) / match.fps, 1),
        f"runs of {LONG_FROZEN_S:g} s or more": sum(r >= long_frames for r in runs),
    }


def tag_report(match):
    report = {"tags": len(match.tags)}
    for type_ in (*deadball.RESTART_TYPES, "kick off (start)"):
        report[type_] = len(select_tags(match.tags, type=type_))
    return report


def check(result, cfg):
    """Every report from a pipeline.run result: plain numbers, safe to print."""
    labels, cleaned = result["labels"], result["cleaned"]
    return {
        "file": {**label_report(labels, cleaned.period, cleaned.fps), "sha256": result["sha256"]},
        "kickoff": kickoff_report(cleaned, labels, result["dead_score"]),
        "still ball": still_report(cleaned, cfg),
        "frozen ball (raw tracking)": frozen_report(result["match"]),
        "tags": tag_report(result["match"]),
    }


def show(report):
    for section, values in report.items():
        print(f"\n== {section}")
        for key, value in values.items():
            if isinstance(value, dict):
                print(f"  half {key}: " + ", ".join(f"{k} {v}" for k, v in value.items()))
            elif isinstance(value, tuple):
                print(f"  {key:<40} {value[0]:>8}   ({value[1]})")
            else:
                print(f"  {key:<40} {value}")
    print("\n(tuples: count, then % or mean seconds)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Label-free checks on our match (aggregates only)")
    parser.add_argument("--config", default="configs/final.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    with tempfile.TemporaryDirectory() as folder:
        result = pipeline.run(cfg, Path(folder) / "check.csv")
        report = check(result, cfg)
    print(f"config  {args.config}")
    show(report)
    return report


if __name__ == "__main__":
    main()
