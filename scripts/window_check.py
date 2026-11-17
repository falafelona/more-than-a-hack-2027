# ws:eval: is the public 5-minute ground truth unusual for a football match? (label-free on ours)
#
# WHY: the public 5 minutes of our match hold 17 possessions of 14.2 s on average, while the
#   IDSSE truth averages 5.5-9 s over whole matches. Either our provider counts possession in
#   longer spells (then the dev set rewards splitting too much), or those 5 minutes are just a
#   calm stretch. This script looks only at the IDSSE truth: how often does a 5-minute window of
#   a real match look like that? It never reads our match or its ground truth (D9).
# HOW: humans only (it reads IDSSE), like devset_eval:
#   .venv\Scripts\python scripts\window_check.py
# Prints aggregates only: counts, shares and durations per match. Writes nothing.

import argparse
from pathlib import Path

import numpy as np
import yaml

from falafelona import devset
from falafelona.vendor import REPO_ROOT

pm = devset.pm
FPS = 25.0
WINDOW_S = 300.0          # 5 minutes of graded play, like the public ground truth
STEP_S = 30.0
PUBLIC_COUNT = 17         # what the public 5 minutes show (from the official scoreboard header)
PUBLIC_MEAN_S = 14.2


def windows(truth, fps=FPS, window_s=WINDOW_S, step_s=STEP_S):
    """(count, mean duration s) of the possessions starting in each 5-minute window of a half."""
    codes = pm.label_codes(truth)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    starts = np.array([a for a, _, _ in spells])
    lengths = np.array([(b - a) / fps for a, b, _ in spells])
    size, step = int(window_s * fps), int(step_s * fps)
    out = []
    for first, stop in runs:
        for a in range(first, stop - size + 1, step):
            inside = (starts >= a) & (starts < a + size)
            if inside.any():
                out.append((int(inside.sum()), float(lengths[inside].mean())))
    return out, float(lengths.mean()) if len(lengths) else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description="How often does a 5-minute window look calm?")
    parser.add_argument("--devset", default=str(REPO_ROOT / "configs" / "devset.yaml"))
    args = parser.parse_args(argv)
    matches = yaml.safe_load(Path(args.devset).read_text())["matches"]
    root = devset.idsse_dir()
    every = []
    print(f"5-minute windows of the IDSSE truth (step {STEP_S:g} s); the public 5 minutes of our "
          f"match show {PUBLIC_COUNT} possessions of {PUBLIC_MEAN_S} s")
    for match_id in matches:
        dev = devset.load_idsse(match_id, root)
        found, whole = windows(dev.truth)
        every += found
        counts = np.array([c for c, _ in found])
        means = np.array([m for _, m in found])
        print(f"  {match_id}: whole match mean {whole:.1f} s; per window: possessions median "
              f"{np.median(counts):.0f} (min {counts.min()}), mean duration median "
              f"{np.median(means):.1f} s (max {means.max():.1f}); windows with <= {PUBLIC_COUNT} "
              f"possessions {np.mean(counts <= PUBLIC_COUNT):.0%}, with mean >= {PUBLIC_MEAN_S} s "
              f"{np.mean(means >= PUBLIC_MEAN_S):.0%}")
    counts = np.array([c for c, _ in every])
    means = np.array([m for _, m in every])
    print(f"all {len(every)} windows: <= {PUBLIC_COUNT} possessions {np.mean(counts <= PUBLIC_COUNT):.1%}, "
          f"mean >= {PUBLIC_MEAN_S} s {np.mean(means >= PUBLIC_MEAN_S):.1%}, both "
          f"{np.mean((counts <= PUBLIC_COUNT) & (means >= PUBLIC_MEAN_S)):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
