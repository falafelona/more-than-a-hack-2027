# ws:poss: do the Smart Tagging phases say which team has the ball? (label-free check)
#
# WHY: every Smart Tagging phase carries a Team tag (DATA.md), and the phases also cover the ball
# gaps, where our team evidence is weakest. Before using them we need to know, per code, whether
# the tagged team is the team WITH the ball (BUILD UP, PROGRESSION, ...), the team WITHOUT it
# (DEFENDING IN ... THIRD), or neither. IDSSE has no Smart Tagging, so the dev set cannot tell us.
#
# WHAT: compares each phase code's team with OUR OWN labels (a submission CSV, e.g. the final), on
#   graded frames where the raw ball is seen (there our team call is right ~90% of the time) and
#   our label is H or A. Per code: tags, frames, how often the tag team == our team ("agree"), and
#   how many tactical gap frames (raw ball missing) the code covers. Agree near 0.9 = "that team
#   has the ball"; near 0.1 = "the other team has it"; near 0.5 = says nothing.
#   Then, for the codes with a clear direction: share of gap frames covered by at least one.
# HOW: humans only, it reads the match and a submission file:
#   $env:MTAH_DATA_DIR = "C:\...\mtah-data"
#   .venv\Scripts\python scripts\tag_team_check.py --labels outputs\final-dead3.csv
# Counts and shares only: no frame numbers, no positions. Uses no possession truth.

import argparse
from collections import defaultdict

import numpy as np

from falafelona.io import load_match

CLEAR = 0.25            # a code "points" if agree >= 1 - CLEAR or <= CLEAR


def read_labels(path, n):
    """str[n] labels from a frame,label CSV (header line skipped)."""
    labels = np.full(n, "X", dtype="<U1")
    with open(path) as handle:
        handle.readline()
        for line in handle:
            frame, label = line.strip().split(",")[:2]
            i = int(frame)
            if 0 <= i < n:
                labels[i] = label.strip()[:1]
    return labels


def tag_report(match, labels):
    period = np.asarray(match.period)
    graded = period > 0
    seen = np.isfinite(match.ball_xy[:, 0])
    ours = graded & seen & np.isin(labels, ["H", "A"])
    gap = graded & ~seen
    per_code = defaultdict(lambda: {"tags": 0, "frames": 0, "agree": 0, "compared": 0,
                                    "gap": np.zeros(match.n_frames, dtype=bool)})
    no_team = 0
    for event in match.tags:
        if event.team not in ("H", "A"):
            no_team += 1
            continue
        a, b = max(0, event.start), min(match.n_frames, event.end)
        if b <= a:
            continue
        row = per_code[event.code.upper()]
        row["tags"] += 1
        row["frames"] += b - a
        window = slice(a, b)
        compare = ours[window]
        row["compared"] += int(compare.sum())
        row["agree"] += int((labels[window][compare] == event.team).sum())
        row["gap"][window] |= gap[window]

    with_team = sum(r["tags"] for r in per_code.values())
    lines = [f"tags with a team: {with_team}, without (N/A or unknown): {no_team}",
             f"graded gap frames (raw ball missing): {int(gap.sum())}",
             "per code: tags, frames, agree with our team where the ball is seen, gap frames"]
    pointing = {"with": np.zeros(match.n_frames, dtype=bool),
                "without": np.zeros(match.n_frames, dtype=bool)}
    for code in sorted(per_code, key=lambda c: -per_code[c]["frames"]):
        row = per_code[code]
        share = row["agree"] / row["compared"] if row["compared"] else float("nan")
        covered = int(row["gap"].sum())
        sign = ""
        if row["compared"] >= 500 and share >= 1 - CLEAR:
            sign, key = "  -> team WITH the ball", "with"
            pointing[key] |= row["gap"]
        elif row["compared"] >= 500 and share <= CLEAR:
            sign, key = "  -> team WITHOUT the ball", "without"
            pointing[key] |= row["gap"]
        lines.append(f"  {code:30s} {row['tags']:4d} tags, {row['frames']:6d} frames, agree "
                     f"{share:.2f} (on {row['compared']}), gap frames {covered}{sign}")
    union = pointing["with"] | pointing["without"]
    lines.append(f"gap frames covered by a code with a clear direction: {int(union.sum())} "
                 f"({union.sum() / max(1, gap.sum()):.1%} of graded gap frames)")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smart Tagging team vs our labels, aggregates.")
    parser.add_argument("--labels", required=True, help="a submission CSV (frame,label)")
    args = parser.parse_args(argv)
    match = load_match()                              # $MTAH_DATA_DIR, outside the repo (D1)
    print(tag_report(match, read_labels(args.labels, match.n_frames)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
