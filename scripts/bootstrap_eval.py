# ws:eval: checks a submission file and prints totals only (tasks 5.1, 5.2 and 5.3).
#
# WHAT:
#   5.1 pipeline check: score on the public 5 minutes with the official metric. A CHECK ONLY,
#       never used to tune or choose between versions (D9). A score near 0 means a bug.
#   5.2 label-free stats for the whole match: number of possessions, median length,
#       H / A / D / X shares, and flicker (possessions shorter than 1 s).
#   5.3 plausibility: possessions per graded minute (public 5 min: ~3.4), and with --tags,
#       how often we agree with the team of Smart Tagging's attacking phases.
# HOW: humans only on real files, in your own terminal with NO AI tool open (CLAUDE.md):
#   .venv/bin/python scripts/bootstrap_eval.py outputs/Falafelona.csv [--bootstrap 1000] [--tags]
# --tags reads Smart Tagging from $MTAH_DATA_DIR. Prints aggregates only. The tests run it on
# made-up files via --gt.

import argparse

import numpy as np

from falafelona import io
from falafelona.vendor import EVALUATION_DIR, games
from falafelona.vendor import possession_metrics as pm

FPS = 25

# Smart Tagging phases where the tagged team is the team WITH the ball. Left out on purpose:
# DEFENDING ... phases (they tag the defending side) and SET PIECES (restarts, often N/A).
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                   "ATTACKING TRANSITION"}


# Finds the public 5-minute answers that ship with the organisers' code (vendor/challenge).
def public_gt_path():
    return next((EVALUATION_DIR / "gt").glob("*_possession_gt_public.csv"))


# Length (in frames) of every possession: each unbroken run of H or of A.
# 'H H D D H H' gives [2, 2], two possessions, like the official metric.
def possession_lengths(labels):
    lengths = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] in ("H", "A"):
                lengths.append(i - start)
            start = i
    return lengths


# 5.2: stats that need no answers. Possession count, median length, flicker, label shares.
def match_stats(labels):
    seconds = np.array(possession_lengths(labels)) / FPS
    stats = {
        "possessions": len(seconds),
        "median_s": round(float(np.median(seconds)), 1) if len(seconds) else 0.0,
        "flicker_under_1s": int(np.sum(seconds < 1)),
    }
    for label in "HADX":
        stats[f"share_{label}"] = round(float(np.mean(labels == label)), 3)
    return stats


# 5.1: score on the public 5 minutes with the official metric. With bootstrap > 0 it also
# gives a 5-95% range, which shows how little one score on 5 minutes means.
def public_check(labels, gt_path, bootstrap=0):
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    result = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=bootstrap)
    check = {"spellF1": round(result["spell_f1_mean"], 3),
             "macroF1": round(result["macro_f1"], 3)}
    if bootstrap:
        check["spellF1_5_95"] = (round(result["spell_bootstrap_low"], 3),
                                 round(result["spell_bootstrap_high"], 3))
    return check


# 5.3: possessions per minute of graded play. A real match has about 3.4 (public 5 minutes:
# 17 possessions). Much higher means we split possessions (flicker); much lower, we merge them.
def possessions_per_minute(labels, period):
    graded = period > 0
    minutes = np.sum(graded) / FPS / 60
    if minutes == 0:
        return None
    return round(float(len(possession_lengths(labels[graded])) / minutes), 2)


# 5.3: inside Smart Tagging's attacking phases, the share of our H/A frames that name the tagged
# team. Frames we call D are skipped (a phase can contain a short stoppage). Tags are automatic,
# not truth, so ~0.8-0.9 is good and a low value points to swapped teams or a bug.
def tag_agreement(labels, tags):
    agree = total = 0
    for tag in tags:
        if tag.code.upper() not in ATTACKING_CODES or tag.team is None:
            continue
        ours = labels[tag.start:tag.end]
        live = ours[(ours == "H") | (ours == "A")]
        agree += int(np.sum(live == tag.team))
        total += len(live)
    return round(agree / total, 3) if total else None


# Reads the file, checks its format with the official validator, then prints the reports.
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--gt", default=None, help="answers file; default: the public 5 minutes")
    parser.add_argument("--tags", action="store_true", help="also compare with Smart Tagging")
    args = parser.parse_args(argv)

    table = pm.read_label_csv(args.submission)
    gt_path = args.gt or public_gt_path()
    n_rows = len(pm.read_label_csv(gt_path))  # 199,319 for the real match
    problem = pm.validate_submission(table, n_rows)
    if problem:
        raise SystemExit(f"invalid submission: {problem}")

    labels = np.array([str(v).strip().upper() for v in table["label"]])
    print("public 5-min check (pipeline check only, D9):",
          public_check(labels, gt_path, args.bootstrap))
    print("label-free match stats:", match_stats(labels))

    plausibility = {"possessions_per_min": possessions_per_minute(
        labels, io.period_of_frames(n_rows, games.GAME))}
    if args.tags:
        tags = io.load_tags(games.GAME, io.data_dir(), n_rows)
        plausibility["smart_tagging_agreement"] = tag_agreement(labels, tags)
    print("plausibility:", plausibility)


if __name__ == "__main__":
    main()