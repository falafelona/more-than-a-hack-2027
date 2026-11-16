# ws:dead: fit the learned dead-ball models on the IDSSE matches (plan B, Mon 16 Nov).
#
# WHAT: for every dev match, cuts the ball like ours (devset.degrade_ball, both gap placements,
#   training seeds that are NOT the dev-set seeds), cleans it like the pipeline, builds the
#   per-frame features (falafelona.learned_dead) and fits ridge logistic regression:
#     models/deadball_lomo_<match>.json   fitted on the OTHER six matches (what the dev set uses)
#     models/deadball_all.json            fitted on all seven (what a run and the submission use)
#   The dev set (configs/exp/poss-learned*.yaml) points at the first kind with {match_id}.
# HOW: humans only, in your own terminal with NO AI tool open (it reads IDSSE, like devset_eval):
#   export IDSSE_DATA_DIR=~/idsse-data
#   .venv/bin/python scripts/fit_deadball.py
# Prints aggregates only: row counts, log losses and the fitted coefficients (model numbers, not
# data). Writes only the JSON files under --out (default models/). Same data, same files.
# Needs src/falafelona/learned_dead.py and the patched deadball.py / devset.py (all in this PR).
# Variants (review round): --placements random|stoppage trains on one kind of gap only,
#   --drop gap:log2_gap_s leaves a feature out, --tag NAME writes models/deadball_NAME_*.json.

import argparse
from pathlib import Path

import numpy as np
import yaml

from falafelona import devset, learned_dead
from falafelona.ball import DataCleaner
from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "exp" / "poss-learned-final.yaml"
DEFAULT_DEVSET = REPO_ROOT / "configs" / "devset.yaml"


def merge(parts):
    """One training_rows result from several (the same match cut in different ways)."""
    return {name: (np.vstack([p[name][0] for p in parts]),
                   np.concatenate([p[name][1] for p in parts])) for name in learned_dead.PARTS}


def match_rows(dev, cfg, dev_cfg, seeds, stride, placements=None):
    """Training rows of one dev match: the gap placements (both by default) x every seed."""
    section = cfg["deadball"]
    players = learned_dead.player_inputs(dev.match, section)    # the same for every cut
    parts = []
    for placement in placements or devset.PLACEMENTS:
        degrade = {**dev_cfg["degrade"], "placement": placement}
        for seed in seeds:
            cut, _ = devset.degrade_ball(dev.match, degrade,
                                         devset.match_seed(seed, dev.match_id), dev.truth)
            cleaned = DataCleaner(cut, cfg).return_cleaned_data()
            parts.append(learned_dead.training_rows(cleaned, dev.truth, section, stride, players))
    return merge(parts)


def drop_features(rows, drops):
    """Zero the columns named in drops ("gap:log2_gap_s"), so their fitted weight is 0.

    A constant column fits to 0 (learned_dead.fit_logistic), so the model file keeps its format
    and the pipeline needs no change. Edits rows in place.
    """
    for drop in drops:
        part, _, name = drop.partition(":")
        if part not in learned_dead.PARTS or name not in learned_dead.PARTS[part]:
            raise SystemExit(f"--drop {drop!r}: use part:feature, for example gap:log2_gap_s")
        column = learned_dead.PARTS[part].index(name)
        for match_rows_ in rows.values():
            match_rows_[part][0][:, column] = 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the learned dead-ball models on IDSSE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="pipeline config: its ball and deadball settings build the features")
    parser.add_argument("--devset", default=str(DEFAULT_DEVSET))
    parser.add_argument("--matches", nargs="+", default=None, help="only these match ids")
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    parser.add_argument("--stride", type=int, default=5, help="use every n-th frame")
    parser.add_argument("--l2", type=float, default=10.0, help="ridge on the coefficients")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[100, 101],
                        help="seeds of the training cuts (the dev set uses 0 and 1)")
    parser.add_argument("--placements", choices=[*devset.PLACEMENTS, "both"], default="both",
                        help="train on one kind of gap only (default: both)")
    parser.add_argument("--drop", nargs="+", default=[], metavar="PART:FEATURE",
                        help="leave these features out, for example gap:log2_gap_s")
    parser.add_argument("--tag", default="",
                        help="name part of the files: models/deadball_<tag>_lomo_<match>.json")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.l2 < 0:
        parser.error("--stride must be at least 1 and --l2 0 or more")

    cfg = load_config(args.config)
    placements = None if args.placements == "both" else (args.placements,)
    dev_cfg = yaml.safe_load(Path(args.devset).read_text())
    ids = args.matches or dev_cfg["matches"]
    root = devset.idsse_dir()

    rows = {}
    for match_id in ids:
        rows[match_id] = match_rows(devset.load_idsse(match_id, root=root), cfg, dev_cfg,
                                    args.train_seeds, args.stride, placements)
        print(f"{match_id}: " + ", ".join(
            f"{part} rows {len(rows[match_id][part][1])} "
            f"(dead {rows[match_id][part][1].mean():.2f})" for part in learned_dead.PARTS))

    drop_features(rows, args.drop)
    folds = learned_dead.fit_folds(rows, args.l2)
    prefix = f"deadball_{args.tag}_" if args.tag else "deadball_"
    out = Path(args.out)
    for name, fold in folds.items():
        path = out / (f"{prefix}all.json" if name == "all" else f"{prefix}lomo_{name}.json")
        learned_dead.save_model(path, fold["fits"], fold["trained_on"], args.l2, fold["rows"])
        held = ", ".join(f"{part} log loss {loss:.3f} vs base rate {base:.3f}"
                         for part, (loss, base, _) in fold["holdout"].items())
        print(f"{path.name}: trained on {len(fold['trained_on'])} matches"
              + (f"; held out {name}: {held}" if held else ""))

    print("coefficients of deadball_all.json (standardised features; positive = more likely dead):")
    for part, features in learned_dead.PARTS.items():
        fit = folds["all"]["fits"][part]
        pairs = sorted(zip(features, fit["coef"], strict=True), key=lambda p: -abs(p[1]))
        print(f"  {part}: intercept {fit['intercept']:+.2f}; "
              + ", ".join(f"{name} {value:+.2f}" for name, value in pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
