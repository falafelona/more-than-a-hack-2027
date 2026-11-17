# ws:poss: fit the learned team models on IDSSE (idea #1, step 1).
#
# WHAT: for every dev match, cuts the ball like ours (devset.degrade_ball, both gap placements,
#   training seeds that are NOT the dev-set seeds), cleans it like the pipeline, builds the
#   per-frame features (falafelona.learned_team) and fits ridge logistic regression on the frames
#   whose truth is H or A (y = 1 for H):
#     models/team_lomo_<match>.json   fitted on the OTHER six matches (what the dev set uses)
#     models/team_all.json            fitted on all seven (what a run and the submission use)
# HOW: humans only, in your own terminal with NO AI tool open (it reads IDSSE, like devset_eval):
#   export IDSSE_DATA_DIR=~/idsse-data
#   .venv/bin/python scripts/fit_team.py
# Prints aggregates only: row counts, held-out log loss and accuracy, and the coefficients
# (model numbers, not data). Writes only the JSON files under --out (default models/).

import argparse
from pathlib import Path

import numpy as np
import yaml

from falafelona import devset, learned_team
from falafelona.ball import DataCleaner
from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final.yaml"
DEFAULT_DEVSET = REPO_ROOT / "configs" / "devset.yaml"


def merge(parts):
    """One training_rows result from several (the same match cut in different ways)."""
    return {name: (np.vstack([p[name][0] for p in parts]),
                   np.concatenate([p[name][1] for p in parts])) for name in learned_team.PARTS}


def match_rows(dev, cfg, dev_cfg, seeds, stride):
    """Training rows of one dev match: both gap placements x every seed."""
    parts = []
    for placement in devset.PLACEMENTS:
        degrade = {**dev_cfg["degrade"], "placement": placement}
        for seed in seeds:
            cut, _ = devset.degrade_ball(dev.match, degrade,
                                         devset.match_seed(seed, dev.match_id), dev.truth)
            cleaned = DataCleaner(cut, cfg).return_cleaned_data()
            parts.append(learned_team.training_rows(cleaned, dev.truth, stride))
    return merge(parts)

def drop_features(rows, drops):
    """Zero the columns named in drops ("seen:spread_diff" or just "spread_diff" for both parts),
    in the as-is AND the swapped rows, so their fitted weight is 0 and they do nothing."""
    for drop in drops:
        part, sep, name = drop.partition(":")
        targets = [(part, name)] if sep else [(p, part) for p in learned_team.PARTS]
        hit = False
        for p, n in targets:
            if p in learned_team.PARTS and n in learned_team.PARTS[p]:
                column = learned_team.PARTS[p].index(n)
                for match_rows_ in rows.values():
                    match_rows_[p][0][:, column] = 0.0
                hit = True
        if not hit:
            raise SystemExit(f"--drop {drop!r}: use part:feature or a feature name, "
                             "for example spread_diff or gap:spread_diff")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the learned team models on IDSSE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="pipeline config: its ball settings clean the cut ball")
    parser.add_argument("--devset", default=str(DEFAULT_DEVSET))
    parser.add_argument("--matches", nargs="+", default=None, help="only these match ids")
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    parser.add_argument("--stride", type=int, default=5, help="use every n-th frame")
    parser.add_argument("--l2", type=float, default=10.0, help="ridge on the coefficients")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[100, 101],
                        help="seeds of the training cuts (the dev set uses 0 and 1)")
    parser.add_argument("--drop", nargs="+", default=[], metavar="FEATURE",
                        help="leave these features out, e.g. spread_diff or gap:spread_diff")
    parser.add_argument("--tag", default="",
                        help="name part of the files: models/team_<tag>_lomo_<match>.json")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.l2 < 0:
        parser.error("--stride must be at least 1 and --l2 0 or more")

    cfg = load_config(args.config)
    dev_cfg = yaml.safe_load(Path(args.devset).read_text())
    ids = args.matches or dev_cfg["matches"]
    root = devset.idsse_dir()

    rows = {}
    for match_id in ids:
        rows[match_id] = match_rows(devset.load_idsse(match_id, root=root), cfg, dev_cfg,
                                    args.train_seeds, args.stride)
        print(f"{match_id}: " + ", ".join(
            f"{part} rows {len(rows[match_id][part][1])} (home {rows[match_id][part][1].mean():.2f})"
            for part in learned_team.PARTS))

    drop_features(rows, args.drop)
    folds = learned_team.fit_folds(rows, args.l2)
    out = Path(args.out)
    prefix = f"team_{args.tag}_" if args.tag else "team_"
    for name, fold in folds.items():
        path = out / (f"{prefix}all.json" if name == "all" else f"{prefix}lomo_{name}.json")
        learned_team.save_model(path, fold["fits"], fold["trained_on"], args.l2, fold["rows"])
        held = ", ".join(f"{part} log loss {loss:.3f} (base {base:.3f}), accuracy {acc:.3f}"
                         for part, (loss, base, acc, _) in fold["holdout"].items())
        print(f"{path.name}: trained on {len(fold['trained_on'])} matches"
              + (f"; held out {name}: {held}" if held else ""))

    print(f"coefficients of {prefix}all.json (standardised features; positive = more likely home):")
    for part, features in learned_team.PARTS.items():
        fit = folds["all"]["fits"][part]
        pairs = sorted(zip(features, fit["coef"], strict=True), key=lambda p: -abs(p[1]))
        print(f"  {part}: intercept {fit['intercept']:+.2f}; "
              + ", ".join(f"{name} {value:+.2f}" for name, value in pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())