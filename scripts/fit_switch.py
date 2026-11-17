# ws:poss: fit the learned switch (hazard) model on IDSSE: where possession changes in a gap.
#
# WHAT: for every dev match, cuts the ball like ours (both gap placements, training seeds that
#   are NOT the dev-set seeds), cleans it like the pipeline and fits ridge logistic regression on
#   gap frames whose truth is H or A (y = 1 where possession changes within +/- 0.5 s):
#     models/switch_lomo_<match>.json   fitted on the OTHER six matches (what the dev set uses)
#     models/switch_all.json            fitted on all seven (what a run and the submission use)
# HOW: humans only, in your own terminal with NO AI tool open (it reads IDSSE):
#   .venv\Scripts\python scripts\fit_switch.py --devset $env:TEMP\devset-smooth.yaml
# Prints aggregates only: row counts, held-out log loss and AUC, and the coefficients.
# --tag NAME writes models/switch_NAME_*.json instead.

import argparse
from pathlib import Path

import numpy as np
import yaml

from falafelona import devset, learned_switch
from falafelona.ball import DataCleaner
from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final.yaml"
DEFAULT_DEVSET = REPO_ROOT / "configs" / "devset.yaml"


def match_rows(dev, cfg, dev_cfg, seeds, stride):
    """Training rows of one dev match: both gap placements x every seed."""
    xs, ys = [], []
    for placement in devset.PLACEMENTS:
        degrade = {**dev_cfg["degrade"], "placement": placement}
        for seed in seeds:
            cut, _ = devset.degrade_ball(dev.match, degrade,
                                         devset.match_seed(seed, dev.match_id), dev.truth)
            cleaned = DataCleaner(cut, cfg).return_cleaned_data()
            x, y = learned_switch.training_rows(cleaned, dev.truth, stride)
            xs.append(x)
            ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the learned switch model on IDSSE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="pipeline config: its ball settings clean the cut ball")
    parser.add_argument("--devset", default=str(DEFAULT_DEVSET))
    parser.add_argument("--matches", nargs="+", default=None, help="only these match ids")
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    parser.add_argument("--stride", type=int, default=5, help="use every n-th frame")
    parser.add_argument("--l2", type=float, default=10.0, help="ridge on the coefficients")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[100, 101],
                        help="seeds of the training cuts (the dev set uses 0 and 1)")
    parser.add_argument("--tag", default="", help="models/switch_<tag>_lomo_<match>.json")
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
        print(f"{match_id}: gap rows {len(rows[match_id][1])} "
              f"(change near {rows[match_id][1].mean():.3f})")

    folds = learned_switch.fit_folds(rows, args.l2)
    out = Path(args.out)
    prefix = f"switch_{args.tag}_" if args.tag else "switch_"
    aucs = []
    for name, fold in folds.items():
        path = out / (f"{prefix}all.json" if name == "all" else f"{prefix}lomo_{name}.json")
        learned_switch.save_model(path, fold, args.l2)
        line = f"{path.name}: trained on {len(fold['trained_on'])} matches"
        if "holdout" in fold:
            loss, base, area, _ = fold["holdout"]
            aucs.append(area)
            line += f"; held out {name}: log loss {loss:.3f} (base {base:.3f}), AUC {area:.3f}"
        print(line)
    print(f"mean held-out AUC {np.mean(aucs):.3f} (0.5 = no signal)")

    fit = folds["all"]["fit"]
    pairs = sorted(zip(learned_switch.FEATURES, fit["coef"], strict=True), key=lambda p: -abs(p[1]))
    print(f"coefficients of {prefix}all.json (standardised; positive = change more likely): "
          f"intercept {fit['intercept']:+.2f}; "
          + ", ".join(f"{name} {value:+.2f}" for name, value in pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
