# ws:poss: fit the learned touch model on IDSSE: which team has the ball after a touch.
#
# WHAT: for every dev match, cuts the ball like ours (both gap placements, training seeds that
#   are NOT the dev-set seeds), cleans it like the pipeline, finds the touches with the config's
#   possession.touch settings and fits ridge logistic regression on touches whose truth just
#   after (0.3 s) is H or A:
#     models/touch_lomo_<match>.json   fitted on the OTHER six matches (what the dev set uses)
#     models/touch_all.json            fitted on all seven (what a run and the submission use)
# HOW: humans only, in your own terminal with NO AI tool open (it reads IDSSE):
#   .venv\Scripts\python scripts\fit_touch.py --devset $env:TEMP\devset-smooth.yaml
# Prints aggregates only: touch counts, held-out log loss and accuracy next to the current
# nearest-player rule's accuracy, and the coefficients. --tag NAME writes models/touch_NAME_*.

import argparse
from pathlib import Path

import numpy as np
import yaml

from falafelona import devset, learned_touch
from falafelona.ball import DataCleaner
from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final.yaml"
DEFAULT_DEVSET = REPO_ROOT / "configs" / "devset.yaml"


def match_rows(dev, cfg, dev_cfg, seeds):
    """Training rows of one dev match: both gap placements x every seed."""
    xs, ys, rules = [], [], []
    for placement in devset.PLACEMENTS:
        degrade = {**dev_cfg["degrade"], "placement": placement}
        for seed in seeds:
            cut, _ = devset.degrade_ball(dev.match, degrade,
                                         devset.match_seed(seed, dev.match_id), dev.truth)
            cleaned = DataCleaner(cut, cfg).return_cleaned_data()
            x, y, rule = learned_touch.training_rows(cleaned, dev.truth, cfg["possession"])
            xs.append(x)
            ys.append(y)
            rules.append(rule)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(rules)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the learned touch model on IDSSE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="pipeline config: ball cleaning and possession.touch / velocity")
    parser.add_argument("--devset", default=str(DEFAULT_DEVSET))
    parser.add_argument("--matches", nargs="+", default=None, help="only these match ids")
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    parser.add_argument("--l2", type=float, default=10.0, help="ridge on the coefficients")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[100, 101],
                        help="seeds of the training cuts (the dev set uses 0 and 1)")
    parser.add_argument("--tag", default="", help="models/touch_<tag>_lomo_<match>.json")
    args = parser.parse_args(argv)
    if args.l2 < 0:
        parser.error("--l2 must be 0 or more")

    cfg = load_config(args.config)
    if "touch" not in cfg["possession"] or "velocity" not in cfg["possession"]:
        parser.error("the config needs possession.touch and possession.velocity blocks")
    dev_cfg = yaml.safe_load(Path(args.devset).read_text())
    ids = args.matches or dev_cfg["matches"]
    root = devset.idsse_dir()

    rows = {}
    for match_id in ids:
        rows[match_id] = match_rows(devset.load_idsse(match_id, root=root), cfg, dev_cfg,
                                    args.train_seeds)
        print(f"{match_id}: touch rows {len(rows[match_id][1])} (home {rows[match_id][1].mean():.2f})")

    folds = learned_touch.fit_folds(rows, args.l2)
    out = Path(args.out)
    prefix = f"touch_{args.tag}_" if args.tag else "touch_"
    model_acc, rule_acc = [], []
    for name, fold in folds.items():
        path = out / (f"{prefix}all.json" if name == "all" else f"{prefix}lomo_{name}.json")
        learned_touch.save_model(path, fold, args.l2)
        line = f"{path.name}: trained on {len(fold['trained_on'])} matches"
        if "holdout" in fold:
            loss, acc, r_acc, r_cover, _ = fold["holdout"]
            model_acc.append(acc)
            rule_acc.append((r_acc, r_cover))
            line += (f"; held out {name}: log loss {loss:.3f}, accuracy {acc:.3f} "
                     f"(rule: {r_acc:.3f} on the {r_cover:.0%} of touches it names)")
        print(line)
    print(f"mean held-out accuracy {np.mean(model_acc):.3f} on every touch; rule "
          f"{np.mean([r for r, _ in rule_acc]):.3f} on the "
          f"{np.mean([c for _, c in rule_acc]):.0%} it names")

    fit = folds["all"]["fit"]
    pairs = sorted(zip(learned_touch.FEATURES, fit["coef"], strict=True), key=lambda p: -abs(p[1]))
    print(f"coefficients of {prefix}all.json (standardised; positive = more likely home): "
          f"intercept {fit['intercept']:+.2f}; "
          + ", ".join(f"{name} {value:+.2f}" for name, value in pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
