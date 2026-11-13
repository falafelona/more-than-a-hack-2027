# ws:eval: score pipeline configs on the dev set (IDSSE) with the official metric (task N5, #23).
#
# WHAT: loads the IDSSE matches listed in configs/devset.yaml, degrades their ball track to look
#   like ours (gap lengths from the EDA of our match, fixed seed), runs the pipeline stages with
#   each --config and prints spellF1 / macroF1 per match and on average, plus the breakdown that
#   explains them (possession precision/recall, alive/dead F1, team accuracy, switch timing).
#   With several configs it also prints the paired difference against the first one. The dev set
#   is what we compare on (D15); the public 5 minutes never chooses between versions (D9).
# HOW: humans only on real files, in your own terminal with NO AI tool open (CLAUDE.md):
#   export IDSSE_DATA_DIR=~/idsse-data          # outside the repo
#   .venv/bin/python scripts/devset_eval.py --config configs/baseline.yaml \
#       configs/exp/ball-interpolate.yaml
# Other options:
#   --clean-ball   do not degrade the ball (an upper bound: what the gaps cost us)
#   --gaps stoppage  put each gap over a dead-ball spell instead of at random (the default comes
#                  from configs/devset.yaml). Run both: a change we keep must win under both.
#   --matches J03WPY J03WMX   only these matches (a quick run)
#   --bootstrap N  a 5-95% interval for spellF1 only, computed per match with N draws (0 = off).
#                  Slow: matches x configs x N draws. Use it for the final choice between configs.
#   --summary      only shapes and label counts per match (checks that the files load)
#   --inspect FILE print the element and attribute NAMES of one IDSSE XML file (no values)
# Prints aggregates only: no frame numbers, no positions, and it writes no files except the
# parsed-data cache under IDSSE_DATA_DIR/cache. The tests run it on made-up files.

import argparse
from pathlib import Path

import yaml

from falafelona import devset
from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT


def load_dev_config(path):
    """Read configs/devset.yaml. Returns a dict with seed, matches and degrade."""
    with Path(path).open() as handle:
        cfg = yaml.safe_load(handle)
    for key in ("seed", "matches", "degrade"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing '{key}'")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score pipeline configs on the dev set.")
    parser.add_argument("--config", nargs="+", default=[],
                        help="pipeline configs; the first is the reference")
    parser.add_argument("--devset", default=str(REPO_ROOT / "configs" / "devset.yaml"))
    parser.add_argument("--matches", nargs="+", default=None, help="only these IDSSE match ids")
    parser.add_argument("--clean-ball", action="store_true", help="do not degrade the ball")
    parser.add_argument("--gaps", choices=devset.PLACEMENTS, default=None,
                        help="where the gaps go (default: degrade.placement in the devset config)")
    parser.add_argument("--summary", action="store_true", help="only shapes and counts")
    parser.add_argument("--inspect", default=None, help="print the XML structure of one file")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="draws for a 5-95%% spellF1 interval, per match (0 = off; slow: matches x configs x "
        "draws, so use e.g. 1000 only for the final choice between configs)",
    )
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be 0 or more")

    if args.inspect:
        print(devset.inspect_structure(args.inspect))
        return 0

    dev_cfg = load_dev_config(args.devset)
    if args.gaps:
        dev_cfg["degrade"] = {**dev_cfg["degrade"], "placement": args.gaps}
    ids = args.matches or dev_cfg["matches"]
    root = devset.idsse_dir()
    matches = [devset.load_idsse(match_id, root=root) for match_id in ids]

    if args.summary:
        for dev in matches:
            print(devset.summary(dev))
        return 0
    if not args.config:
        parser.error("give at least one --config (or use --summary / --inspect)")

    configs = {Path(path).stem: load_config(path) for path in args.config}
    if len(configs) != len(args.config):
        parser.error("two configs have the same file name")
    degrade = not args.clean_ball
    result = devset.run_devset(
        configs, dev_cfg, matches, degrade=degrade, bootstrap_n=args.bootstrap
    )
    print(devset.format_report(result, degrade=degrade))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())