# ws:eval: sweep of the H / A / D decoder costs, with a leave-one-match-out check.
#
# WHY: switch_cost (3.5) was chosen in #45, before the still-ball clue, the restart and touch
#   evidence and the learned dead-score blend changed how strong the per-frame evidence is.
#   to_dead_cost / from_dead_cost (6 / 6) are still the placeholder costs of #33, never swept.
#   A synthetic study (experiments/pgm_study, untracked) found that tuning these costs mattered
#   far more than the decoder family, and that a best value on the edge of the grid hides a
#   better one outside it. This script makes that check routine.
#
# TWO STEPS:
#   generate  writes configs/exp/cost-sweep/cost-*.yaml: the dev-set config of the submitted
#             version (configs/exp/poss-blend-learned70-stopmodel.yaml = final.yaml with the
#             leave-one-match-out dead model) with ONLY switch_cost, to_dead_cost, from_dead_cost
#             and dead_weight changed. No data is read.
#   analyse   reads the --scores-json files written by scripts/devset_eval.py (per-match spellF1
#             means only) and prints: the best configs in-sample, a warning for every setting
#             whose best value is on the edge of the grid (with a next grid to try), the
#             leave-one-match-out gain (scripts/touch_lomo.py) and the Go/No-Go verdict.
#             Reads no match data, so it may run anywhere.
#
# HOW (humans run devset_eval on the real IDSSE files, in a terminal with NO AI tool open):
#   .venv/bin/python scripts/cost_sweep.py generate                       # stage 1 grid
#   REF=configs/exp/poss-blend-learned70-stopmodel.yaml
#   sed 's/^seed: 0/seed: 1/' configs/devset.yaml > /tmp/devset-seed1.yaml
#   mkdir -p ~/cost-sweep
#   for gaps in random stoppage; do
#     .venv/bin/python scripts/devset_eval.py --gaps $gaps --scores-json ~/cost-sweep/s0-$gaps.json \
#         --config $REF configs/exp/cost-sweep/cost-*.yaml
#     .venv/bin/python scripts/devset_eval.py --gaps $gaps --devset /tmp/devset-seed1.yaml \
#         --scores-json ~/cost-sweep/s1-$gaps.json --config $REF configs/exp/cost-sweep/cost-*.yaml
#   done
#   .venv/bin/python scripts/cost_sweep.py analyse --run s0-random=~/cost-sweep/s0-random.json \
#       s0-stoppage=~/cost-sweep/s0-stoppage.json s1-random=~/cost-sweep/s1-random.json \
#       s1-stoppage=~/cost-sweep/s1-stoppage.json
# The config is CHOSEN on the s0 runs only; s1 and the held-out match only check it.

import argparse
import importlib.util
import itertools
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "configs" / "exp" / "poss-blend-learned70-stopmodel.yaml"
REFERENCE = BASE.stem
OUT_DIR = REPO_ROOT / "configs" / "exp" / "cost-sweep"
PREFIX = "cost-"
KEYS = ("switch_cost", "to_dead_cost", "from_dead_cost", "dead_weight")
SHORT = {"switch_cost": "sw", "to_dead_cost": "td", "from_dead_cost": "fd", "dead_weight": "dw"}

# Stage 1: the H <-> A switch against the (tied) dead costs, dead_weight as shipped. 6 x 5 = 30
# configs, and the shipped point (3.5, 6, 6, 2) is one of them as a check (its gain must be 0).
STAGE1 = {"switch": [2.5, 3.5, 5.0, 7.0, 10.0, 14.0], "dead": [4.0, 6.0, 9.0, 13.0, 18.0],
          "dead_weight": [2.0]}


def _touch_lomo():
    spec = importlib.util.spec_from_file_location(
        "touch_lomo", REPO_ROOT / "scripts" / "touch_lomo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmt(value):
    """3.5 -> '3.5', 6.0 -> '6' (file names and grids)."""
    return f"{float(value):g}"


def config_name(values):
    return PREFIX + "-".join(f"{SHORT[k]}{fmt(values[k])}" for k in KEYS)


def grid_points(switch, to_dead, from_dead, dead_weight):
    """[{key: value}] for every combination; from_dead None = the same as to_dead (tied)."""
    points = []
    for sw, td, dw in itertools.product(switch, to_dead, dead_weight):
        for fd in (from_dead if from_dead else [td]):
            points.append({"switch_cost": float(sw), "to_dead_cost": float(td),
                           "from_dead_cost": float(fd), "dead_weight": float(dw)})
    return points


def with_costs(base_text, values):
    """base_text with the four cost lines of the possession section set to `values`.

    Only those four lines change, so every other setting (and the leave-one-match-out dead
    model path) stays exactly as in the base config.
    """
    lines = base_text.splitlines()
    try:
        start = lines.index("possession:")
    except ValueError:
        raise SystemExit("base config has no top-level 'possession:' line") from None
    done = set()
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i].startswith(" "):
            break                                       # next top-level section
        match = re.match(r"^  (\w+):", lines[i])
        if match and match.group(1) in values:
            key = match.group(1)
            lines[i] = f"  {key}: {fmt(values[key])}  # SWEEP (base: {lines[i].split(':', 1)[1].strip()})"
            done.add(key)
    missing = set(values) - done
    if missing:
        raise SystemExit(f"base config has no possession setting(s) {sorted(missing)}")
    return "\n".join(lines) + "\n"


def generate(points, out_dir=OUT_DIR, base=BASE):
    """Write one config per point; returns the paths. Refuses to overwrite a different file."""
    base_text = Path(base).read_text()
    body = [line for line in base_text.splitlines() if not line.startswith("#")]
    header = (f"# Decoder cost sweep (scripts/cost_sweep.py): GENERATED, do not edit by hand.\n"
              f"# {Path(base).relative_to(REPO_ROOT)} with only {', '.join(KEYS)} changed.\n")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for values in points:
        text = header + with_costs("\n".join(body) + "\n", values)
        path = out_dir / f"{config_name(values)}.yaml"
        if path.exists() and path.read_text() != text:
            raise SystemExit(f"{path} exists with other content; remove it first")
        path.write_text(text)
        paths.append(path)
    return paths


def read_params(config_dir):
    """{config name: {key: value}} from the sweep configs (the YAML, not the file name)."""
    params = {}
    for path in sorted(Path(config_dir).glob(f"{PREFIX}*.yaml")):
        possession = yaml.safe_load(path.read_text())["possession"]
        params[path.stem] = {k: float(possession[k]) for k in KEYS}
    return params


def edges(best, params):
    """{key: 'low' / 'high'} for each setting whose best value is the grid's lowest / highest."""
    out = {}
    for key in KEYS:
        grid = sorted({p[key] for p in params.values()})
        if len(grid) < 2:
            continue
        if best[key] == grid[0]:
            out[key] = "low"
        elif best[key] == grid[-1]:
            out[key] = "high"
    return out


def next_values(key, best, params):
    """A next grid for one setting: extend past an edge, else refine around the best value."""
    grid = sorted({p[key] for p in params.values()})
    v = best[key]
    if len(grid) < 2:
        return [v]
    i = grid.index(v)
    if i == len(grid) - 1:
        ratio = grid[-1] / grid[-2]
        return [round(v * ratio ** k, 2) for k in (0, 1, 2)]
    if i == 0:
        ratio = grid[1] / grid[0]
        return sorted(round(v / ratio ** k, 2) for k in (0, 1, 2))
    lo, hi = grid[i - 1], grid[i + 1]
    return [round((lo + v) / 2, 2), v, round((v + hi) / 2, 2)]


def in_sample(g, select, configs):
    """[(mean gain over the select runs and all matches, config)], best first."""
    rows = []
    for name in configs:
        vals = [v for run in select for v in g[run][name].values()]
        rows.append((sum(vals) / len(vals), name))
    return sorted(rows, key=lambda r: (-r[0], r[1]))


def load_runs(pairs):
    """{run: {config: {match: spellF1}}} from name=path[,path...] pairs.

    Several files for one run (stage 1 and stage 2 of the same seed and gaps) are merged; the
    reference must score the same in each, else they are not the same run.
    """
    runs = {}
    for pair in pairs:
        name, sep, paths = pair.partition("=")
        if not sep or not name or not paths:
            raise SystemExit(f"--run wants name=path[,path...], got {pair!r}")
        merged = {}
        for path in paths.split(","):
            scores = json.loads(Path(path).expanduser().read_text())
            for config, per in scores.items():
                if config in merged and merged[config] != per:
                    raise SystemExit(f"run {name}: {config} scores differently in {path}; "
                                     "the files are not from the same run (seed, gaps)")
                merged[config] = per
        runs[name] = merged
    return runs


def analyse(runs, reference, select, params, top=10):
    tl = _touch_lomo()
    g = tl.gains(runs, reference, PREFIX)
    for run in select:
        if run not in g:
            raise SystemExit(f"--select run {run!r} not given with --run")
    configs = sorted(next(iter(g.values())))
    unknown = [c for c in configs if c not in params]
    if unknown:
        raise SystemExit(f"no config file for {unknown[:3]} in the --configs folder")
    params = {c: params[c] for c in configs}           # edges of the grid that actually ran
    ranked = in_sample(g, select, configs)
    lines = [(f"Decoder cost sweep: {len(configs)} configs vs {reference}; "
              f"chosen on runs {', '.join(select)}"), "",
             (f"Top {min(top, len(ranked))} in-sample (mean gain on the select runs; per run: "
              "mean, better on n/7):")]
    for mean, name in ranked[:top]:
        parts = []
        for run in g:
            m, wins, _, n = tl.summary(g[run][name].values())
            parts.append(f"{run} {m:+.3f} ({wins}/{n})")
        lines.append(f"  {name}: {mean:+.3f} | " + ", ".join(parts))

    best = params[ranked[0][1]]
    lines.append("")
    hit = edges(best, params)
    if hit:
        lines.append("GRID EDGE: the best config is on the edge of the grid for "
                     + ", ".join(f"{k} ({side})" for k, side in hit.items())
                     + ". A better value may lie outside: run the next grid before choosing.")
    else:
        lines.append("Grid edges: none (the best value of every setting is inside its grid).")
    nxt = {k: next_values(k, best, params) for k in KEYS}
    weights = sorted({1.5, 2.0, 3.0} | set(nxt["dead_weight"]))
    cmd = ("  .venv/bin/python scripts/cost_sweep.py generate"
           f" --switch {' '.join(map(fmt, nxt['switch_cost']))}"
           f" --to-dead {' '.join(map(fmt, nxt['to_dead_cost']))}"
           f" --from-dead {' '.join(map(fmt, nxt['from_dead_cost']))}"
           f" --dead-weight {' '.join(map(fmt, weights))}")
    lines.append("Next grid (refines around the best, extends past an edge, unties the dead costs "
                 "and opens dead_weight):")
    lines.append(cmd)

    lines.append("")
    if len(runs[select[0]][reference]) < 2:
        lines.append("Leave-one-match-out: needs at least 2 matches; skipped.")
    else:
        lines.append(tl.report(runs, reference, PREFIX, select, [ranked[0][1]])
                     .replace("touch configs", "cost configs"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decoder cost sweep: generate configs, analyse.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write configs/exp/cost-sweep/cost-*.yaml")
    gen.add_argument("--switch", nargs="+", type=float, default=STAGE1["switch"])
    gen.add_argument("--to-dead", nargs="+", type=float, default=STAGE1["dead"])
    gen.add_argument("--from-dead", nargs="+", type=float, default=None,
                     help="default: tied to --to-dead (one value for both)")
    gen.add_argument("--dead-weight", nargs="+", type=float, default=STAGE1["dead_weight"])
    gen.add_argument("--out", default=str(OUT_DIR))
    ana = sub.add_parser("analyse", help="read devset_eval --scores-json files and report")
    ana.add_argument("--run", nargs="+", required=True,
                     help="name=path[,path...] of --scores-json files (one run = one seed + gaps)")
    ana.add_argument("--reference", default=REFERENCE)
    ana.add_argument("--select", nargs="+", default=None,
                     help="runs used to choose (default: those whose name starts with s0)")
    ana.add_argument("--configs", default=str(OUT_DIR), help="folder of the sweep configs")
    ana.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "generate":
        points = grid_points(args.switch, args.to_dead, args.from_dead, args.dead_weight)
        paths = generate(points, args.out)
        print(f"wrote {len(paths)} configs to {args.out}")
        return 0
    runs = load_runs(args.run)
    select = args.select or [r for r in runs if r.startswith("s0")] or list(runs)
    print(analyse(runs, args.reference, select, read_params(args.configs), args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
