# ws:eval: leave-one-match-out check of the touch settings (is +0.056 overfitted to 7 matches?).
#
# WHY: the touch settings were picked from a sweep on the 7 dev matches (seed 0 cuts) and
#   "confirmed" on seed 1 cuts. But seed only moves the ball gaps; touch works where the ball is
#   seen, and those frames are the same under both seeds. So seed 1 is not a real hold-out.
# WHAT: for each match M, choose the touch config with the best mean paired gain (vs --reference)
#   on the OTHER 6 matches, using only the --select runs (default: the seed 0 runs, as the sweep
#   was tuned); then report that config's gain on M, in every run. M never helps choose its own
#   config, so the mean of these gains is an honest estimate of what tuning-then-applying gives.
# IN: --scores-json files written by scripts/devset_eval.py (one per run, all configs in each):
#   name=path pairs, e.g. s0-random=/path/s0-random.json. Per-match means only; no match data.
# HOW: humans run devset_eval (real IDSSE files); this script only reads the small JSON files,
#   so it may run anywhere. The tests run it on made-up numbers.

import argparse
import json
from pathlib import Path

GO_GAIN = 0.04       # Go/No-Go: mean paired gain at least this ...
GO_WINS = 6          # ... and better on at least this many of the 7 matches, in every run


def load_runs(pairs):
    """{run: {config: {match: spellF1}}} from name=path pairs."""
    runs = {}
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--run wants name=path, got {pair!r}")
        runs[name] = json.loads(Path(path).read_text())
    return runs


def gains(runs, reference, prefix):
    """{run: {config: {match: gain vs reference}}} for the configs whose name starts with prefix."""
    out = {}
    for run, scores in runs.items():
        if reference not in scores:
            raise SystemExit(f"run {run}: reference {reference!r} missing")
        ref = scores[reference]
        out[run] = {
            name: {m: per[m] - ref[m] for m in ref}
            for name, per in scores.items() if name.startswith(prefix) and name != reference
        }
    return out


def lomo(g, select):
    """[(held-out match, chosen config, {run: gain on that match})], deterministic ties by name."""
    first = g[select[0]]
    configs = sorted(first)
    matches = sorted(next(iter(first.values())))
    rows = []
    for held in matches:
        others = [m for m in matches if m != held]

        fit = {name: sum(g[r][name][m] for r in select for m in others)
               / (len(select) * len(others)) for name in configs}
        best = max(configs, key=lambda name: (fit[name], name))
        rows.append((held, best, {run: g[run][best][held] for run in g}))
    return rows


def summary(values):
    vals = list(values)
    mean = sum(vals) / len(vals)
    return mean, sum(v > 0 for v in vals), sum(v >= GO_GAIN for v in vals), len(vals)


def report(runs, reference, prefix, select, compare):
    g = gains(runs, reference, prefix)
    for run in select:
        if run not in g:
            raise SystemExit(f"--select run {run!r} not given with --run")
    rows = lomo(g, select)
    lines = [(f"Leave-one-match-out over {len(next(iter(g.values())))} touch configs, vs "
              f"{reference}; chosen on runs {', '.join(select)}")]
    for held, best, per in rows:
        lines.append(f"  held out {held}: chose {best}; gain on {held}: "
                     + ", ".join(f"{run} {v:+.3f}" for run, v in per.items()))
    lines.append("LOMO gain (each match scored by a config chosen without it):")
    passes = True
    for run in g:
        mean, wins, big, n = summary(per[run] for _, _, per in rows)
        passes &= mean >= GO_GAIN and wins >= GO_WINS
        lines.append(f"  {run}: mean {mean:+.3f}, better on {wins}/{n}, >= +{GO_GAIN} on {big}/{n}")
    lines.append("In-sample, same config on every match:")
    for name in compare:
        if name not in next(iter(g.values())):
            lines.append(f"  {name}: not in the runs")
            continue
        parts = []
        for run in g:
            mean, wins, _, n = summary(g[run][name].values())
            parts.append(f"{run} {mean:+.3f} ({wins}/{n})")
        lines.append(f"  {name}: " + ", ".join(parts))
    chosen = {best for _, best, _ in rows}
    lines.append(f"configs chosen: {', '.join(sorted(chosen))}")
    lines.append(f"verdict: {'Go' if passes else 'No-Go'} (LOMO mean >= +{GO_GAIN} and better on "
                 f">= {GO_WINS}/7 in every run)")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Leave-one-match-out check of the touch sweep.")
    parser.add_argument("--run", nargs="+", required=True, help="name=path of --scores-json files")
    parser.add_argument("--reference", default="poss-place-late",
                        help="config the gains are against (the final.yaml before touch)")
    parser.add_argument("--prefix", default="touch-", help="candidate configs start with this")
    parser.add_argument("--select", nargs="+", default=None,
                        help="runs used to choose (default: those whose name starts with s0)")
    parser.add_argument("--compare", nargs="+",
                        default=["touch-r4-off0.12-margin0.1-hold0.8", "touch-r2-wide-cost2"])
    args = parser.parse_args(argv)
    runs = load_runs(args.run)
    select = args.select or [r for r in runs if r.startswith("s0")] or list(runs)
    print(report(runs, args.reference, args.prefix, select, args.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
