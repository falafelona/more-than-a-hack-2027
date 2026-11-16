# ws:eval: compare two possession files, aggregates only. No ground truth, no match data printed.
"""Compare two submission files (frame,label CSVs) and print only aggregates.

    python scripts/compare_submissions.py A.csv B.csv

Prints how many frames change label from ref (A) to new (B), which kinds of change they are, how many
stretches they form, how many H <-> A switches each file has, and the possession count of each
file, counted exactly as vendor/challenge/evaluation/validate_submission.py counts them.
Nothing per frame is printed: no frame numbers, no labels, no rows.

Used for task 2: "how many frames change label vs final, and do the possessions stay the same".
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_metrics():
    """The organisers' possession_metrics.py (vendor/challenge/evaluation), as validate uses it."""
    sys.path.insert(0, str(ROOT / "vendor" / "challenge" / "evaluation"))
    import possession_metrics
    return possession_metrics


pm = _load_metrics()


def possessions(codes):
    """(count, mean seconds): what validate_submission.py reports (spells = runs of H or A)."""
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    return len(spells), (alive / float(pm.FPS) / len(spells) if spells else 0.0)


def team_switches(codes):
    """Number of places where H is followed directly by A or A by H."""
    before, after = codes[:-1], codes[1:]
    return int(np.sum((before <= pm.A) & (after <= pm.A) & (before != after)))


def stretch_lengths(changed):
    """Lengths (frames) of the runs of consecutive changed frames."""
    edges = np.diff(np.concatenate(([0], changed.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def compare(a, b):
    """Aggregates for two code arrays of the same length (a = ref, the reference)."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)} rows")
    cm = pm.confusion(a, b)
    changed = a != b
    lengths = stretch_lengths(changed)
    moves = {(pm.LABELS[i], pm.LABELS[j]): int(cm[i, j])
             for i in range(pm.N_LABELS) for j in range(pm.N_LABELS) if i != j and cm[i, j]}
    return {
        "rows": len(a),
        "changed": int(changed.sum()),
        "moves": moves,
        "stretches": len(lengths),
        "stretch_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "stretch_max": int(lengths.max()) if len(lengths) else 0,
        "counts": [[int(np.sum(codes == k)) for k in range(pm.N_LABELS)] for codes in (a, b)],
        "switches": [team_switches(a), team_switches(b)],
        "possessions": [possessions(a), possessions(b)],
    }


def report(name_a, name_b, r):
    """The text to print: aggregates only."""
    lines = [f"ref = {name_a}", f"new = {name_b}", f"rows {r['rows']}"]
    share = 100.0 * r["changed"] / r["rows"] if r["rows"] else 0.0
    lines.append(f"frames with a different label: {r['changed']} ({share:.3f}% of rows)")
    for (src, dst), n in sorted(r["moves"].items()):
        lines.append(f"  ref {src} -> new {dst}: {n}")
    lines.append(f"changed stretches: {r['stretches']} (mean {r['stretch_mean']:.1f} frames, "
                 f"longest {r['stretch_max']})")
    for name, counts in zip(("ref", "new"), r["counts"], strict=True):
        pct = " ".join(f"{lab} {c} ({100.0 * c / r['rows']:.1f}%)" if r["rows"] else f"{lab} 0"
                       for lab, c in zip(pm.LABELS, counts, strict=True))
        lines.append(f"label counts {name}: {pct}")
    lines.append(f"H<->A switches: ref {r['switches'][0]}, new {r['switches'][1]}")
    (na, ma), (nb, mb) = r["possessions"]
    lines.append(f"possessions: ref {na} (mean {ma:.1f} s), new {nb} (mean {mb:.1f} s), "
                 f"new minus ref = {nb - na:+d}")
    return lines


def load(path, expected=None):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, len(table) if expected is None else expected)
    if reason:
        raise ValueError(f"{Path(path).name}: {reason}")
    return pm.label_codes(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", help="the reference file (e.g. the final.yaml submission)")
    parser.add_argument("b", help="the file to compare with it")
    args = parser.parse_args(argv)
    try:
        a = load(args.a)
        b = load(args.b, expected=len(a))
    except (OSError, ValueError) as error:
        print(f"FAIL  {error}")
        return 1
    print("\n".join(report(Path(args.a).name, Path(args.b).name, compare(a, b))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
