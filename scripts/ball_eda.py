# ws:ball: EDA of the ball track, aggregates only (task 2.1, team-board issue #13).
#
# WHAT: describes the ball track of the real match with numbers only.
#   - gap lengths: runs of frames where the ball is missing, counted inside each half. Gaps
#     that touch the first or last graded frame of a half are reported on their own: their true
#     length is cut off by the edge, and the second half is graded from before its kickoff.
#   - missing share per half (graded frames only)
#   - speed distribution: the ATD speed field as given, and the speed implied by the move
#     between two consecutive frames (m/s), plus their ratio (it shows the unit of the field)
#   - jump share: how many frame-to-frame moves imply a speed above each of a few limits
#   - extra: share of seen ball positions that lie outside the pitch
# These numbers show what #19 (false detections) and #30 (short gaps) can work with. They only
# describe this match. The thresholds those tasks use are still tuned on other matches (D15).
# HOW: in your terminal:
#   export MTAH_DATA_DIR=~/mtah-data
#   .venv/bin/python scripts/ball_eda.py
# Paste the printed block into the issue. It prints aggregates only: no frame numbers, no
# positions, and it writes no files. The tests run it on made-up data.

import argparse
import itertools
import math

import numpy as np

from falafelona import vendor
from falafelona.io import load_match

# Reporting grids. They only decide which numbers are printed; they do not change any label.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
GAP_EDGES_S = (0.2, 0.5, 1.0, 2.0, 5.0)  # gap length bins in seconds
JUMP_SPEEDS_MS = (20.0, 30.0, 40.0, 50.0)  # a move faster than this (m/s) counts as a jump


# Where each unbroken run of True starts and ends (end is the first index after the run) in a
# 1-D boolean array: [F T T F T] gives starts [1, 4] and ends [3, 5].
def gap_runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


# The frames of one half as a slice, or None if the half is not in the data.
def half_slice(period, half):
    frames = np.flatnonzero(period == half)
    return None if frames.size == 0 else slice(int(frames[0]), int(frames[-1]) + 1)


# The QUANTILES of some values, rounded, or None when there are no values. With whole_values
# every quantile is a length that really occurs (a gap of 1.15 frames makes no sense).
def quantiles(values, whole_values=False):
    if len(values) == 0:
        return None
    method = "inverted_cdf" if whole_values else "linear"
    return [round(float(q), 2) for q in np.quantile(values, QUANTILES, method=method)]


# Computes every number of the report from a Match. Returns a dict of plain numbers.
def ball_eda(match, jump_speeds=JUMP_SPEEDS_MS, pitch=None):
    fps = float(match.fps)
    length, width = pitch or vendor.games.GAMES[vendor.games.GAME]["pitch"]
    seen = ~np.isnan(match.ball_xy).any(axis=1)

    # Per half: missing frames, the gaps (runs of missing frames) and the move between each
    # pair of consecutive frames that both have a ball. Frames outside the two halves are
    # ignored, and a gap or a move never crosses a half boundary.
    halves, inside_lists, edge_lists, step_lists = {}, [], [], []
    for half in (1, 2):
        part = half_slice(match.period, half)
        if part is None:
            continue
        seen_half, xy = seen[part], match.ball_xy[part]
        halves[half] = {"frames": int(seen_half.size), "missing": int((~seen_half).sum())}
        starts, ends = gap_runs(~seen_half)
        at_edge = (starts == 0) | (ends == seen_half.size)
        inside_lists.append((ends - starts)[~at_edge])
        edge_lists.append((ends - starts)[at_edge])
        both = seen_half[1:] & seen_half[:-1]
        steps = np.linalg.norm(xy[1:] - xy[:-1], axis=1)[both]
        step_lists.append(steps * fps)  # metres per frame -> metres per second
    inside = np.concatenate(inside_lists) if inside_lists else np.array([], dtype=int)
    edge = np.concatenate(edge_lists) if edge_lists else np.array([], dtype=int)
    derived = np.concatenate(step_lists) if step_lists else np.array([])

    graded_seen = ((match.period == 1) | (match.period == 2)) & seen
    given = match.ball_speed[graded_seen]
    given = given[~np.isnan(given)]
    xy = match.ball_xy[graded_seen]
    outside = (xy[:, 0] < 0) | (xy[:, 0] > length) | (xy[:, 1] < 0) | (xy[:, 1] > width)

    frames = sum(h["frames"] for h in halves.values())
    missing = sum(h["missing"] for h in halves.values())

    # Gap length bins. A gap of L frames lasts L / fps seconds, so it is "up to s seconds"
    # when L <= floor(s * fps). Rounding up would put a 13-frame gap (0.52 s) in the 0.5 s bin.
    edges = [math.floor(s * fps + 1e-9) for s in GAP_EDGES_S]
    bounds = [0, *edges, np.inf]
    labels = [f"up to {GAP_EDGES_S[0]:g} s"]
    labels += [f"{a:g}-{b:g} s" for a, b in itertools.pairwise(GAP_EDGES_S)]
    labels += [f"over {GAP_EDGES_S[-1]:g} s"]
    bins = []
    for label, lo, hi in zip(labels, bounds[:-1], bounds[1:]):
        in_bin = inside[(inside > lo) & (inside <= hi)]
        bins.append(
            {
                "label": label,
                "gaps": int(in_bin.size),
                "share_of_missing": float(in_bin.sum() / missing) if missing else 0.0,
            }
        )

    median_given = float(np.median(given)) if given.size else None
    median_derived = float(np.median(derived)) if derived.size else None
    ratio = median_given / median_derived if median_given is not None and median_derived else None

    return {
        "fps": fps,
        "graded_frames": frames,
        "missing_frames": missing,
        "missing_share": missing / frames if frames else None,
        "halves": halves,
        "gaps": {
            "inside": {
                "count": int(inside.size),
                "length_quantiles": quantiles(inside, whole_values=True),
                "length_max": int(inside.max()) if inside.size else None,
                "bins": bins,
            },
            "at_edge": {
                "count": int(edge.size),
                "frames": int(edge.sum()),
                "share_of_missing": float(edge.sum() / missing) if missing else 0.0,
                "length_max": int(edge.max()) if edge.size else None,
            },
        },
        "speed_given": {
            "n": int(given.size),
            "quantiles": quantiles(given),
            "max": round(float(given.max()), 2) if given.size else None,
        },
        "speed_derived": {
            "n": int(derived.size),
            "quantiles": quantiles(derived),
            "max": round(float(derived.max()), 2) if derived.size else None,
        },
        "median_ratio_given_to_derived": None if ratio is None else round(ratio, 2),
        "jumps": {
            "pairs": int(derived.size),
            "by_limit": [
                {
                    "limit": float(v),
                    "count": int((derived > v).sum()),
                    "share": float((derived > v).mean()) if derived.size else None,
                }
                for v in jump_speeds
            ],
        },
        "outside_pitch": {
            "count": int(outside.sum()),
            "seen": int(outside.size),
            "share": float(outside.mean()) if outside.size else None,
        },
    }


def _pct(value, digits=1):
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def _qline(name, values, unit):
    if values is None:
        return f"  {name}: n/a"
    cells = ", ".join(f"p{round(q * 100)} {v:g}" for q, v in zip(QUANTILES, values))
    return f"  {name}: {cells}" + (f" ({unit})" if unit else "")


# Turns the dict into the text block that gets pasted into the issue.
def format_report(r):
    lines = [f"Ball track EDA, aggregates only ({r['fps']:g} fps, graded frames only)"]
    if not r["graded_frames"]:
        lines.append("no graded frames found: nothing to report")
        return "\n".join(lines)

    lines.append(
        f"graded frames: {r['graded_frames']:,}, ball missing on {r['missing_frames']:,} "
        f"({_pct(r['missing_share'])})"
    )
    for half, h in sorted(r["halves"].items()):
        lines.append(
            f"  half {half}: {h['frames']:,} frames, missing {h['missing']:,} "
            f"({_pct(h['missing'] / h['frames'])})"
        )

    inside, edge = r["gaps"]["inside"], r["gaps"]["at_edge"]
    lines.append(
        f"gaps (runs of missing ball): {inside['count'] + edge['count']:,} "
        f"({inside['count']:,} inside a half, {edge['count']:,} at a half edge)"
    )
    if inside["count"]:
        lines.append(_qline("inside a half, length", inside["length_quantiles"], "frames"))
        longest = inside["length_max"]
        lines.append(f"  longest: {longest:,} frames ({longest / r['fps']:.1f} s)")
        for b in inside["bins"]:
            lines.append(
                f"  {b['label']}: {b['gaps']:,} gaps, "
                f"{_pct(b['share_of_missing'])} of the missing frames"
            )
    if edge["count"]:
        lines.append(
            f"  touching the first or last graded frame of a half: {edge['count']:,} gaps, "
            f"{_pct(edge['share_of_missing'])} of the missing frames, "
            f"longest {edge['length_max']:,} frames ({edge['length_max'] / r['fps']:.1f} s)"
        )

    for key, title in (
        ("speed_given", "ball speed, ATD field as given"),
        ("speed_derived", "ball speed implied by the move between two frames"),
    ):
        s = r[key]
        lines.append(f"{title}: {s['n']:,} values")
        lines.append(_qline("quantiles", s["quantiles"], "m/s" if key == "speed_derived" else ""))
        lines.append(f"  max: {'n/a' if s['max'] is None else s['max']}")
    ratio = r["median_ratio_given_to_derived"]
    lines.append(
        f"median(as given) / median(implied): {'n/a' if ratio is None else ratio}  "
        "(about 1 means the field is in m/s, about 3.6 means km/h)"
    )

    j = r["jumps"]
    lines.append(f"jump share, among {j['pairs']:,} moves between two consecutive seen frames:")
    for item in j["by_limit"]:
        lines.append(
            f"  faster than {item['limit']:g} m/s: {item['count']:,} ({_pct(item['share'], 2)})"
        )

    o = r["outside_pitch"]
    lines.append(
        f"seen ball positions outside the pitch: {o['count']:,} of {o['seen']:,} "
        f"({_pct(o['share'], 2)})"
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ball track EDA, aggregates only.")
    parser.add_argument(
        "--jump-speeds",
        type=float,
        nargs="+",
        default=list(JUMP_SPEEDS_MS),
        help="limits in m/s for the jump share",
    )
    parser.add_argument("--n-frames", type=int, default=None, help="tests only: first N frames")
    args = parser.parse_args(argv)

    match = load_match(n_frames=args.n_frames)  # reads $MTAH_DATA_DIR, outside the repo (D1)
    print(format_report(ball_eda(match, args.jump_speeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())