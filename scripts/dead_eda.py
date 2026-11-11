# ws:dead: EDA of Smart Tagging restarts vs ball gaps vs out of bounds (task 3.3, team-board #17).
#
# WHAT: three signals can mark a dead ball. This counts each one and how often they line up:
#   - restarts: Smart Tagging SET PIECES tags (throw-in, corner, goal kick, free kick, kick off)
#   - long gaps: the ball is missing for at least LONG_GAP_S seconds
#   - out of bounds: the seen ball is outside the pitch for at least MIN_OUT_FRAMES frames
# A restart "lines up" with a gap or out-of-bounds stretch if they touch within a window:
# from WINDOW_BEFORE_S before the restart to WINDOW_AFTER_S after it.
# Only graded frames count. These numbers describe this match only; dead-ball thresholds are
# still tuned on other matches (D15).
# HOW: humans only, in your own terminal with NO AI tool open (CLAUDE.md):
#   export MTAH_DATA_DIR=~/mtah-data
#   .venv/bin/python scripts/dead_eda.py
# Prints counts and shares only: no frame numbers, no positions. The tests use made-up matches.

from collections import Counter

import numpy as np

from falafelona.io import load_match

# Reporting choices. They only decide which numbers are printed; they change no label.
LONG_GAP_S = 2.0       # a missing-ball run at least this long counts as a long gap
MIN_OUT_FRAMES = 5     # an out-of-bounds run at least this long counts (0.2 s)
WINDOW_BEFORE_S = 10.0  # look this far before a restart
WINDOW_AFTER_S = 2.0    # and this far after it


# Start and end (end = first frame after) of every unbroken run of True in a 1-D array.
# [F T T F T] gives [(1, 3), (4, 5)].
def runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


# True if any run touches the window around `frame`.
def near(run_list, frame, before, after):
    return any(end >= frame - before and start <= frame + after for start, end in run_list)


# Every number of the report, from a Match. Returns a dict of plain numbers.
def dead_eda(match):
    fps = match.fps
    length, width = match.pitch
    graded = match.period > 0
    x, y = match.ball_xy[:, 0], match.ball_xy[:, 1]

    missing = np.isnan(x) & graded
    with np.errstate(invalid="ignore"):  # NaN comparisons are False, which is what we want
        outside = ~np.isnan(x) & graded & ((x < 0) | (x > length) | (y < 0) | (y > width))

    long_gaps = [(s, e) for s, e in runs(missing) if e - s >= LONG_GAP_S * fps]
    outs = [(s, e) for s, e in runs(outside) if e - s >= MIN_OUT_FRAMES]
    restarts = [t for t in match.tags
                if t.code.upper() == "SET PIECES" and 0 <= t.start < len(graded)
                and graded[t.start]]

    before, after = int(WINDOW_BEFORE_S * fps), int(WINDOW_AFTER_S * fps)
    gap_hit = [near(long_gaps, t.start, before, after) for t in restarts]
    out_hit = [near(outs, t.start, before, after) for t in restarts]
    starts = [t.start for t in restarts]

    # The other way round: a restart comes during the run, or within the window after it.
    def restart_near(start, end):
        return any(start - after <= r <= end + before for r in starts)

    def share(count, total):
        return round(count / total, 2) if total else None

    return {
        "restarts": len(restarts),
        "restarts_by_type": dict(Counter(t.type or "unknown" for t in restarts)),
        "restarts_without_team": sum(t.team is None for t in restarts),
        "long_gaps": len(long_gaps),
        "out_of_bounds_stretches": len(outs),
        "restarts_near_long_gap": share(sum(gap_hit), len(restarts)),
        "restarts_near_out_of_bounds": share(sum(out_hit), len(restarts)),
        "restarts_near_either": share(sum(g or o for g, o in zip(gap_hit, out_hit)),
                                      len(restarts)),
        "long_gaps_near_restart": share(sum(restart_near(s, e) for s, e in long_gaps),
                                        len(long_gaps)),
        "out_of_bounds_near_restart": share(sum(restart_near(s, e) for s, e in outs),
                                            len(outs)),
    }


def main():
    match = load_match()  # reads $MTAH_DATA_DIR, outside the repo (D1)
    for key, value in dead_eda(match).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()