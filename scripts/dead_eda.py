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
#
# The question it answers for the possession decoder: are our long ball gaps STOPPAGES (then D
# inside a gap is right) or OCCLUSIONS in open play (then D splits a possession)? Restarts are
# frequent, so a gap can sit near one by chance. Every "near a restart" share therefore comes
# with a CHANCE level: the same share with all restarts shifted by SHIFTS_S and averaged. Only
# a share well above its chance level says "stoppage". The gap shares are also given per gap
# length and weighted by missing frames (what the score feels), plus a clue that needs no tags:
# was the last seen ball near a line, against how often the seen ball is near a line at all.
#
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
SHIFTS_S = (-120.0, -60.0, 60.0, 120.0)  # restarts moved by these for the chance level
GAP_BINS_S = ((2.0, 5.0), (5.0, 20.0), (20.0, np.inf))  # gap lengths reported on their own
LINE_MARGIN_M = 2.0    # a ball this close to a touchline or goal line (or outside) is "near a line"


# Start and end (end = first frame after) of every unbroken run of True in a 1-D array.
# [F T T F T] gives [(1, 3), (4, 5)].
def runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


# True if any run touches the window around `frame`.
def near(run_list, frame, before, after):
    return any(end >= frame - before and start <= frame + after for start, end in run_list)


# True if a restart comes during the run, or within the window after it.
def restart_near(start, end, starts, before, after):
    return any(start - after <= r <= end + before for r in starts)


# count / total rounded, or None when there is nothing to count.
def share(count, total):
    return round(count / total, 2) if total else None


# Mean of the values that exist, rounded, or None.
def mean_share(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 2) if values else None


# The restart frames moved by `shift` frames, kept only where they still land on a graded frame.
def shifted(starts, shift, graded):
    moved = [r + shift for r in starts]
    return [r for r in moved if 0 <= r < len(graded) and graded[r]]


# How the long gaps line up with the given restart frames: share of gaps, and share of their
# missing frames, near a restart.
def gap_alignment(gaps, starts, before, after):
    hit = [restart_near(s, e, starts, before, after) for s, e in gaps]
    frames = [e - s for s, e in gaps]
    return {
        "gaps": share(sum(hit), len(gaps)),
        "frames": share(sum(f for f, h in zip(frames, hit) if h), sum(frames)),
    }


# The same alignment with every restart shifted by SHIFTS_S, averaged: what chance gives.
def chance_alignment(gaps, starts, before, after, graded, fps):
    tries = [gap_alignment(gaps, shifted(starts, round(s * fps), graded), before, after)
             for s in SHIFTS_S]
    return {key: mean_share([t[key] for t in tries]) for key in ("gaps", "frames")}


# bool[n]: the seen ball is within LINE_MARGIN_M of a touchline or goal line, or outside.
def near_line(ball_xy, pitch):
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        inside_distance = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))
        return ~np.isnan(x) & (inside_distance <= LINE_MARGIN_M)


# Share of gaps whose last seen ball (the frame just before the gap) was near a line. Gaps with
# no seen, graded frame just before them are left out.
def last_seen_near_line(gaps, line, seen, graded):
    known = [s for s, _ in gaps if s > 0 and seen[s - 1] and graded[s - 1]]
    return share(sum(bool(line[s - 1]) for s in known), len(known))


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

    real = gap_alignment(long_gaps, starts, before, after)
    chance = chance_alignment(long_gaps, starts, before, after, graded, fps)

    seen = ~np.isnan(x) & graded
    line = near_line(match.ball_xy, match.pitch)
    missing_total = int(missing.sum())
    by_length = {}
    for low, high in GAP_BINS_S:
        part = [(s, e) for s, e in long_gaps if low * fps <= e - s < high * fps]
        name = f"{low:g}-{high:g} s" if np.isfinite(high) else f"over {low:g} s"
        part_real = gap_alignment(part, starts, before, after)
        part_chance = chance_alignment(part, starts, before, after, graded, fps)
        by_length[name] = {
            "gaps": len(part),
            "share_of_missing_frames": share(sum(e - s for s, e in part), missing_total),
            "near_restart": part_real["gaps"],
            "near_restart_by_chance": part_chance["gaps"],
            "last_seen_near_line": last_seen_near_line(part, line, seen, graded),
        }

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
        "long_gaps_near_restart": real["gaps"],
        "long_gaps_near_restart_by_chance": chance["gaps"],
        "long_gap_frames_near_restart": real["frames"],
        "long_gap_frames_near_restart_by_chance": chance["frames"],
        "long_gaps_by_length": by_length,
        "long_gaps_last_seen_near_line": last_seen_near_line(long_gaps, line, seen, graded),
        "seen_ball_near_line": share(int((line & seen).sum()), int(seen.sum())),
        "out_of_bounds_near_restart": share(
            sum(restart_near(s, e, starts, before, after) for s, e in outs), len(outs)),
    }


def main():
    match = load_match()  # reads $MTAH_DATA_DIR, outside the repo (D1)
    for key, value in dead_eda(match).items():
        if isinstance(value, dict) and key == "long_gaps_by_length":
            print(f"{key}:")
            for name, numbers in value.items():
                print(f"  {name}: {numbers}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
