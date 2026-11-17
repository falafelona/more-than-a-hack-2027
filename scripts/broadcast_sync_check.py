# ws:ball: can the broadcast tracking fill the tactical ball gaps? (checklist items 3-5)
#
# WHY: the tactical ball is missing on ~41% of graded frames. The broadcast recording is a second
# camera with its own tracking (ball seen on ~50% of its frames), but it is NOT synchronised to
# tactical: its own timeline, cuts, replays, no mapping table (DATA.md). Slots are not people
# either, so players cannot be matched one by one.
#
# WHAT (aggregates only):
#   1. both recordings: frames, ball seen, homography valid, players tracked per team
#   2. sync: the broadcast is cut into stretches (no homography, or the visible players jump = a
#      camera cut). Every stretch of >= MIN_STRETCH_S is slid along the whole tactical match
#      (every STEP-th frame, by FFT) to find where the two BALL tracks agree best (the ball is the
#      one thing both see in the same place, whatever part of the pitch the TV shows). The best
#      place is then checked with the players: every player the broadcast sees should have a
#      tactical player of the same team close by (median distance <= MAX_PLAYER_M). It also has to
#      be clearly better than the best place elsewhere (>= UNIQUE_S away). Both pitch
#      orientations are tried.
#   3. ball: on matched frames, where both see the ball, how far apart they are; and how many
#      tactical gap frames (raw ball missing, graded) get a broadcast ball.
# HOW: humans only, it reads the match:
#   $env:MTAH_DATA_DIR = "C:\...\mtah-data"
#   .venv\Scripts\python scripts\broadcast_sync_check.py
# Prints counts, shares and distances in metres: no frame numbers, no positions. Writes no files.
# Label-free: tracking and homography only, never the possession truth.

import argparse
import warnings

import numpy as np

from falafelona import io
from falafelona.io import load_match

STEP = 5                 # frames: the signatures are compared on every 5th frame (0.2 s)
MIN_STRETCH_S = 4.0      # broadcast stretches shorter than this are not matched
CUT_JUMP_M = 8.0         # the visible players' centre jumping this much in one frame = a cut
MIN_OVERLAP = 0.2        # a placement needs the ball seen in both on this share of the stretch
MAX_PLAYER_M = 2.0       # matched: broadcast players are this close to a tactical one (median)
UNIQUE_S = 5.0           # ... and the ball fits clearly better than anywhere this far away
UNIQUE_RATIO = 0.5       # best ball rms <= this x the best ball rms elsewhere
QUANTILES = (0.25, 0.5, 0.75, 0.9)


# --- loading -----------------------------------------------------------------------------------


def load_broadcast(game=None):
    """(ball_xy, players_xy, slot_team, homography_valid, pitch) of the broadcast recording."""
    from falafelona import vendor

    games = vendor.games
    game = game or games.GAME
    root = str(io.data_dir())
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, root, source="broadcast")
    n = games.source_frames("broadcast", game)
    pitch = games.GAMES[game]["pitch"]
    slot_team = io.slot_teams(meta_path)
    ball_xy, players_xy, _, _ = io.parse_raw(raw_path, n, len(slot_team), pitch)
    _, _, valid = vendor.homography.load(game, root, source="broadcast")
    valid = np.asarray(valid[:n], dtype=bool)
    if len(valid) < n:
        valid = np.concatenate([valid, np.zeros(n - len(valid), dtype=bool)])
    return ball_xy, players_xy, slot_team, valid, pitch


# --- signatures and stretches ------------------------------------------------------------------


def visible_centre(players_xy):
    """float[n, 2]: centre of every tracked player in the frame; NaN where fewer than 3."""
    count = np.isfinite(players_xy[..., 0]).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(players_xy, axis=1)
    centre[count < 3] = np.nan
    return centre


def stretches(signature, valid, fps):
    """(first, stop) of broadcast stretches: homography valid, signature known, no jump."""
    known = np.asarray(valid, dtype=bool) & np.isfinite(signature).all(axis=1)
    jump = np.zeros(len(signature), dtype=bool)
    step = np.abs(np.diff(signature, axis=0)).max(axis=1)
    jump[1:] = ~(step <= CUT_JUMP_M)            # NaN counts as a jump
    out, start = [], None
    for i in range(len(signature)):
        if known[i] and (start is None or not jump[i]):
            if start is None:
                start = i
            continue
        if start is not None:
            out.append((start, i))
        start = i if known[i] else None
    if start is not None:
        out.append((start, len(signature)))
    min_len = round(MIN_STRETCH_S * fps)
    return [(a, b) for a, b in out if b - a >= min_len]


def _corr(a, b, size):
    """Cross-correlation sum_t a[t] b[t + k] for every shift k >= 0 (via FFT)."""
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    return np.fft.irfft(np.conj(fa) * fb, size)


def match_stretch(piece, target):
    """Mean squared distance (per signature value) of `piece` placed at every start in `target`.

    piece float[m, k], target float[n, k], NaN = unknown. Returns float[n - m + 1] (inf where
    fewer than half of the piece's values overlap known target values).
    """
    m, n = len(piece), len(target)
    size = 1 << int(np.ceil(np.log2(n + m)))
    total = np.zeros(n - m + 1)
    count = np.zeros(n - m + 1)
    for k in range(piece.shape[1]):
        p, t = piece[:, k], target[:, k]
        pm, tm = np.isfinite(p).astype(float), np.isfinite(t).astype(float)
        p0, t0 = np.nan_to_num(p), np.nan_to_num(t)
        sq = (_corr(pm * p0 * p0, tm, size) + _corr(pm, tm * t0 * t0, size)
              - 2.0 * _corr(pm * p0, tm * t0, size))[: n - m + 1]
        total += sq
        count += _corr(pm, tm, size)[: n - m + 1]
    count = np.round(count)
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(count >= MIN_OVERLAP * m * piece.shape[1], np.maximum(total, 0) / count,
                       np.inf)
    return mse


def flip(xy, pitch):
    """Positions (..., 2) with the pitch turned 180 degrees (the other orientation)."""
    out = np.array(xy, dtype=float, copy=True)
    out[..., 0] = pitch[0] - out[..., 0]
    out[..., 1] = pitch[1] - out[..., 1]
    return out


def player_fit(bro_players, bro_slots, tac_players, tac_slots, frames, offset):
    """Median distance (m) from each broadcast player to the nearest tactical player of the
    same team, at tactical frame = broadcast frame + offset (every STEP-th frame)."""
    bro_slots, tac_slots = np.asarray(bro_slots), np.asarray(tac_slots)
    out = []
    for b in frames[::STEP]:
        t = b + offset
        if not 0 <= t < len(tac_players):
            continue
        for team in ("H", "A"):
            bp = bro_players[b, bro_slots == team]
            tp = tac_players[t, tac_slots == team]
            bp = bp[np.isfinite(bp[:, 0])]
            tp = tp[np.isfinite(tp[:, 0])]
            if len(bp) and len(tp):
                d = np.hypot(bp[:, None, 0] - tp[None, :, 0], bp[:, None, 1] - tp[None, :, 1])
                out.extend(d.min(axis=1).tolist())
    return float(np.median(out)) if out else np.inf


def sync(tac, bro_ball, bro_players, bro_slots, bro_valid, pitch):
    """[(first, stop, offset, player_m, flipped)] for matched broadcast stretches, plus counts.

    offset: tactical frame = broadcast frame + offset.
    """
    fps = float(tac.fps)
    runs = stretches(visible_centre(bro_players), bro_valid, fps)
    tac_ball = np.array(tac.ball_xy, dtype=float, copy=True)
    tac_ball[np.asarray(tac.period) <= 0] = np.nan
    target = tac_ball[::STEP]
    unique = max(1, round(UNIQUE_S * fps / STEP))
    matched, rejected = [], 0
    for first, stop in runs:
        best = None
        for flipped in (False, True):
            piece = bro_ball[first:stop]
            piece = flip(piece, pitch) if flipped else piece
            piece = piece[::STEP]
            if len(piece) >= len(target) or not np.isfinite(piece[:, 0]).any():
                continue
            mse = match_stretch(piece, target)
            k = int(np.argmin(mse))
            if not np.isfinite(mse[k]):
                continue
            far = mse.copy()
            far[max(0, k - unique):k + unique + 1] = np.inf
            second = float(np.min(far)) if np.isfinite(far).any() else np.inf
            rms = float(np.sqrt(mse[k]))
            if best is None or rms < best[0]:
                best = (rms, float(np.sqrt(second)), k * STEP - first, flipped)
        if best is None:
            rejected += 1
            continue
        rms, second, offset, flipped = best
        players = flip(bro_players, pitch) if flipped else bro_players
        fit = player_fit(players, bro_slots, tac.players_xy, tac.slot_team,
                         np.arange(first, stop), offset)
        if fit <= MAX_PLAYER_M and rms <= UNIQUE_RATIO * second:
            matched.append((first, stop, offset, fit, flipped))
        else:
            rejected += 1
    return matched, len(runs), rejected, runs


def borrow(tac, bro_players, bro_slots, runs, matched, pitch):
    """Second pass: a stretch the ball could not place takes the offset of the nearest matched
    stretch before or after it (the offset drifts slowly between replays), if the players agree
    there (median distance <= MAX_PLAYER_M). Returns the extra matches."""
    done = {(m[0], m[1]) for m in matched}
    ordered = sorted(matched)
    extra = []
    for first, stop in runs:
        if (first, stop) in done:
            continue
        before = [m for m in ordered if m[1] <= first]
        after = [m for m in ordered if m[0] >= stop]
        best = None
        for m in (before[-1:] + after[:1]):
            offset, flipped = m[2], m[4]
            players = flip(bro_players, pitch) if flipped else bro_players
            fit = player_fit(players, bro_slots, tac.players_xy, tac.slot_team,
                             np.arange(first, stop), offset)
            if fit <= MAX_PLAYER_M and (best is None or fit < best[3]):
                best = (first, stop, offset, fit, flipped)
        if best is not None:
            extra.append(best)
    return extra


# --- the report --------------------------------------------------------------------------------


def _q(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "none"
    return ", ".join(f"p{round(100 * p)} {np.quantile(v, p):.1f}" for p in QUANTILES)


def describe(name, ball_xy, players_xy, slot_team, valid, graded=None):
    label = "graded" if graded is not None else "all"
    graded = np.ones(len(ball_xy), dtype=bool) if graded is None else graded
    seen = np.isfinite(ball_xy[:, 0])
    slot_team = np.asarray(slot_team)
    per_team = [np.isfinite(players_xy[:, slot_team == t, 0]).sum(axis=1) for t in ("H", "A")]
    return (f"  {name}: {len(ball_xy)} frames, ball seen {seen[graded].mean():.1%}"
            f" (of {label}), homography valid "
            f"{np.mean(valid):.1%}, players tracked per frame: home median "
            f"{np.median(per_team[0][graded]):.0f}, away median {np.median(per_team[1][graded]):.0f}")


def report(tac, bro_ball, bro_players, bro_slots, bro_valid, pitch):
    fps = float(tac.fps)
    graded = np.asarray(tac.period) > 0
    lines = ["recordings:",
             describe("tactical", tac.ball_xy, tac.players_xy, tac.slot_team,
                      tac.homography_valid if tac.homography_valid is not None
                      else np.ones(tac.n_frames, bool), graded),
             describe("broadcast", bro_ball, bro_players, bro_slots, bro_valid)]

    matched, n_runs, rejected, runs = sync(tac, bro_ball, bro_players, bro_slots, bro_valid,
                                           pitch)
    extra = borrow(tac, bro_players, bro_slots, runs, matched, pitch)
    bro_total = len(bro_ball)
    covered = sum(b - a for a, b, *_ in matched)
    lines += ["sync (ball tracks placed, players checked):",
              (f"  stretches >= {MIN_STRETCH_S:.0f} s: {n_runs}; matched {len(matched)}, "
               f"rejected {rejected}"),
              f"  broadcast frames in matched stretches: {covered} ({covered / bro_total:.1%})",
              (f"  second pass (offset borrowed from a neighbour, players agree): "
               f"{len(extra)} more stretches, {sum(b - a for a, b, *_ in extra)} frames; "
               f"player distance (m) {_q([m[3] for m in extra])}"),
              (f"  matched stretches: player distance (m) {_q([m[3] for m in matched])}; "
               f"length (s) {_q([(m[1] - m[0]) / fps for m in matched])}; "
               f"flipped orientation {sum(m[4] for m in matched)} of {len(matched)}")]
    if matched:
        offsets = np.array([m[2] for m in matched]) / fps
        steps = np.diff(offsets[np.argsort([m[0] for m in matched])])
        lines.append(f"  offset changes between consecutive stretches (s): {_q(np.abs(steps))}")

    lines.append("ball (first pass only, then with the second pass):")
    lines += ball_lines(tac, bro_ball, matched, pitch, graded)
    lines += ball_lines(tac, bro_ball, matched + extra, pitch, graded)
    return "\n".join(lines)


def ball_lines(tac, bro_ball, matched, pitch, graded):
    # broadcast ball onto tactical frames
    filled = np.full((tac.n_frames, 2), np.nan)
    hits = np.zeros(tac.n_frames, dtype=int)
    covered = np.zeros(tac.n_frames, dtype=bool)
    for first, stop, offset, _rms, flipped in matched:
        frames = np.arange(first, stop)
        target = frames + offset
        ok = (target >= 0) & (target < tac.n_frames)
        covered[target[ok]] = True
        ball = bro_ball[frames[ok]].copy()
        if flipped:
            ball[:, 0] = pitch[0] - ball[:, 0]
            ball[:, 1] = pitch[1] - ball[:, 1]
        have = np.isfinite(ball[:, 0])
        filled[target[ok][have]] = ball[have]
        hits[target[ok][have]] += 1
    tac_seen = np.isfinite(tac.ball_xy[:, 0])
    bro_seen = np.isfinite(filled[:, 0])
    both = graded & tac_seen & bro_seen
    gap = graded & ~tac_seen
    d = np.hypot(*(tac.ball_xy[both] - filled[both]).T)
    agree = (f"  where both see the ball ({int(both.sum())} frames): distance (m) {_q(d)}; "
             f"<= 2 m {np.mean(d <= 2):.0%}, <= 5 m {np.mean(d <= 5):.0%}") if both.any() \
        else "  no frame where both see the ball"
    in_cover = gap & covered
    return [(f"  tactical graded frames covered by a matched broadcast frame with a ball: "
               f"{np.mean(bro_seen[graded]):.1%}"),
              agree,
              (f"  tactical gap frames (graded, raw ball missing): {int(gap.sum())}; "
               f"filled by broadcast {int((gap & bro_seen).sum())} "
               f"({(gap & bro_seen).sum() / max(1, gap.sum()):.1%}); of the gap frames inside "
               f"matched stretches ({int(in_cover.sum())}): "
               f"{(in_cover & bro_seen).sum() / max(1, in_cover.sum()):.1%}"),
            (f"  tactical frames hit by more than one broadcast stretch (replays): "
             f"{int((hits > 1).sum())}")]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Broadcast vs tactical sync check, aggregates.")
    parser.parse_args(argv)
    tac = load_match()                                # $MTAH_DATA_DIR, outside the repo (D1)
    print(report(tac, *load_broadcast()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())