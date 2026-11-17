# ws:ball: does the tactical camera follow the ball closely enough to help inside ball gaps?
#
# WHAT: the tactical camera pans with play, and its homography (one matrix per frame, valid on
#   every frame) says where it points. Two "camera aim" points per frame, in metres:
#     centre    the pitch point at the centre of the picture (960, 540)
#     visible   the middle of the part of the pitch that is in the picture (a grid of pitch points,
#               kept where they land inside the 1920 x 1080 image)
#   and, to compare, the players' centre (the mean of every tracked player), which the models
#   already use. On graded frames where the RAW ball is seen, prints how far each point is from the
#   ball: quantiles and the share within 5 / 10 / 20 m, plus how often the camera beats the
#   players' centre. --shift tries moving the camera aim earlier / later (the camera lags play).
#   The camera only helps the gaps if it is clearly closer to the ball than the players' centre.
# HOW: humans only, it reads the match (like ball_eda.py):
#   $env:MTAH_DATA_DIR = "C:\...\mtah-data"
#   .venv\Scripts\python scripts\camera_aim_check.py
# Aggregates only: distances in metres, no frame numbers, no positions. Writes no files.
# Label-free: it uses the tracking and the homography, never the possession truth.

import argparse
import warnings

import numpy as np

from falafelona.io import load_match

IMAGE_W, IMAGE_H = 1920.0, 1080.0
GRID = (43, 28)                 # pitch points tested for "visible": every 2.5 m or so
SHIFTS_S = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
QUANTILES = (0.25, 0.5, 0.75, 0.9)
WITHIN_M = (5.0, 10.0, 20.0)


def _apply(matrices, x, y):
    """Points through per-frame 3x3 matrices: matrices[n, 3, 3], x / y [n] or [n, k]."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    h = matrices
    if x.ndim == 2:
        h = matrices[:, None, :, :]
    den = h[..., 2, 0] * x + h[..., 2, 1] * y + h[..., 2, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((h[..., 0, 0] * x + h[..., 0, 1] * y + h[..., 0, 2]) / den,
                (h[..., 1, 0] * x + h[..., 1, 1] * y + h[..., 1, 2]) / den)


def camera_points(homography, valid, pitch):
    """(centre_xy, visible_xy) float[n, 2] in metres; NaN where there is no matrix."""
    n = len(homography)
    length, width = pitch
    centre = np.full((n, 2), np.nan)
    visible = np.full((n, 2), np.nan)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(homography).all(axis=(1, 2))
    if not ok.any():
        return centre, visible
    h = homography[ok]
    with np.errstate(invalid="ignore"):
        inverse = np.linalg.inv(h)
    cx, cy = _apply(inverse, np.full(len(h), IMAGE_W / 2), np.full(len(h), IMAGE_H / 2))
    inside = (cx >= 0) & (cx <= 1) & (cy >= 0) & (cy <= 1)
    centre[ok] = np.column_stack([np.where(inside, cx, np.nan) * length,
                                  np.where(inside, cy, np.nan) * width])
    gx, gy = np.meshgrid(np.linspace(0, 1, GRID[0]), np.linspace(0, 1, GRID[1]))
    gx, gy = gx.ravel(), gy.ravel()
    xs = np.broadcast_to(gx, (len(h), gx.size))
    ys = np.broadcast_to(gy, (len(h), gy.size))
    px, py = _apply(h, xs, ys)
    seen = (px >= 0) & (px <= IMAGE_W) & (py >= 0) & (py <= IMAGE_H)
    count = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = np.where(count > 0, (seen * gx).sum(axis=1) / count, np.nan)
        vy = np.where(count > 0, (seen * gy).sum(axis=1) / count, np.nan)
    visible[ok] = np.column_stack([vx * length, vy * width])
    return centre, visible


def players_centre(players_xy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(players_xy, axis=1)


def shifted(points, period, frames):
    """points[t + frames] inside the same half; NaN where that leaves the half."""
    n = len(points)
    index = np.arange(n) + frames
    ok = (index >= 0) & (index < n)
    out = np.full_like(points, np.nan)
    out[ok] = points[index[ok]]
    same = np.zeros(n, dtype=bool)
    same[ok] = period[index[ok]] == period[ok]
    out[~same] = np.nan
    return out


def summary(distance):
    d = distance[np.isfinite(distance)]
    if not d.size:
        return "no frames"
    q = ", ".join(f"p{round(100 * p)} {np.quantile(d, p):.1f}" for p in QUANTILES)
    w = ", ".join(f"<={m:.0f} m {np.mean(d <= m):.0%}" for m in WITHIN_M)
    return f"{q} m; {w} ({d.size} frames)"


def camera_aim_report(match):
    period = np.asarray(match.period)
    ball = np.asarray(match.ball_xy, dtype=float)
    seen = (period > 0) & np.isfinite(ball).all(axis=1)
    centre, visible = camera_points(match.homography, match.homography_valid, match.pitch)
    team = players_centre(match.players_xy)

    def dist(points):
        return np.where(seen, np.hypot(*(points - ball).T), np.nan)

    share = seen.sum() / max(1, (period > 0).sum())
    lines = [f"graded frames with the raw ball seen: {int(seen.sum())} ({share:.0%} of graded)",
             "distance to the ball (m):"]
    d_team = dist(team)
    for name, points in (("players' centre", team), ("camera centre", centre),
                         ("camera visible", visible)):
        lines.append(f"  {name:16s} {summary(dist(points))}")
    for name, points in (("camera centre", centre), ("camera visible", visible)):
        d = dist(points)
        both = np.isfinite(d) & np.isfinite(d_team)
        if both.any():
            lines.append(f"  {name} closer than the players' centre on "
                         f"{np.mean(d[both] < d_team[both]):.0%} of frames")
    lines.append("camera visible, moved in time (negative = earlier camera frame), median m:")
    fps = float(match.fps)
    parts = []
    for s in SHIFTS_S:
        d = dist(shifted(visible, period, round(s * fps)))
        d = d[np.isfinite(d)]
        parts.append(f"{s:+.1f} s {np.median(d):.1f}" if d.size else f"{s:+.1f} s -")
    lines.append("  " + ", ".join(parts))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Camera aim vs ball, aggregates only.")
    parser.add_argument("--n-frames", type=int, default=None, help="tests only: first N frames")
    args = parser.parse_args(argv)
    match = load_match(n_frames=args.n_frames)   # reads $MTAH_DATA_DIR, outside the repo (D1)
    if match.homography is None:
        raise SystemExit("this match has no homography")
    print(camera_aim_report(match))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
