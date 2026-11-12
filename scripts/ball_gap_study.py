"""Straight line or Kalman smoother for ball gaps? A study on SYNTHETIC ball paths (task 2.5).

No match data is read, and nothing here is a model of a real match. The generator makes a
football-like ball (passes along the ground, flying shots with little drag, dribbles with a
wandering direction), adds tracking noise, hides a stretch of known length and asks both fills
for the hidden positions. The truth is known, so the error is exact. Aggregates only.

What it answers:
  1. how far off is each fill, per gap length (q is tuned on one seed and scored on another)
  2. is the smoother's own uncertainty (sigma) honest, and does it rank the errors
It cannot say how possession labels change: that is the dev-set run (scripts/devset_eval.py).

    python scripts/ball_gap_study.py            # about half a minute
    python scripts/ball_gap_study.py --gaps 50  # quick look
"""

import argparse

import numpy as np

from falafelona import ball

FPS = 25.0
DT = 1.0 / FPS
PITCH = (105.0, 68.0)
CONTEXT = 100  # frames of ball seen on each side of a gap (4 s)
GAP_SECONDS = (0.2, 0.5, 1.0, 2.0, 3.0, 5.0)
KALMAN_Q = (1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
MEASUREMENT_NOISE = 0.3  # what the smoother is told; the simulated tracking noise is --noise


def simulate(n, rng):
    """A ball path of n frames, float[n, 2] metres, kept on the pitch."""
    xy = np.zeros((n, 2))
    pos = np.array([PITCH[0] / 2, PITCH[1] / 2])
    t = 0
    while t < n:
        kind = rng.choice(["ground", "air", "dribble"], p=[0.40, 0.25, 0.35])
        frames = int(max(5, rng.exponential(2.0 if kind == "dribble" else 2.2) * FPS))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        wiggle = 0.0
        if kind == "dribble":
            speed, decel, wiggle = rng.uniform(2.0, 6.5), 0.0, 0.9
        elif kind == "ground":
            speed, decel = rng.lognormal(np.log(11.0), 0.45), rng.uniform(1.5, 3.0)
        else:
            speed, decel = rng.lognormal(np.log(15.0), 0.4), rng.uniform(0.2, 0.6)
        speed = min(speed, 32.0)
        for k in range(min(frames, n - t)):
            angle += wiggle * rng.normal() * np.sqrt(DT)
            pos = pos + speed * DT * np.array([np.cos(angle), np.sin(angle)])
            for axis in (0, 1):  # bounce off the boundary
                if not 0.0 <= pos[axis] <= PITCH[axis]:
                    pos[axis] = np.clip(pos[axis], 0.0, PITCH[axis])
                    angle = np.pi - angle if axis == 0 else -angle
            speed = max(0.5, speed - decel * DT)
            xy[t + k] = pos
        t += frames
    return xy


def make_gaps(count, seconds, rng, noise):
    """`count` windows [CONTEXT seen, gap, CONTEXT seen]: (truth, seen-with-noise, gap frames)."""
    gap = round(seconds * FPS)
    size = 2 * CONTEXT + gap
    path = simulate(count * size, rng)
    windows = []
    for i in range(count):
        truth = path[i * size : (i + 1) * size]
        seen = truth + rng.normal(0.0, noise, truth.shape)
        seen[CONTEXT : CONTEXT + gap] = np.nan
        windows.append((truth, seen, gap))
    return windows


def fill_linear(seen, gap):
    """A straight line between the last ball seen before the gap and the first after it."""
    before, after = CONTEXT - 1, CONTEXT + gap
    t = (np.arange(CONTEXT, CONTEXT + gap) - before) / (after - before)
    return seen[before] * (1.0 - t)[:, None] + seen[after] * t[:, None]


def fill_kalman(seen, gap, q):
    """What method `kalman` puts in the gap, and its sigma (m) for every gap frame."""
    observed = ~np.isnan(seen[:, 0])
    est, sigma = ball._smooth_run(seen, observed, DT, q, MEASUREMENT_NOISE, 15.0)
    return est[CONTEXT : CONTEXT + gap], sigma[CONTEXT : CONTEXT + gap]


def errors(windows, fill):
    """Per window, the error in metres of every gap frame (a list of arrays)."""
    out = []
    for truth, seen, gap in windows:
        est = fill(seen, gap)
        est = est[0] if isinstance(est, tuple) else est
        out.append(np.hypot(*(est - truth[CONTEXT : CONTEXT + gap]).T))
    return out


def rmse(windows, fill):
    """Root mean square error per window, averaged over windows."""
    return float(np.mean([np.sqrt(np.mean(e**2)) for e in errors(windows, fill)]))


def build(count, seed, noise):
    rng = np.random.default_rng(seed)
    return {s: make_gaps(count, s, rng, noise) for s in GAP_SECONDS}


def spearman(a, b):
    rank = lambda x: np.argsort(np.argsort(x))
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def report(args):
    tune, test = build(args.gaps, args.tune_seed, args.noise), build(args.gaps, args.test_seed, args.noise)
    pooled = {
        q: np.mean([rmse(tune[s], lambda z, g, q=q: fill_kalman(z, g, q)) for s in GAP_SECONDS])
        for q in KALMAN_Q
    }
    best = min(pooled, key=pooled.get)
    print(f"synthetic ball, tracking noise {args.noise} m, {args.gaps} gaps per length, 25 fps")
    print(f"process_noise tuned on seed {args.tune_seed} (pooled RMSE, m): "
          + "  ".join(f"q={q}: {v:.2f}" for q, v in pooled.items()))
    print(f"-> q = {best}. Scored on seed {args.test_seed}:\n")
    print(f"{'gap':>5} {'linear':>8} {'q=1.6':>7} {'q=' + str(best):>7} {'gain':>6}   90th percentile, linear / kalman")
    for s in GAP_SECONDS:
        lin = rmse(test[s], fill_linear)
        k_low = rmse(test[s], lambda z, g: fill_kalman(z, g, 1.6))
        k_best = rmse(test[s], lambda z, g: fill_kalman(z, g, best))
        p_lin = np.percentile(np.concatenate(errors(test[s], fill_linear)), 90)
        p_kal = np.percentile(np.concatenate(errors(test[s], lambda z, g: fill_kalman(z, g, best))), 90)
        print(f"{s:>4}s {lin:8.2f} {k_low:7.2f} {k_best:7.2f} {100 * (lin - k_best) / lin:5.1f}%   {p_lin:5.2f} / {p_kal:5.2f}")

    print(f"\nIs sigma honest? (q = {best}; share of gap frames whose error is within k sigma)")
    print("a Gaussian 2-D error would give 39% / 86% / 99%")
    all_err, all_sigma = [], []
    for s in GAP_SECONDS:
        e = np.concatenate(errors(test[s], lambda z, g: fill_kalman(z, g, best)))
        sig = np.concatenate([fill_kalman(z, g, best)[1] for _, z, g in test[s]])
        all_err.append(e)
        all_sigma.append(sig)
        print(f"  gap {s:>4}s: {100 * np.mean(e <= sig):4.0f}% / {100 * np.mean(e <= 2 * sig):4.0f}% / {100 * np.mean(e <= 3 * sig):4.0f}%")
    e, sig = np.concatenate(all_err), np.concatenate(all_sigma)
    print(f"Spearman(sigma, error) over all gap frames: {spearman(sig, e):.2f}")
    print("\nError of the filled frames by sigma (all gaps pooled):")
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]:
        m = (sig >= lo) & (sig < hi)
        print(f"  sigma {lo:g}-{hi:g} m: {100 * m.mean():4.1f}% of frames, mean error {e[m].mean():5.2f} m, "
              f"90th percentile {np.percentile(e[m], 90):5.2f} m")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gaps", type=int, default=300, help="gaps per gap length")
    parser.add_argument("--noise", type=float, default=0.25, help="simulated tracking noise, m")
    parser.add_argument("--tune-seed", type=int, default=1)
    parser.add_argument("--test-seed", type=int, default=2)
    report(parser.parse_args(argv))


if __name__ == "__main__":
    main()
