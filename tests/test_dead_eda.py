# ws:dead: tests for scripts/dead_eda.py (task 3.3). Synthetic data only.
#
# The match here is made up inside the test: no match data, no Smart Tagging file.



import importlib.util
from pathlib import Path
from types import SimpleNamespace as Fake

import numpy as np

# scripts/ is not a package, so load the script as a module
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dead_eda.py"
spec = importlib.util.spec_from_file_location("dead_eda", SCRIPT)
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)


def fake_match():
    """60 s of graded play: a 3 s ball gap before a throw-in at 20 s, the ball out of bounds
    before a corner at 40 s, and a long gap at 50 s with no restart (lost in play)."""
    n = 60 * 25
    ball = np.full((n, 2), [50.0, 30.0])
    ball[17 * 25:20 * 25] = np.nan          # 3 s gap, then the throw-in
    ball[38 * 25:40 * 25] = [110.0, 30.0]   # 2 s past the goal line, then the corner
    ball[50 * 25:53 * 25] = np.nan          # 3 s gap, no restart
    tags = [
        Fake(code="SET PIECES", type="Throw-in", team=None, start=20 * 25),
        Fake(code="SET PIECES", type="Corner Kick", team="H", start=40 * 25),
        Fake(code="BUILD UP", type=None, team="A", start=5 * 25),  # not a restart
    ]
    return Fake(fps=25.0, pitch=(105.0, 68.0), period=np.ones(n, dtype=np.int8),
                ball_xy=ball, tags=tags)


def test_runs():
    assert de.runs(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]


def test_counts_and_overlaps():
    r = de.dead_eda(fake_match())
    assert r["restarts"] == 2
    assert r["restarts_by_type"] == {"Throw-in": 1, "Corner Kick": 1}
    assert r["restarts_without_team"] == 1
    assert r["long_gaps"] == 2
    assert r["out_of_bounds_stretches"] == 1
    assert r["restarts_near_long_gap"] == 0.5        # the throw-in
    assert r["restarts_near_out_of_bounds"] == 0.5   # the corner
    assert r["restarts_near_either"] == 1.0
    assert r["long_gaps_near_restart"] == 0.5        # the gap at 50 s has no restart
    assert r["out_of_bounds_near_restart"] == 1.0


def test_ungraded_frames_are_ignored():
    match = fake_match()
    match.period[:] = 0
    r = de.dead_eda(match)
    assert r["restarts"] == 0 and r["long_gaps"] == 0
    assert r["restarts_near_long_gap"] is None