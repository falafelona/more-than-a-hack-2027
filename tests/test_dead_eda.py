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


def long_match():
    """10 minutes of graded play. Two 3 s gaps are each followed by a throw-in (stoppages); a
    third 3 s gap and a 30 s gap have no restart near them (lost in play). The ball is seen at
    the centre, except just before the first gap, where it is 1 m from the touchline."""
    n = 600 * 25
    ball = np.full((n, 2), [50.0, 30.0])
    ball[99 * 25:100 * 25] = [50.0, 1.0]     # near the touchline, then the ball is gone
    ball[100 * 25:103 * 25] = np.nan          # gap, then a throw-in at 103 s
    ball[300 * 25:303 * 25] = np.nan          # gap, then a throw-in at 303 s
    ball[450 * 25:453 * 25] = np.nan          # gap, no restart
    ball[500 * 25:530 * 25] = np.nan          # 30 s gap, no restart
    tags = [Fake(code="SET PIECES", type="Throw-in", team=None, start=103 * 25),
            Fake(code="SET PIECES", type="Throw-in", team=None, start=303 * 25)]
    return Fake(fps=25.0, pitch=(105.0, 68.0), period=np.ones(n, dtype=np.int8),
                ball_xy=ball, tags=tags)


def test_the_chance_level_comes_from_shifted_restarts():
    r = de.dead_eda(long_match())
    assert r["long_gaps"] == 4
    assert r["long_gaps_near_restart"] == 0.5               # the two gaps before a throw-in
    assert r["long_gaps_near_restart_by_chance"] == 0.0     # shifted by 60 or 120 s: none line up


def test_gap_frames_are_weighted_by_length():
    r = de.dead_eda(long_match())
    # 75 + 75 frames near a restart, out of 75 * 3 + 750 missing frames in long gaps
    assert r["long_gap_frames_near_restart"] == round(150 / 975, 2)
    assert r["long_gap_frames_near_restart_by_chance"] == 0.0


def test_gaps_are_split_by_length():
    bins = de.dead_eda(long_match())["long_gaps_by_length"]
    assert bins["2-5 s"]["gaps"] == 3 and bins["2-5 s"]["near_restart"] == round(2 / 3, 2)
    assert bins["5-20 s"]["gaps"] == 0 and bins["5-20 s"]["near_restart"] is None
    assert bins["over 20 s"]["gaps"] == 1 and bins["over 20 s"]["near_restart"] == 0.0
    assert bins["over 20 s"]["share_of_missing_frames"] == round(750 / 975, 2)


def test_last_seen_ball_near_a_line_against_its_base_rate():
    r = de.dead_eda(long_match())
    assert r["long_gaps_last_seen_near_line"] == 0.25        # only the first gap
    assert r["seen_ball_near_line"] < 0.01                    # 25 of ~14,000 seen frames


def test_a_chance_restart_is_counted_by_chance_too():
    """A restart 60 s after a gap: never near it for real, but the shift of -60 s lands on it."""
    match = long_match()
    match.tags = [Fake(code="SET PIECES", type="Throw-in", team=None, start=513 * 25)]
    r = de.dead_eda(match)
    assert r["long_gaps_near_restart"] == 0.25               # the 30 s gap (restart inside it)
    assert r["long_gaps_near_restart_by_chance"] > 0.0
