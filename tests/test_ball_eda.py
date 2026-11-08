# ws:ball: tests for scripts/ball_eda.py (task 2.1, team-board issue #13). Synthetic data only.
#
# The script describes the real ball track with numbers. Here it runs on made-up data where we
# know the right answer, so we can check the counting is correct before a human runs it on the
# real match. Two kinds of made-up data are used:
#   - make_match(): a tiny 35-frame match built by hand, with known gaps, one jump and one
#     ball outside the pitch. Its numbers are checked against a slow, simple loop (`naive`).
#   - the `bundle` fixture (tests/conftest.py): the 500-frame fake match from the loader
#     tests, with a ball gap on frames 100-139 and the ball outside the pitch on 250-264.

import importlib.util
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from falafelona import io
from falafelona.io import Match

# scripts/ is not a package, so load the script as a module
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ball_eda.py"
spec = importlib.util.spec_from_file_location("ball_eda", SCRIPT)
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)

PITCH = (105.0, 68.0)


def as_match(xy, period, speed=9.0):
    """Wrap ball positions in a Match with no players. The speed field is `speed` where seen."""
    n = len(xy)
    return Match(
        n_frames=n,
        fps=25.0,
        period=np.asarray(period, dtype=np.int8),
        ball_xy=xy,
        ball_speed=np.where(np.isnan(xy[:, 0]), np.nan, speed),
        players_xy=np.full((n, 0, 2), np.nan),
        players_speed=np.full((n, 0), np.nan),
        slot_team=np.array([], dtype="<U1"),
    )


def make_match():
    """A 35-frame match built by hand. The ball moves 0.1 m per frame (2.5 m/s at 25 fps).

    frames 0-14   half 1
    frames 15-19  not graded (ball placed 500 m away: must be ignored everywhere)
    frames 20-34  half 2
    Ball missing on frames 3-5 (gap of 3), 10-11 (gap of 2), 20-21 (gap of 2, at the start of
    half 2, so it touches the edge of the half) and 30 (gap of 1). On frame 25 the ball is 30 m
    off the path (a jump, so the moves into and out of that frame are very fast). On frame 2 the
    ball is outside the pitch (x < 0). The speed field says 9.0 on every seen frame.
    """
    n = 35
    xy = np.zeros((n, 2))
    xy[:, 0] = 0.1 * np.arange(n)
    xy[:, 1] = 0.5
    xy[15:20, 0] = 500.0
    xy[2, 0] = -0.5
    xy[25, 0] += 30.0
    for lo, hi in ((3, 6), (10, 12), (20, 22), (30, 31)):
        xy[lo:hi] = np.nan
    return as_match(xy, [1] * 15 + [0] * 5 + [2] * 15)


def match_with_gaps(*gaps, n=100):
    """One graded half of n frames, ball seen everywhere except the given (start, stop) ranges."""
    xy = np.zeros((n, 2))
    xy[:, 0] = 0.1 * np.arange(n)
    for lo, hi in gaps:
        xy[lo:hi] = np.nan
    return as_match(xy, [1] * n)


def naive(match, pitch):
    """The same numbers computed the slow, obvious way, with plain loops. Our reference."""
    gaps, moves, outside, seen_count = [], [], 0, 0
    for half in (1, 2):
        frames = [i for i in range(match.n_frames) if match.period[i] == half]
        run = 0
        for i in frames:
            if np.isnan(match.ball_xy[i, 0]):
                run += 1
            else:
                if run:
                    gaps.append(run)
                run = 0
                seen_count += 1
                x, y = match.ball_xy[i]
                outside += int(x < 0 or x > pitch[0] or y < 0 or y > pitch[1])
        if run:
            gaps.append(run)
        for a, b in itertools.pairwise(frames):
            if not (np.isnan(match.ball_xy[a, 0]) or np.isnan(match.ball_xy[b, 0])):
                dx, dy = match.ball_xy[b] - match.ball_xy[a]
                moves.append(float(np.hypot(dx, dy)) * match.fps)
    return {"gaps": sorted(gaps), "moves": moves, "outside": outside, "seen": seen_count}


# --- the hand-built match -------------------------------------------------------------------


def test_gap_lengths_and_missing_share_per_half():
    """Gaps of 3 and 2 in half 1, 1 inside half 2, and a gap of 2 at the start of half 2."""
    result = eda.ball_eda(make_match(), pitch=PITCH)
    inside, edge = result["gaps"]["inside"], result["gaps"]["at_edge"]

    assert result["halves"][1] == {"frames": 15, "missing": 5}
    assert result["halves"][2] == {"frames": 15, "missing": 3}
    assert result["missing_share"] == pytest.approx(8 / 30)

    # gaps that do not touch the edge of a half: 3, 2 and 1 frames (6 of the 8 missing frames)
    assert inside["count"] == 3
    assert inside["length_max"] == 3
    assert inside["length_quantiles"] == [1, 1, 2, 3, 3, 3]  # whole frames, not 1.15 etc.
    assert inside["bins"][0]["gaps"] == 3  # all three are 0.2 s (5 frames) or shorter
    assert inside["bins"][0]["share_of_missing"] == pytest.approx(6 / 8)
    assert sum(b["gaps"] for b in inside["bins"][1:]) == 0
    # the 2-frame gap at the very start of half 2 is reported on its own (2 of the 8 frames)
    assert edge["count"] == 1 and edge["frames"] == 2 and edge["length_max"] == 2
    assert edge["share_of_missing"] == pytest.approx(2 / 8)


def test_ungraded_frames_and_half_boundaries_are_ignored():
    """Frames 15-19 sit between the halves, with the ball far away. They must not count."""
    result = eda.ball_eda(make_match(), pitch=PITCH)

    assert result["graded_frames"] == 30  # 15 + 15, not 35
    # the 500 m placement would be "outside the pitch" and would give an enormous move
    assert result["outside_pitch"]["count"] == 1  # only frame 2
    assert result["speed_derived"]["max"] < 1000


def test_jumps_and_speeds_match_the_slow_reference():
    """Move counts, jump counts, outside-pitch count and speed quantiles equal the naive loop."""
    match = make_match()
    expected = naive(match, PITCH)
    result = eda.ball_eda(match, jump_speeds=(20.0, 40.0), pitch=PITCH)

    assert expected["gaps"] == [1, 2, 2, 3]  # the naive count of all gaps, edge ones included
    assert len(expected["moves"]) == 17  # 7 pairs in half 1, 10 in half 2
    assert result["jumps"]["pairs"] == 17
    for item in result["jumps"]["by_limit"]:
        want = sum(m > item["limit"] for m in expected["moves"])
        assert item["count"] == want
    # the jump on frame 25 makes exactly two very fast moves (into it and out of it)
    assert [i["count"] for i in result["jumps"]["by_limit"]] == [2, 2]
    assert result["jumps"]["by_limit"][0]["share"] == pytest.approx(2 / 17)

    assert result["outside_pitch"] == {
        "count": expected["outside"],
        "seen": expected["seen"],
        "share": expected["outside"] / expected["seen"],
    }
    want_q = [round(float(q), 2) for q in np.quantile(expected["moves"], eda.QUANTILES)]
    assert result["speed_derived"]["quantiles"] == want_q


def test_ratio_of_given_to_implied_speed_shows_the_unit():
    """Field says 9.0 and the ball really moves at 2.5 m/s, so the ratio is 3.6 (km/h-like)."""
    result = eda.ball_eda(make_match(), pitch=PITCH)

    assert result["speed_given"]["quantiles"] == [9.0] * 6
    assert result["median_ratio_given_to_derived"] == 3.6


# --- gaps: where they start and end, and which bin they fall in ------------------------------


def test_gap_runs_gives_start_and_end_of_each_gap():
    """The helper that finds gaps: [T F T T F T] has gaps at 0, at 2-3 and at 5."""
    starts, ends = eda.gap_runs(np.array([True, False, True, True, False, True]))
    assert list(starts) == [0, 2, 5] and list(ends) == [1, 4, 6]  # end = first index after

    starts, ends = eda.gap_runs(np.array([True, True, True]))  # a gap covering everything
    assert list(starts) == [0] and list(ends) == [3]
    for empty in (np.array([False, False]), np.array([], dtype=bool)):
        starts, ends = eda.gap_runs(empty)
        assert len(starts) == 0 and len(ends) == 0


def test_gap_at_the_start_or_end_of_a_half_is_reported_separately():
    """A gap that touches the first or last graded frame has a cut-off length: own line."""
    result = eda.ball_eda(match_with_gaps((0, 4), (50, 53), (96, 100)), pitch=PITCH)
    inside, edge = result["gaps"]["inside"], result["gaps"]["at_edge"]

    assert inside["count"] == 1 and inside["length_max"] == 3  # only frames 50-52
    assert edge["count"] == 2 and edge["frames"] == 8 and edge["length_max"] == 4
    assert result["missing_frames"] == 3 + 8


def test_gap_bins_follow_the_length_in_seconds():
    """At 25 fps: 5 frames = 0.2 s, 12 = 0.48 s, 13 = 0.52 s. A gap goes in the bin that holds it.

    The 13-frame gap is longer than 0.5 s, so it must NOT be counted in the "0.2-0.5 s" bin
    (rounding 0.5 s * 25 = 12.5 frames up to 13 would put it there).
    """
    match = match_with_gaps((10, 15), (30, 36), (50, 62), (70, 83))  # 5, 6, 12, 13 frames
    bins = {
        b["label"]: b["gaps"] for b in eda.ball_eda(match, pitch=PITCH)["gaps"]["inside"]["bins"]
    }

    assert bins["up to 0.2 s"] == 1  # 5 frames
    assert bins["0.2-0.5 s"] == 2  # 6 and 12 frames
    assert bins["0.5-1 s"] == 1  # 13 frames
    assert sum(bins.values()) == 4


def test_gap_bins_with_a_frame_rate_that_does_not_divide_evenly():
    """At 24 fps, 0.2 s is 4.8 frames. A 4-frame gap lasts 0.17 s, a 5-frame gap 0.21 s.

    So 4 frames is "up to 0.2 s" and 5 frames is "0.2-0.5 s". Rounding 4.8 to 5 would
    put both in the first bin, which is wrong for the 5-frame gap.
    """
    match = replace(match_with_gaps((10, 14), (30, 35)), fps=24.0)  # gaps of 4 and 5 frames
    bins = {
        b["label"]: b["gaps"] for b in eda.ball_eda(match, pitch=PITCH)["gaps"]["inside"]["bins"]
    }

    assert bins["up to 0.2 s"] == 1
    assert bins["0.2-0.5 s"] == 1


# --- edge cases ------------------------------------------------------------------------------


@pytest.mark.parametrize("seen", [False, True], ids=["ball never seen", "ball never missing"])
def test_extreme_cases_do_not_crash(seen):
    """No ball at all, or no gap at all: the report is still produced (with n/a where empty)."""
    match = make_match()
    if seen:
        match.ball_xy[:] = 0.1 * np.arange(match.n_frames)[:, None]
        match.ball_speed[:] = 9.0
    else:
        match.ball_xy[:] = np.nan
        match.ball_speed[:] = np.nan

    text = eda.format_report(eda.ball_eda(match, pitch=PITCH))

    assert "graded frames: 30" in text
    if seen:
        assert "gaps (runs of missing ball): 0 (0 inside a half, 0 at a half edge)" in text
    else:  # each half is one gap that touches both of its edges
        assert "gaps (runs of missing ball): 2 (0 inside a half, 2 at a half edge)" in text


def test_no_graded_frames_says_so():
    """A match with no graded frames (as in the 500-frame fake bundle) gives a clear message."""
    match = replace(make_match(), period=np.zeros(35, dtype=np.int8))
    text = eda.format_report(eda.ball_eda(match, pitch=PITCH))

    assert "no graded frames found" in text


def test_same_input_gives_the_same_numbers():
    """Running it twice gives identical results (runs must be repeatable)."""
    assert eda.ball_eda(make_match(), pitch=PITCH) == eda.ball_eda(make_match(), pitch=PITCH)


# --- the fake bundle, loaded the way the real match is loaded --------------------------------


@pytest.fixture
def bundle_match(bundle):
    """The 500-frame fake match with half 1 = frames 0-249 and half 2 = frames 250-499."""
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    return replace(match, period=np.array([1] * 250 + [2] * 250, dtype=np.int8))


def test_bundle_numbers(bundle, bundle_match):
    """The fake bundle has one 40-frame gap and 15 frames with the ball outside the pitch."""
    result = eda.ball_eda(bundle_match)  # pitch from the official games.py, like the real run
    inside = result["gaps"]["inside"]

    assert result["graded_frames"] == 500
    assert result["missing_frames"] == 40
    assert inside["count"] == 1 and inside["length_max"] == 40
    assert result["gaps"]["at_edge"]["count"] == 0
    # 40 frames = 1.6 s, so it falls in the "1-2 s" bin
    assert [b["gaps"] for b in inside["bins"]] == [0, 0, 0, 1, 0, 0]
    assert result["outside_pitch"]["count"] == 15
    assert result["outside_pitch"]["seen"] == 460
    # moves between consecutive seen frames: 99 + 109 in half 1 (the gap splits it), 249 in half 2
    assert result["jumps"]["pairs"] == 99 + 109 + 249


def test_output_is_small_and_shows_no_coordinates(bundle, bundle_match):
    """The report is a short block of totals. None of the ball positions may appear in it."""
    text = eda.format_report(eda.ball_eda(bundle_match))

    assert len(text.splitlines()) <= 40
    seen = bundle["ball_xy"][~np.isnan(bundle["ball_xy"][:, 0])]
    for x, y in seen[:50]:
        assert f"{x:.3f}" not in text and f"{y:.3f}" not in text


# --- running it from the command line --------------------------------------------------------


def test_main_prints_the_report(bundle, capsys):
    """main() loads the fake bundle through io.load_match and prints the report."""
    assert eda.main(["--n-frames", str(bundle["n_frames"])]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Ball track EDA, aggregates only")
    assert "no graded frames found" in out  # 500 frames are all before the first graded half


def test_main_with_graded_frames_and_custom_jump_limits(monkeypatch, capsys):
    """The whole path (options -> numbers -> text) on a match that has graded frames."""
    monkeypatch.setattr(eda, "load_match", lambda n_frames=None: make_match())

    assert eda.main(["--jump-speeds", "20", "1000"]) == 0

    out = capsys.readouterr().out
    assert "graded frames: 30, ball missing on 8 (26.7%)" in out
    assert "faster than 20 m/s: 2 (11.76%)" in out  # the two moves around the jump
    assert "faster than 1000 m/s: 0 (0.00%)" in out  # nothing moves that fast


def test_main_needs_the_data_folder_variable(monkeypatch):
    """Without MTAH_DATA_DIR the script stops with the loader's clear error (D1)."""
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)

    with pytest.raises(RuntimeError, match="not set"):
        eda.main([])