# ws:ball: tests for ball method `interpolate` (task 2.4, team-board issue #30).
# Synthetic data only: every ball here is a made-up path.
#
# What the method does: a short stretch of missing ball is filled with a straight line between
# the last ball seen before it and the first ball seen after it. A long stretch stays missing
# (NaN), so the dead-ball stage still treats it as "no ball".
#
# The issue's "done when" is "Synthetic gaps filled": see test_short_gaps_are_filled_...
#
# Most tests use a ball that walks in a straight line along the pitch at 0.3 m per frame
# (7.5 m/s at 25 fps), with a few frames blanked out (set to NaN). Because the walk is a
# straight line at a steady pace, the right answer for a blanked frame is the original position,
# so the tests can check the filled values exactly.
#
# The max_gap used in most tests is 0.5 s. At 25 fps that is 12 frames (0.5 * 25 = 12.5, and
# only whole frames count, so a gap of 12 frames is filled and a gap of 13 is not).

from dataclasses import replace

import numpy as np
import pytest

from falafelona import ball, deadball, io, vendor
from falafelona.ball import INTERPOLATED, OBSERVED, REJECTED, UNKNOWN, DataCleaner, clean
from falafelona.config import load_config
from falafelona.io import Match

CONFIGS = vendor.REPO_ROOT / "configs"
PITCH = vendor.games.GAMES[vendor.games.GAME]["pitch"]
NAN = float("nan")


def cfg(**ball_settings):
    """A config for method `interpolate`, with the given settings in the ball section."""
    return {"ball": {"method": "interpolate", **ball_settings}}


def walk(n, per_frame=0.3, start=(10.0, 30.0)):
    """A ball path of n frames: it moves per_frame metres along the pitch on every frame."""
    xy = np.zeros((n, 2))
    xy[:, 0] = start[0] + per_frame * np.arange(n)
    xy[:, 1] = start[1]
    return xy


def blank(xy, *gaps):
    """A copy of the path with the given (start, stop) ranges of frames set to NaN."""
    out = xy.copy()
    for lo, hi in gaps:
        out[lo:hi] = np.nan
    return out


def match_from(xy, period=None, fps=25.0, speed=5.0):
    """A Match around a ball path, with no players. `period` defaults to one half, all graded."""
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    period = np.ones(n, dtype=np.int8) if period is None else np.asarray(period, dtype=np.int8)
    speed = np.broadcast_to(np.asarray(speed, dtype=float), (n,)).copy()
    speed[np.isnan(xy[:, 0])] = np.nan
    return Match(
        n_frames=n,
        fps=fps,
        period=period,
        ball_xy=xy,
        ball_speed=speed,
        players_xy=np.full((n, 0, 2), np.nan),
        players_speed=np.full((n, 0), np.nan),
        slot_team=np.array([], dtype="<U1"),
        pitch=PITCH,
    )


def frames_with(track, source):
    """The frame numbers whose source is `source`."""
    return np.flatnonzero(track.source == source).tolist()


# ---------------------------------------------------------------------------------------------
# Filling short gaps
# ---------------------------------------------------------------------------------------------


def test_short_gaps_are_filled_with_the_path_the_ball_took():
    """Issue #30 "done when": synthetic gaps are filled. Gaps of 1, 5 and 12 frames."""
    truth = walk(100)
    gaps = [(10, 11), (30, 35), (60, 72)]
    match = match_from(blank(truth, *gaps))

    track = clean(match, cfg(max_gap=0.5))

    np.testing.assert_allclose(track.xy, truth)  # every frame is back on the true path
    filled = [frame for lo, hi in gaps for frame in range(lo, hi)]
    assert frames_with(track, INTERPOLATED) == filled
    assert track.source.dtype == np.int8
    assert (np.delete(track.source, filled) == OBSERVED).all()  # the seen frames are untouched


def test_the_filled_ball_is_spaced_evenly_in_time():
    """A gap of 3 frames between (0, 0) and (4, 8): the ball is a quarter of the way at a time."""
    xy = np.array([[0.0, 0.0], [NAN, NAN], [NAN, NAN], [NAN, NAN], [4.0, 8.0]])

    track = clean(match_from(xy), cfg(max_gap=1.0))

    expected = [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
    np.testing.assert_allclose(track.xy, expected)


@pytest.mark.parametrize(
    "gap, fps, filled",
    [
        (12, 25.0, True),  # 12 frames = 0.48 s <= 0.5 s
        (13, 25.0, False),  # 13 frames = 0.52 s > 0.5 s
        (25, 50.0, True),  # the same 0.5 s is 25 frames at 50 fps
        (26, 50.0, False),
    ],
)
def test_max_gap_is_in_seconds_so_it_depends_on_the_frame_rate(gap, fps, filled):
    xy = blank(walk(100), (40, 40 + gap))
    track = clean(match_from(xy, fps=fps), cfg(max_gap=0.5))

    expected = list(range(40, 40 + gap)) if filled else []
    assert frames_with(track, INTERPOLATED) == expected


def test_a_long_gap_stays_missing_and_unknown():
    """Long gaps are left alone, so later stages still see "no ball" there."""
    truth = walk(200)
    xy = blank(truth, (20, 24), (60, 120))  # a short gap (4 frames) and a long one (60 frames)

    track = clean(match_from(xy), cfg(max_gap=0.5))

    assert frames_with(track, INTERPOLATED) == [20, 21, 22, 23]
    assert np.isnan(track.xy[60:120]).all()
    assert (track.source[60:120] == UNKNOWN).all()


@pytest.mark.parametrize(
    "gap",
    [(0, 5), (95, 100)],
    ids=["gap at the start of the data", "gap at the end of the data"],
)
def test_a_gap_with_no_ball_on_one_side_is_not_filled(gap):
    """The ball is only known on one side, so a line between two points cannot be drawn."""
    track = clean(match_from(blank(walk(100), gap)), cfg(max_gap=1.0))

    assert frames_with(track, INTERPOLATED) == []
    assert np.isnan(track.xy[gap[0] : gap[1]]).all()


def test_the_ball_is_never_filled_across_the_edge_of_a_half():
    """Half 1, a stretch outside the halves (period 0), then half 2. Gaps may not cross these."""
    period = [1] * 40 + [0] * 20 + [2] * 40
    truth = walk(100)
    xy = blank(
        truth,
        (38, 42),  # runs from half 1 into the stretch outside the halves
        (58, 62),  # runs from outside the halves into half 2
        (20, 24),  # inside half 1: fine
        (80, 84),  # inside half 2: fine
    )

    track = clean(match_from(xy, period=period), cfg(max_gap=1.0))

    assert frames_with(track, INTERPOLATED) == [20, 21, 22, 23, 80, 81, 82, 83]


def test_a_gap_between_two_halves_is_not_filled_even_if_it_is_short():
    """Half 1 ends and half 2 starts with 4 missing frames in between: do not draw a line."""
    period = [1] * 50 + [2] * 50
    track = clean(match_from(blank(walk(100), (48, 52)), period=period), cfg(max_gap=1.0))

    assert frames_with(track, INTERPOLATED) == []


def test_a_gap_outside_the_graded_halves_is_not_filled():
    """Frames with period 0 are not graded, so there is no point filling them."""
    period = [0] * 100
    track = clean(match_from(blank(walk(100), (40, 44)), period=period), cfg(max_gap=1.0))

    assert frames_with(track, INTERPOLATED) == []


def test_two_gaps_next_to_each_other_are_filled_separately():
    """One seen frame between two gaps: each gap is filled from its own two ends."""
    truth = walk(60)
    track = clean(match_from(blank(truth, (20, 23), (24, 27))), cfg(max_gap=0.5))

    np.testing.assert_allclose(track.xy, truth)
    assert frames_with(track, INTERPOLATED) == [20, 21, 22, 24, 25, 26]


def test_a_track_with_no_gaps_comes_back_unchanged():
    truth = walk(50)
    track = clean(match_from(truth), cfg(max_gap=1.0))

    np.testing.assert_array_equal(track.xy, truth)
    assert (track.source == OBSERVED).all()


def test_a_track_with_no_ball_at_all_comes_back_unchanged():
    track = clean(match_from(np.full((50, 2), np.nan)), cfg(max_gap=1.0))

    assert np.isnan(track.xy).all()
    assert (track.source == UNKNOWN).all()


# ---------------------------------------------------------------------------------------------
# The speed guard (max_speed): do not draw a path the ball cannot have taken
# ---------------------------------------------------------------------------------------------


def test_a_gap_is_not_filled_when_the_ball_could_not_have_crossed_it():
    """4 frames gap = 5 steps = 0.2 s. At 40 m/s the ball covers 8 m. The ends are 20 m apart."""
    xy = walk(40)
    xy[20:] += (20.0, 0.0)  # the ball is 20 m further on after the gap
    xy = blank(xy, (20, 24))

    guarded = clean(match_from(xy), cfg(max_gap=1.0, max_speed=40.0, speed_slack=0.0))
    unguarded = clean(match_from(xy), cfg(max_gap=1.0))

    assert frames_with(guarded, INTERPOLATED) == []
    assert np.isnan(guarded.xy[20:24]).all()
    assert frames_with(unguarded, INTERPOLATED) == [20, 21, 22, 23]  # no guard: fills anyway


def test_the_speed_guard_uses_metres_per_second_so_the_time_of_the_gap_counts():
    """The same 12 m jump is possible over 0.8 s (a long gap) and impossible over 0.2 s."""
    short = blank(walk(60), (20, 24))
    short[24:] += (12.0, 0.0)
    long = blank(walk(60), (20, 39))  # 19 frames, 20 steps = 0.8 s
    long[39:] += (12.0, 0.0)

    settings = cfg(
        max_gap=1.0, max_speed=40.0, speed_slack=0.0
    )  # 40 m/s: 8 m in 0.2 s, 32 m in 0.8 s

    assert frames_with(clean(match_from(short), settings), INTERPOLATED) == []
    assert len(frames_with(clean(match_from(long), settings), INTERPOLATED)) == 19


def test_the_speed_slack_is_extra_room_for_tracking_noise():
    """Ends 8.5 m apart over 0.2 s at 40 m/s (8 m): refused with no slack, allowed with 1 m."""
    xy = blank(walk(40, per_frame=0.0), (20, 24))
    xy[24:] += (8.5, 0.0)

    no_slack = clean(match_from(xy), cfg(max_gap=1.0, max_speed=40.0, speed_slack=0.0))
    some_slack = clean(match_from(xy), cfg(max_gap=1.0, max_speed=40.0, speed_slack=1.0))

    assert frames_with(no_slack, INTERPOLATED) == []
    assert frames_with(some_slack, INTERPOLATED) == [20, 21, 22, 23]


def test_a_gap_whose_ends_are_exactly_at_the_speed_limit_is_filled():
    """The limit is "no more than": exactly 8 m in 0.2 s at 40 m/s is allowed."""
    xy = blank(walk(40, per_frame=0.0), (20, 24))
    xy[24:] += (8.0, 0.0)

    track = clean(match_from(xy), cfg(max_gap=1.0, max_speed=40.0, speed_slack=0.0))

    assert frames_with(track, INTERPOLATED) == [20, 21, 22, 23]


# ---------------------------------------------------------------------------------------------
# The speed field
# ---------------------------------------------------------------------------------------------


def test_the_speed_field_is_filled_as_a_line_between_the_two_ends():
    """The ATD speed field is a number per frame. Gap frames get a value between the two ends."""
    xy = blank(walk(10), (3, 6))
    speed = np.array([0.0, 0.0, 0.0, NAN, NAN, NAN, 8.0, 8.0, 8.0, 8.0])
    speed[:3] = [2.0, 2.0, 2.0]
    match = replace(match_from(xy), ball_speed=speed)

    track = clean(match, cfg(max_gap=1.0))

    np.testing.assert_allclose(track.speed[3:6], [3.5, 5.0, 6.5])  # 2 -> 8 in four steps
    np.testing.assert_array_equal(track.speed[:3], speed[:3])  # seen frames keep their own


def test_a_filled_frame_has_no_speed_if_one_end_has_none():
    xy = blank(walk(10), (3, 6))
    speed = np.full(10, 5.0)
    speed[2] = np.nan  # the last seen frame before the gap has a position but no speed
    speed[3:6] = np.nan
    match = replace(match_from(xy), ball_speed=speed)

    track = clean(match, cfg(max_gap=1.0))

    assert np.isnan(track.speed[3:6]).all()
    assert (track.source[3:6] == INTERPOLATED).all()  # the position is still filled


# ---------------------------------------------------------------------------------------------
# Running `reject` first (reject_first)
# ---------------------------------------------------------------------------------------------

REJECT_SPEED = {"max_speed": 40.0, "speed_slack": 0.0, "max_false_run": 2.0}


def with_one_false_ball():
    """A steady walk with a single fake position on frame 50, 30 m off the path."""
    truth = walk(100)
    xy = truth.copy()
    xy[50] = (80.0, 5.0)
    return truth, xy


def test_by_default_a_false_ball_is_left_alone():
    """Without reject_first this method only fills gaps. It does not judge the balls it sees."""
    _, xy = with_one_false_ball()

    track = clean(match_from(xy), cfg(max_gap=0.5, **REJECT_SPEED))

    assert frames_with(track, INTERPOLATED) == []
    assert frames_with(track, REJECTED) == []
    np.testing.assert_array_equal(track.xy[50], xy[50])


def test_reject_first_turns_a_false_ball_into_a_gap_and_fills_it():
    truth, xy = with_one_false_ball()

    track = clean(match_from(xy), cfg(max_gap=0.5, reject_first=True, **REJECT_SPEED))

    np.testing.assert_allclose(track.xy, truth)  # the fake is gone, the real path is back
    assert frames_with(track, INTERPOLATED) == [50]
    assert frames_with(track, REJECTED) == []


def test_a_rejected_stretch_longer_than_max_gap_stays_rejected():
    """Six false frames in a row (0.24 s) with max_gap 0.1 s: too long to fill, so stay NaN."""
    truth = walk(100)
    xy = truth.copy()
    xy[50:56] = (80.0, 5.0)

    track = clean(match_from(xy), cfg(max_gap=0.1, reject_first=True, **REJECT_SPEED))

    assert frames_with(track, REJECTED) == [50, 51, 52, 53, 54, 55]
    assert np.isnan(track.xy[50:56]).all()


def test_reject_first_also_fills_the_short_gaps_that_were_already_there():
    truth, xy = with_one_false_ball()
    xy[20:23] = np.nan

    track = clean(match_from(xy), cfg(max_gap=0.5, reject_first=True, **REJECT_SPEED))

    np.testing.assert_allclose(track.xy, truth)
    assert frames_with(track, INTERPOLATED) == [20, 21, 22, 50]


# ---------------------------------------------------------------------------------------------
# What the next stages get
# ---------------------------------------------------------------------------------------------


def test_datacleaner_gives_the_next_stage_a_match_with_the_gaps_filled():
    """The raw match is left as it was; the players are shared, not copied."""
    xy = blank(walk(100), (30, 34), (60, 90))  # a short gap and a long one
    raw = match_from(xy)
    before = xy.copy()

    out = DataCleaner(raw, cfg(max_gap=0.5)).return_cleaned_data()

    assert out is not raw
    assert not np.isnan(out.ball_xy[30:34]).any()  # filled
    assert np.isnan(out.ball_xy[60:90]).all()  # still missing
    np.testing.assert_array_equal(raw.ball_xy, before)  # the raw match did not change
    assert out.players_xy is raw.players_xy


def test_datacleaner_hands_back_the_same_match_when_no_gap_was_filled():
    raw = match_from(blank(walk(100), (20, 80)))  # the only gap is long

    assert DataCleaner(raw, cfg(max_gap=0.5)).return_cleaned_data() is raw


def test_filling_does_not_change_the_track_it_was_given():
    """The filling step returns a new track. The track that goes in keeps its gaps."""
    match = match_from(blank(walk(50), (10, 14)))
    track = ball._none(match, cfg())
    settings = ball._interpolate_settings(cfg(max_gap=0.5)["ball"])

    filled = ball._fill(track, match.period, match.fps, settings)

    assert np.isnan(track.xy[10:14]).all()  # still missing in the input
    assert (track.source[10:14] == UNKNOWN).all()
    assert (filled.source[10:14] == INTERPOLATED).all()


def test_the_same_input_always_gives_the_same_answer():
    xy = blank(walk(100), (10, 13), (40, 60))
    first = clean(match_from(xy), cfg(max_gap=0.5))
    second = clean(match_from(xy), cfg(max_gap=0.5))

    np.testing.assert_array_equal(first.xy, second.xy)
    np.testing.assert_array_equal(first.source, second.source)


def test_a_filled_gap_is_no_longer_dead_ball_for_the_ball_missing_deadball():
    """With deadball `ball_missing`, a short gap now scores 0 (alive) and a long gap still 1."""
    xy = blank(walk(200), (30, 34), (80, 140))
    cleaned = DataCleaner(match_from(xy), cfg(max_gap=0.5)).return_cleaned_data()

    score = deadball.detect(cleaned, None, {"deadball": {"method": "ball_missing"}})

    assert (score[30:34] == 0).all()
    assert (score[80:140] == 1).all()


# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ball_settings, text",
    [
        ({}, "max_gap"),
        ({"max_gap": 0.0}, "max_gap"),
        ({"max_gap": -0.5}, "max_gap"),
        ({"max_gap": float("nan")}, "max_gap"),
        ({"max_gap": float("inf")}, "max_gap"),
        ({"max_gap": 0.5, "max_speed": 40.0}, "speed_slack"),
        ({"max_gap": 0.5, "max_speed": 40.0, "speed_slack": -1.0}, "speed_slack"),
        ({"max_gap": 0.5, "max_speed": 0.0, "speed_slack": 0.0}, "max_speed"),
        ({"max_gap": 0.5, "max_speed": float("nan"), "speed_slack": 0.0}, "max_speed"),
        ({"max_gap": 0.5, "reject_first": "yes"}, "reject_first"),
        ({"max_gap": 0.5, "reject_first": 1}, "reject_first"),
        ({"max_gap": 0.5, "reject_first": True}, "does nothing"),  # no reject keys given
        (
            {"max_gap": 0.5, "reject_first": True, "max_speed": 40.0, "speed_slack": 0.0},
            "max_false_run",
        ),
    ],
)
def test_a_wrong_or_incomplete_config_fails_loudly(ball_settings, text):
    """A missing or silly setting must not quietly turn the filling off or make it wild."""
    match = match_from(walk(20))

    with pytest.raises(ValueError, match=text):
        clean(match, cfg(**ball_settings))


def test_the_experiment_config_loads_and_runs(bundle):
    """configs/exp/ball-interpolate.yaml is the real file: it must name this method and work."""
    config = load_config(CONFIGS / "exp" / "ball-interpolate.yaml")
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    gap_start, gap_stop = bundle["ball_gap"]  # the fake match has one 40-frame (1.6 s) gap

    track = clean(match, config)

    assert config["ball"]["method"] == "interpolate"
    assert "interpolate" in ball.METHODS
    # the gap is longer than max_gap in the config, so it must stay missing
    assert (track.source[gap_start:gap_stop] == UNKNOWN).all()
    assert (track.source[:gap_start] == OBSERVED).all()
    assert config["ball"]["max_gap"] * match.fps < gap_stop - gap_start


def test_baseline_keeps_the_untouched_ball_and_final_uses_interpolate():
    """baseline.yaml keeps `ball: method: none`. final.yaml uses interpolate with the same settings
    as configs/exp/ball-interpolate.yaml (dev-set decision, experiments/RESULTS.md)."""
    assert load_config(CONFIGS / "baseline.yaml")["ball"]["method"] == "none"
    final = load_config(CONFIGS / "final.yaml")["ball"]
    assert final == load_config(CONFIGS / "exp" / "ball-interpolate.yaml")["ball"]


# ---------------------------------------------------------------------------------------------
# Many random cases against a slow, obvious version
# ---------------------------------------------------------------------------------------------


def slow_fill(xy, period, fps, max_gap, max_speed=None, slack=0.0):
    """The same rule written the slow, obvious way, one gap at a time. Our reference.

    Returns (xy, filled): the new positions and the set of frames that were filled.
    """
    out = xy.copy()
    filled = set()
    n = len(xy)
    frame = 0
    while frame < n:
        if not np.isnan(xy[frame, 0]):
            frame += 1
            continue
        stop = frame
        while stop < n and np.isnan(xy[stop, 0]):
            stop += 1
        before, after = frame - 1, stop  # the seen frames on both sides of this gap
        ok = before >= 0 and after < n
        ok = ok and after - before - 1 <= int(np.floor(max_gap * fps + 1e-9))
        ok = ok and period[before] != 0 and all(period[before : after + 1] == period[before])
        if ok and max_speed is not None:
            distance = float(np.hypot(*(xy[after] - xy[before])))
            ok = distance <= max_speed * (after - before) / fps + slack
        if ok:
            for k in range(frame, stop):
                t = (k - before) / (after - before)
                out[k] = xy[before] * (1 - t) + xy[after] * t
                filled.add(k)
        frame = stop
    return out, filled


def random_case(rng):
    """A random path with random gaps, random halves, and (sometimes) big jumps."""
    n = int(rng.integers(30, 160))
    xy = np.cumsum(rng.normal(0.0, 0.4, (n, 2)), axis=0) + 50.0
    jumps = rng.random(n) < 0.05
    xy[jumps] += rng.normal(0.0, 15.0, (jumps.sum(), 2))
    for _ in range(int(rng.integers(0, 8))):
        lo = int(rng.integers(0, n))
        xy[lo : lo + int(rng.integers(1, 20))] = np.nan
    cut1, cut2 = sorted(rng.integers(0, n, 2))
    period = np.concatenate([np.ones(cut1), np.zeros(cut2 - cut1), np.full(n - cut2, 2)]).astype(
        np.int8
    )
    if rng.random() < 0.3:
        period[:] = 1
    return xy, period


@pytest.mark.parametrize("guard", [False, True], ids=["no speed guard", "with speed guard"])
def test_it_agrees_with_the_slow_version_on_random_cases(guard):
    rng = np.random.default_rng(7)
    fills = 0
    for _ in range(300):
        xy, period = random_case(rng)
        fps = float(rng.choice([25.0, 50.0]))
        max_gap = float(rng.choice([0.2, 0.4, 0.8]))
        settings = {"max_gap": max_gap}
        slow_args = {"max_speed": None, "slack": 0.0}
        if guard:
            settings |= {"max_speed": 30.0, "speed_slack": 0.5}
            slow_args = {"max_speed": 30.0, "slack": 0.5}

        track = clean(match_from(xy, period=period, fps=fps), cfg(**settings))
        want_xy, want_filled = slow_fill(xy, period, fps, max_gap, **slow_args)

        np.testing.assert_allclose(track.xy, want_xy)
        assert set(frames_with(track, INTERPOLATED)) == want_filled
        fills += len(want_filled)
    assert fills > 100  # the comparison actually filled things; it is not comparing empty answers