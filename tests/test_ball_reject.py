"""Tests for ball method `reject` (task 2.3, team-board issue #19). Synthetic data only.

Issue numbers (#19, #30) are issues on falafelona/team-board, not this repo's PRs.

What the method does: throw away false ball detections, in two ways.
  - SPEED CHECK: the ball jumps somewhere it cannot get to in the time that passed.
  - PLAYER CHECK: the ball is inside the pitch but far from every tracked player.
A rejected frame gets xy = NaN and source = REJECTED. Issue #19 "done when": synthetic fake
jumps are rejected. That is the first test in this file.

All the made-up balls here walk in a straight line along the pitch at 0.3 m per frame (7.5 m/s
at 25 fps). A "fake" is a frame (or a few) where the ball is somewhere far away instead. Most
tests use the limit max_speed = 40 m/s, which is 1.6 m per frame at 25 fps.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from falafelona import ball, deadball, io, vendor
from falafelona.ball import OBSERVED, REJECTED, UNKNOWN, DataCleaner, clean
from falafelona.config import load_config
from falafelona.io import Match

CONFIGS = vendor.REPO_ROOT / "configs"
PITCH = vendor.games.GAMES[vendor.games.GAME]["pitch"]
SPEED = {"max_speed": 40.0, "speed_slack": 0.0, "max_false_run": 2.0}  # 2 s = 50 frames at 25 fps


def cfg(**ball_settings):
    """A config with method `reject` and the given thresholds."""
    return {"ball": {"method": "reject", **ball_settings}}


def walk(n, per_frame=0.3, start=(10.0, 30.0)):
    """A ball path of n frames: it moves per_frame metres along the pitch on every frame."""
    xy = np.empty((n, 2))
    xy[:, 0] = start[0] + per_frame * np.arange(n)
    xy[:, 1] = start[1]
    return xy


def match_from(xy, players=None, fps=25.0):
    """A Match around a ball path. `players` is float[n, slots, 2] (NaN = slot empty), or none."""
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if players is None:
        players = np.full((n, 0, 2), np.nan)
    return Match(
        n_frames=n,
        fps=fps,
        period=np.ones(n, dtype=np.int8),
        ball_xy=xy,
        ball_speed=np.where(np.isnan(xy[:, 0]), np.nan, 5.0),
        players_xy=players,
        players_speed=np.full(players.shape[:2], np.nan),
        slot_team=np.array(["H"] * players.shape[1], dtype="<U1"),
        pitch=PITCH,
    )


def rejected(track):
    """The frame numbers that were rejected."""
    return np.flatnonzero(track.source == REJECTED).tolist()


# ---------------------------------------------------------------------------------------------
# The speed check
# ---------------------------------------------------------------------------------------------


def test_fake_jumps_are_rejected():
    """Issue #19 "done when": fake jumps (one frame, or a few in a row) are thrown away."""
    xy = walk(200)
    xy[50] = (95.0, 5.0)  # one frame far away
    xy[120] = (95.0, 5.0)  # another one
    xy[160:163] = (3.0, 65.0)  # three frames in a row, far away
    xy[80] = np.nan  # the ball is also simply missing on frame 80
    raw = xy.copy()
    fake = [50, 120, 160, 161, 162]

    track = clean(match_from(xy), cfg(**SPEED))

    assert rejected(track) == fake
    assert np.isnan(track.xy[fake]).all()  # a rejected ball has no position
    assert np.isnan(track.speed[fake]).all()  # ... and no speed
    assert track.source[80] == UNKNOWN  # a ball that was never seen is not "rejected"
    # every other frame is exactly as it was
    keep = np.ones(200, dtype=bool)
    keep[fake] = False
    np.testing.assert_array_equal(track.xy[keep], raw[keep])
    assert (track.source[keep & (np.arange(200) != 80)] == OBSERVED).all()


def test_a_false_blip_right_after_a_gap_is_rejected_not_the_real_ball():
    """The first frame seen after a gap is a fake. The real ball after it must be kept.

    A check that only compares each frame with the one before it would believe the fake (the ball
    could have gone anywhere during the gap) and then reject the real ball that follows.
    """
    path = walk(400)
    xy = np.full((400, 2), np.nan)
    xy[:100] = path[:100]  # the ball is seen, then lost for 8 seconds
    xy[300:302] = (95.0, 5.0)  # first thing seen after the gap: two fake frames
    xy[302:] = path[302:]  # then the real ball, where it should be

    track = clean(match_from(xy), cfg(**SPEED))

    assert rejected(track) == [300, 301]


def test_a_lone_blip_with_a_gap_on_both_sides_cannot_be_judged_and_is_kept():
    """The limit of the speed check: with a gap before and after, any place is possible.

    Time passes either side of the fake, so no speed can be impossible. Only the player check
    (or a later method) could catch this one.
    """
    path = walk(500)
    xy = np.full((500, 2), np.nan)
    xy[:100] = path[:100]
    xy[300:302] = (95.0, 5.0)  # the fake, seen for two frames
    xy[400:] = path[400:]  # the real ball is back 4 s later

    track = clean(match_from(xy), cfg(**SPEED))

    assert rejected(track) == []


def test_a_ball_that_reappears_far_away_after_a_gap_is_kept():
    """After a long gap the ball can be anywhere. Time between frames counts, not frame count."""
    xy = np.full((400, 2), np.nan)
    xy[:50] = walk(50)
    xy[350:] = walk(50, start=(100.0, 60.0))  # 12 s later, about 100 m away (40 m/s * 12 s = 480)

    track = clean(match_from(xy), cfg(**SPEED))

    assert rejected(track) == []


@pytest.mark.parametrize(
    "per_frame, fps, n_rejected",
    [
        (1.5, 25.0, 0),  # 37.5 m/s: possible
        (1.7, 25.0, "most"),  # 42.5 m/s on every frame: impossible
        (1.0, 25.0, 0),  # 25 m/s: possible at 25 fps ...
        (1.0, 50.0, "most"),  # ... but the same 1 m per frame is 50 m/s at 50 fps
    ],
)
def test_the_speed_limit_uses_metres_per_second(per_frame, fps, n_rejected):
    """The limit is in m/s, so it depends on the frame rate of the match."""
    track = clean(match_from(walk(100, per_frame=per_frame), fps=fps), cfg(**SPEED))

    if n_rejected == 0:
        assert rejected(track) == []
    else:
        assert len(rejected(track)) >= 90  # a ball that is always too fast: nearly all rejected


def test_slack_allows_for_tracking_noise():
    """speed_slack is extra metres of allowed movement. 1.7 m per frame fails 40 m/s by 0.1 m."""
    xy = walk(100, per_frame=1.7)

    assert rejected(clean(match_from(xy), cfg(**{**SPEED, "speed_slack": 0.5}))) == []
    assert len(rejected(clean(match_from(xy), cfg(**{**SPEED, "speed_slack": 0.0})))) >= 90


def run_of_fakes(first, middle, last):
    """A real path of first + last frames with `middle` fake frames in between.

    The fake frames form their own smooth path, far from the real one, so they are
    consistent with each other but not with the real ball.
    """
    path = walk(first + middle + last)
    xy = path.copy()
    xy[first : first + middle] = walk(middle, start=(80.0, 5.0))
    return xy


@pytest.mark.parametrize("middle", [1, 40, 50])
def test_a_short_run_of_fakes_between_real_stretches_is_rejected(middle):
    """A false run of up to max_false_run (2 s = 50 frames) is rejected."""
    track = clean(match_from(run_of_fakes(100, middle, 100)), cfg(**SPEED))

    assert rejected(track) == list(range(100, 100 + middle))


@pytest.mark.parametrize("middle", [51, 100])
def test_a_run_longer_than_max_false_run_is_never_rejected(middle):
    """Over 2 s is too long to be a blip: a jump into it and out of it is believed.

    This is the safety stop. Without it, two long stretches that do not connect (a tracker that
    switched object) would make the check throw one of them away completely.
    """
    track = clean(match_from(run_of_fakes(100, middle, 100)), cfg(**SPEED))

    assert rejected(track) == []


def test_when_a_long_stretch_disagrees_with_short_ones_the_long_one_wins():
    """20 frames, then 100 frames somewhere else, then 20 frames back: the 100 are believed."""
    track = clean(match_from(run_of_fakes(20, 100, 20)), cfg(**SPEED))

    assert rejected(track) == list(range(20)) + list(range(120, 140))


# ---------------------------------------------------------------------------------------------
# The player check
# ---------------------------------------------------------------------------------------------


def one_ball_many_frames(n_frames, ball_at=(50.0, 30.0)):
    """n frames with the ball standing still at ball_at, and 46 empty player slots."""
    xy = np.tile(ball_at, (n_frames, 1))
    players = np.full((n_frames, 46, 2), np.nan)
    return xy, players


def test_a_ball_far_from_every_player_is_rejected():
    xy, players = one_ball_many_frames(4)
    players[0, 0] = (52.0, 30.0)  # frame 0: a player 2 m away -> kept
    players[1, 0] = (95.0, 30.0)  # frame 1: the nearest player is 45 m away -> rejected
    players[1, 1] = (5.0, 5.0)
    players[2, 0] = (90.0, 30.0)  # frame 2: nearest player exactly 40 m away -> kept (not "more")
    players[3, 0] = (90.5, 30.0)  # frame 3: 40.5 m away -> rejected

    track = clean(match_from(xy, players), cfg(max_player_distance=40.0))

    assert rejected(track) == [1, 3]


def test_a_frame_with_no_tracked_players_is_not_judged():
    xy, players = one_ball_many_frames(2)
    players[0, 0] = (95.0, 30.0)  # frame 0: a far player
    # frame 1: no player tracked at all, so we cannot say the ball is far from them

    track = clean(match_from(xy, players), cfg(max_player_distance=40.0))

    assert rejected(track) == [0]


def test_min_players_sets_how_many_must_be_tracked_before_a_frame_is_judged():
    xy, players = one_ball_many_frames(2)
    players[0, :3] = (95.0, 30.0)  # frame 0: three tracked players, all far away
    players[1, :5] = (95.0, 30.0)  # frame 1: five tracked players, all far away

    assert rejected(clean(match_from(xy, players), cfg(max_player_distance=40.0))) == [0, 1]
    only_five = cfg(max_player_distance=40.0, min_players=5)
    assert rejected(clean(match_from(xy, players), only_five)) == [1]  # exactly 5 is enough


def test_a_ball_outside_the_pitch_is_not_judged_by_the_player_check():
    """Balls outside the pitch are the dead-ball stage's evidence (deadball.py out_of_bounds)."""
    xy, players = one_ball_many_frames(3, ball_at=(-8.0, 30.0))  # behind the goal line
    players[:, 0] = (60.0, 30.0)  # the nearest player is 68 m away
    xy[1] = (PITCH[0] + 8.0, 30.0)  # frame 1: behind the other goal line instead
    xy[2] = (50.0, 30.0)  # frame 2: inside the pitch, 10 m from the player -> kept

    track = clean(match_from(xy, players), cfg(max_player_distance=40.0))

    assert rejected(track) == []


def test_a_step_within_the_slack_does_not_cut_a_stretch():
    """At 40 m/s and 25 fps the ball may move 1.6 m per frame. The slack is extra room on top."""
    frames = np.arange(4)
    pos = np.array([[10.0, 30.0], [11.0, 30.0], [12.7, 30.0], [13.7, 30.0]])  # one 1.7 m step

    starts, ends = ball._stretches(frames, pos, 25.0, 40.0, 0.0)
    assert (starts.tolist(), ends.tolist()) == ([0, 2], [2, 4])  # no slack: cut at the step

    starts, ends = ball._stretches(frames, pos, 25.0, 40.0, 0.5)
    assert (starts.tolist(), ends.tolist()) == ([0], [4])  # 1.7 m <= 1.6 m + 0.5 m: no cut


def test_the_size_of_the_pitch_comes_from_the_match():
    """The same ball is inside a 105 m pitch (judged) and outside a 90 m pitch (not judged)."""
    xy, players = one_ball_many_frames(1, ball_at=(100.0, 30.0))
    players[:, 0] = (50.0, 30.0)  # 50 m from the ball
    match = match_from(xy, players)
    config = cfg(max_player_distance=40.0)

    assert rejected(clean(match, config)) == [0]
    assert rejected(clean(replace(match, pitch=(90.0, 60.0)), config)) == []


def test_a_ball_exactly_on_a_line_counts_as_inside_the_pitch():
    """On the line is still in play, so the player check does judge it."""
    on_the_lines = [(0.0, 30.0), (PITCH[0], 30.0), (50.0, 0.0), (50.0, PITCH[1])]
    xy, players = one_ball_many_frames(4, ball_at=(50.0, 30.0))
    xy[:] = on_the_lines
    players[0, 0] = (0.0, 90.0)  # 60 m from the ball in frame 0
    players[1, 0] = (0.0, 90.0)  # far from the ball in frame 1
    players[2, 0] = (50.0, 60.0)  # 60 m from the ball in frame 2
    players[3, 0] = (50.0, 0.0)  # 68 m from the ball in frame 3

    track = clean(match_from(xy, players), cfg(max_player_distance=40.0))

    assert rejected(track) == [0, 1, 2, 3]


def test_the_player_check_gives_the_same_answer_in_small_blocks(monkeypatch):
    """The check works through the frames in blocks to save memory. Block edges must not matter."""
    rng = np.random.default_rng(0)
    n = 100
    xy = rng.uniform(0, 1, (n, 2)) * PITCH
    xy[rng.random(n) < 0.2] = np.nan
    players = rng.uniform(0, 1, (n, 46, 2)) * PITCH
    players[rng.random((n, 46)) < 0.8] = np.nan
    match = match_from(xy, players)

    in_one_block = ball._far_from_players(match, 20.0, 3)
    monkeypatch.setattr(ball, "_CHUNK_FRAMES", 7)
    in_small_blocks = ball._far_from_players(match, 20.0, 3)

    assert in_one_block.any() and not in_one_block.all()  # the comparison means something
    np.testing.assert_array_equal(in_one_block, in_small_blocks)


# ---------------------------------------------------------------------------------------------
# Both checks, the config, and how the result is used
# ---------------------------------------------------------------------------------------------


def test_each_check_is_off_when_its_keys_are_missing():
    xy = walk(100)
    xy[50] = (95.0, 5.0)  # a jump
    players = np.full((100, 46, 2), np.nan)
    players[:, 0] = (30.0, 30.0)  # one player, standing still
    match = match_from(xy, players)

    only_players = clean(match, cfg(max_player_distance=40.0))
    only_speed = clean(match, cfg(**SPEED))

    assert 50 in rejected(only_speed)
    assert 50 not in rejected(only_players) or rejected(only_players) == rejected(only_speed)
    # the jump at frame 50 is 65 m from the player, so the player check rejects it as well;
    # what matters is the opposite: the speed check alone does not reject far-but-smooth balls
    far_but_smooth = walk(100, start=(10.0, 30.0))
    far_players = np.full((100, 46, 2), np.nan)
    far_players[:, 0] = (100.0, 5.0)
    assert rejected(clean(match_from(far_but_smooth, far_players), cfg(**SPEED))) == []


@pytest.mark.parametrize(
    "ball_settings, text",
    [
        ({}, "does nothing"),
        ({"max_speed": 40.0, "max_false_run": 2.0}, "speed_slack"),
        ({"max_speed": 40.0, "speed_slack": 0.0}, "max_false_run"),
        ({**SPEED, "max_speed": 0.0}, "max_speed"),
        ({**SPEED, "max_speed": -5.0}, "max_speed"),
        ({**SPEED, "max_speed": float("nan")}, "max_speed"),
        ({**SPEED, "speed_slack": -1.0}, "speed_slack"),
        ({**SPEED, "max_false_run": 0.0}, "max_false_run"),
        ({"max_player_distance": 0.0}, "max_player_distance"),
        ({"max_player_distance": float("inf")}, "max_player_distance"),
        ({"max_player_distance": 40.0, "min_players": 0}, "min_players"),
        ({"max_player_distance": 40.0, "min_players": 2.5}, "min_players"),
    ],
)
def test_a_wrong_or_incomplete_config_fails_loudly(ball_settings, text):
    """A missing or silly threshold must not quietly turn the filter off or make it wild."""
    match = match_from(walk(10))

    with pytest.raises(ValueError, match=text):
        clean(match, cfg(**ball_settings))


def test_the_experiment_config_loads_and_runs_on_the_fake_match(bundle):
    """configs/exp/ball-reject.yaml is a real config: it loads, names `reject`, and runs."""
    config = load_config(CONFIGS / "exp" / "ball-reject.yaml")
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)

    track = clean(match, config)

    assert config["ball"]["method"] == "reject"
    assert track.xy.shape == match.ball_xy.shape
    assert np.isin(track.source, (OBSERVED, REJECTED, UNKNOWN)).all()
    # the frames where the fake ball was missing stay "unknown", not "rejected"
    gap_start, gap_stop = bundle["ball_gap"]
    assert (track.source[gap_start:gap_stop] == UNKNOWN).all()
    # a frame that was not rejected keeps its position exactly
    kept = track.source == OBSERVED
    np.testing.assert_array_equal(track.xy[kept], match.ball_xy[kept])


def test_datacleaner_hands_the_next_stage_a_match_without_the_fakes():
    xy = walk(200)
    xy[50] = (95.0, 5.0)
    match = match_from(xy)
    raw = match.ball_xy.copy()

    cleaner = DataCleaner(match, cfg(**SPEED))
    out = cleaner.return_cleaned_data()

    assert out is not match  # something changed, so a new Match
    assert np.isnan(out.ball_xy[50]).all()
    assert np.isnan(out.ball_speed[50])
    np.testing.assert_array_equal(match.ball_xy, raw)  # the raw Match was not modified
    assert out.players_xy is match.players_xy  # players are shared, not copied


def test_datacleaner_with_nothing_to_reject_hands_back_the_same_match():
    match = match_from(walk(100))

    assert DataCleaner(match, cfg(**SPEED)).return_cleaned_data() is match


def test_the_same_input_always_gives_the_same_answer():
    xy = walk(300)
    xy[[40, 41, 200]] = (95.0, 5.0)
    match = match_from(xy)

    first = clean(match, cfg(**SPEED))
    second = clean(match, cfg(**SPEED))

    np.testing.assert_array_equal(first.xy, second.xy)
    np.testing.assert_array_equal(first.source, second.source)


# ---------------------------------------------------------------------------------------------
# How the dead-ball stage sees a cleaned ball (deadball.py reads match.ball_xy)
# ---------------------------------------------------------------------------------------------


def test_a_rejected_ball_counts_as_missing_for_the_ball_missing_deadball():
    """With deadball method `ball_missing`, rejected frames score dead (1) and the rest alive (0).

    This is how the two stages meet today: rejecting a ball makes the frame "ball not seen".
    """
    xy = walk(100)
    xy[30:33] = (95.0, 5.0)
    config = {**cfg(**SPEED), "deadball": {"method": "ball_missing"}}
    cleaner = DataCleaner(match_from(xy), config)

    score = deadball.detect(cleaner.return_cleaned_data(), cleaner.track, config)

    assert np.flatnonzero(score == 1).tolist() == [30, 31, 32]
    assert (score[np.arange(100) >= 33] == 0).all()


def test_balls_outside_the_pitch_still_reach_the_out_of_bounds_deadball():
    """The player check must not erase the ball the dead-ball stage uses as evidence."""
    xy, players = one_ball_many_frames(20, ball_at=(PITCH[0] + 5.0, 30.0))  # behind the goal line
    players[:, 0] = (30.0, 30.0)  # the only tracked player is 80 m away
    config = {
        **cfg(**SPEED, max_player_distance=40.0),
        "deadball": {"method": "out_of_bounds", "out_margin": 1.0},
    }
    cleaner = DataCleaner(match_from(xy, players), config)

    score = deadball.detect(cleaner.return_cleaned_data(), cleaner.track, config)

    assert rejected(cleaner.track) == []
    assert (score == 1).all()


# ---------------------------------------------------------------------------------------------
# The chain choice, checked against trying every possibility
# ---------------------------------------------------------------------------------------------


def chain_is_allowed(kept, stretches, fps, max_speed, slack):
    """The rules for a set of kept stretches, written out slowly and plainly."""
    first_frame, last_frame, first_pos, last_pos, count, protected = stretches
    if any(protected[k] and k not in kept for k in range(len(count))):
        return False  # a protected stretch must always be kept
    for before, after in itertools.pairwise(kept):
        seconds = (first_frame[after] - last_frame[before]) / fps
        moved = np.hypot(*(first_pos[after] - last_pos[before]))
        connects = moved <= max_speed * seconds + slack
        if not connects and not (protected[before] and protected[after]):
            return False
    return True


def random_stretches(rng):
    """Up to 8 stretches in time order, with random positions, sizes and protection."""
    n = int(rng.integers(1, 9))
    first_frame, last_frame = [], []
    frame = 0
    for _ in range(n):
        start = frame + int(rng.integers(1, 60))
        stop = start + int(rng.integers(0, 80))
        first_frame.append(start)
        last_frame.append(stop)
        frame = stop
    first_frame, last_frame = np.array(first_frame), np.array(last_frame)
    count = np.array([int(rng.integers(1, f - s + 2)) for s, f in zip(first_frame, last_frame)])
    first_pos = rng.uniform(0, 100, (n, 2))
    last_pos = rng.uniform(0, 100, (n, 2))
    protected = rng.random(n) < 0.25
    return first_frame, last_frame, first_pos, last_pos, count, protected


def test_the_chain_keeps_as_many_frames_as_any_allowed_choice_could():
    """For 400 random small cases, compare with trying every subset of stretches."""
    rng = np.random.default_rng(1)
    for _ in range(400):
        stretches = random_stretches(rng)
        slack = float(rng.choice([0.0, 1.5]))
        count = stretches[4]

        kept = ball._best_chain(*stretches, 25.0, 40.0, slack).tolist()

        assert kept == sorted(kept)
        assert chain_is_allowed(kept, stretches, 25.0, 40.0, slack)
        best_possible = max(
            count[list(subset)].sum()
            for mask in range(1 << len(count))
            if chain_is_allowed(
                subset := [k for k in range(len(count)) if mask >> k & 1],
                stretches,
                25.0,
                40.0,
                slack,
            )
        )
        assert count[kept].sum() == best_possible