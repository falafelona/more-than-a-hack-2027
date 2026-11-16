# ws:dead: tests for still_ball, a dead ball the tracker still SEES. Made-up tracks only.

import numpy as np
import pytest

from falafelona import deadball, io

FPS = 25.0
BALL_SPEED = {"half_window_s": 0.2, "min_points": 5, "min_each_side": 2}
SETTINGS = {"still_ball_speed": 0.5, "still_ball_seconds": 1.5, "ball_speed": BALL_SPEED}
EDGE = 6          # frames of slack at the edges: the fit window must see mostly still frames


def match_with_ball(xy):
    n = len(xy)
    return io.Match(
        n_frames=n, fps=FPS, period=np.ones(n, dtype=np.int8),
        ball_xy=np.asarray(xy, dtype=float), ball_speed=np.full(n, np.nan),
        players_xy=np.full((n, 46, 2), np.nan), players_speed=np.full((n, 46), np.nan),
        slot_team=np.array(["H"] * 23 + ["A"] * 23), pitch=(105.0, 68.0),
    )


def moving(n, speed, start=(20.0, 34.0)):
    """A ball rolling along x at `speed` m/s, with 1 cm jitter."""
    rng = np.random.default_rng(0)
    x = start[0] + speed * np.arange(n) / FPS
    xy = np.column_stack([x, np.full(n, start[1])])
    return xy + rng.normal(0.0, 0.01, size=xy.shape)


def still_score(xy):
    cfg = {"deadball": {"method": "still_ball", **SETTINGS}}
    return deadball.DeadBall(match_with_ball(xy), None, cfg).score()


def test_a_ball_lying_still_for_long_enough_is_dead():
    xy = np.concatenate([moving(100, 8.0, start=(28.0, 34.0)),     # rolls to x = 60
                         np.tile([60.0, 34.0], (100, 1)),            # lies still for 4 s
                         moving(100, 8.0, start=(60.0, 34.0))])
    score = still_score(xy)
    assert (score[100 + EDGE:200 - EDGE] == 1.0).all()
    assert (score[:100 - EDGE] == 0.0).all() and (score[200 + EDGE:] == 0.0).all()


def test_a_short_still_moment_in_play_is_not_dead():
    xy = np.concatenate([moving(100, 8.0, start=(-2.0, 34.0)),    # rolls to x = 30
                         np.tile([30.0, 34.0], (25, 1)),             # still for only 1 s
                         moving(100, 8.0, start=(30.0, 34.0))])
    assert (still_score(xy) == 0.0).all()


def test_a_dribbled_ball_is_not_still():
    assert (still_score(moving(300, 3.0)) == 0.0).all()


def test_a_missing_ball_is_never_still():
    assert (still_score(np.full((200, 2), np.nan)) == 0.0).all()


def test_still_ball_adds_its_weight_in_combined():
    xy = np.tile([52.5, 34.0], (200, 1))
    cfg = {"deadball": {"method": "combined", "weights": {"still_ball": 0.4}, **SETTINGS}}
    score = deadball.DeadBall(match_with_ball(xy), None, cfg).score()
    assert np.allclose(score[EDGE:-EDGE], 0.4)


@pytest.mark.parametrize("drop", ["still_ball_speed", "still_ball_seconds", "ball_speed"])
def test_missing_still_ball_settings_fail_clearly(drop):
    section = {"method": "still_ball", **SETTINGS}
    del section[drop]
    with pytest.raises(ValueError, match=drop):
        deadball.check_config(section, FPS)


def test_a_bad_ball_speed_block_fails_clearly():
    section = {"method": "still_ball", **SETTINGS, "ball_speed": {**BALL_SPEED, "min_points": 99}}
    with pytest.raises(ValueError, match="deadball.ball_speed"):
        deadball.check_config(section, FPS)




# --- still_ball_at_spot and still_ball_before_restart ----------------------------------------

SPOT = {"spot_line_m": 2.0, "spot_centre_m": 3.0, "goal_area_depth_m": 6.0,
        "goal_area_half_width_m": 10.0}


@pytest.mark.parametrize("point, at_spot", [
    ((50.0, 0.5), 1.0),     # on the touchline: throw-in
    ((104.5, 67.5), 1.0),   # corner
    ((4.0, 36.0), 1.0),     # goal area: goal kick
    ((52.5, 34.0), 1.0),    # centre spot: kickoff
    ((30.0, 20.0), 0.0),    # open play
    ((15.0, 34.0), 0.0),    # penalty area but outside the goal area
    ((np.nan, np.nan), 0.0),
])
def test_restart_spot_finds_where_restarts_are_taken(point, at_spot):
    xy = np.array([point])
    assert deadball.restart_spot(xy, (105.0, 68.0), SPOT)[0] == at_spot


def spot_score(xy, method="still_ball_at_spot", tags=()):
    cfg = {"deadball": {"method": method, **SETTINGS, **SPOT,
                        "restart_seconds": {"throw-in": 4}}}
    match = match_with_ball(xy)
    match.tags = list(tags)
    return deadball.DeadBall(match, None, cfg).score()


def test_a_still_ball_on_the_touchline_is_dead_and_in_open_play_is_not():
    assert (spot_score(np.tile([50.0, 0.5], (200, 1)))[EDGE:-EDGE] == 1.0).all()
    assert (spot_score(np.tile([30.0, 20.0], (200, 1))) == 0.0).all()


def test_a_still_ball_counts_only_before_a_restart_tag():
    throw_in = io.Event(start=150, end=151, code="SET PIECES", type="Throw-in", team=None, half=1)
    score = spot_score(np.tile([30.0, 20.0], (200, 1)), "still_ball_before_restart", [throw_in])
    assert (score[150 - 100 + 1:150] == 1.0).all()          # the 4 s before the tag
    assert (score[:150 - 100] == 0.0).all() and (score[150:] == 0.0).all()


def test_without_tags_still_ball_before_restart_gives_nothing():
    score = spot_score(np.tile([30.0, 20.0], (200, 1)), "still_ball_before_restart")
    assert (score == 0.0).all()


def test_the_spot_clue_needs_its_settings():
    section = {"method": "still_ball_at_spot", **SETTINGS}
    with pytest.raises(ValueError, match="spot_line_m"):
        deadball.check_config(section, FPS)
