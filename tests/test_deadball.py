
import pytest

from falafelona import deadball, io
from falafelona.config import load_config

# Read the real config files, so the tests break if a config stops working.
CONFIGS = io.vendor.REPO_ROOT / "configs"
BALL_MISSING = load_config(CONFIGS / "baseline.yaml")
OUT_OF_BOUNDS = load_config(CONFIGS / "exp" / "dead-out-of-bounds.yaml")
PLAYERS_STILL = load_config(CONFIGS / "exp" / "dead-players-still.yaml")
COMBINED = {
    "deadball": {
        "method": "combined",
        "weights": {"out_of_bounds": 1.0, "players_still": 0.4, "ball_missing": 0.3},
        "out_margin": 1.0,
        "still_speed": 1.0,
        "still_seconds": 3,
    }
}


def test_ball_missing_marks_the_gap_as_dead(bundle):
    # `bundle` is Tala's fake match. It has a stretch where the ball is missing.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    score = deadball.detect(match, None, BALL_MISSING)
    gap_start, gap_end = bundle["ball_gap"]

    # check there is one score per frame       
    assert len(score) == match.n_frames
    # check the score is 1 inside the gap      
    assert (score[gap_start:gap_end] == 1).all()
    # check the score is 0 just before the gap 
    assert score[gap_start - 1] == 0


def test_out_of_bounds_marks_the_ball_past_the_line_as_dead(bundle):
    # The fake match puts the ball past the far goal line for a stretch of frames.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    score = deadball.detect(match, None, OUT_OF_BOUNDS)
    out_start, out_end = bundle["ball_out"]
    gap_start, gap_end = bundle["ball_gap"]

    # one score per frame
    assert len(score) == match.n_frames
    # the ball is past the line: dead
    assert (score[out_start:out_end] == 1).all()
    # the frame before is inside the pitch: alive
    assert score[out_start - 1] == 0
    # a missing ball is not evidence of out of bounds (ball_missing handles it)
    assert (score[gap_start:gap_end] == 0).all()


def test_players_still_marks_a_long_stoppage_but_not_a_short_slow_moment(bundle):
    # The fake match has every player at 1 m/s the whole time, so we make our own stoppage.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    match.players_speed *= 5.0              # everyone runs at 5 m/s (empty slots stay NaN)
    match.players_speed[100:200] *= 0.1     # 4 s stoppage at 0.5 m/s
    match.players_speed[300:325] *= 0.1     # 1 s slow moment in live play
    score = deadball.detect(match, None, PLAYERS_STILL)

    # one score per frame
    assert len(score) == match.n_frames
    # the long stoppage is dead
    assert (score[100:200] == 1).all()
    # the frames around it are alive
    assert score[99] == 0
    assert score[200] == 0
    # the short slow moment is too short: alive
    assert (score[300:325] == 0).all()


def test_combined_adds_up_the_clues(bundle):
    # Fixed weights here, so tuning the config later does not break the test.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    gap_start, gap_end = bundle["ball_gap"]
    out_start, out_end = bundle["ball_out"]
    match.players_speed *= 5.0                              # everyone runs at 5 m/s
    still_start, still_end = gap_start + 20, gap_start + 120
    match.players_speed[still_start:still_end] *= 0.1       # 4 s stoppage, starts inside the ball gap
    score = deadball.detect(match, None, COMBINED)

    # ball missing only: 0.3, not enough on its own
    assert score[gap_start:still_start] == pytest.approx(0.3)
    # ball missing and players still: 0.3 + 0.4, enough together
    assert score[still_start:gap_end] == pytest.approx(0.7)
    # players still but the ball is seen: 0.4, not enough on its own
    assert score[gap_end:still_end] == pytest.approx(0.4)
    # ball out: 1.0
    assert score[out_start:out_end] == pytest.approx(1.0)
    # no clue: 0
    assert score[still_end] == 0
    # never above 1
    assert score.max() <= 1.0


def test_unknown_method_is_rejected(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    with pytest.raises(ValueError):
        deadball.detect(match, None, {"deadball": {"method": "nope"}})