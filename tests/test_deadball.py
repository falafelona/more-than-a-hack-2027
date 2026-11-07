
import pytest

from falafelona import deadball, io

BALL_MISSING = {"deadball": {"method": "ball_missing"}}
OUT_OF_BOUNDS = {"deadball": {"method": "out_of_bounds", "out_margin": 1.0}}


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


def test_unknown_method_is_rejected(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    with pytest.raises(ValueError):
        deadball.detect(match, None, {"deadball": {"method": "nope"}})