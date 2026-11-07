
import pytest

from falafelona import deadball, io

CFG = {"deadball": {"method": "ball_missing"}}


def test_ball_missing_marks_the_gap_as_dead(bundle):
    # `bundle` is Tala's fake match. It has a stretch where the ball is missing.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    score = deadball.detect(match, None, CFG)
    gap_start, gap_end = bundle["ball_gap"]

    # check there is one score per frame       
    assert len(score) == match.n_frames
    # check the score is 1 inside the gap      
    assert (score[gap_start:gap_end] == 1).all()
    # check the score is 0 just before the gap 
    assert score[gap_start - 1] == 0

def test_unknown_method_is_rejected(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    with pytest.raises(ValueError):
        deadball.detect(match, None, {"deadball": {"method": "nope"}})