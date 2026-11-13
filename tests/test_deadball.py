
import copy
import warnings

import numpy as np
import pytest

from falafelona import deadball, io, pipeline
from falafelona.config import load_config

# Read the real config files, so the tests break if a config stops working.
CONFIGS = io.vendor.REPO_ROOT / "configs"
BALL_MISSING = load_config(CONFIGS / "baseline.yaml")
OUT_OF_BOUNDS = load_config(CONFIGS / "exp" / "dead-out-of-bounds.yaml")
PLAYERS_STILL = load_config(CONFIGS / "exp" / "dead-players-still.yaml")
ALL_CLUES = load_config(CONFIGS / "exp" / "dead-combined.yaml")
HMM_PAIR = load_config(CONFIGS / "exp" / "poss-hmm-margin-dead-combined.yaml")
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


def test_before_kickoff_marks_the_second_half_wait_as_dead(bundle):
    # The fake match has no graded halves (period is all 0), so we make our own.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    kickoff = next(tag.start for tag in match.tags
                   if tag.type == "Kick Off (Start)" and tag.half == 2)
    graded_from = kickoff - 50                  # the second half is graded 2 s before its kickoff
    match.period[:graded_from - 100] = 1        # first half
    match.period[graded_from:] = 2              # second half
    score = deadball.detect(match, None, {"deadball": {"method": "before_kickoff"}})

    # one score per frame
    assert len(score) == match.n_frames
    # graded but before the kickoff: dead
    assert (score[graded_from:kickoff] == 1).all()
    # from the kickoff on: alive
    assert (score[kickoff:] == 0).all()
    # the first half and the break are not touched
    assert (score[:graded_from] == 0).all()


def test_before_kickoff_without_a_tag_gives_no_evidence(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    match.period[:] = 2
    match.tags = []
    score = deadball.detect(match, None, {"deadball": {"method": "before_kickoff"}})
    assert (score == 0).all()


def test_before_restart_marks_the_wait_before_a_throw_in(bundle):
    # The fake match has one throw-in tag. Kickoff tags are not in the list, so they are ignored.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    throw_in = next(tag.start for tag in match.tags if tag.type == "Throw-in")
    cfg = {"deadball": {"method": "before_restart", "restart_seconds": {"throw-in": 2}}}
    score = deadball.detect(match, None, cfg)
    wait = 2 * 25                               # 2 s at 25 frames per second

    # one score per frame
    assert len(score) == match.n_frames
    # the 2 s before the throw: dead
    assert (score[throw_in - wait:throw_in] == 1).all()
    # just before that wait, and from the throw on: alive
    assert score[throw_in - wait - 1] == 0
    assert score[throw_in] == 0
    # nothing else is dead (the kickoff tags are ignored)
    assert score.sum() == wait


def test_unknown_method_is_rejected(bundle):
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    with pytest.raises(ValueError):
        deadball.detect(match, None, {"deadball": {"method": "nope"}})

def test_out_of_bounds_uses_the_match_pitch(bundle):
    # Dev-set matches have their own pitch size. On a bigger pitch the fake "out" ball is inside.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    match.pitch = (120.0, 80.0)
    score = deadball.detect(match, None, OUT_OF_BOUNDS)
    out_start, out_end = bundle["ball_out"]
    assert (score[out_start:out_end] == 0).all()


def test_players_still_with_no_speeds_is_never_still_and_quiet(bundle):
    # If the speed field is empty, players_still must say "no evidence", not crash or warn.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    match.players_speed[:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        score = deadball.detect(match, None, PLAYERS_STILL)
    assert (score == 0).all()


def test_combined_without_tags_just_loses_the_tag_clues(bundle):
    # The dev set has no Smart Tagging: the tag clues give 0 and the rest works as usual.
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    match.tags = []
    no_tag_clues = copy.deepcopy(ALL_CLUES)
    for clue in ("before_kickoff", "before_restart"):
        del no_tag_clues["deadball"]["weights"][clue]

    with_tag_clues = deadball.detect(match, None, ALL_CLUES)
    without_them = deadball.detect(match, None, no_tag_clues)
    assert (with_tag_clues == without_them).all()


def _section(**changes):
    """The shipped combined section with some keys changed (None removes a key)."""
    section = copy.deepcopy(ALL_CLUES["deadball"])
    for key, value in changes.items():
        if value is None:
            del section[key]
        else:
            section[key] = value
    return section


@pytest.mark.parametrize("section, message", [
    (_section(method="nope"), "unknown deadball method"),
    (_section(weights={"ball_missingg": 0.3}), "is not a clue"),
    (_section(weights={"combined": 1.0}), "is not a clue"),
    (_section(weights={"ball_missing": -0.3}), "must be a number >= 0"),
    (_section(still_seconds=None), "needs 'still_seconds'"),
    (_section(out_margin="1 m"), "must be a number >= 0"),
    (_section(restart_seconds={"throw in": 8}), "is not a restart type"),
    (_section(restart_seconds=None), "needs 'restart_seconds'"),
])
def test_config_mistakes_fail_with_a_clear_message(section, message):
    with pytest.raises(ValueError, match=message):
        deadball.check_config(section)


def test_shipped_dead_configs_pass_the_check():
    for cfg in (BALL_MISSING, OUT_OF_BOUNDS, PLAYERS_STILL, ALL_CLUES, HMM_PAIR):
        deadball.check_config(cfg["deadball"])


def test_hmm_with_the_combined_dead_score_runs_end_to_end(bundle, tmp_path):
    # The pair from the 13 Nov review: the HMM reads the combined score as a chance of dead ball.
    out = pipeline.run(HMM_PAIR, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert out["rows"] == bundle["n_frames"]
    assert ((out["dead_score"] >= 0) & (out["dead_score"] <= 1)).all()
    assert set(out["labels"]) <= {"H", "A", "D"}


def test_a_ball_gap_alone_leans_dead_in_the_hmm_pair():
    # Review of #38: the HMM calls D only above a dead score of 0.5. With ball_missing at 0.3 a
    # gap with no other clue scored 0.3, so the HMM never called a gap dead ball, and on our
    # match (player speeds possibly empty) no other clue would lift it. A gap alone must lean D.
    assert HMM_PAIR["deadball"]["weights"]["ball_missing"] > 0.5
