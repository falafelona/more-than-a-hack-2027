import copy
import warnings

import numpy as np
import pytest

from falafelona import deadball, io, pipeline, velocity
from falafelona.config import load_config

# Read the real config files, so the tests break if a config stops working.
CONFIGS = io.vendor.REPO_ROOT / "configs"
BALL_MISSING = load_config(CONFIGS / "baseline.yaml")
OUT_OF_BOUNDS = load_config(CONFIGS / "exp" / "dead-out-of-bounds.yaml")
PLAYERS_STILL = load_config(CONFIGS / "exp" / "dead-players-still.yaml")
ALL_CLUES = load_config(CONFIGS / "exp" / "dead-combined.yaml")
HMM_PAIR = load_config(CONFIGS / "exp" / "poss-hmm-margin-dead-combined.yaml")
FPS = 25.0
# Player speed from positions: 5 frames each side, so a still stretch is found up to about 5
# frames late at its start and ends up to 5 frames early (the window must see mostly still).
SPEED = {"half_window_s": 0.2, "min_points": 5, "min_each_side": 2, "max_speed": 12.0}
EDGE = 6                                        # frames of slack at the edges of a still stretch
COMBINED = {
    "deadball": {
        "method": "combined",
        "weights": {"out_of_bounds": 1.0, "players_still": 0.4, "ball_missing": 0.3},
        "out_margin": 1.0,
        "still_speed": 1.0,
        "still_seconds": 3,
        "speed": SPEED,
    }
}


def made_up_match(n=500):
    """A match built here: one graded half, ball seen mid-pitch, 46 empty slots, no tags."""
    return io.Match(
        n_frames=n,
        fps=FPS,
        period=np.ones(n, dtype=np.int8),
        ball_xy=np.tile([52.5, 34.0], (n, 1)),
        ball_speed=np.full(n, np.nan),
        players_xy=np.full((n, 46, 2), np.nan),
        players_speed=np.full((n, 46), np.nan),
        slot_team=np.array(["H"] * 23 + ["A"] * 23),
        pitch=(105.0, 68.0),
    )


def run_players(match, speed, slots=range(22), seed=0):
    """Every given slot moves in its own straight line at speed[frame] m/s, with 2 cm jitter.

    speed: one value per frame. Positions are what the stage reads; the speed field stays NaN.
    """
    rng = np.random.default_rng(seed)
    distance = np.concatenate([[0.0], np.cumsum(np.asarray(speed, dtype=float)[:-1])]) / FPS
    for slot in slots:
        start = rng.uniform([20.0, 10.0], [85.0, 58.0])
        angle = rng.uniform(0.0, 2.0 * np.pi)
        line = start + distance[:, None] * [np.cos(angle), np.sin(angle)]
        match.players_xy[:, slot] = line + rng.normal(0.0, 0.02, size=line.shape)


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


def test_players_still_marks_a_long_stoppage_but_not_a_short_slow_moment():
    # 22 players run at 5 m/s, with a 4 s stoppage and a 1 s slow moment at 0.5 m/s.
    match = made_up_match()
    speed = np.full(match.n_frames, 5.0)
    speed[100:200] = 0.5                    # 4 s stoppage
    speed[300:325] = 0.5                    # 1 s slow moment in live play
    run_players(match, speed)
    score = deadball.detect(match, None, PLAYERS_STILL)

    # one score per frame
    assert len(score) == match.n_frames
    # the long stoppage is dead (its edges within the fit's half window)
    assert (score[100 + EDGE:200 - EDGE] == 1).all()
    # the play around it is alive
    assert (score[:100] == 0).all()
    assert (score[200:300] == 0).all()
    # the short slow moment is too short: alive
    assert (score[300:] == 0).all()


def test_players_still_uses_the_median_so_a_few_joggers_do_not_hide_a_stoppage():
    # During the stoppage 18 players stand and 4 jog back into position at 7 m/s. Their mean
    # speed is over 1 m/s, so a mean would miss the stoppage; the median sees it.
    match = made_up_match()
    speed = np.full(match.n_frames, 5.0)
    speed[100:300] = 0.0
    run_players(match, speed, slots=range(18))
    jog = np.full(match.n_frames, 5.0)
    jog[100:300] = 7.0
    run_players(match, jog, slots=range(18, 22), seed=1)
    score = deadball.detect(match, None, PLAYERS_STILL)

    speeds = velocity.player_speed(match.players_xy, match.period, FPS,
                                   PLAYERS_STILL["deadball"]["speed"])
    assert np.nanmean(speeds[150:250], axis=1).min() > PLAYERS_STILL["deadball"]["still_speed"]
    assert (score[100 + EDGE:300 - EDGE] == 1).all()
    assert (score[:100] == 0).all() and (score[300:] == 0).all()


def test_players_still_survives_tracklet_swaps():
    # In the middle of a 4 s stoppage, 12 slots are taken over by someone standing 3 m away (as
    # when the tracker re-acquires players). A line fit across a 3 m step reads about 10 m/s,
    # under max_speed, so without cutting the tracks the stoppage splits into two short halves.
    match = made_up_match()
    speed = np.full(match.n_frames, 5.0)
    speed[100:200] = 0.0
    run_players(match, speed)
    match.players_xy[150:, :12, 0] += 3.0
    no_cut = copy.deepcopy(PLAYERS_STILL)
    no_cut["deadball"]["speed"]["max_speed"] = 1000.0       # nothing is a jump, nothing capped

    assert (deadball.detect(match, None, no_cut)[100:200] == 0).any()
    assert (deadball.detect(match, None, PLAYERS_STILL)[100 + EDGE:200 - EDGE] == 1).all()


def test_players_still_ignores_the_speed_field():
    # The ATD's speed field says everyone stands still, the positions say they run: alive.
    match = made_up_match()
    run_players(match, np.full(match.n_frames, 5.0))
    match.players_speed[:, :22] = 0.0
    assert (deadball.detect(match, None, PLAYERS_STILL) == 0).all()


def test_players_still_gives_nothing_outside_the_graded_halves():
    # Speeds are only computed inside the graded halves, so ungraded frames are never still.
    match = made_up_match()
    run_players(match, np.zeros(match.n_frames))
    match.period[:] = 0
    assert (deadball.detect(match, None, PLAYERS_STILL) == 0).all()


def test_combined_adds_up_the_clues():
    # Fixed weights here, so tuning the config later does not break the test.
    match = made_up_match()
    gap_start, gap_end = 100, 140
    out_start, out_end = 250, 265
    match.ball_xy[gap_start:gap_end] = np.nan                # ball missing
    match.ball_xy[out_start:out_end, 0] = 108.0              # past the far goal line
    still_start, still_end = gap_start + 20, gap_start + 120
    speed = np.full(match.n_frames, 5.0)                     # everyone runs at 5 m/s
    speed[still_start:still_end] = 0.5                       # 4 s stoppage, starts inside the gap
    run_players(match, speed)
    score = deadball.detect(match, None, COMBINED)

    # ball missing only: 0.3, not enough on its own
    assert score[gap_start:still_start] == pytest.approx(0.3)
    # ball missing and players still: 0.3 + 0.4, enough together
    assert score[still_start + EDGE:gap_end] == pytest.approx(0.7)
    # players still but the ball is seen: 0.4, not enough on its own
    assert score[gap_end:still_end - EDGE] == pytest.approx(0.4)
    # ball out: 1.0
    assert score[out_start:out_end] == pytest.approx(1.0)
    # no clue: 0
    assert (score[still_end:out_start] == 0).all()
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


def test_players_still_with_no_players_is_never_still_and_quiet():
    # If no slot has a position, players_still must say "no evidence", not crash or warn.
    match = made_up_match()
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
    (_section(speed=None), "needs a 'speed' block"),
    (_section(speed={"half_window_s": 0.2, "min_points": 5, "min_each_side": 2}),
     r"deadball.speed: missing \['max_speed'\]"),
])
def test_config_mistakes_fail_with_a_clear_message(section, message):
    with pytest.raises(ValueError, match=message):
        deadball.check_config(section)


@pytest.mark.parametrize("change, message", [
    ({"max_speed": 0}, "max_speed must be a positive number"),
    ({"min_points": 99}, "min_points"),                   # more than the 11-frame window
    ({"half_window_s": "0.2"}, "half_window_s"),
])
def test_speed_block_mistakes_are_caught_once_the_frame_rate_is_known(change, message):
    section = _section()
    section["speed"] = {**section["speed"], **change}
    with pytest.raises(ValueError, match=f"deadball.speed: .*{message}"):
        deadball.check_config(section, FPS)
    with pytest.raises(ValueError, match=f"deadball.speed: .*{message}"):
        deadball.detect(made_up_match(), None, {"deadball": section})   # DeadBall passes fps


def test_shipped_dead_configs_pass_the_check():
    for cfg in (BALL_MISSING, OUT_OF_BOUNDS, PLAYERS_STILL, ALL_CLUES, HMM_PAIR):
        deadball.check_config(cfg["deadball"])
        deadball.check_config(cfg["deadball"], FPS)


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
