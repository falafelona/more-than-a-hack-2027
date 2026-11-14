# ws:dead: tests for the graded gap score (deadball method gap_chance). Synthetic data only.
#
# Every match here is made up inside the test (made_up_match in test_deadball.py): no match data.
# The shipped numbers in configs/exp/poss-hmm-margin-dead-gap.yaml are used where the test is
# about what the HMM will do with them: it reads a dead score above 0.5 as "leans dead".

import copy
import warnings

import numpy as np
import pytest
from test_deadball import FPS, made_up_match, run_players

from falafelona import deadball, io, pipeline
from falafelona.config import load_config

CONFIGS = io.vendor.REPO_ROOT / "configs"
GAP_PAIR = load_config(CONFIGS / "exp" / "poss-hmm-margin-dead-gap.yaml")
MID_PITCH = (30.0, 20.0)                        # far from every line and from the centre spot


def gap_match(gaps, before=MID_PITCH, after=MID_PITCH, players_speed=5.0):
    """A made-up match with ball gaps at `gaps` [(first, stop), ...] and 22 players.

    The ball is seen at MID_PITCH everywhere else; `before` / `after` set the last ball before
    and the first ball after every gap. Players move at players_speed m/s throughout.
    """
    match = made_up_match(n=1000)
    match.ball_xy[:] = MID_PITCH
    for first, stop in gaps:
        match.ball_xy[first - 1] = before
        match.ball_xy[stop] = after
        match.ball_xy[first:stop] = np.nan
    run_players(match, np.full(match.n_frames, float(players_speed)))
    return match


def gap_config(**weights):
    """The shipped dead-ball section, run as gap_chance alone, with some clue weights changed."""
    section = copy.deepcopy(GAP_PAIR["deadball"])
    section["method"] = "gap_chance"
    section["gap"]["weights"].update(weights)
    return {"deadball": section}


def chance(match, **weights):
    return deadball.detect(match, None, gap_config(**weights))


EIGHT_S = (300, 500)                            # an 8 s gap


def test_one_chance_per_gap_and_zero_where_the_ball_is_seen():
    match = gap_match([EIGHT_S, (700, 750)])    # 8 s and 2 s
    score = chance(match)

    assert len(np.unique(score[300:500])) == 1 and len(np.unique(score[700:750])) == 1
    assert (score[:300] == 0).all() and (score[500:700] == 0).all() and (score[750:] == 0).all()
    assert 0 < score[700] < score[300] < 1      # a longer gap leans more to dead


def test_a_gap_with_no_sign_of_a_stoppage_keeps_its_possession():
    # The point of the graded score: with ball_missing every gap scored 0.6 and became D.
    assert chance(gap_match([EIGHT_S]))[400] < 0.5


def test_a_gap_last_seen_near_a_line_leans_dead():
    near = gap_match([EIGHT_S], before=(30.0, 1.0))           # 1 m from the touchline
    past = gap_match([EIGHT_S], before=(-0.5, 20.0))          # behind the goal line
    assert chance(near)[400] > 0.5 and chance(past)[400] > 0.5


def test_where_the_ball_comes_back_counts():
    throw_in = gap_match([EIGHT_S], after=(60.0, 0.3))
    kickoff = gap_match([EIGHT_S], after=(52.5, 34.0))        # the centre spot of 105 x 68
    nothing = gap_match([EIGHT_S])
    assert chance(throw_in)[400] > 0.5 and chance(kickoff)[400] > 0.5
    assert chance(nothing)[400] < 0.5


def test_players_standing_during_the_gap_lean_dead_and_running_does_not():
    standing = gap_match([EIGHT_S])
    speed = np.full(standing.n_frames, 5.0)
    speed[300:500] = 0.0
    run_players(standing, speed)

    assert chance(standing)[400] > 0.5
    assert chance(gap_match([EIGHT_S]))[400] < 0.5
    assert chance(standing, players_still=0.0)[400] == pytest.approx(
        chance(gap_match([EIGHT_S]))[400])                    # weight 0: the clue is off


def test_a_restart_tag_after_the_gap_is_off_unless_given_a_weight():
    match = gap_match([EIGHT_S])
    match.tags = [io.Event(start=520, end=530, code="SET PIECES", type="Throw-in", team=None,
                           half=1)]                           # 0.8 s after the gap
    late = gap_match([EIGHT_S])
    late.tags = [io.Event(start=700, end=710, code="SET PIECES", type="Throw-in", team=None,
                          half=1)]                            # 8 s after: too late

    assert GAP_PAIR["deadball"]["gap"]["weights"]["restart_tag_after"] == 0   # no dev-set test
    assert chance(match)[400] == pytest.approx(chance(gap_match([EIGHT_S]))[400])
    assert chance(match, restart_tag_after=1.0)[400] > chance(match)[400]
    assert chance(late, restart_tag_after=1.0)[400] == pytest.approx(chance(late)[400])


def test_gaps_are_cut_at_the_half_and_the_edge_of_a_half_gives_no_clue():
    match = gap_match([(300, 500)], before=(30.0, 1.0), after=(30.0, 1.0))
    match.period[400:] = 2                      # the half changes inside the gap
    score = chance(match)

    assert deadball.gap_runs(np.isnan(match.ball_xy[:, 0]), match.period) == [(300, 400),
                                                                               (400, 500)]
    # the first piece only has the ball before it, the second only the ball after it: one clue
    # each, so they score the same, and the same as one 4 s gap with one clue
    one_clue = chance(gap_match([(300, 400)], before=(30.0, 1.0)))[350]
    assert score[350] == pytest.approx(score[450]) == pytest.approx(one_clue)


def test_a_ball_outside_the_half_is_not_where_it_comes_back():
    # The half ends with the gap; the next frame (half-time, not graded) shows a ball near a
    # line. That ball says nothing about how the gap ended.
    match = gap_match([(300, 400)], after=(30.0, 1.0))
    match.period[400:] = 0
    no_clue = chance(gap_match([(300, 400)]))[350]
    assert chance(match)[350] == pytest.approx(no_clue)


def test_no_players_gives_no_still_clue_and_no_warning():
    match = gap_match([EIGHT_S])
    match.players_xy[:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = chance(match)
    assert score[400] == pytest.approx(chance(gap_match([EIGHT_S]))[400])
    assert ((score >= 0) & (score <= 1)).all()


def test_the_graded_pair_replaces_ball_missing():
    weights = GAP_PAIR["deadball"]["weights"]
    assert "ball_missing" not in weights and "players_still" not in weights   # inside gap_chance
    assert weights["gap_chance"] == 1.0         # a chance already: added as it is


def test_shipped_gap_config_passes_the_check():
    deadball.check_config(GAP_PAIR["deadball"])
    deadball.check_config(GAP_PAIR["deadball"], FPS)


def test_without_the_still_clue_no_player_settings_are_needed():
    section = gap_config(players_still=0.0)["deadball"]
    del section["speed"], section["still_speed"]
    deadball.check_config(section, FPS)


def _gap_section(**changes):
    """The shipped gap_chance section with `gap` keys changed (None removes a key)."""
    section = gap_config()["deadball"]
    for key, value in changes.items():
        if value is None:
            del section["gap"][key]
        else:
            section["gap"][key] = value
    return section


@pytest.mark.parametrize("section, message", [
    ({**gap_config()["deadball"], "gap": None}, "needs a 'gap' block"),
    (_gap_section(intercept="-1"), "intercept must be a number"),
    (_gap_section(intercept=None), "intercept must be a number"),
    (_gap_section(line_distance=-2.0), "line_distance must be a number >= 0"),
    (_gap_section(tag_seconds=None), "needs 'tag_seconds'"),
    (_gap_section(weights={"last_seen_near_lines": 1.0}), "is not a gap clue"),
    (_gap_section(weights={"last_seen_near_line": 1.0}), "needs 'reappears_near_line'"),
    (_gap_section(weights=[1.0, 2.0]), "weights must give a number"),
])
def test_gap_config_mistakes_fail_with_a_clear_message(section, message):
    with pytest.raises(ValueError, match=message):
        deadball.check_config(section, FPS)


def test_the_still_clue_needs_the_player_settings():
    section = gap_config()["deadball"]
    del section["speed"]
    with pytest.raises(ValueError, match="needs a 'speed' block"):
        deadball.check_config(section, FPS)


def test_the_graded_pair_runs_end_to_end(bundle, tmp_path):
    out = pipeline.run(GAP_PAIR, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert out["rows"] == bundle["n_frames"]
    assert ((out["dead_score"] >= 0) & (out["dead_score"] <= 1)).all()
    assert set(out["labels"]) <= {"H", "A", "D"}
