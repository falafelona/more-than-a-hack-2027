# ws:poss: tests for the gap-reappear evidence (possession.gap_reappear). Made-up numbers only.
#
# The idea: after a long ball gap in LIVE play, possession may have changed hands unseen, so
# where the ball reappears the team near it is favoured for a few frames. No match data here.

from types import SimpleNamespace

import numpy as np
import pytest

from falafelona import possession

H, A, N = possession.Possession.HOME, possession.Possession.AWAY, possession.Possession.NOBODY
HS, AS = possession.H_STATE, possession.A_STATE

# a gap counts from 4 frames, live if mean dead score < 0.5, look 3 frames, boost 5 frames
SMALL = {"min_gap": 4, "live_max": 0.5, "look": 3, "boost": 5, "cost": 2.0}


def costs(codes, seen, dead, period=None, **settings):
    period = np.ones(len(codes), dtype=np.int8) if period is None else np.asarray(period)
    return possession.gap_reappear_team_costs(
        np.asarray(codes), np.asarray(seen, dtype=bool), np.asarray(dead, dtype=float), period,
        **{**SMALL, **settings})


# --- gap_reappear_team_costs ------------------------------------------------------------------


def test_after_a_live_gap_the_team_near_the_reappearing_ball_is_favoured():
    codes = [H] * 4 + [N] * 5 + [A] * 8          # H had it, ball lost, reappears with A
    seen = [1] * 4 + [0] * 5 + [1] * 8
    dead = [0] * 17                                 # live play throughout
    out = costs(codes, seen, dead)

    assert (out[9:14, HS] == 2.0).all()             # boost: H costs, A is free
    assert (out[:, AS] == 0).all()
    assert (out[:9] == 0).all() and (out[14:] == 0).all()


def test_a_stoppage_gap_is_left_alone():
    codes = [H] * 4 + [N] * 5 + [A] * 8
    seen = [1] * 4 + [0] * 5 + [1] * 8
    dead = [0] * 4 + [1] * 5 + [0] * 8             # the gap was dead ball
    assert (costs(codes, seen, dead) == 0).all()


def test_a_short_gap_is_left_alone():
    codes = [H] * 4 + [N] * 3 + [A] * 8             # 3 frames, below min_gap 4
    seen = [1] * 4 + [0] * 3 + [1] * 8
    assert (costs(codes, seen, [0] * 15) == 0).all()


def test_a_tie_gives_no_evidence():
    codes = [H] * 4 + [N] * 5 + [A, H, N] + [N] * 5   # one each in the look window
    seen = [1] * 4 + [0] * 5 + [1] * 8
    assert (costs(codes, seen, [0] * 17) == 0).all()


def test_gaps_outside_the_graded_halves_and_across_halves_are_ignored():
    codes = [H] * 4 + [N] * 5 + [A] * 8
    seen = [1] * 4 + [0] * 5 + [1] * 8
    ungraded = [0] * 17
    assert (costs(codes, seen, [0] * 17, period=ungraded) == 0).all()
    across = [1] * 6 + [2] * 11                     # the gap starts in one half, ends in the next
    assert (costs(codes, seen, [0] * 17, period=across) == 0).all()


def test_the_boost_is_cut_at_the_end_of_the_match():
    codes = [H] * 4 + [N] * 5 + [A] * 2
    seen = [1] * 4 + [0] * 5 + [1] * 2
    out = costs(codes, seen, [0] * 11)
    assert (out[9:, HS] == 2.0).all() and len(out) == 11


# --- in the decoder: the switch comes sooner ---------------------------------------------------


def test_brief_evidence_after_a_live_gap_is_no_longer_ignored():
    # H has the ball, it is lost for 20 frames of live play, and reappears with A for only 8
    # frames before it is lost again. Those 8 frames of weak evidence (0.3 each, 2.4 in total)
    # cannot pay the switch cost (3.5), so the decoder keeps H throughout and misses A's
    # possession. With the boost at the reappearance, the switch to A pays off.
    codes = np.array([H] * 40 + [N] * 20 + [A] * 8 + [N] * 20)
    seen = codes != N
    dead = np.zeros(len(codes))
    team = possession.nearest_team_costs(codes, 0.3)
    table = possession.switch_table(3.5, 6.0, 6.0)

    def frames_of_a(team_costs):
        path = possession.viterbi(possession.add_dead_costs(team_costs, dead, 2.0), table)
        return int(np.sum(path == AS))

    boost = possession.gap_reappear_team_costs(codes, seen, dead, np.ones(len(codes)),
                                               min_gap=10, live_max=0.5, look=5, boost=10,
                                               cost=0.5)
    assert frames_of_a(team) == 0
    assert frames_of_a(team + boost) > 0


# --- the config block -------------------------------------------------------------------------

BLOCK = {"min_gap_s": 2.0, "live_max": 0.5, "look_s": 1.0, "boost_s": 1.0, "cost": 0.5}


def fake_match():
    """60 frames at 25 fps, one player per team, ball missing on frames 10-59."""
    n = 60
    ball = np.full((n, 2), np.nan)
    ball[:10] = [50.0, 30.0]
    players = np.full((n, 2, 2), [[49.0, 30.0], [80.0, 30.0]])
    return SimpleNamespace(fps=25.0, period=np.ones(n, dtype=np.int8), ball_xy=ball,
                           players_xy=players, slot_team=np.array(["H", "A"]))


def test_off_when_the_block_is_absent():
    assert possession._gap_reappear(fake_match(), np.zeros(60), {"method": "hmm"}) is None


def test_on_when_the_block_is_present():
    out = possession._gap_reappear(fake_match(), np.zeros(60), {"gap_reappear": BLOCK})
    assert out.shape == (60, 2)


@pytest.mark.parametrize("change, message", [
    ({"cost": 0}, "cost must be a number > 0"),
    ({"min_gap_s": True}, "min_gap_s must be a number > 0"),
    ({"live_max": 1.5}, "live_max is a dead chance"),
    ({"boost_seconds": 1.0}, "unknown keys"),           # a typo must fail loudly
])
def test_bad_settings_are_named(change, message):
    with pytest.raises(ValueError, match=message):
        possession._gap_reappear(fake_match(), np.zeros(60), {"gap_reappear": {**BLOCK, **change}})


def test_not_a_mapping():
    with pytest.raises(TypeError):
        possession._gap_reappear(fake_match(), np.zeros(60), {"gap_reappear": 1.0})
