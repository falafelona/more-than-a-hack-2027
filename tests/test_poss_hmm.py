# ws:poss: tests for the 3-state H / A / D HMM (Dudu). Made-up numbers only, no match data.

from types import SimpleNamespace

import numpy as np
import pytest

from falafelona import possession
from falafelona.config import load_config


def test_margin_costs_favour_the_closer_team_and_are_neutral_when_equal():
    d_home = np.array([1.0, 3.0, 2.0])
    d_away = np.array([3.0, 1.0, 2.0])
    costs = possession.margin_team_costs(
        d_home, d_away, team_cost=1.0, margin_scale=1.0, near_distance=5.0
    )

    assert costs[0, possession.H_STATE] < costs[0, possession.A_STATE]  # home closer
    assert costs[1, possession.A_STATE] < costs[1, possession.H_STATE]  # away closer
    assert costs[2, 0] == pytest.approx(costs[2, 1])  # a 50/50 duel


def test_margin_costs_give_no_evidence_without_a_ball_or_with_nobody_close():
    d_home = np.array([np.inf, 6.0])  # ball missing; nearest player 6 m away
    d_away = np.array([np.inf, 9.0])
    costs = possession.margin_team_costs(
        d_home, d_away, team_cost=1.0, margin_scale=1.0, near_distance=5.0
    )

    assert (costs == 0).all()  # the current owner just holds


def test_team_distances_finds_the_nearest_player_of_each_team():
    players = np.array([[[1.0, 0.0], [5.0, 0.0], [0.0, 2.0], [np.nan, np.nan]]])
    ball = np.array([[0.0, 0.0]])
    d_home, d_away = possession.team_distances(players, ball, np.array(["H", "H", "A", "A"]))

    assert d_home[0] == pytest.approx(1.0) and d_away[0] == pytest.approx(2.0)


def test_dead_costs_add_a_d_column_that_follows_the_dead_score():
    team = np.zeros((2, 2))
    costs = possession.add_dead_costs(team, np.array([0.0, 1.0]), dead_weight=2.0)

    assert costs.shape == (2, 3)
    assert costs[0, possession.D_STATE] == pytest.approx(2.0)  # alive: D is costly
    assert costs[1, possession.D_STATE] == pytest.approx(0.0)  # dead: D is free
    assert costs[1, possession.H_STATE] == pytest.approx(2.0)  # dead: H and A are costly


def test_the_old_nearest_evidence_is_unchanged():
    codes = np.array([possession.Possession.HOME, possession.Possession.AWAY,
                      possession.Possession.NOBODY])
    costs = possession.nearest_team_costs(codes, team_cost=1.0)

    assert costs.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]


def test_an_unknown_team_evidence_is_refused():
    section = {
        "method": "hmm",
        "team_evidence": "telepathy",
        "team_cost": 1.0,
        "dead_weight": 2.0,
        "switch_cost": 10.0,
        "to_dead_cost": 6.0,
        "from_dead_cost": 6.0,
    }

    with pytest.raises(ValueError, match="team_evidence"):
        possession._hmm(None, None, section)

def test_the_margin_evidence_is_really_used(monkeypatch):
    """Home 0.5 m and away 1.5 m from the ball on every frame: nearest says "home" at full
    cost, the margin evidence is weaker. The two kinds must give different costs. (Dudu)"""
    n = 50
    players = np.zeros((n, 2, 2))
    players[:, 0] = [0.5, 0.0]
    players[:, 1] = [1.5, 0.0]
    match = SimpleNamespace(players_xy=players, ball_xy=np.zeros((n, 2)),
                            slot_team=np.array(["H", "A"]))
    seen = []
    real = possession.add_dead_costs
    monkeypatch.setattr(possession, "add_dead_costs",
                        lambda team, dead, weight: seen.append(team.copy()) or real(team, dead, weight))

    possession._hmm(match, np.zeros(n), {**HMM, "team_evidence": "nearest"})
    possession._hmm(match, np.zeros(n), {**HMM, "team_evidence": "margin",
                                         "margin_scale": 1.0, "near_distance": 5.0})

    assert not np.allclose(seen[0], seen[1])


# --- decoding behaviour: the cases that broke the earlier decoders ------------------------------
# Settings of configs/exp/poss-hmm.yaml; the dead score is 1 where the ball is missing, as with
# deadball method ball_missing.

H, A, N = possession.Possession.HOME, possession.Possession.AWAY, possession.Possession.NOBODY
HMM = {"method": "hmm", "team_evidence": "nearest", "team_cost": 1.0, "dead_weight": 2.0,
       "switch_cost": 10.0, "to_dead_cost": 6.0, "from_dead_cost": 6.0}


def fake_match(codes):
    """A Match-like object where the given team is nearest on every frame, NaN ball where N."""
    codes = np.asarray(codes)
    players = np.zeros((len(codes), 2, 2))
    players[:, 0] = [10.0, 10.0]          # slot 0: home
    players[:, 1] = [90.0, 50.0]          # slot 1: away
    ball = np.where((codes == H)[:, None], [11.0, 10.0], [89.0, 50.0])
    ball[codes == N] = np.nan
    return SimpleNamespace(players_xy=players, ball_xy=ball, slot_team=np.array(["H", "A"]))


def decode(codes, dead=None, **settings):
    """predict() with the HMM; dead defaults to 1 where the ball is missing. One string out."""
    codes = np.asarray(codes)
    dead_score = (codes == N).astype(float) if dead is None else np.asarray(dead, dtype=float)
    labels = possession.predict(fake_match(codes), dead_score,
                                {"possession": {**HMM, **settings}})
    return "".join(labels)


def test_a_team_change_across_a_long_gap_gives_no_fake_possession():
    assert decode([H] * 100 + [N] * 300 + [A] * 100) == "H" * 100 + "D" * 300 + "A" * 100


def test_d_covers_exactly_the_gap_with_no_bleed_into_play():
    assert decode([H] * 200 + [N] * 400 + [H] * 200) == "H" * 200 + "D" * 400 + "H" * 200


def test_one_wrong_frame_after_a_dead_ball_is_not_a_possession():
    codes = [H] * 20 + [A] + [H] * 100
    dead = [1.0] * 20 + [0.0] * 101
    assert decode(codes, dead) == "D" * 20 + "H" * 101


def test_a_noisy_hand_over_still_switches_where_it_starts():
    assert decode([H] * 50 + ([A] * 10 + [H]) * 20) == "H" * 50 + "A" * 220


def test_a_short_flicker_is_ignored():
    assert decode([H] * 50 + [A] * 4 + [H] * 50) == "H" * 104


def test_a_ball_missing_for_a_few_frames_does_not_split_the_possession():
    assert decode([H] * 50 + [N] * 3 + [H] * 50) == "H" * 103


def test_a_real_change_switches():
    assert decode([H] * 50 + [A] * 50) == "H" * 50 + "A" * 50

# --- evidence_lag: move the H / A evidence later ---------------------------------------------


def test_lag_moves_the_team_costs_later_and_leaves_the_start_empty():
    team = np.arange(10, dtype=float).reshape(5, 2)
    shifted = possession.lag_team_costs(team, 2)

    assert (shifted[:2] == 0).all()
    assert np.array_equal(shifted[2:], team[:3])


def test_a_lag_longer_than_the_match_gives_no_evidence():
    assert (possession.lag_team_costs(np.ones((3, 2)), 5) == 0).all()


def test_no_lag_is_the_default_and_changes_nothing():
    codes = [H] * 50 + [A] * 50
    assert decode(codes) == decode(codes, evidence_lag=0)


def test_a_lag_moves_the_switch_later_by_the_same_number_of_frames():
    assert decode([H] * 50 + [A] * 50, evidence_lag=4) == "H" * 54 + "A" * 46


def test_a_lag_does_not_move_the_dead_ball():
    labels = decode([H] * 100 + [N] * 300 + [H] * 100, evidence_lag=6)
    assert labels[100:400] == "D" * 300


@pytest.mark.parametrize("value", [-1, 2.5, True])
def test_a_bad_lag_is_refused(value):
    with pytest.raises(ValueError, match="evidence_lag"):
        possession._hmm(fake_match([H] * 3), np.zeros(3), {**HMM, "evidence_lag": value})


# --- fast_ball_speed: no team evidence while a pass or shot is in the air --------------------

VELOCITY = {"half_window_s": 0.12, "min_points": 4, "min_each_side": 1}


def test_fast_ball_frames_are_above_the_threshold_and_nan_is_not_fast():
    speed = np.array([5.0, 12.0, 20.0, np.nan])
    assert list(possession.fast_ball_frames(speed, 10.0)) == [False, True, True, False]


def test_fast_ball_frames_lose_their_team_evidence_and_others_keep_it():
    team = np.ones((3, 2))
    costs = possession.drop_fast_ball_evidence(team, np.array([False, True, False]))
    assert (costs[1] == 0).all() and (costs[[0, 2]] == 1).all()
    assert (team == 1).all()  # the input is not changed


def fast_match(codes):
    """fake_match plus the fields ball_velocity needs."""
    match = fake_match(codes)
    match.period = np.ones(len(codes), dtype=np.int8)
    match.fps = 25.0
    return match


def decode_with_speed(codes, speed, **settings):
    """predict() with the HMM where ball_velocity returns the given speed per frame."""
    codes = np.asarray(codes)
    real = possession.velocity.ball_velocity
    possession.velocity.ball_velocity = lambda *a, **k: (None, np.asarray(speed, dtype=float))
    try:
        labels = possession.predict(fast_match(codes), np.zeros(len(codes)),
                                    {"possession": {**HMM, **settings}})
    finally:
        possession.velocity.ball_velocity = real
    return "".join(labels)


def test_a_ball_flying_past_an_opponent_does_not_switch():
    """Home has it, the ball flies fast past an away player for 30 frames, home receives."""
    codes = [H] * 50 + [A] * 30 + [H] * 50
    speed = [3.0] * 50 + [20.0] * 30 + [3.0] * 50
    low_cost = {"switch_cost": 2.0}

    assert "A" in decode_with_speed(codes, speed, **low_cost)        # rule off: it switches
    assert decode_with_speed(codes, speed, fast_ball_speed=10.0, velocity=VELOCITY,
                             **low_cost) == "H" * 130


def test_a_slow_ball_received_by_the_other_team_still_switches():
    codes = [H] * 50 + [A] * 50
    speed = [3.0] * 100
    assert decode_with_speed(codes, speed, fast_ball_speed=10.0, velocity=VELOCITY) == \
        "H" * 50 + "A" * 50


def test_no_fast_ball_speed_means_the_rule_is_off():
    codes = [H] * 50 + [A] * 50
    assert decode(codes) == "H" * 50 + "A" * 50     # fake_match has no fps / period: not read


@pytest.mark.parametrize("value", [0, -5, "12", True])
def test_a_bad_fast_ball_speed_is_refused(value):
    with pytest.raises(ValueError, match="fast_ball_speed"):
        possession._hmm(fast_match([H] * 3), np.zeros(3),
                        {**HMM, "fast_ball_speed": value, "velocity": VELOCITY})


def test_fast_ball_speed_without_a_velocity_block_is_refused():
    with pytest.raises(ValueError, match="velocity"):
        possession._hmm(fast_match([H] * 3), np.zeros(3), {**HMM, "fast_ball_speed": 10.0})


def test_the_real_velocity_fit_marks_a_fast_ball_on_a_made_up_track():
    """No monkeypatch: a ball moving 20 m/s for 30 frames between two still stretches."""
    n = 130
    x = np.concatenate([np.full(50, 10.0), 10.0 + 0.8 * np.arange(1, 31), np.full(50, 34.0)])
    match = fast_match([H] * n)
    match.ball_xy = np.column_stack([x, np.full(n, 10.0)])
    _, speed = possession.velocity.ball_velocity(match.ball_xy, match.period, match.fps, VELOCITY)
    fast = possession.fast_ball_frames(speed, 10.0)
    assert fast[55:75].all() and not fast[:45].any() and not fast[90:].any()


# --- velocity_match: the player who moves with the ball controls it ---------------------------

MATCH = {"max_diff": 2.0, "control_distance": 2.0, "cost": 1.0}


def test_velocity_match_gives_evidence_only_to_the_team_moving_with_the_ball():
    v_ball = np.array([[3.0, 0.0]] * 4)
    v_home = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 0.0], [np.nan, np.nan]])
    v_away = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
    near = np.ones(4)
    costs = possession.velocity_match_costs(v_ball, v_home, v_away, near, near, **MATCH)

    assert list(costs[0]) == [0.0, 1.0]   # home matches: against A
    assert list(costs[1]) == [0.0, 0.0]   # both match: no evidence
    assert list(costs[2]) == [1.0, 0.0]   # away matches: against H
    assert list(costs[3]) == [0.0, 0.0]   # home NaN, away does not match


def test_velocity_match_needs_the_player_close_to_the_ball():
    v = np.array([[3.0, 0.0]])
    costs = possession.velocity_match_costs(v, v, np.array([[0.0, 0.0]]),
                                            np.array([5.0]), np.array([1.0]), **MATCH)
    assert (costs == 0).all()             # home moves with it but is 5 m away


def test_nearest_slots_finds_the_nearest_player_of_each_team():
    players = np.array([[[1.0, 0.0], [5.0, 0.0], [0.0, 2.0], [np.nan, np.nan]],
                        [[1.0, 0.0], [5.0, 0.0], [0.0, 2.0], [0.0, 1.0]]])
    ball = np.array([[0.0, 0.0], [np.nan, np.nan]])
    i_home, i_away = possession.nearest_slots(players, ball, np.array(["H", "H", "A", "A"]))
    assert list(i_home) == [0, -1] and list(i_away) == [2, -1]


def test_player_velocities_follow_a_running_player():
    n = 30
    players = np.zeros((n, 2, 2))
    players[:, 0, 0] = 0.2 * np.arange(n)           # 5 m/s along x at 25 fps
    players[:, 1] = np.nan                          # an empty slot
    v = possession.player_velocities(players, np.ones(n, dtype=np.int8), 25.0, VELOCITY)
    assert np.allclose(v[5:25, 0], [5.0, 0.0])
    assert np.isnan(v[:, 1]).all()


def dribble_match():
    """Home dribbles at 3 m/s along y = 0, 1.5 m from the ball. A still away player stands at
    x = 15, 0.8 m off the path, so for a stretch he is the nearest without touching it."""
    n = 250
    ball = np.column_stack([0.12 * np.arange(n), np.zeros(n)])
    players = np.zeros((n, 2, 2))
    players[:, 0] = ball + [0.0, 1.5]               # home: moves with the ball
    players[:, 1] = [15.0, 0.8]                     # away: standing still
    return SimpleNamespace(players_xy=players, ball_xy=ball, slot_team=np.array(["H", "A"]),
                           period=np.ones(n, dtype=np.int8), fps=25.0)


def test_a_dribble_past_a_still_opponent_does_not_switch_with_velocity_match():
    settings = {**HMM, "team_evidence": "margin", "margin_scale": 1.0, "near_distance": 5.0,
                "team_cost": 0.5, "switch_cost": 1.0}
    match = dribble_match()
    off = "".join(possession._hmm(match, np.zeros(250), settings))
    on = "".join(possession._hmm(match, np.zeros(250),
                                 {**settings, "velocity": VELOCITY, "velocity_match": MATCH}))

    assert "A" in off                     # margin alone: the opponent "takes" it
    assert set(on) == {"H"}               # home moves with the ball: no switch


@pytest.mark.parametrize("block, key", [
    ({**MATCH, "cost": 0}, "cost"),
    ({**MATCH, "max_diff": "2"}, "max_diff"),
    ({"max_diff": 2.0, "cost": 1.0}, "control_distance"),
    ("yes", "velocity_match"),
])
def test_bad_velocity_match_settings_are_refused(block, key):
    with pytest.raises((TypeError, ValueError), match=key):
        possession._hmm(fast_match([H] * 3), np.zeros(3),
                        {**HMM, "velocity": VELOCITY, "velocity_match": block})


def test_velocity_match_without_a_velocity_block_is_refused():
    with pytest.raises(ValueError, match="velocity block"):
        possession._hmm(fast_match([H] * 3), np.zeros(3), {**HMM, "velocity_match": MATCH})

# --- settings and configs -----------------------------------------------------------------------


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown possession method 'decodr'"):
        possession.predict(None, np.zeros(3), {"possession": {"method": "decodr"}})


@pytest.mark.parametrize("key", ["team_cost", "dead_weight", "switch_cost", "to_dead_cost",
                                 "from_dead_cost"])
def test_every_cost_is_required(key):
    section = {k: v for k, v in HMM.items() if k != key}
    with pytest.raises(ValueError, match=key):
        possession._hmm(fake_match([H] * 3), np.zeros(3), section)


@pytest.mark.parametrize("value", [-1, "10", True])
def test_bad_costs_are_refused(value):
    with pytest.raises(ValueError, match="switch_cost"):
        possession._hmm(fake_match([H] * 3), np.zeros(3), {**HMM, "switch_cost": value})


@pytest.mark.parametrize("name", ["poss-hmm.yaml", "poss-hmm-margin.yaml",
                                  "poss-switch-cost.yaml"])
def test_the_experiment_configs_differ_from_the_old_final_only_in_possession(name):
    """They were made from the old final.yaml (baseline + interpolate), which is
    configs/exp/ball-interpolate.yaml: the dev-set rows compare against it."""
    exp = load_config(f"configs/exp/{name}")
    old_final = load_config("configs/exp/ball-interpolate.yaml")
    assert exp["possession"]["method"] in possession.METHODS
    assert {k: v for k, v in exp.items() if k != "possession"} == {
        k: v for k, v in old_final.items() if k != "possession"
    }


@pytest.mark.parametrize("name", ["poss-hmm.yaml", "poss-hmm-margin.yaml"])
def test_runs_through_the_pipeline(bundle, tmp_path, name):
    from falafelona import pipeline

    cfg = load_config(f"configs/exp/{name}")
    result = pipeline.run(cfg, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert result["rows"] == bundle["n_frames"]
    assert result["counts"].get("X", 0) == 0
    assert sum(result["counts"].values()) == bundle["n_frames"]



def test_the_best_pair_config_runs_end_to_end(bundle, tmp_path):
    """configs/exp/poss-best-dead-gap.yaml: our possession settings with the graded gap score."""
    from falafelona import pipeline

    cfg = load_config("configs/exp/poss-best-dead-gap.yaml")
    assert cfg["possession"]["evidence_lag"] == 12
    assert cfg["possession"]["fast_ball_speed"] == 12
    result = pipeline.run(cfg, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert result["rows"] == bundle["n_frames"]