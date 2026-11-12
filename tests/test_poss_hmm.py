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
def test_the_experiment_configs_differ_from_final_only_in_possession(name):
    exp = load_config(f"configs/exp/{name}")
    final = load_config("configs/final.yaml")
    assert exp["possession"]["method"] in possession.METHODS
    assert {k: v for k, v in exp.items() if k != "possession"} == {
        k: v for k, v in final.items() if k != "possession"
    }


@pytest.mark.parametrize("name", ["poss-hmm.yaml", "poss-hmm-margin.yaml"])
def test_runs_through_the_pipeline(bundle, tmp_path, name):
    from falafelona import pipeline

    cfg = load_config(f"configs/exp/{name}")
    result = pipeline.run(cfg, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert result["rows"] == bundle["n_frames"]
    assert result["counts"].get("X", 0) == 0
    assert sum(result["counts"].values()) == bundle["n_frames"]
