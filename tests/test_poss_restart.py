# ws:poss: tests for the restart-taker evidence (possession.restart). Made-up numbers only.

from types import SimpleNamespace

import numpy as np
import pytest

from falafelona import possession

H, A, N = possession.Possession.HOME, possession.Possession.AWAY, possession.Possession.NOBODY
HS, AS = possession.H_STATE, possession.A_STATE

# threshold 0.5, a dead spell counts from 3 frames, look 4 frames each side, boost 5 frames
SMALL = {"threshold": 0.5, "min_dead": 3, "look": 4, "boost": 5, "cost": 2.0}


def costs(codes, dead, **settings):
    return possession.restart_team_costs(np.asarray(codes), np.asarray(dead, dtype=float),
                                         **{**SMALL, **settings})


# --- restart_team_costs -------------------------------------------------------------------------


def test_the_team_near_the_ball_at_the_end_of_a_dead_spell_gets_the_restart():
    codes = [A] * 4 + [N] * 4 + [H] * 6
    dead = [0] * 4 + [1] * 4 + [0] * 6
    out = costs(codes, dead)

    assert (out[8:13, AS] == 2.0).all()       # boost frames after the end: A costs, H is free
    assert (out[8:13, HS] == 0.0).all()
    assert (out[:8] == 0).all() and (out[13:] == 0).all()   # nothing outside the boost


def test_away_taking_the_restart_puts_the_cost_on_home():
    codes = [N] * 4 + [A] * 6
    dead = [1] * 4 + [0] * 6
    out = costs(codes, dead)

    assert (out[4:9, HS] == 2.0).all() and (out[:, AS] == 0).all()


def test_a_dead_spell_shorter_than_min_dead_gives_no_evidence():
    codes = [N] * 2 + [H] * 6
    dead = [1] * 2 + [0] * 6
    assert (costs(codes, dead) == 0).all()


def test_a_tie_near_the_end_gives_no_evidence():
    codes = [N] * 4 + [H, A, H, A]
    dead = [1] * 4 + [0] * 4
    assert (costs(codes, dead) == 0).all()


def test_a_dead_spell_running_to_the_end_of_the_match_gives_no_evidence():
    codes = [H] * 4 + [N] * 4
    dead = [0] * 4 + [1] * 4
    assert (costs(codes, dead) == 0).all()


def test_the_dead_score_must_reach_the_threshold_and_nan_counts_as_alive():
    codes = [N] * 4 + [H] * 6
    assert (costs(codes, [0.4] * 4 + [0] * 6) == 0).all()             # below 0.5: not dead
    assert (costs(codes, [np.nan] * 4 + [0] * 6) == 0).all()          # unknown: not dead
    assert costs(codes, [0.5] * 4 + [0] * 6)[4, AS] == 2.0            # at 0.5: dead


def test_the_boost_stops_at_the_end_of_the_match():
    codes = [N] * 4 + [H] * 2
    dead = [1] * 4 + [0] * 2
    out = costs(codes, dead)
    assert out.shape == (6, 2) and (out[4:, AS] == 2.0).all()


def test_two_dead_spells_each_give_their_own_restart():
    codes = [N] * 3 + [H] * 6 + [N] * 3 + [A] * 6
    dead = [1] * 3 + [0] * 6 + [1] * 3 + [0] * 6
    out = costs(codes, dead)
    assert (out[3:8, AS] == 2.0).all() and (out[12:17, HS] == 2.0).all()


# --- the rule inside the HMM --------------------------------------------------------------------

HMM = {"method": "hmm", "team_evidence": "nearest", "team_cost": 1.0, "dead_weight": 2.0,
       "switch_cost": 10.0, "to_dead_cost": 6.0, "from_dead_cost": 6.0}
# 25 fps: dead spell from 25 frames, look 50 frames each side, boost 50 frames
RESTART = {"threshold": 0.5, "min_dead_s": 1.0, "look_s": 2.0, "boost_s": 2.0, "cost": 2.0}


def fake_match(codes):
    """The given team is nearest on every frame, ball missing where N; 25 fps, one half."""
    codes = np.asarray(codes)
    players = np.zeros((len(codes), 2, 2))
    players[:, 0] = [10.0, 10.0]          # slot 0: home
    players[:, 1] = [90.0, 50.0]          # slot 1: away
    ball = np.where((codes == H)[:, None], [11.0, 10.0], [89.0, 50.0])
    ball[codes == N] = np.nan
    return SimpleNamespace(players_xy=players, ball_xy=ball, slot_team=np.array(["H", "A"]),
                           period=np.ones(len(codes), dtype=np.int8), fps=25.0)


def decode(codes, **settings):
    codes = np.asarray(codes)
    dead = (codes == N).astype(float)
    return "".join(possession.predict(fake_match(codes), dead,
                                      {"possession": {**HMM, **settings}}))


def test_a_restart_taken_by_home_is_not_given_to_away_for_a_moment():
    """Home attacks, the ball goes out, away is nearest for 15 frames at the restart, then home
    plays on. Home is nearest most of the time around the restart, so it takes it."""
    codes = [H] * 50 + [N] * 50 + [A] * 15 + [H] * 60

    assert "A" in decode(codes)                                  # rule off: away for a moment
    assert decode(codes, restart=RESTART) == "H" * 50 + "D" * 50 + "H" * 75


def test_no_restart_block_means_the_rule_is_off():
    codes = [H] * 10
    assert possession._restart(fake_match(codes), np.zeros(len(codes)), HMM) is None


@pytest.mark.parametrize("key", possession.RESTART_KEYS)
@pytest.mark.parametrize("value", [0, -1.0, True, "2", float("nan"), None])
def test_bad_restart_settings_are_refused(key, value):
    codes = [H] * 3
    with pytest.raises(ValueError, match=f"restart.{key}"):
        possession._restart(fake_match(codes), np.zeros(3), {**HMM, "restart": {**RESTART,
                                                                               key: value}})


def test_a_restart_block_that_is_not_a_mapping_is_refused():
    with pytest.raises(TypeError, match="mapping"):
        possession._restart(fake_match([H] * 3), np.zeros(3), {**HMM, "restart": 2.0})


# --- the experiment configs ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["best-still-ball.yaml", "best-restart.yaml",
                                  "best-still-ball-restart.yaml"])
def test_the_still_ball_and_restart_configs_run_end_to_end(bundle, tmp_path, name):
    """Each config is poss-best-dead-gap.yaml plus the new clue(s), and runs on synthetic data."""
    from falafelona import pipeline
    from falafelona.config import load_config

    cfg = load_config(f"configs/exp/{name}")
    best = load_config("configs/exp/poss-best-dead-gap.yaml")
    assert cfg["ball"] == best["ball"] and cfg["output"] == best["output"]
    assert ("still_ball_at_spot" in cfg["deadball"]["weights"]) == ("still" in name)
    assert ("restart" in cfg["possession"]) == ("restart" in name)

    result = pipeline.run(cfg, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert result["rows"] == bundle["n_frames"]
    assert sum(result["counts"].values()) == bundle["n_frames"]
