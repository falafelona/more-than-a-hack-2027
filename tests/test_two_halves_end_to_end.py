# ws:int: configs/final.yaml end to end on a synthetic match with two graded halves (task M2).
# The small bundle never reaches the graded halves, so the still-ball, fast-ball, restart and
# half logic only run here. Made-up numbers only (tests/two_halves.py).

import numpy as np
import pytest
import two_halves as th

from falafelona import deadball, pipeline, possession, vendor
from falafelona.config import load_config

FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")


def run(monkeypatch, path, **match_options):
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match(**match_options))
    return pipeline.run(FINAL, path, n_frames=th.N)


@pytest.fixture
def result(monkeypatch, tmp_path):
    return run(monkeypatch, tmp_path / "out.csv")


def share(mask):
    return float(np.mean(mask))


def test_the_file_has_one_label_per_frame_and_only_h_a_d(result):
    labels = result["labels"]
    assert result["rows"] == th.N
    assert set(np.unique(labels)) <= {"H", "A", "D"}
    assert sum(result["counts"].values()) == th.N


def test_two_runs_write_the_same_file(monkeypatch, tmp_path):
    first = run(monkeypatch, tmp_path / "a.csv")
    second = run(monkeypatch, tmp_path / "b.csv")
    assert first["sha256"] == second["sha256"]


def test_both_teams_get_the_ball_in_both_halves(result):
    labels, period = result["labels"], th.make_match().period
    for half in (1, 2):
        assert {"H", "A"} <= set(labels[period == half])


def test_the_middle_of_every_scripted_spell_goes_to_the_right_team(result):
    for first, stop, team in th.SPELLS:
        middle = (first + stop) // 2
        assert result["labels"][middle] == team, (first, stop, team)


def test_the_second_half_is_dead_until_its_kickoff(result):
    wait = slice(th.HALF_2[0], th.KICKOFF_2)
    assert (result["dead_score"][wait] == 1.0).all()
    assert (result["labels"][wait] == "D").all()


def test_the_ball_out_of_play_is_dead(result):
    assert (result["labels"][slice(*th.OUT)] == "D").all()


def test_the_still_ball_at_the_touchline_is_found_and_called_dead(result):
    """still_ball_at_spot fires on a SEEN ball lying at the line before a throw-in."""
    clue = deadball.DeadBall(result["cleaned"], None, FINAL).run("still_ball_at_spot")
    lying = slice(*th.STILL_AT_LINE)
    assert share(clue[lying] == 1.0) >= 0.9
    assert share(result["labels"][lying] == "D") >= 0.8
    assert clue[th.HALF_2[0]:th.HALF_2[1]].sum() > 0       # and only where the ball lies still
    assert share(clue[th.KICKOFF_2 + 50:th.STILL_AT_LINE[0] - 50] == 0.0) == 1.0


def test_fast_passes_are_found(result):
    fast = possession._fast_ball(result["cleaned"], FINAL["possession"])
    assert share(fast[355:385]) >= 0.9                     # the home pass from 350 to 390
    assert share(fast[400:540]) == 0.0                     # a slow dribble is not fast


def test_the_restart_rule_fires_after_a_stoppage(result):
    costs = possession._restart(result["cleaned"], result["dead_score"], FINAL["possession"])
    assert costs is not None and (costs > 0).any()


def test_a_missing_second_half_kickoff_tag_is_reported(monkeypatch, tmp_path):
    with pytest.warns(RuntimeWarning, match="no second-half Kick Off"):
        out = run(monkeypatch, tmp_path / "out.csv", with_second_kickoff=False)
    with pytest.warns(RuntimeWarning, match="no second-half Kick Off"):
        clue = deadball.DeadBall(out["cleaned"], None, FINAL).run("before_kickoff")
    assert (clue == 0).all()                               # the clue gives nothing without it
