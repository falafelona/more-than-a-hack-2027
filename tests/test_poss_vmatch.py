# ws:poss: tests for the ball-carrier experiment configs (possession.velocity_match switched on
# in the final config). Made-up numbers only: nothing here reads match data.

import numpy as np
import pytest
import two_halves as th

from falafelona import pipeline, possession, vendor
from falafelona.config import load_config

EXP = vendor.REPO_ROOT / "configs" / "exp"
FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")
CONFIGS = [("poss-vmatch-low", 0.25), ("poss-vmatch-mid", 0.5), ("poss-vmatch-high", 1.0)]


def run(monkeypatch, cfg, path):
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    return pipeline.run(cfg, path, n_frames=th.N)


def test_final_yaml_does_not_use_velocity_match():
    assert "velocity_match" not in FINAL["possession"]     # the experiment is off in the submission


@pytest.mark.parametrize("name, cost", CONFIGS)
def test_the_configs_are_final_with_velocity_match_switched_on(name, cost):
    cfg = load_config(EXP / f"{name}.yaml")
    block = cfg["possession"].pop("velocity_match")
    assert block == {"max_diff": 2.0, "control_distance": 2.0, "cost": cost}
    assert cfg == FINAL


def test_the_costs_are_neighbours_of_the_middle_one():
    costs = [load_config(EXP / f"{name}.yaml")["possession"]["velocity_match"]["cost"]
             for name, _ in CONFIGS]
    assert costs == [0.25, 0.5, 1.0]


@pytest.mark.parametrize("name, cost", CONFIGS)
def test_the_configs_run_end_to_end_and_the_clue_fires(monkeypatch, tmp_path, name, cost):
    cfg = load_config(EXP / f"{name}.yaml")
    out = run(monkeypatch, cfg, tmp_path / "out.csv")
    assert out["rows"] == th.N
    assert set(np.unique(out["labels"])) <= {"H", "A", "D"}
    extra = possession._velocity_match(out["cleaned"], cfg["possession"])
    assert extra.shape == (th.N, 2)
    assert (extra > 0).any()                               # the owner moves with the ball
    assert extra.max() == pytest.approx(cost)              # and only that cost is ever added


def test_a_velocity_match_config_writes_the_same_file_twice(monkeypatch, tmp_path):
    cfg = load_config(EXP / "poss-vmatch-mid.yaml")
    first = run(monkeypatch, cfg, tmp_path / "a.csv")
    second = run(monkeypatch, cfg, tmp_path / "b.csv")
    assert first["sha256"] == second["sha256"]
