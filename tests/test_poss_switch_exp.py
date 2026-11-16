# ws:poss: tests for the switch-cost experiment configs (possession.switch_cost 3.0 and 4.0 around
# the final 3.5). Made-up numbers only: nothing here reads match data.

import numpy as np
import pytest
import two_halves as th

from falafelona import pipeline, vendor
from falafelona.config import load_config

EXP = vendor.REPO_ROOT / "configs" / "exp"
FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")
CONFIGS = [("poss-switch-3.0", 3.0), ("poss-switch-4.0", 4.0)]


def run(monkeypatch, cfg, path):
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    return pipeline.run(cfg, path, n_frames=th.N)


def test_final_yaml_switch_cost_is_the_middle_one():
    assert FINAL["possession"]["switch_cost"] == 3.5       # the two experiments bracket it


@pytest.mark.parametrize("name, cost", CONFIGS)
def test_the_configs_are_final_with_only_the_switch_cost_changed(name, cost):
    cfg = load_config(EXP / f"{name}.yaml")
    assert cfg["possession"].pop("switch_cost") == cost
    final = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")
    final["possession"].pop("switch_cost")
    assert cfg == final


@pytest.mark.parametrize("name, cost", CONFIGS)
def test_the_configs_run_end_to_end(monkeypatch, tmp_path, name, cost):
    cfg = load_config(EXP / f"{name}.yaml")
    out = run(monkeypatch, cfg, tmp_path / "out.csv")
    assert out["rows"] == th.N
    assert set(np.unique(out["labels"])) <= {"H", "A", "D"}


def test_a_switch_cost_config_writes_the_same_file_twice(monkeypatch, tmp_path):
    cfg = load_config(EXP / "poss-switch-3.0.yaml")
    first = run(monkeypatch, cfg, tmp_path / "a.csv")
    second = run(monkeypatch, cfg, tmp_path / "b.csv")
    assert first["sha256"] == second["sha256"]
