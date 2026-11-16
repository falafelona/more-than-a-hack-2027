# ws:poss: tests for the dead-ball sweep configs (task 7): cost of entering and leaving dead ball
# at 4 instead of 6, and the dead-score weight at 1.5 and 3.0 instead of 2.0. Made-up numbers
# only: nothing here reads match data.

import copy

import numpy as np
import pytest
import two_halves as th

from falafelona import pipeline, vendor
from falafelona.config import load_config

EXP = vendor.REPO_ROOT / "configs" / "exp"
FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")
NOSTILL = load_config(EXP / "poss-place-late-nostill.yaml")     # final without the still-ball clue
CONFIGS = [
    ("poss-dead-cost-4.0", {"to_dead_cost": 4.0, "from_dead_cost": 4.0}),
    ("poss-dead-weight-1.5", {"dead_weight": 1.5}),
    ("poss-dead-weight-3.0", {"dead_weight": 3.0}),
]


def run(monkeypatch, cfg, path):
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    return pipeline.run(cfg, path, n_frames=th.N)


@pytest.mark.parametrize("base", [FINAL, NOSTILL])
def test_the_bases_have_the_middle_values(base):
    possession = base["possession"]
    assert possession["to_dead_cost"] == 6.0       # the experiments move one setting off these
    assert possession["from_dead_cost"] == 6.0
    assert possession["dead_weight"] == 2.0


@pytest.mark.parametrize("suffix, base", [("", FINAL), ("-nostill", NOSTILL)])
@pytest.mark.parametrize("name, changes", CONFIGS)
def test_the_configs_are_their_base_with_only_those_settings_changed(name, changes, suffix, base):
    cfg = load_config(EXP / f"{name}{suffix}.yaml")
    for key, value in changes.items():
        assert cfg["possession"].pop(key) == value
    rest = copy.deepcopy(base)
    for key in changes:
        rest["possession"].pop(key)
    assert cfg == rest


@pytest.mark.parametrize("name, changes", CONFIGS)
def test_the_final_based_configs_run_end_to_end(monkeypatch, tmp_path, name, changes):
    cfg = load_config(EXP / f"{name}.yaml")
    out = run(monkeypatch, cfg, tmp_path / "out.csv")
    assert out["rows"] == th.N
    assert set(np.unique(out["labels"])) <= {"H", "A", "D"}


def test_a_sweep_config_writes_the_same_file_twice(monkeypatch, tmp_path):
    cfg = load_config(EXP / "poss-dead-cost-4.0.yaml")
    first = run(monkeypatch, cfg, tmp_path / "a.csv")
    second = run(monkeypatch, cfg, tmp_path / "b.csv")
    assert first["sha256"] == second["sha256"]
