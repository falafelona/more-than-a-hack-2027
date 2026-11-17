# ws:dead: the learned dead-ball score inside the pipeline: the `learned` method, its configs and
# the dev set's leave-one-match-out hook. Made-up numbers only: nothing here reads match data.

import copy
import inspect

import numpy as np
import pytest
import two_halves as th
import yaml
from test_learned_dead import toy_fits

from falafelona import deadball, devset, learned_dead, pipeline, vendor
from falafelona.config import ConfigError, load_config

EXP = vendor.REPO_ROOT / "configs" / "exp"
FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")
# The configs of #54 were made from final.yaml before touch (= poss-place-late.yaml); the touch
# ones from final.yaml with touch (= poss-touch-best.yaml).
PRE_TOUCH = load_config(EXP / "poss-place-late.yaml")
DEV_NAMES = ["poss-learned", "poss-learned-dw1.5", "poss-learned-dw3.0", "poss-learned-touch"]
NAMES = [*DEV_NAMES, "poss-learned-final", "poss-learned-touch-final"]
DEAD_WEIGHT = {"poss-learned": 2.0, "poss-learned-dw1.5": 1.5, "poss-learned-dw3.0": 3.0,
               "poss-learned-final": 2.0, "poss-learned-touch": 2.0,
               "poss-learned-touch-final": 2.0}
TOUCH = load_config(EXP / "poss-touch-best.yaml")     # final.yaml before the blend
BASE = {name: TOUCH if "touch" in name else PRE_TOUCH for name in NAMES}
WEIGHTS = {"before_kickoff": 1.0, "out_of_bounds": 1.0, "before_restart": 0.6, "learned": 1.0}


def with_toy_model(cfg, tmp_path):
    """A copy of cfg whose learned model is a made-up one in tmp_path."""
    cfg = copy.deepcopy(cfg)
    learned_dead.save_model(tmp_path / "m.json", toy_fits(), ["toy"], 10.0,
                            {"gap": 1, "visible": 1})
    cfg["deadball"]["learned"]["model"] = str(tmp_path / "m.json")
    return cfg


def test_final_yaml_blends_the_full_stoppage_model():
    assert FINAL["deadball"]["weights"]["learned"] == 0.7      # exception taken (RESULTS.md)
    assert FINAL["deadball"]["learned"]["model"] == "models/deadball_stoppage3_all.json"
    learned_dead.load_model(learned_dead.model_path(FINAL["deadball"]))   # the file is committed


@pytest.mark.parametrize("name", NAMES)
def test_the_configs_are_final_with_the_learned_clue_swapped_in(name):
    cfg = load_config(EXP / f"{name}.yaml")
    assert cfg["deadball"].pop("weights") == WEIGHTS
    assert cfg["deadball"].pop("learned")["model"]
    assert cfg["possession"].pop("dead_weight") == DEAD_WEIGHT[name]
    final = copy.deepcopy(BASE[name])
    final["deadball"].pop("weights")
    final["possession"].pop("dead_weight")
    assert cfg == final


def test_the_dev_set_configs_use_leave_one_match_out_models_and_the_final_one_the_full_model():
    for name in DEV_NAMES:
        model = load_config(EXP / f"{name}.yaml")["deadball"]["learned"]["model"]
        assert model == "models/deadball_lomo_{match_id}.json"
    for name in ("poss-learned-final", "poss-learned-touch-final"):
        assert load_config(EXP / f"{name}.yaml")["deadball"]["learned"]["model"] \
            == "models/deadball_all.json"


def test_the_dev_set_fills_in_the_held_out_match_before_it_scores():
    assert "for_match" in inspect.getsource(devset.run_devset)


def test_learned_is_a_deadball_method_and_a_clue_of_combined():
    assert "learned" in deadball.METHODS
    assert "learned" in deadball.CONFIG_KEYS[""] and "learned" in deadball.CONFIG_KEYS
    assert deadball.SETTINGS["learned"] == learned_dead.SETTINGS


@pytest.mark.parametrize("name", ["poss-learned-final"])
def test_the_score_is_a_chance_on_every_frame_and_the_same_every_time(tmp_path, name):
    cfg = with_toy_model(load_config(EXP / f"{name}.yaml"), tmp_path)
    match = th.make_match()

    first = deadball.detect(match, None, cfg)
    second = deadball.detect(match, None, cfg)

    assert first.shape == (match.n_frames,) and np.isfinite(first).all()
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert np.array_equal(first, second)


def test_the_learned_method_alone_also_runs(tmp_path):
    cfg = with_toy_model(load_config(EXP / "poss-learned-final.yaml"), tmp_path)
    cfg["deadball"]["method"] = "learned"
    cfg["deadball"].pop("weights")

    out = deadball.detect(th.make_match(), None, cfg)

    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_a_config_with_learned_runs_end_to_end_and_writes_the_same_file_twice(monkeypatch,
                                                                              tmp_path):
    cfg = with_toy_model(load_config(EXP / "poss-learned-final.yaml"), tmp_path)
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())

    first = pipeline.run(cfg, tmp_path / "a.csv", n_frames=th.N)
    second = pipeline.run(cfg, tmp_path / "b.csv", n_frames=th.N)

    assert first["rows"] == th.N
    assert set(np.unique(first["labels"])) <= {"H", "A", "D"}
    assert first["sha256"] == second["sha256"]


def test_a_dev_set_config_is_refused_outside_the_dev_set():
    cfg = load_config(EXP / "poss-learned.yaml")

    with pytest.raises(ValueError, match="match_id"):
        deadball.detect(th.make_match(), None, cfg)


def test_a_missing_model_file_stops_the_run_with_a_clear_message(tmp_path):
    cfg = load_config(EXP / "poss-learned-final.yaml")
    cfg["deadball"]["learned"]["model"] = str(tmp_path / "nope.json")

    with pytest.raises(FileNotFoundError, match="fit_deadball"):
        deadball.detect(th.make_match(), None, cfg)


def test_a_misspelt_learned_setting_stops_the_build(tmp_path):
    cfg = copy.deepcopy(load_config(EXP / "poss-learned-final.yaml"))
    cfg["deadball"]["learned"]["modle"] = "x"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ConfigError, match=r"deadball\.learned\.modle"):
        load_config(path)


def test_learned_needs_its_block_and_the_speed_blocks():
    cfg = load_config(EXP / "poss-learned-final.yaml")
    for drop, message in (("learned", "learned"), ("speed", "speed"), ("ball_speed", "ball_speed")):
        section = copy.deepcopy(cfg["deadball"])
        section.pop(drop)
        with pytest.raises(ValueError, match=message):
            deadball.check_config(section, 25.0)


def test_the_committed_submission_model_loads():
    path = vendor.REPO_ROOT / "models" / "deadball_all.json"
    if not path.is_file():
        pytest.skip("models/deadball_all.json is not committed yet (scripts/fit_deadball.py)")
    doc = learned_dead.load_model(path)
    assert len(doc["trained_on"]) == 7
