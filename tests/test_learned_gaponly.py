# ws:dead: the gap-only option of the learned dead-ball score (learned.visible: false) and its
# experiment config. Made-up numbers only.

import copy

import numpy as np
import pytest
import two_halves as th
from test_learned_dead_pipeline import with_toy_model

from falafelona import deadball, learned_dead, vendor
from falafelona.config import load_config

EXP = vendor.REPO_ROOT / "configs" / "exp"
FINAL = load_config(vendor.REPO_ROOT / "configs" / "exp" / "poss-place-late.yaml")   # final before touch


def test_the_gap_only_config_is_final_with_gap_chance_swapped_for_learned_on_gaps():
    cfg = load_config(EXP / "poss-learned-gaponly.yaml")
    weights = cfg["deadball"].pop("weights")
    final_weights = copy.deepcopy(FINAL["deadball"]["weights"])
    assert final_weights.pop("gap_chance") == 1.0
    assert weights == {**final_weights, "learned": 1.0}
    learned = cfg["deadball"].pop("learned")
    assert learned == {"model": "models/deadball_lomo_{match_id}.json", "visible": False}
    final = copy.deepcopy(FINAL)
    final["deadball"].pop("weights")
    assert cfg == final                      # dead_weight and everything else as in final.yaml


def test_visible_false_scores_only_the_gap_frames(tmp_path):
    cfg = with_toy_model(load_config(EXP / "poss-learned-gaponly.yaml"), tmp_path)
    cfg["deadball"]["method"] = "learned"
    cfg["deadball"].pop("weights")
    both = copy.deepcopy(cfg)
    both["deadball"]["learned"]["visible"] = True
    match = th.make_match()
    gap = np.isnan(match.ball_xy).any(axis=1)

    only_gaps = deadball.detect(match, None, cfg)
    everything = deadball.detect(match, None, both)

    assert gap.any() and (~gap).any()
    assert np.array_equal(only_gaps[gap], everything[gap])
    assert (only_gaps[~gap] == 0).all()
    assert (everything[~gap] > 0).any()


def test_visible_must_be_true_or_false():
    section = copy.deepcopy(load_config(EXP / "poss-learned-gaponly.yaml")["deadball"])
    section["learned"]["visible"] = "no"
    with pytest.raises(TypeError, match="visible"):
        learned_dead.check_settings(section)
