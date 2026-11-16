# configs/*.yaml load and have the shape every stage expects.
from pathlib import Path

import pytest

from falafelona.config import ConfigError, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_configs_load():
    assert load_config(CONFIGS / "baseline.yaml")["possession"]["method"] == "nearest_majority"
    assert load_config(CONFIGS / "final.yaml")["possession"]["method"] == "hmm"


def test_final_is_the_blend_config():
    """final.yaml carries the same settings as configs/exp/poss-blend-learned70-stopmodel-final.yaml."""
    assert load_config(CONFIGS / "final.yaml") == load_config(
        CONFIGS / "exp" / "poss-blend-learned70-stopmodel-final.yaml")


def test_final_is_touch_best_with_the_dead_score_blend():
    final = load_config(CONFIGS / "final.yaml")
    assert final["deadball"].pop("learned") == {"model": "models/deadball_stoppage_all.json"}
    assert final["deadball"]["weights"].pop("learned") == 0.7
    assert final["deadball"]["weights"]["gap_chance"] == 0.3
    final["deadball"]["weights"]["gap_chance"] = 1.0
    assert final == load_config(CONFIGS / "exp" / "poss-touch-best.yaml")


def test_touch_best_is_the_best_dev_set_config_before_the_blend():
    """The previous final.yaml: same settings as configs/exp/poss-touch-best.yaml."""
    assert load_config(CONFIGS / "exp" / "poss-touch-best.yaml") == load_config(
        CONFIGS / "exp" / "poss-touch-best.yaml")


def test_final_is_switch_placement_plus_touch_evidence():
    """The step before: poss-place-late.yaml, plus the touch block and nothing else."""
    final = load_config(CONFIGS / "exp" / "poss-touch-best.yaml")   # final before the blend
    assert final["possession"].pop("touch") == {"offset_s": 0.12, "min_change": 3.0,
                                                "distance": 2.0, "margin": 0.1, "hold_s": 0.8,
                                                "cost": 2.0}
    assert final == load_config(CONFIGS / "exp" / "poss-place-late.yaml")


def test_final_is_still_ball_restart_plus_switch_placement_plus_touch():
    """Two steps before: best-still-ball-restart.yaml, plus switch_placement 0.75 and touch."""
    final = load_config(CONFIGS / "exp" / "poss-touch-best.yaml")   # final before the blend
    assert final["possession"].pop("switch_placement") == 0.75
    final["possession"].pop("touch")
    assert final == load_config(CONFIGS / "exp" / "best-still-ball-restart.yaml")


def test_baseline_matches_official_smooth50():
    cfg = load_config(CONFIGS / "baseline.yaml")
    assert cfg["ball"]["method"] == "none"
    assert cfg["deadball"]["method"] == "ball_missing"
    assert cfg["possession"]["window"] == 50


def test_missing_section_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 0\nball: {method: none}\n")
    with pytest.raises(ConfigError, match="missing section"):
        load_config(bad)


def test_stage_without_method_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "seed: 0\nball: {method: none}\ndeadball: {method: ball_missing}\n"
        "possession: {window: 50}\noutput: {}\n"
    )
    with pytest.raises(ConfigError, match="'possession' needs a 'method'"):
        load_config(bad)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
