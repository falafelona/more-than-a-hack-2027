# configs/*.yaml load and have the shape every stage expects.
from pathlib import Path

import pytest

from falafelona.config import ConfigError, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_configs_load():
    assert load_config(CONFIGS / "baseline.yaml")["possession"]["method"] == "nearest_majority"
    assert load_config(CONFIGS / "final.yaml")["possession"]["method"] == "hmm"

def test_final_is_the_smoothed_ball_config():
    """final.yaml carries the same settings as configs/exp/poss-smooth-dist4-lag24-final.yaml."""
    load_config(CONFIGS / "exp" / "poss-smooth-dist4-lag24-final.yaml")


def test_final_is_the_blend_with_new_costs_and_timing():
    """The step before: the blend config, with only the decoder costs and the switch timing changed."""
    final = load_config(CONFIGS / "exp" / "poss-cost-timing-final.yaml")
    changed = {"switch_cost": 10, "to_dead_cost": 18, "from_dead_cost": 18,
               "evidence_lag": 18, "switch_placement": 1.0}
    assert {key: final["possession"][key] for key in changed} == changed
    blend = load_config(CONFIGS / "exp" / "poss-blend-learned70-stopmodel-final.yaml")
    final["possession"].update({key: blend["possession"][key] for key in changed})
    assert final == blend


def test_blend_is_touch_best_with_the_dead_score_blend():
    blend = load_config(CONFIGS / "exp" / "poss-blend-learned70-stopmodel-final.yaml")
    assert blend["deadball"].pop("learned") == {"model": "models/deadball_stoppage_all.json"}
    assert blend["deadball"]["weights"].pop("learned") == 0.7
    assert blend["deadball"]["weights"]["gap_chance"] == 0.3
    blend["deadball"]["weights"]["gap_chance"] = 1.0
    assert blend == load_config(CONFIGS / "exp" / "poss-touch-best.yaml")


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

def test_final_is_cost_timing_with_wider_touch_and_longer_lag():
    """The previous final (cost + timing) with touch distance 4.0 and evidence_lag 24 (D26)."""
    final = load_config(CONFIGS / "exp" / "poss-smooth-dist4-lag24-final.yaml")
    old = load_config(CONFIGS / "exp" / "poss-cost-timing-final.yaml")
    old["possession"]["evidence_lag"] = 24
    old["possession"]["touch"]["distance"] = 4.0
    assert final == old


def test_final_is_the_learned_touch_config_with_hold_2():
    """final.yaml (with its comments) = poss-touch-hold2.0-final.yaml, which is the learned-touch
    step (poss-touch-final.yaml) with touch.hold_s 2.0; a sweep tool cannot silently rewrite it."""
    final = load_config(CONFIGS / "final.yaml")
    assert final == load_config(CONFIGS / "exp" / "poss-touch-hold2.0-final.yaml")
    assert final["possession"]["touch"].pop("hold_s") == 2.0
    step = load_config(CONFIGS / "exp" / "poss-touch-final.yaml")
    assert step["possession"]["touch"].pop("hold_s") == 0.8
    assert final == step
    assert "SWEEP" not in (CONFIGS / "final.yaml").read_text()
