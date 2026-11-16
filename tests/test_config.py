# configs/*.yaml load and have the shape every stage expects.
from pathlib import Path

import pytest

from falafelona.config import ConfigError, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_configs_load():
    assert load_config(CONFIGS / "baseline.yaml")["possession"]["method"] == "nearest_majority"
    assert load_config(CONFIGS / "final.yaml")["possession"]["method"] == "hmm"


def test_final_is_the_best_dev_set_config():
    """final.yaml carries the same settings as configs/exp/poss-place-late.yaml."""
    assert load_config(CONFIGS / "final.yaml") == load_config(
        CONFIGS / "exp" / "poss-place-late.yaml")


def test_final_is_still_ball_restart_plus_switch_placement():
    """The step before: best-still-ball-restart.yaml, plus switch_placement 0.75 and nothing else."""
    final = load_config(CONFIGS / "final.yaml")
    assert final["possession"].pop("switch_placement") == 0.75
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
