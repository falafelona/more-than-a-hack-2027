# ws:int: the typo guard (task M1). A misspelt setting must stop the build, not switch a
# feature off. Config files only, no match data.

import inspect

import pytest
import yaml

from falafelona import deadball, possession, vendor
from falafelona.config import ConfigError, load_config, unknown_keys

CONFIGS = vendor.REPO_ROOT / "configs"
# devset.yaml configures the dev-set tool, not the pipeline, so it has no stage sections.
# devset*.yaml configure the dev-set tool, not the pipeline, so they have no stage sections.
PIPELINE_CONFIGS = sorted(p for p in [*CONFIGS.glob("*.yaml"), *CONFIGS.glob("exp/*.yaml")]
                          if not p.name.startswith("devset"))

def write(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def final():
    return yaml.safe_load((CONFIGS / "final.yaml").read_text())


@pytest.mark.parametrize("path", PIPELINE_CONFIGS, ids=lambda p: p.name)
def test_every_config_in_the_repo_uses_only_known_settings(path):
    load_config(path)


@pytest.mark.parametrize(("where", "typo", "named"), [
    (("possession",), "evidence_lagg", "possession.evidence_lagg (did you mean 'evidence_lag'?)"),
    (("possession",), "fast_bal_speed", "possession.fast_bal_speed"),
    (("possession", "restart"), "boost", "possession.restart.boost"),
    (("possession", "velocity"), "half_window", "possession.velocity.half_window"),
    (("deadball",), "still_ball_sped", "deadball.still_ball_sped"),
    (("deadball", "speed"), "max_sped", "deadball.speed.max_sped"),
    (("deadball", "ball_speed"), "min_point", "deadball.ball_speed.min_point"),
    (("deadball", "gap"), "intercpt", "deadball.gap.intercpt"),
])
def test_a_misspelt_setting_stops_the_build(tmp_path, where, typo, named):
    cfg = final()
    block = cfg
    for key in where:
        block = block[key]
    block[typo] = 1.0

    with pytest.raises(ConfigError, match="unknown setting") as error:
        load_config(write(tmp_path, cfg))
    assert named in str(error.value)


def test_a_misspelt_optional_feature_is_caught_even_when_the_rest_is_valid():
    """The case that motivated the guard: `restrat:` would quietly turn the restart rule off."""
    cfg = final()
    cfg["possession"]["restrat"] = cfg["possession"].pop("restart")
    assert unknown_keys(cfg) == ["possession.restrat (did you mean 'restart'?)"]


def test_the_final_config_has_no_unknown_settings():
    assert unknown_keys(final()) == []


@pytest.mark.parametrize("module", [deadball, possession], ids=lambda m: m.__name__)
def test_every_listed_setting_is_read_somewhere_in_its_module(module):
    """A name in CONFIG_KEYS that the code never reads would be accepted and silently ignored."""
    source = inspect.getsource(module)
    for key in module.CONFIG_KEYS[""]:
        assert f'"{key}"' in source, f"{module.__name__}.CONFIG_KEYS lists {key!r} but never reads it"
