"""ws:int: load a pipeline config from configs/*.yaml.

Every threshold, window and method choice lives in a config file, never hard-coded.

    cfg = load_config("configs/baseline.yaml")
    cfg["possession"]["window"]   # -> 50
"""

from pathlib import Path

import yaml

# Sections every config must have. Stage sections must also name a `method`.
STAGES = ("ball", "deadball", "possession")
REQUIRED = ("seed", *STAGES, "output")


class ConfigError(ValueError):
    """A config file is missing a section or has the wrong shape."""


def load_config(path):
    """Read and check one YAML config. Returns a dict."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")

    with path.open() as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing section(s) {missing}")

    if not isinstance(cfg["seed"], int):
        raise ConfigError(f"{path}: seed must be an integer")

    for stage in (*STAGES, "output"):
        if not isinstance(cfg[stage], dict):
            raise ConfigError(f"{path}: section {stage!r} must be a mapping")
    for stage in STAGES:
        if "method" not in cfg[stage]:
            raise ConfigError(f"{path}: section {stage!r} needs a 'method'")

    return cfg
