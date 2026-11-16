"""ws:int: load a pipeline config from configs/*.yaml.

Every threshold, window and method choice lives in a config file, never hard-coded.

    cfg = load_config("configs/baseline.yaml")
    cfg["possession"]["window"]   # -> 50
"""

import difflib
from pathlib import Path

import yaml

from falafelona import deadball, possession

# Sections every config must have. Stage sections must also name a `method`.
STAGES = ("ball", "deadball", "possession")
REQUIRED = ("seed", *STAGES, "output")
# The sections whose every setting name is checked, and where each stage lists its names.
KNOWN_KEYS = {"deadball": deadball.CONFIG_KEYS, "possession": possession.CONFIG_KEYS}


class ConfigError(ValueError):
    """A config file is missing a section or has the wrong shape."""


def unknown_keys(cfg):
    """['possession.evidence_lagg', ...]: setting names that no stage reads, with a close match.

    Several features switch on just by being present, so a misspelt name would switch one off
    without an error. Each entry is the dotted name, plus " (did you mean ...?)" when a known
    name is close.
    """
    found = []
    for stage, blocks in KNOWN_KEYS.items():
        for block, allowed in blocks.items():
            part = cfg[stage] if block == "" else cfg[stage].get(block)
            if not isinstance(part, dict):
                continue
            prefix = stage if block == "" else f"{stage}.{block}"
            for key in part:
                if key in allowed:
                    continue
                close = difflib.get_close_matches(str(key), allowed, n=1)
                hint = f" (did you mean {close[0]!r}?)" if close else ""
                found.append(f"{prefix}.{key}{hint}")
    return found


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

    unknown = unknown_keys(cfg)
    if unknown:
        raise ConfigError(f"{path}: unknown setting(s) {', '.join(unknown)}. A new setting "
                          f"must be added to CONFIG_KEYS in deadball.py or possession.py")

    return cfg
