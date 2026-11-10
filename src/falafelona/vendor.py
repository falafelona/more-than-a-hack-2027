"""ws:int: access to the official challenge code in vendor/challenge (pinned git submodule).

The rest of the package imports the organisers' modules from here, so we use their code exactly
as the official scorer does:

    from falafelona.vendor import games, possession_metrics

Never copy or edit the organisers' code (docs/decisions.md D3). Importing this module opens no
match data: `games` only builds paths.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_DIR = REPO_ROOT / "vendor" / "challenge"
EVALUATION_DIR = CHALLENGE_DIR / "evaluation"

# Official scripts, run as-is (e.g. by `make validate`).
VALIDATOR = EVALUATION_DIR / "validate_submission.py"
OFFICIAL_BASELINE = CHALLENGE_DIR / "starter" / "baseline.py"
OFFICIAL_SMART_TAGGING = CHALLENGE_DIR / "starter" / "load_smart_tagging.py"
OFFICIAL_HOMOGRAPHY = CHALLENGE_DIR / "starter" / "load_homography.py"

if not (CHALLENGE_DIR / "games.py").is_file():
    raise ImportError(
        f"official challenge code not found in {CHALLENGE_DIR}.\n"
        "Run: git submodule update --init"
    )

# Same import paths the official scripts set up for themselves.
for _path in (str(CHALLENGE_DIR), str(EVALUATION_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games
import possession_metrics


def _load_script(name, path):
    """An official starter script as a module. Scripts are not a package, so they load by path.

    Importing one only defines functions: its __main__ block, which reads the data, does not run.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The organisers' reference baseline. We use its write_submission so our CSV format is theirs.
baseline = _load_script("official_baseline", OFFICIAL_BASELINE)
# The organisers' Smart Tagging and homography readers, used by io.py (task 1.7).
smart_tagging = _load_script("official_load_smart_tagging", OFFICIAL_SMART_TAGGING)
homography = _load_script("official_load_homography", OFFICIAL_HOMOGRAPHY)

__all__ = [
    "CHALLENGE_DIR",
    "EVALUATION_DIR",
    "OFFICIAL_BASELINE",
    "OFFICIAL_HOMOGRAPHY",
    "OFFICIAL_SMART_TAGGING",
    "REPO_ROOT",
    "VALIDATOR",
    "baseline",
    "games",
    "homography",
    "possession_metrics",
    "smart_tagging",
]
