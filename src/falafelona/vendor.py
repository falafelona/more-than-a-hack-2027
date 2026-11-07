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

__all__ = [
    "CHALLENGE_DIR",
    "EVALUATION_DIR",
    "OFFICIAL_BASELINE",
    "VALIDATOR",
    "games",
    "possession_metrics",
]
