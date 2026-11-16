# ws:poss: the config for the "still-ball clue off" check of switch_placement (task 2). Made-up
# data only: these tests only read the config files.
import copy

import pytest

from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT

EXP = REPO_ROOT / "configs" / "exp"
FINAL = load_config(REPO_ROOT / "configs" / "final.yaml")
BASE = load_config(EXP / "best-restart.yaml")
STILL_BALL_KEYS = ("still_ball_speed", "still_ball_seconds", "spot_line_m", "spot_centre_m",
                   "goal_area_depth_m", "goal_area_half_width_m", "ball_speed")


def test_best_restart_is_final_without_the_still_ball_clue():
    """So 'still-ball off' is exactly final.yaml minus the clue, and the paired run is fair."""
    final = copy.deepcopy(FINAL)
    final["possession"].pop("switch_placement")                # final.yaml has it on, best-restart not
    deadball = final["deadball"]
    deadball["weights"].pop("still_ball_at_spot")
    for key in STILL_BALL_KEYS:
        deadball.pop(key)
    assert final == BASE


def test_the_nostill_config_is_best_restart_plus_switch_placement():
    cfg = load_config(EXP / "poss-place-late-nostill.yaml")
    assert cfg["possession"].pop("switch_placement") == 0.75
    assert cfg == BASE


@pytest.mark.parametrize("name", ["best-restart", "poss-place-late-nostill"])
def test_no_still_ball_clue_in_the_nostill_pair(name):
    cfg = load_config(EXP / f"{name}.yaml")
    assert "still_ball_at_spot" not in cfg["deadball"]["weights"]
