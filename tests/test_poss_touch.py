# ws:poss: touch evidence (possession.touch). Made-up numbers only, no match data.

import numpy as np
import pytest
import synthetic_idsse as syn

from falafelona import devset, possession, vendor
from falafelona.config import load_config
from falafelona.io import Match

FPS = 25.0
H, A = possession.H_STATE, possession.A_STATE
# final.yaml without any touch block, so the tests hold before and after final.yaml takes it
FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")
FINAL = {**FINAL, "possession": {k: v for k, v in FINAL["possession"].items() if k != "touch"}}
TOUCH = {"offset_s": 0.2, "min_change": 5.0, "distance": 1.5, "margin": 1.0, "hold_s": 0.4,
         "cost": 0.5}


def make_match(ball_xy, players_xy, slot_team, period=None):
    n = len(ball_xy)
    return Match(
        n_frames=n, fps=FPS,
        period=np.ones(n, dtype=np.int8) if period is None else period,
        ball_xy=np.asarray(ball_xy, dtype=float), ball_speed=np.full(n, np.nan),
        players_xy=np.asarray(players_xy, dtype=float),
        players_speed=np.full(players_xy.shape[:2], np.nan),
        slot_team=np.asarray(slot_team), pitch=(105.0, 68.0),
    )


# --- touches --------------------------------------------------------------------------------


def straight_then_back(n=100, turn=50, speed=10.0):
    """Ball moving +x at `speed` m/s, turned straight back at frame `turn`."""
    t = np.arange(n)
    x = np.where(t < turn, t, 2 * turn - t) * speed / FPS + 50.0
    return np.column_stack([x, np.full(n, 34.0)])


def test_a_turned_ball_is_one_touch_at_the_turn_and_a_straight_ball_is_none():
    velocity = FINAL["possession"]["velocity"]
    period = np.ones(100, dtype=np.int8)
    vxy, _ = possession.velocity.ball_velocity(straight_then_back(), period, FPS, velocity)
    touches = possession.ball_touches(vxy, offset=5, min_change=5.0)

    assert len(touches) == 1 and abs(touches[0] - 50) <= 1

    straight = np.column_stack([50.0 + np.arange(100) * 0.4, np.full(100, 34.0)])
    vxy, _ = possession.velocity.ball_velocity(straight, period, FPS, velocity)
    assert len(possession.ball_touches(vxy, offset=5, min_change=5.0)) == 0


def test_no_touch_where_the_velocity_is_unknown():
    vxy = np.full((50, 2), np.nan)
    vxy[:20] = [10.0, 0.0]
    assert len(possession.ball_touches(vxy, offset=5, min_change=1.0)) == 0


def test_the_toucher_must_be_close_and_clearly_closer_than_the_other_team():
    d_home = np.array([0.5, 3.0, 1.0, 2.5])
    d_away = np.array([3.0, 0.5, 1.2, 3.0])
    teams = possession.touch_teams(np.arange(4), d_home, d_away, distance=1.5, margin=1.0)

    assert teams.tolist() == [H, A, -1, -1]          # home, away, a duel, nobody close


def test_touch_costs_go_on_the_other_team_for_hold_frames():
    costs = possession.touch_team_costs(20, np.array([2, 10]), np.array([H, -1]), hold=4,
                                        cost=0.5)

    assert costs[2:6, A].tolist() == [0.5] * 4 and costs[:, H].sum() == 0
    assert costs[6:, A].sum() == 0                   # the unclear touch gives nothing


def touch_scene(swap=False, mirror=False):
    """Home player 0 carries the ball along, away player 1 waits; the ball turns at frame 50
    next to the away player (an interception)."""
    n = 100
    ball = straight_then_back(n)
    players = np.full((n, 2, 2), np.nan)
    players[:, 0] = ball + [-0.5, 0.0]
    players[:, 0, 0] = np.minimum(players[:, 0, 0], ball[49, 0] - 6.0)   # home stays behind
    players[:, 1] = ball[50] + [0.3, 0.0]                               # away at the turn
    slot = np.array(["H", "A"])
    if swap:
        slot = slot[::-1]
    if mirror:
        ball = ball * [-1, 1] + [105, 0]
        players = players * [-1, 1] + [105, 0]
    return make_match(ball, players, slot)


@pytest.mark.parametrize("swap, mirror", [(False, False), (True, False), (False, True)])
def test_touch_team_follows_the_player_and_not_the_side_or_the_slot_order(swap, mirror):
    section = {**FINAL["possession"], "touch": TOUCH}
    frames, teams = possession.touch_events(touch_scene(swap, mirror), section)

    assert len(frames) == 1
    assert teams[0] == (H if swap else A)


def test_touch_needs_a_velocity_block_and_checks_its_numbers():
    match = touch_scene()
    with pytest.raises(ValueError, match="velocity"):
        possession.touch_events(match, {"touch": TOUCH})
    with pytest.raises(ValueError, match="touch.cost"):
        possession.touch_events(match, {**FINAL["possession"], "touch": {**TOUCH, "cost": 0}})
    with pytest.raises(TypeError, match="mapping"):
        possession.touch_events(match, {**FINAL["possession"], "touch": 1.0})


# --- in the decoder -----------------------------------------------------------------------------


@pytest.fixture
def dev(tmp_path):
    syn.write_idsse(tmp_path, "J03WTEST")
    return devset.load_idsse("J03WTEST", root=tmp_path)


def with_blocks(**blocks):
    return {**FINAL, "possession": {**FINAL["possession"], **blocks}}


def test_evidence_that_never_fires_leaves_every_label_as_it_was(dev):
    cut, _ = devset.degrade_ball(dev.match, {"missing_share": 0.3, "gaps": [
        {"min_s": 0.5, "max_s": 8.0, "share": 1.0}]}, seed=0)
    base = devset.predict_labels(cut, FINAL)
    never = with_blocks(touch={**TOUCH, "min_change": 1e9})

    assert np.array_equal(base, devset.predict_labels(cut, never))


def test_a_flat_topped_change_gives_its_middle_frame():
    vxy = np.zeros((40, 2))
    vxy[20:] = [-10.0, 0.0]
    vxy[:20] = [10.0, 0.0]
    # change is 20 on frames 17..22 (offset 3), the turn is between 19 and 20
    touches = possession.ball_touches(vxy, offset=3, min_change=5.0)

    assert touches.tolist() == [20]


def test_the_decoder_adds_exactly_the_touch_costs(monkeypatch):
    seen = {}
    real = possession.viterbi

    def spy(costs, table):
        seen["costs"] = costs.copy()
        return real(costs, table)

    monkeypatch.setattr(possession, "viterbi", spy)
    match = touch_scene()
    dead = np.zeros(match.n_frames)
    possession.predict(match, dead, FINAL)
    base = seen["costs"]
    possession.predict(match, dead, with_blocks(touch=TOUCH))
    extra = seen["costs"] - base

    hold = round(TOUCH["hold_s"] * FPS)
    assert extra[:, A].sum() == 0                                   # away touched it
    assert extra[:, H].sum() == pytest.approx(TOUCH["cost"] * hold)


def test_evidence_check_counts_and_report(dev):
    cfg = with_blocks(touch=TOUCH)
    dev_cfg = {"seed": 0, "matches": ["J03WTEST"], "degrade": {"missing_share": 0.3, "gaps": [
        {"min_s": 0.5, "max_s": 8.0, "share": 1.0}]}}
    result = devset.run_devset({"final": FINAL, "both": cfg}, dev_cfg, [dev], diagnose=True)
    d = result["diagnosis"]["both"]["J03WTEST"]
    text = devset.format_report(result)

    assert "touches" not in result["diagnosis"]["final"]["J03WTEST"]
    assert 0 <= d["touches_right"] <= d["touches_scored"] <= d["touches_clear"] <= d["touches"]
    assert 0 <= d["switches_touched"] <= d["switches_seen"]
    assert 0 <= d["touches_claim_right"] <= d["touches_claim"] <= d["touches_scored"]
    assert "a claimed switch" in text
    assert "touch evidence:" in text
