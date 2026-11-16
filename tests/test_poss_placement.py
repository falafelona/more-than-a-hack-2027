# ws:poss: tests for the switch placement (possession.switch_placement, task 2). Made-up tracks
# only: nothing here reads match data.

import copy
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from falafelona import possession
from falafelona.config import load_config
from falafelona.vendor import REPO_ROOT

H, A, D = possession.H_STATE, possession.A_STATE, possession.D_STATE
FINAL = load_config(REPO_ROOT / "configs" / "final.yaml")
PITCH = (105.0, 68.0)
FPS = 25.0
TABLE = possession.switch_table(3.5, 6.0, 6.0)


def path_cost(path, costs, table=TABLE):
    path = np.asarray(path)
    total = costs[np.arange(len(path)), path].sum()
    return total + sum(table[a, b] for a, b in pairwise(path))


def evidence_gap(n=40, gap=(10, 30), left=H, right=A):
    """H evidence (cost against A) before the gap, A evidence after it, nothing inside."""
    costs = np.zeros((n, 3))
    costs[:gap[0], A] = 0.5 if left == H else 0.0
    costs[:gap[0], H] = 0.5 if left == A else 0.0
    costs[gap[1]:, H] = 0.5 if right == A else 0.0
    costs[gap[1]:, A] = 0.5 if right == H else 0.0
    costs[:, D] = 5.0                                         # never worth going dead here
    return costs


# --- place_switches -----------------------------------------------------------------------------


def test_viterbi_puts_the_switch_at_the_earliest_frame_of_the_gap():
    path = possession.viterbi(evidence_gap(), TABLE)
    assert (path[:10] == H).all() and (path[10:] == A).all()


@pytest.mark.parametrize("fraction, first_away", [(0.0, 10), (0.5, 20), (1.0, 30), (0.25, 15)])
def test_the_switch_moves_by_the_fraction_of_the_gap(fraction, first_away):
    costs = evidence_gap()
    path = possession.place_switches(possession.viterbi(costs, TABLE), costs, fraction)
    assert (path[:first_away] == H).all() and (path[first_away:] == A).all()


def test_the_total_cost_does_not_change():
    costs = evidence_gap()
    path = possession.viterbi(costs, TABLE)
    for fraction in (0.0, 0.3, 0.5, 1.0):
        moved = possession.place_switches(path, costs, fraction)
        assert path_cost(moved, costs) == pytest.approx(path_cost(path, costs))


def test_fraction_zero_changes_nothing():
    costs = evidence_gap()
    path = possession.viterbi(costs, TABLE)
    assert (possession.place_switches(path, costs, 0.0) == path).all()


def test_the_input_is_not_modified():
    costs = evidence_gap()
    path = possession.viterbi(costs, TABLE)
    before = path.copy()
    possession.place_switches(path, costs, 1.0)
    assert (path == before).all()


def test_a_switch_with_evidence_on_every_frame_stays():
    costs = np.zeros((20, 3))
    costs[:10, A] = 0.5
    costs[10:, H] = 0.5
    costs[:, D] = 5.0
    path = possession.viterbi(costs, TABLE)
    assert (possession.place_switches(path, costs, 1.0) == path).all()


def test_a_switch_through_dead_is_not_touched():
    costs = np.zeros((40, 3))
    costs[:10, A] = 0.5
    costs[10:30, :2] = 5.0                                   # dead: H and A cost the same, D is free
    costs[30:, H] = 0.5
    path = possession.viterbi(costs, TABLE)
    assert set(path[10:30]) == {D}
    assert (possession.place_switches(path, costs, 1.0) == path).all()


def test_two_switches_never_move_into_evidence_frames():
    costs = np.zeros((70, 3))
    costs[:10, A] = 1.0          # H evidence
    costs[28:40, H] = 1.0        # A evidence
    costs[52:, A] = 1.0          # H evidence again
    costs[:, D] = 9.0
    path = possession.viterbi(costs, TABLE)
    assert (path[10:28] == A).all() and (path[40:52] == H).all()     # both at their earliest frame
    moved = possession.place_switches(path, costs, 1.0)
    assert path_cost(moved, costs) == pytest.approx(path_cost(path, costs))
    assert (moved[:28] == H).all()
    assert (moved[28:52] == A).all()
    assert (moved[52:] == H).all()


def test_a_switch_towards_home_moves_the_same_way():
    costs = evidence_gap(left=A, right=H)
    path = possession.viterbi(costs, TABLE)
    moved = possession.place_switches(path, costs, 1.0)
    assert (moved[:30] == A).all() and (moved[30:] == H).all()


def test_short_paths_pass_through():
    for n in (0, 1):
        assert len(possession.place_switches(np.zeros(n, dtype=int), np.zeros((n, 3)), 1.0)) == n


@pytest.mark.parametrize("seed", range(20))
def test_random_costs_keep_the_cost_and_only_move_tied_frames(seed):
    rng = np.random.default_rng(seed)
    n = 300
    costs = np.zeros((n, 3))
    team = rng.random(n) < 0.35                               # a third of the frames have evidence
    costs[team, H] = rng.choice([0.0, 0.5], size=team.sum())
    costs[team, A] = 0.5 - costs[team, H]
    costs[:, D] = rng.choice([1.0, 3.0], size=n)
    path = possession.viterbi(costs, TABLE)
    for fraction in (0.0, 0.5, 1.0):
        moved = possession.place_switches(path, costs, fraction)
        assert path_cost(moved, costs) == pytest.approx(path_cost(path, costs))
        changed = moved != path
        assert (costs[changed, H] == costs[changed, A]).all()
        assert (moved[path == D] == D).all() and ((moved == D) == (path == D)).all()


# --- inside the HMM -----------------------------------------------------------------------------


def long_ball_match(pass_speed=20.0):
    """Home holds the ball at x = 30, plays it long, an away player at x = 60 receives it.

    One graded half, 25 fps, two players, no ball gaps. Returns the match and the frame where the
    ball arrives. The fast-ball rule removes the evidence while it flies.
    """
    hold, keep = 50, 100
    flight = round(30.0 / pass_speed * FPS)
    x = np.concatenate([np.full(hold, 30.5),
                        30.5 + pass_speed * np.arange(1, flight + 1) / FPS,
                        np.full(keep, 60.5)])
    n = len(x)
    ball = np.column_stack([x, np.full(n, 34.0)])
    players = np.zeros((n, 2, 2))
    players[:, 0] = [30.0, 34.0]
    players[:, 1] = [60.0, 34.0]
    match = SimpleNamespace(players_xy=players, ball_xy=ball, slot_team=np.array(["H", "A"]),
                            period=np.ones(n, dtype=np.int8), fps=FPS, pitch=PITCH)
    return match, hold + flight


def decode(match, placement=None):
    section = copy.deepcopy(FINAL["possession"])
    if placement is not None:
        section["switch_placement"] = placement
    labels = possession.predict(match, np.zeros(len(match.ball_xy)), {"possession": section})
    return "".join(labels)


def test_a_long_ball_switches_later_with_placement():
    match, _ = long_ball_match()
    early, late = decode(match), decode(match, 1.0)
    assert early.index("A") < late.index("A")
    assert late.count("A") + (late.index("A") - early.index("A")) == early.count("A")
    # every frame outside the flight is labelled as before
    assert early[:early.index("A")] == late[:early.index("A")]
    assert early[late.index("A"):] == late[late.index("A"):]


def test_placement_half_sits_between_the_two():
    match, _ = long_ball_match()
    first = {p: decode(match, p).index("A") for p in (None, 0.5, 1.0)}
    assert first[None] <= first[0.5] <= first[1.0]
    assert first[None] < first[1.0]


def test_placement_zero_is_the_unplaced_labels():
    match, _ = long_ball_match()
    assert decode(match, 0.0) == decode(match)


def test_evidence_on_every_frame_leaves_nothing_to_place():
    """The ball stays put; the home player next to it is replaced by an away player. Both teams
    have evidence all the time (after the lag), so there is no stretch to move the switch in."""
    n, change = 200, 100
    ball = np.tile([30.5, 34.0], (n, 1))
    players = np.zeros((n, 2, 2))
    players[:change, 0] = [30.0, 34.0]
    players[change:, 0] = [60.0, 34.0]
    players[:change, 1] = [60.0, 34.0]
    players[change:, 1] = [31.0, 34.0]
    match = SimpleNamespace(players_xy=players, ball_xy=ball, slot_team=np.array(["H", "A"]),
                            period=np.ones(n, dtype=np.int8), fps=FPS, pitch=PITCH)
    plain = decode(match)
    assert plain.startswith("H") and plain.endswith("A")
    assert decode(match, 1.0) == plain


def test_the_label_counts_move_by_the_stretch_only():
    match, _ = long_ball_match()
    early, late = decode(match), decode(match, 1.0)
    assert len(early) == len(late)
    assert set(early) == set(late) == {"H", "A"}


# --- settings -----------------------------------------------------------------------------------


def test_no_switch_placement_means_the_rule_is_off():
    assert possession._placement(FINAL["possession"]) is None


def test_final_yaml_does_not_use_switch_placement():
    assert "switch_placement" not in FINAL["possession"]


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 1, 1.0])
def test_good_placements_are_read(value):
    assert possession._placement({"switch_placement": value}) == float(value)


@pytest.mark.parametrize("value", [-0.1, 1.01, 2, True, False, "0.5", float("nan"),
                                   float("inf"), None, [0.5]])
def test_bad_placements_are_refused(value):
    with pytest.raises(ValueError, match="switch_placement"):
        possession._placement({"switch_placement": value})


# --- the experiment configs ---------------------------------------------------------------------


@pytest.mark.parametrize("name, value", [("poss-place-half", 0.5), ("poss-place-late", 0.75),
                                         ("poss-place-end", 1.0)])
def test_the_placement_configs_are_final_plus_the_placement(name, value):
    cfg = load_config(REPO_ROOT / "configs" / "exp" / f"{name}.yaml")
    assert cfg["possession"].pop("switch_placement") == value
    assert cfg == FINAL


@pytest.mark.parametrize("name", ["poss-place-half", "poss-place-late", "poss-place-end"])
def test_the_placement_configs_run_end_to_end(bundle, tmp_path, name):
    from falafelona import pipeline

    cfg = load_config(REPO_ROOT / "configs" / "exp" / f"{name}.yaml")
    result = pipeline.run(cfg, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert result["rows"] == bundle["n_frames"]
    assert sum(result["counts"].values()) == bundle["n_frames"]
