# ws:poss: the learned switch (hazard) model (learned_switch.py). Made-up numbers only.

import copy

import numpy as np
import pytest
import two_halves as th

from falafelona import ball, pipeline, possession, vendor
from falafelona import learned_switch as ls
from falafelona.config import load_config

FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")


def truth():
    out = np.full(th.N, "X", dtype="<U1")
    for first, stop in (th.HALF_1, th.HALF_2):
        out[first:stop] = "D"
    for first, stop, team in th.SPELLS:
        out[first:stop] = team
    out[th.GAP[0]:th.GAP[0] + 30] = "H"         # a change in the middle of the gap
    out[th.GAP[0] + 30:th.GAP[1]] = "A"
    return out


def cleaned():
    return ball.DataCleaner(th.make_match(), FINAL).return_cleaned_data()


def toy_model(tmp_path):
    rows = ls.training_rows(cleaned(), truth(), stride=1)
    folds = ls.fit_folds({"a": rows, "b": rows})
    path = tmp_path / "switch.json"
    ls.save_model(path, folds["all"], 10.0)
    return path, folds


def test_the_target_marks_frames_near_a_change_only():
    t = np.array(list("HHHHHDDDAAAAAAAAAA"))
    near = ls.switch_target(t, np.ones(len(t)), fps=4.0)        # +/- 2 frames
    assert near[6:11].all() and not near[:6].any() and not near[11:].any()
    halves = ls.switch_target(np.array(list("HHAA")), np.array([1, 1, 2, 2]), fps=4.0)
    assert not halves.any()                                       # a new half is not a change


def test_features_live_in_gaps_only_and_have_no_nan():
    from falafelona import learned_team as lt
    in_gap, x = ls.frame_features(lt.build_inputs(cleaned()))
    assert x.shape == (th.N, len(ls.FEATURES)) and np.isfinite(x).all()
    assert in_gap.any() and not x[~in_gap].any()


def test_auc_is_one_for_a_perfect_ranking_and_a_half_for_none():
    assert ls.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert ls.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_a_saved_model_loads_back_and_a_wrong_one_is_refused(tmp_path):
    path, folds = toy_model(tmp_path)
    doc = ls.load_model(path)
    assert doc["features"] == list(ls.FEATURES) and 0 < doc["base"] < 1
    assert "holdout" in folds["a"] and "holdout" not in folds["all"]
    path.write_text(path.read_text().replace(f'"version": {ls.VERSION}', '"version": 99'))
    with pytest.raises(ValueError):
        ls.load_model(path)


def test_switch_costs_only_get_cheaper_inside_gaps():
    in_gap = np.array([False, True, True, True])
    q = np.array([0.9, 0.05, 0.5, 0.99])
    cost = ls.team_switch_costs(in_gap, q, base=0.1, switch_cost=18.0, cost=4.0, min_cost=4.0)
    assert cost[0] == 18.0 and cost[1] == 18.0             # outside a gap / below the base
    assert 4.0 < cost[2] < 18.0 and cost[3] == 4.0         # lower, never under min_cost


def test_viterbi_switches_where_the_team_switch_is_cheap():
    costs = np.zeros((10, 3))
    costs[:, 2] = 100.0
    costs[:5, 1] = costs[5:, 0] = 1.0                       # H then A, 5 cost on each side
    flat = possession.viterbi(costs, possession.switch_table(18.0, 50.0, 50.0))
    assert set(flat[:5]) == set(flat[5:])                   # too expensive: one team
    cheap = np.full(10, 18.0)
    cheap[5] = 2.0
    path = possession.viterbi(costs, possession.switch_table(18.0, 50.0, 50.0), cheap)
    assert list(path) == [0] * 5 + [1] * 5


def test_for_match_fills_the_placeholder_and_bad_blocks_are_refused():
    cfg = {"possession": {"hazard": {"model": "m/switch_lomo_{match_id}.json"}}}
    assert ls.for_match(cfg, "X1")["possession"]["hazard"]["model"] == "m/switch_lomo_X1.json"
    for block in ({"model": "a.json"}, {"model": "a.json", "cost": -1, "min_cost": 1},
                  {"model": "a.json", "cost": 1, "min_cost": 1, "other": 2}):
        with pytest.raises(ValueError):
            ls.check_settings({"hazard": block})


def test_the_hazard_runs_end_to_end_and_writes_the_same_file_twice(monkeypatch, tmp_path):
    path, _ = toy_model(tmp_path)
    cfg = copy.deepcopy(FINAL)
    cfg["possession"]["hazard"] = {"model": str(path), "cost": 4.0, "min_cost": 4.0}
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    first = pipeline.run(cfg, tmp_path / "a.csv", n_frames=th.N)
    second = pipeline.run(cfg, tmp_path / "b.csv", n_frames=th.N)
    assert first["rows"] == th.N and first["sha256"] == second["sha256"]
    assert possession.CONFIG_KEYS["hazard"] == ls.CONFIG_KEYS
