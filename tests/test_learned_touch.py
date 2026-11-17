# ws:poss: the learned touch model (learned_touch.py). Made-up numbers only.

import copy

import numpy as np
import pytest
import two_halves as th

from falafelona import ball, pipeline, possession, vendor
from falafelona import learned_touch as lt
from falafelona.config import load_config

FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")


def truth():
    out = np.full(th.N, "X", dtype="<U1")
    for first, stop in (th.HALF_1, th.HALF_2):
        out[first:stop] = "D"
    for first, stop, team in th.SPELLS:
        out[first:stop] = team
    return out


def cleaned():
    return ball.DataCleaner(th.make_match(), FINAL).return_cleaned_data()


def section(path=None):
    sec = copy.deepcopy(FINAL["possession"])
    sec["touch"]["min_change"] = 1.0              # more touches on the scripted match
    if path is not None:
        sec["touch_learned"] = {"model": str(path), "cost": 1.0}
    return sec


def toy_model(tmp_path):
    rows = lt.training_rows(cleaned(), truth(), section())
    folds = lt.fit_folds({"a": rows, "b": rows})
    path = tmp_path / "touch.json"
    lt.save_model(path, folds["all"], 10.0)
    return path, folds, rows


def test_features_are_finite_and_the_rule_column_is_the_rule():
    match = cleaned()
    touches, x = lt.touch_features(match, section())
    assert x.shape == (len(touches), len(lt.FEATURES)) and np.isfinite(x).all()
    assert len(touches) > 0
    _, rule = possession.touch_events(match, section())
    sign = np.where(rule == possession.H_STATE, 1, np.where(rule == possession.A_STATE, -1, 0))
    assert np.array_equal(x[:, lt.FEATURES.index("rule_team")], sign)


def test_swap_flips_only_the_team_columns_and_twice_is_nothing():
    x = np.arange(len(lt.FEATURES), dtype=float)[None, :] + 1.0
    assert np.array_equal(lt.swap(lt.swap(x)), x)
    flipped = lt.swap(x)[0] < 0
    assert [f for f, flip in zip(lt.FEATURES, flipped, strict=True) if not flip] == [
        "log_speed_before", "log_speed_after", "log_change", "log_near", "log_to_next",
        "log_since_prev", "seen_after"]


def test_training_rows_are_doubled_and_labelled_from_just_after_the_touch(tmp_path):
    match = cleaned()
    _, y, rule = lt.training_rows(match, truth(), section())
    touches, _ = lt.touch_features(match, section())
    label = lt.labels_after(truth(), touches, match.fps)
    kept = np.isin(label, ["H", "A"]) & (match.period[touches] > 0)
    assert len(y) == 2 * kept.sum() and len(rule) == len(y)
    assert y[:kept.sum()].tolist() == (label[kept] == "H").astype(float).tolist()
    assert np.array_equal(y[kept.sum():], 1.0 - y[:kept.sum()])


def test_the_chance_is_exactly_symmetric_and_a_saved_model_loads_back(tmp_path):
    path, folds, rows = toy_model(tmp_path)
    model = lt.load_model(path)
    x = rows[0][: len(rows[0]) // 2]
    p, q = lt.symmetric_chance(model, x), lt.symmetric_chance(model, lt.swap(x))
    assert np.allclose(p + q, 1.0)
    assert "holdout" in folds["a"] and "holdout" not in folds["all"]
    path.write_text(path.read_text().replace(f'"version": {lt.VERSION}', '"version": 99'))
    with pytest.raises(ValueError):
        lt.load_model(path)


def test_touch_costs_put_cost_only_on_the_unlikely_team_for_hold_frames(tmp_path):
    path, _, _ = toy_model(tmp_path)
    sec = section(path)
    costs = lt.touch_costs(cleaned(), sec)
    assert costs.shape == (th.N, 2) and (costs >= 0).all()
    assert not ((costs[:, 0] > 0) & (costs[:, 1] > 0)).any()     # never both teams at once
    assert (costs > 0).any()


def test_bad_blocks_are_refused_and_for_match_fills_the_placeholder():
    for block in ({"model": "a.json"}, {"model": "a.json", "cost": -1},
                  {"model": "a.json", "cost": 1, "other": 2}, {"model": "", "cost": 1}):
        with pytest.raises(ValueError):
            lt.check_settings({"touch": {}, "touch_learned": block})
    with pytest.raises(ValueError, match="touch block"):
        lt.check_settings({"touch_learned": {"model": "a.json", "cost": 1}})
    cfg = {"possession": {"touch_learned": {"model": "m/touch_lomo_{match_id}.json"}}}
    assert lt.for_match(cfg, "X1")["possession"]["touch_learned"]["model"] == \
        "m/touch_lomo_X1.json"
    assert possession.CONFIG_KEYS["touch_learned"] == lt.CONFIG_KEYS


def test_the_learned_touches_run_end_to_end_and_write_the_same_file_twice(monkeypatch,
                                                                          tmp_path):
    path, _, _ = toy_model(tmp_path)
    cfg = copy.deepcopy(FINAL)
    cfg["possession"] = section(path)
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    first = pipeline.run(cfg, tmp_path / "a.csv", n_frames=th.N)
    second = pipeline.run(cfg, tmp_path / "b.csv", n_frames=th.N)
    assert first["rows"] == th.N and first["sha256"] == second["sha256"]


def test_the_fit_script_builds_rows_from_both_gap_placements():
    import importlib.util
    from types import SimpleNamespace

    import yaml

    script = vendor.REPO_ROOT / "scripts" / "fit_touch.py"
    spec = importlib.util.spec_from_file_location("fit_touch", script)
    fit_touch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_touch)
    dev_cfg = yaml.safe_load((vendor.REPO_ROOT / "configs" / "devset.yaml").read_text())
    dev = SimpleNamespace(match=th.make_match(), truth=truth(), match_id="toy")
    cfg = copy.deepcopy(FINAL)
    cfg["possession"] = section()
    x, y, rule = fit_touch.match_rows(dev, cfg, dev_cfg, seeds=[100])
    assert x.shape[1] == len(lt.FEATURES) and len(x) == len(y) == len(rule)
