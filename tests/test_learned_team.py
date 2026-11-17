# ws:poss: the learned team evidence (learned_team.py). Made-up numbers only.

import copy

import numpy as np
import pytest
import two_halves as th

from falafelona import ball, pipeline, possession, vendor
from falafelona import learned_team as lt
from falafelona.config import load_config

FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")


def truth():
    """Per-frame H / A / D / X of the scripted match: its spells, D elsewhere in the halves."""
    out = np.full(th.N, "X", dtype="<U1")
    for first, stop in (th.HALF_1, th.HALF_2):
        out[first:stop] = "D"
    for first, stop, team in th.SPELLS:
        out[first:stop] = team
    out[th.GAP[0]:th.GAP[1]] = "A"              # so the gap part has rows to fit on too
    return out


def cleaned():
    return ball.DataCleaner(th.make_match(), FINAL).return_cleaned_data()


def toy_model(tmp_path):
    """Fit on the scripted match (twice, so leave-one-out has two 'matches') and save it."""
    rows = lt.training_rows(cleaned(), truth(), stride=1)
    folds = lt.fit_folds({"a": rows, "b": rows})
    path = tmp_path / "team.json"
    lt.save_model(path, folds["all"]["fits"], ["a", "b"], 10.0, folds["all"]["rows"])
    return path, folds


def learned_cfg(path):
    cfg = copy.deepcopy(FINAL)
    cfg["possession"]["team_evidence"] = "learned"
    cfg["possession"]["learned"] = {"model": str(path)}
    return cfg


def test_features_have_the_right_shape_and_no_nan():
    in_gap, seen, gap = lt.frame_features(lt.build_inputs(cleaned()))
    assert seen.shape == (th.N, len(lt.SEEN_FEATURES))
    assert gap.shape == (th.N, len(lt.GAP_FEATURES))
    assert np.isfinite(seen).all() and np.isfinite(gap).all()
    assert in_gap[th.GAP[0]:th.GAP[1]].all() and not in_gap[:th.GAP[0]].any()
    assert not gap[~in_gap].any() and not seen[in_gap].any()


def test_margin_is_positive_when_home_is_closer():
    m = lt._margin(np.array([1.0, 5.0, np.inf, np.inf]), np.array([4.0, 2.0, 3.0, np.inf]))
    assert m[0] == pytest.approx(3.0) and m[1] == pytest.approx(-3.0)
    assert m[2] == -lt.MARGIN_CAP_M and m[3] == 0.0


def test_touch_signal_fades_and_resets_at_a_new_half():
    period = np.array([1] * 30 + [2] * 10, dtype=np.int8)
    signal, sign = lt._touch_signal(40, period, 25.0, np.array([5, 12]), np.array([0, 1]))
    assert signal[5] == 1.0 and signal[12] == -1.0 and 0 < -signal[20] < 1
    assert signal[30:].sum() == 0 and sign[5] == 1 and sign[12] == -1


def test_training_rows_keep_only_alive_graded_frames():
    rows = lt.training_rows(cleaned(), truth(), stride=1)
    alive = np.isin(truth(), ["H", "A"]).sum()
    assert len(rows["seen"][1]) + len(rows["gap"][1]) == 2 * alive     # as is + swapped
    assert set(np.unique(rows["seen"][1])) == {0.0, 1.0}
    assert rows["seen"][1].mean() == pytest.approx(0.5)               # no home lean to learn


def test_swap_flips_only_the_team_features_and_twice_is_nothing():
    for part, features in lt.PARTS.items():
        assert len(lt.SWAP[part]) == len(features)
        x = np.arange(1.0, len(features) + 1)[None, :]
        assert np.array_equal(lt.swap(part, lt.swap(part, x)), x)
    flipped = lt.swap("seen", np.ones((1, len(lt.SEEN_FEATURES))))[0]
    assert flipped[lt.SEEN_FEATURES.index("margin")] == -1
    assert flipped[lt.SEEN_FEATURES.index("log_near")] == 1


def test_the_chance_is_exactly_symmetric_between_the_teams(tmp_path):
    _, folds = toy_model(tmp_path)
    rng = np.random.default_rng(0)
    for part, features in lt.PARTS.items():
        x = rng.normal(size=(50, len(features)))
        fit = folds["all"]["fits"][part]
        p = lt.symmetric_chance(fit, part, x)
        assert np.allclose(lt.symmetric_chance(fit, part, lt.swap(part, x)), 1.0 - p)
        team_blind = np.where(np.asarray(lt.SWAP[part]) > 0, 3.0, 0.0)[None, :]
        assert lt.symmetric_chance(fit, part, team_blind)[0] == pytest.approx(0.5)


def test_a_fitted_model_beats_a_coin_on_the_frames_it_saw(tmp_path):
    _, folds = toy_model(tmp_path)
    loss, base, accuracy, n = folds["a"]["holdout"]["seen"]
    assert n > 0 and loss < base and accuracy > 0.8


def test_a_part_with_one_label_fits_a_flat_model():
    fit = lt._fit(np.ones((5, 3)), np.ones(5), 10.0)
    assert fit["coef"] == [0.0, 0.0, 0.0] and fit["intercept"] > 0


def test_a_saved_model_loads_back_and_a_wrong_one_is_refused(tmp_path):
    path, _ = toy_model(tmp_path)
    doc = lt.load_model(path)
    assert doc["seen"]["features"] == list(lt.SEEN_FEATURES)
    bad = tmp_path / "bad.json"
    bad.write_text(path.read_text().replace(f'"version": {lt.VERSION}', '"version": 1'))
    with pytest.raises(ValueError):
        lt.load_model(bad)
    with pytest.raises(FileNotFoundError):
        lt.load_model(tmp_path / "missing.json")


def test_home_chance_is_a_chance_and_neutral_outside_the_halves(tmp_path):
    path, _ = toy_model(tmp_path)
    p = lt.home_chance(cleaned(), {"learned": {"model": str(path)}})
    assert p.shape == (th.N,) and np.all((p >= 0) & (p <= 1))
    assert np.all(p[:th.HALF_1[0]] == 0.5)
    first_home = th.SPELLS[0]
    assert p[first_home[0] + 100] > 0.5          # a scripted home spell


def test_team_costs_put_zero_on_the_likelier_team():
    costs = lt.team_costs(np.array([0.9, 0.1, 0.5]), 0.5)
    assert costs[0, 0] == 0 and costs[0, 1] > 0
    assert costs[1, 1] == 0 and costs[1, 0] > 0
    assert np.all(costs[2] == 0)


def test_for_match_fills_the_placeholder_only():
    cfg = learned_cfg("models/team_lomo_{match_id}.json")
    filled = lt.for_match(cfg, "J03WPY")
    assert filled["possession"]["learned"]["model"] == "models/team_lomo_J03WPY.json"
    assert cfg["possession"]["learned"]["model"].endswith("{match_id}.json")
    assert lt.for_match(FINAL, "J03WPY") is FINAL
    with pytest.raises(ValueError):
        lt.model_path(cfg["possession"])


def test_bad_learned_blocks_are_refused():
    with pytest.raises(ValueError):
        lt.check_settings({"learned": {}})
    with pytest.raises(ValueError):
        lt.check_settings({"learned": {"model": "m.json", "typo": 1}})


def test_learned_evidence_runs_end_to_end_and_writes_the_same_file_twice(monkeypatch, tmp_path):
    path, _ = toy_model(tmp_path)
    cfg = learned_cfg(path)
    monkeypatch.setattr(pipeline, "load_match", lambda n_frames=None: th.make_match())
    first = pipeline.run(cfg, tmp_path / "a.csv", n_frames=th.N)
    second = pipeline.run(cfg, tmp_path / "b.csv", n_frames=th.N)
    assert first["rows"] == th.N and first["sha256"] == second["sha256"]
    assert set(np.unique(first["labels"])) <= {"H", "A", "D"}


def test_learned_is_a_known_team_evidence():
    assert "learned" in possession.TEAM_EVIDENCE
    assert possession.CONFIG_KEYS["learned"] == ("model", "add_cost", "where")


def test_the_fit_script_builds_rows_from_both_gap_placements():
    import importlib.util
    from types import SimpleNamespace

    import yaml

    script = vendor.REPO_ROOT / "scripts" / "fit_team.py"
    spec = importlib.util.spec_from_file_location("fit_team", script)
    fit_team = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_team)
    dev_cfg = yaml.safe_load((vendor.REPO_ROOT / "configs" / "devset.yaml").read_text())
    dev = SimpleNamespace(match=th.make_match(), truth=truth(), match_id="toy")
    rows = fit_team.match_rows(dev, FINAL, dev_cfg, seeds=[100], stride=5)
    alive = int((np.isin(truth(), ["H", "A"]) & (np.arange(th.N) % 5 == 0)).sum())
    assert len(rows["seen"][1]) + len(rows["gap"][1]) == 4 * alive  # 2 placements x swap



def _diag():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "team_diag", vendor.REPO_ROOT / "scripts" / "team_diag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_diagnostic_prints_aggregates_for_a_match(tmp_path):
    diag = _diag()
    path, _ = toy_model(tmp_path)
    s = diag.summarise(cleaned(), lt.load_model(path))
    assert s["seen"]["frames"] > 0 and s["gap"]["frames"] == th.GAP[1] - th.GAP[0]
    assert 0 <= s["seen"]["p_home"] <= 1 and 0 <= s["nearest_home"] <= 1
    assert s["tracking"]["H"] > 0 and s["tracking"]["A"] > 0
    both = diag.average([s, s])
    assert both["seen"]["frames"] == 2 * s["seen"]["frames"]
    assert both["seen"]["p_mean"] == pytest.approx(s["seen"]["p_mean"])
    text = diag.format_summary("toy", both)
    assert "chance home mean" in text and "margin" in text




def test_the_fit_script_can_leave_a_feature_out():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fit_team", vendor.REPO_ROOT / "scripts" / "fit_team.py")
    fit_team = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_team)
    rows = {"a": lt.training_rows(cleaned(), truth(), stride=1)}
    fit_team.drop_features(rows, ["spread_diff"])
    for part, features in lt.PARTS.items():
        assert not rows["a"][part][0][:, features.index("spread_diff")].any()
    with pytest.raises(SystemExit):
        fit_team.drop_features(rows, ["no_such_feature"])


def test_the_diagnostic_counts_touches_and_ball_speed(tmp_path):
    path, _ = toy_model(tmp_path)
    s = _diag().summarise(cleaned(), lt.load_model(path))
    assert s["touches_per_min"] >= 0 and s["ball_speed_median"] >= 0
    assert "touches per minute" in _diag().format_summary("toy", _diag().average([s]))



def test_the_ball_stats_see_jitter_and_count_the_touch_funnel():
    diag = _diag()
    clean = diag.ball_stats(cleaned())
    match = cleaned()
    noisy_xy = match.ball_xy + np.random.default_rng(0).normal(0, 0.3, match.ball_xy.shape)
    from dataclasses import replace
    noisy = diag.ball_stats(replace(match, ball_xy=noisy_xy))
    assert noisy["jitter_cm"] > 5 * max(clean["jitter_cm"], 0.1)
    assert noisy["speed_pct"][2] > clean["speed_pct"][2]
    for b in (clean, noisy):
        assert 0 <= b["clear_share"] <= 1 and b["peaks_per_seen_min"] >= b["touches_per_seen_min"]
    assert "touch funnel" in diag.format_ball(clean)



def test_the_lag_scan_finds_a_ball_that_runs_behind_the_players():
    from dataclasses import replace

    diag = _diag()
    match = cleaned()
    in_sync = diag.lag_scan(match)
    assert min(in_sync, key=lambda lag: in_sync[lag][0]) == 0
    late = np.full_like(match.ball_xy, np.nan)
    late[6:] = match.ball_xy[:-6]                      # the ball runs 6 frames behind
    scan = diag.lag_scan(replace(match, ball_xy=late))
    assert min(scan, key=lambda lag: scan[lag][0]) == -6
    assert "lowest at" in diag.format_lags(scan)



def test_smoothing_the_ball_rounds_corners_and_keeps_gaps():
    from dataclasses import replace

    diag = _diag()
    match = cleaned()
    same = diag.smooth_ball(match.ball_xy, match.period, 0)
    assert np.array_equal(np.isnan(same), np.isnan(match.ball_xy))
    smooth = diag.smooth_ball(match.ball_xy, match.period, 0.24 * th.FPS)
    assert np.array_equal(np.isnan(smooth[:, 0]), np.isnan(match.ball_xy[:, 0]))
    sharp = diag.ball_stats(match)
    blurred = diag.ball_stats(replace(match, ball_xy=smooth))
    assert blurred["peaks_per_seen_min"] <= sharp["peaks_per_seen_min"]
    assert blurred["jitter_cm"] <= sharp["jitter_cm"] + 1e-9



# --- v6: changes inside the gap, team clues x gap position (non-linear step A) ------------------


def test_v6_gap_features_start_at_zero_and_flip_with_the_teams():
    in_gap, _, gap = lt.frame_features(lt.build_inputs(cleaned()))
    f = list(lt.GAP_FEATURES)
    first = np.flatnonzero(in_gap)[0]
    for name in ("field_turn", "line_turn", "width_turn", "length_turn"):
        assert gap[first, f.index(name)] == 0.0                 # no change at the gap's start
    assert len(lt.SWAP["gap"]) == len(lt.GAP_FEATURES)
    assert all(lt.SWAP["gap"][f.index(n)] == -1 for n in lt.GAP_FEATURES[21:])
    assert np.isfinite(gap).all()


def test_a_v5_model_file_still_loads_and_gives_a_chance(tmp_path):
    import json

    path, _ = toy_model(tmp_path)
    doc = json.loads(path.read_text())
    k = len(lt.GAP_FEATURES_V5)
    doc["version"] = 5
    doc["gap"].update(features=list(lt.GAP_FEATURES_V5), mean=doc["gap"]["mean"][:k],
                      scale=doc["gap"]["scale"][:k], coef=doc["gap"]["coef"][:k])
    path.write_text(json.dumps(doc))
    chance = lt.home_chance(cleaned(), learned_cfg(path)["possession"])
    assert chance.shape == (th.N,) and ((chance > 0) & (chance < 1)).all()