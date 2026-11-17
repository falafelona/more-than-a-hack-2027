# ws:dead: tests for the learned dead-ball score (falafelona.learned_dead). Made-up numbers only:
# nothing here reads match data.

import json
from types import SimpleNamespace

import numpy as np
import pytest

from falafelona import learned_dead as ld

FPS = 25.0
PITCH = (105.0, 68.0)
SECTION = {
    "still_speed": 1.0, "still_ball_speed": 0.5, "still_ball_seconds": 2.0, "spot_line_m": 2.0,
    "spot_centre_m": 3.0, "goal_area_depth_m": 5.5, "goal_area_half_width_m": 9.16,
    "out_margin": 1.0,
    "speed": {"half_window_s": 0.2, "min_points": 5, "min_each_side": 2, "max_speed": 12.0},
    "ball_speed": {"half_window_s": 0.2, "min_points": 5, "min_each_side": 2},
    "learned": {"model": "models/x.json"},
}


def inputs(n=1000, gaps=(), ball=None, period=None, player_speed=None):
    """Hand-made inputs for frame_features: a still ball at the middle, with `gaps` cut out."""
    xy = np.tile(np.array([50.0, 30.0]), (n, 1)) if ball is None else np.array(ball, float)
    for first, stop in gaps:
        xy[first:stop] = np.nan
    period = np.ones(n, dtype=np.int8) if period is None else period
    speed = np.full(n, 2.0) if player_speed is None else np.asarray(player_speed, float)
    missing = np.isnan(xy[:, 0])
    edges = np.flatnonzero((missing[1:] != missing[:-1]) | (period[1:] != period[:-1])) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [n]))
    runs = [(int(a), int(b)) for a, b in zip(starts, stops, strict=True) if missing[a]]
    return {
        "ball_xy": xy, "period": period, "fps": FPS, "pitch": PITCH,
        "ball_speed": np.where(missing, np.nan, 0.0), "slow_ball": ~missing,
        "still_ball": np.zeros(n, bool), "at_spot": np.zeros(n), "out_margin": 1.0,
        "gap_runs": runs, "player_speed": speed, "slow_players": speed < 1.0,
    }


def column(names, name):
    return names.index(name)


# --- the features ------------------------------------------------------------------------------


def test_centred_mean_matches_a_brute_force_mean_and_skips_nan():
    rng = np.random.default_rng(1)
    values = rng.uniform(0, 5, 60)
    values[10:14] = np.nan
    period = np.array([1] * 35 + [2] * 25, dtype=np.int8)

    got = ld.centred_mean(values, period, 4)

    for i in range(60):
        same = np.flatnonzero(period == period[i])
        window = values[max(i - 4, same[0]):min(i + 5, same[-1] + 1)]
        window = window[np.isfinite(window)]
        assert got[i] == pytest.approx(window.mean() if window.size else 0.0)


def test_a_windowed_mean_never_looks_across_the_halves():
    period = np.array([1] * 10 + [2] * 10, dtype=np.int8)
    values = np.array([0.0] * 10 + [10.0] * 10)

    got = ld.centred_mean(values, period, 5)

    assert got[9] == 0.0 and got[10] == 10.0


def test_run_lengths_give_each_frame_the_length_of_its_run():
    flags = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool)

    assert ld._run_lengths(flags).tolist() == [0, 3, 3, 3, 0, 0, 2, 2, 0, 1]


def test_the_gap_features_describe_where_the_ball_left_and_came_back():
    n = 1000
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    ball[399] = (1.0, 30.0)                 # last seen 1 m from the goal line
    ball[500] = (52.5, 34.0)                # comes back on the centre spot
    inp = inputs(n, gaps=[(400, 500)], ball=ball)
    inp["ball_speed"][399] = 3.0

    in_gap, gap_x, vis_x = ld.frame_features(inp)

    row = gap_x[450]
    f = ld.GAP_FEATURES
    assert in_gap.sum() == 100 and not in_gap[399] and not in_gap[500]
    assert row[column(f, "log2_gap_s")] == pytest.approx(2.0)          # 4 s
    assert row[column(f, "last_near_line")] == pytest.approx(np.exp(-1.0 / 3.0))
    assert row[column(f, "next_near_line")] == pytest.approx(np.exp(-34.0 / 3.0))
    assert row[column(f, "next_at_centre")] == pytest.approx(1.0)
    assert row[column(f, "open_ended")] == 0.0
    assert row[column(f, "last_ball_speed")] == pytest.approx(np.log1p(3.0))
    assert row[column(f, "gap_players_slow")] == 0.0
    assert not gap_x[~in_gap].any() and not vis_x[in_gap].any()        # other rows stay zero


def test_a_gap_at_the_edge_of_a_half_is_open_ended_without_a_last_or_next_ball():
    inp = inputs(500, gaps=[(0, 100)])

    _, gap_x, _ = ld.frame_features(inp)

    f = ld.GAP_FEATURES
    assert gap_x[50, column(f, "open_ended")] == 1.0
    assert gap_x[50, column(f, "last_near_line")] == 0.0


def test_the_player_speed_windows_follow_the_players_frame_by_frame_inside_one_gap():
    speed = np.concatenate([np.full(400, 6.0), np.full(300, 0.2), np.full(300, 6.0)])
    inp = inputs(1000, gaps=[(150, 850)], player_speed=speed)             # a 28 s gap

    _, gap_x, _ = ld.frame_features(inp)

    w1 = column(ld.GAP_FEATURES, "players_speed_w1")
    assert gap_x[200, w1] > gap_x[550, w1] + 1.0       # running, then standing, then running
    assert gap_x[800, w1] > gap_x[550, w1] + 1.0
    length = column(ld.GAP_FEATURES, "log2_gap_s")
    assert np.ptp(gap_x[150:850, length]) == 0.0       # one value per gap, not per frame


def test_the_seen_ball_features_mark_a_still_ball_at_a_spot_and_a_ball_out_of_play():
    n = 400
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    ball[100:200] = (0.5, 30.0)             # on the goal line, still
    ball[300:310] = (-3.0, 30.0)            # 3 m past it
    inp = inputs(n, ball=ball)
    inp["still_ball"][100:200] = True
    inp["at_spot"][100:200] = 1.0

    _, _, vis_x = ld.frame_features(inp)

    v = ld.VISIBLE_FEATURES
    assert vis_x[150, column(v, "ball_still")] == 1.0
    assert vis_x[150, column(v, "ball_still_at_spot")] == 1.0
    assert vis_x[150, column(v, "near_line")] == pytest.approx(np.exp(-0.5 / 3.0))
    assert vis_x[150, column(v, "log_slow_run")] == pytest.approx(np.log1p(n / FPS))
    assert vis_x[305, column(v, "out_of_bounds")] == 1.0
    assert vis_x[50, column(v, "out_of_bounds")] == 0.0
    assert vis_x[50, column(v, "ball_still")] == 0.0


def test_features_are_finite_and_the_same_every_time():
    rng = np.random.default_rng(3)
    ball = np.cumsum(rng.normal(0, 0.3, (800, 2)), axis=0) + np.array([50.0, 30.0])
    inp = inputs(800, gaps=[(100, 220), (500, 780)], ball=ball)

    first = ld.frame_features(inp)
    second = ld.frame_features(inp)

    for a, b in zip(first, second, strict=True):
        assert np.isfinite(a).all() and np.array_equal(a, b)


# --- fitting -----------------------------------------------------------------------------------


def test_fit_logistic_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40_000, 3)) * [1.0, 3.0, 0.5] + [0.0, 5.0, -1.0]
    true = np.array([1.5, -0.4, 0.8])
    p = 1 / (1 + np.exp(-(x @ true - 0.7)))
    y = (rng.uniform(size=len(x)) < p).astype(float)

    fit = ld.fit_logistic(x, y, l2=1e-3)

    raw = np.array(fit["coef"]) / np.array(fit["scale"])
    assert raw == pytest.approx(true, abs=0.08)
    assert ld.predict_proba(fit, x).mean() == pytest.approx(y.mean(), abs=0.01)


def test_fit_logistic_is_deterministic_and_survives_separable_data():
    x = np.linspace(-1, 1, 200)[:, None]
    y = (x[:, 0] > 0).astype(float)

    first = ld.fit_logistic(x, y)
    second = ld.fit_logistic(x, y)

    assert first == second
    assert np.isfinite(first["coef"]).all() and np.isfinite(first["intercept"])
    assert ld.predict_proba(first, [[0.9]])[0] > 0.9
    assert ld.predict_proba(first, [[-0.9]])[0] < 0.1


def test_fit_logistic_refuses_one_class_and_bad_shapes():
    with pytest.raises(ValueError, match="same label"):
        ld.fit_logistic(np.zeros((5, 2)), np.ones(5))
    with pytest.raises(ValueError, match="x must be"):
        ld.fit_logistic(np.zeros(5), np.zeros(5))


def test_a_constant_column_gets_no_weight_and_no_division_by_zero():
    rng = np.random.default_rng(2)
    x = np.column_stack([rng.normal(size=500), np.full(500, 7.0)])
    y = (x[:, 0] + rng.normal(size=500) > 0).astype(float)

    fit = ld.fit_logistic(x, y)

    assert abs(fit["coef"][1]) < 1e-6 and np.isfinite(fit["coef"]).all()


def test_log_loss_is_zero_for_sure_right_answers_and_log2_for_a_coin():
    assert ld.log_loss([1.0, 0.0], [1, 0]) < 1e-5
    assert ld.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))


# --- the model file ----------------------------------------------------------------------------


def toy_fits(seed=0):
    rng = np.random.default_rng(seed)
    fits = {}
    for name, features in ld.PARTS.items():
        x = rng.normal(size=(300, len(features)))
        y = (x[:, 0] + rng.normal(size=300) > 0).astype(float)
        fits[name] = ld.fit_logistic(x, y)
    return fits


def test_a_saved_model_loads_back_with_six_digit_numbers(tmp_path):
    fits = toy_fits()
    path = tmp_path / "models" / "m.json"

    ld.save_model(path, fits, ["B", "A"], 10.0, {"gap": 5, "visible": 7})
    doc = ld.load_model(path)

    assert doc["trained_on"] == ["A", "B"] and doc["rows"] == {"gap": 5, "visible": 7}
    assert doc["gap"]["features"] == list(ld.GAP_FEATURES)
    assert doc["visible"]["coef"] == pytest.approx(fits["visible"]["coef"], rel=1e-5)
    assert path.read_text() == path.read_text() and path.read_text().endswith("\n")


def test_saving_twice_gives_the_same_bytes(tmp_path):
    fits = toy_fits()

    ld.save_model(tmp_path / "a.json", fits, ["A"], 10.0, {"gap": 1, "visible": 1})
    ld.save_model(tmp_path / "b.json", fits, ["A"], 10.0, {"gap": 1, "visible": 1})

    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.update(version=99), "not a version"),
    (lambda d: d.update(windows_s=[1.0, 2.0]), "other player-speed windows"),
    (lambda d: d["gap"].update(features=["x"]), "other features"),
    (lambda d: d["visible"].update(coef=[1.0]), "finite numbers"),
    (lambda d: d["gap"].update(scale=[0.0] * len(ld.GAP_FEATURES)), "scale must be above 0"),
    (lambda d: d["gap"].update(intercept=float("nan")), "intercept"),
])
def test_a_model_file_made_for_something_else_is_refused(tmp_path, edit, message):
    path = tmp_path / "m.json"
    ld.save_model(path, toy_fits(), ["A"], 10.0, {"gap": 1, "visible": 1})
    doc = json.loads(path.read_text())
    edit(doc)
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match=message):
        ld.load_model(path)


def test_a_missing_model_file_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_deadball"):
        ld.load_model(tmp_path / "nope.json")


# --- config ------------------------------------------------------------------------------------


def test_the_learned_block_is_checked():
    ld.check_settings({"learned": {"model": "models/x.json"}})

    for bad in ({}, {"learned": {}}, {"learned": {"model": ""}}, {"learned": {"model": 3}},
                {"learned": {"model": "m.json", "extra": 1}}):
        with pytest.raises(ValueError, match="learned"):
            ld.check_settings(bad)


def test_for_match_fills_in_the_held_out_match_and_leaves_other_configs_alone():
    cfg = {"deadball": {"method": "learned", "learned": {"model": "models/lomo_{match_id}.json"}}}
    plain = {"deadball": {"method": "learned", "learned": {"model": "models/all.json"}}}
    other = {"deadball": {"method": "ball_missing"}}

    filled = ld.for_match(cfg, "J03WPY")

    assert filled["deadball"]["learned"]["model"] == "models/lomo_J03WPY.json"
    assert cfg["deadball"]["learned"]["model"] == "models/lomo_{match_id}.json"   # not changed
    assert ld.for_match(plain, "J03WPY") is plain and ld.for_match(other, "J03WPY") is other


def test_a_model_path_with_the_placeholder_is_refused_outside_the_dev_set(tmp_path):
    section = {"learned": {"model": "models/lomo_{match_id}.json"}}

    with pytest.raises(ValueError, match="match_id"):
        ld.model_path(section, tmp_path)
    assert ld.model_path({"learned": {"model": "models/a.json"}}, tmp_path) \
        == tmp_path / "models" / "a.json"


# --- fitting on a made-up match, and scoring with it ---------------------------------------------


def made_up_match(seed=0, n=3000):
    """Two halves; the ball is cut where the players stand still, so there is something to learn."""
    rng = np.random.default_rng(seed)
    period = np.concatenate([np.ones(n // 2), np.full(n - n // 2, 2)]).astype(np.int8)
    spells = ((400, 700), (1900, 2300), (2500, 2700))   # players standing still
    t = np.arange(n)[:, None]
    players = np.zeros((n, 4, 2))
    for slot in range(4):
        phase = rng.uniform(0, 6, 2)
        walk = 50 + 8 * np.sin(t / np.array([60.0, 50.0]) + phase)       # about 3 m/s
        for first, stop in spells:
            walk[first:stop] = walk[first]               # frozen: speed 0
        players[:, slot] = walk + rng.uniform(-20, 20, 2)
    ball = np.tile(np.array([52.5, 34.0]), (n, 1)) + rng.normal(0, 5, (n, 2))
    for first, stop in ((380, 720), (1000, 1100), (1880, 2320)):
        ball[first:stop] = np.nan
    ball[2500:2700] = (0.5, 20.0)                        # seen, standing at a throw-in spot
    truth = np.full(n, "H")
    for first, stop in spells:
        truth[first:stop] = "D"
    return SimpleNamespace(n_frames=n, fps=FPS, period=period, ball_xy=ball, players_xy=players,
                           pitch=PITCH), truth


def test_fitting_on_a_made_up_match_learns_that_still_players_mean_dead_ball(tmp_path):
    match, truth = made_up_match()
    rows = ld.training_rows(match, truth, SECTION, stride=5)
    fits = {name: ld.fit_logistic(*rows[name], l2=1.0) for name in ld.PARTS}
    ld.save_model(tmp_path / "m.json", fits, ["made-up"], 1.0,
                  {name: len(rows[name][1]) for name in ld.PARTS})

    section = {**SECTION, "learned": {"model": str(tmp_path / "m.json")}}
    out = ld.score(match, section)

    assert out.shape == (match.n_frames,) and np.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()
    assert out[400:700].mean() > 0.8                     # players standing, ball gone: dead
    assert out[1000:1100].mean() < 0.2                   # players running, ball gone: alive
    assert out[1900:2300].mean() > 0.8
    assert out[2500:2700].mean() > 0.8                   # ball seen standing at the line: dead
    assert out[2750:2950].mean() < 0.2                   # play on
    assert ld.score(match, section).tolist() == out.tolist()      # same every time


def test_training_rows_keep_only_graded_scored_frames_every_stride_frames():
    match, truth = made_up_match()
    truth = truth.copy()
    truth[:100] = "X"                                    # not scored
    match.period[:50] = 0                                # not graded

    rows = ld.training_rows(match, truth, SECTION, stride=10)

    total = sum(len(rows[name][1]) for name in ld.PARTS)
    assert total == len([i for i in range(100, match.n_frames) if i % 10 == 0])
    assert all(set(np.unique(rows[name][1])) <= {0.0, 1.0} for name in ld.PARTS)
    assert all(rows[name][0].shape[1] == len(ld.PARTS[name]) for name in ld.PARTS)


def test_score_works_on_a_match_with_no_gap_at_all(tmp_path):
    match, truth = made_up_match()
    rows = ld.training_rows(match, truth, SECTION, stride=5)
    fits = {name: ld.fit_logistic(*rows[name], l2=1.0) for name in ld.PARTS}
    ld.save_model(tmp_path / "m.json", fits, ["made-up"], 1.0, {"gap": 1, "visible": 1})
    match.ball_xy[:] = np.array([52.5, 34.0])
    section = {**SECTION, "learned": {"model": str(tmp_path / "m.json")}}

    out = ld.score(match, section)

    assert np.isfinite(out).all() and out.shape == (match.n_frames,)


# --- leave-one-match-out -------------------------------------------------------------------------


def rows_for(seed):
    match, truth = made_up_match(seed)
    return ld.training_rows(match, truth, SECTION, stride=5)


def test_fit_folds_makes_one_model_per_match_and_one_for_all_and_the_holdout_beats_the_base_rate():
    rows = {"A": rows_for(1), "B": rows_for(2), "C": rows_for(3)}

    folds = ld.fit_folds(rows, l2=1.0)

    assert sorted(folds) == ["A", "B", "C", "all"]
    assert folds["A"]["trained_on"] == ["B", "C"] and folds["all"]["trained_on"] == ["A", "B", "C"]
    assert folds["all"]["holdout"] == {}
    for name in "ABC":
        for part in ld.PARTS:
            loss, base, count = folds[name]["holdout"][part]
            assert count == len(rows[name][part][1])
            assert loss < base                    # the held-out match is predicted better than
            #                                       by a constant: the model learned something real
    assert folds["A"]["rows"]["gap"] == len(rows["B"]["gap"][1]) + len(rows["C"]["gap"][1])


def test_fit_folds_needs_a_match_to_hold_out():
    with pytest.raises(ValueError, match="at least two"):
        ld.fit_folds({"A": rows_for(1)})


# --- v3 gap features -----------------------------------------------------------------------------


def test_time_in_the_gap_and_the_edge_clues_fade_away_from_their_edge():
    n = 1000
    ball = np.tile(np.array([50.0, 30.0]), (n, 1))
    ball[199] = (50.0, 0.2)                     # the ball left at the touchline
    ball[700] = (52.5, 34.0)                    # and comes back at the centre spot
    inp = inputs(n, gaps=[(200, 700)], ball=ball)
    _, gap_x, _ = ld.frame_features(inp)
    f = ld.GAP_FEATURES
    early, late = gap_x[205], gap_x[695]
    assert early[column(f, "log_since_s")] < late[column(f, "log_since_s")]
    assert early[column(f, "log_until_s")] > late[column(f, "log_until_s")]
    assert early[column(f, "last_line_early")] > late[column(f, "last_line_early")]
    assert late[column(f, "next_restart_late")] > early[column(f, "next_restart_late")]


def test_the_speed_trend_sees_players_stop_and_start():
    speed = np.concatenate([np.full(300, 6.0), np.full(400, 0.2), np.full(300, 6.0)])
    inp = inputs(1000, gaps=[(100, 900)], player_speed=speed)
    _, gap_x, _ = ld.frame_features(inp)
    trend = column(ld.GAP_FEATURES, "speed_trend")
    assert gap_x[300, trend] < -1.0 and gap_x[700, trend] > 1.0 and abs(gap_x[500, trend]) < 1e-9


def test_the_crowd_counts_players_near_where_the_ball_comes_back():
    n, players = 600, 10
    inp = inputs(n, gaps=[(100, 500)])
    xy = np.full((n, players, 2), 80.0)
    xy[:, :4] = (50.0, 30.0)                    # 4 of 10 next to the ball's spot
    inp["players_xy"] = xy
    _, gap_x, _ = ld.frame_features(inp)
    f = ld.GAP_FEATURES
    assert gap_x[300, column(f, "crowd_next")] == 0.4
    assert gap_x[300, column(f, "crowd_last")] == 0.4
    assert gap_x[300, column(f, "players_spread")] > 0.0
    assert not gap_x[50].any()                  # still zero outside the gap


def test_a_v2_model_file_still_loads_and_scores_with_the_first_twelve_columns(tmp_path):
    path = tmp_path / "m.json"
    fits = toy_fits()
    ld.save_model(path, fits, ["A"], 10.0, {"gap": 1, "visible": 1})
    doc = json.loads(path.read_text())
    k = len(ld.GAP_FEATURES_V2)
    doc["version"] = 2
    doc["gap"].update(features=list(ld.GAP_FEATURES_V2), mean=doc["gap"]["mean"][:k],
                      scale=doc["gap"]["scale"][:k], coef=doc["gap"]["coef"][:k])
    path.write_text(json.dumps(doc))
    assert ld.load_model(path)["version"] == 2
    doc["gap"]["features"] = list(ld.GAP_FEATURES)          # v2 with v3 features: refused
    path.write_text(json.dumps(doc))
    try:
        ld.load_model(path)
    except ValueError as err:
        assert "other features" in str(err)
    else:
        raise AssertionError("a v2 file with v3 features was accepted")