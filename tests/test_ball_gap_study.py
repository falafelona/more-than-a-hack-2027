# ws:ball: tests for scripts/ball_gap_study.py (task 2.5). Synthetic data only.
#
# The study makes up its own ball paths. These tests check that the made-up paths are sane, that
# the two fills are right on a path where the right answer is known, and that the script runs.

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not a package, so load the script as a module
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ball_gap_study.py"
spec = importlib.util.spec_from_file_location("ball_gap_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def straight(n, per_frame=0.3):
    xy = np.zeros((n, 2))
    xy[:, 0] = 10.0 + per_frame * np.arange(n)
    xy[:, 1] = 30.0
    return xy


def test_simulated_paths_stay_on_the_pitch_and_do_not_teleport():
    xy = study.simulate(5000, np.random.default_rng(0))

    assert xy.shape == (5000, 2)
    assert (xy >= 0).all() and (xy[:, 0] <= study.PITCH[0]).all() and (xy[:, 1] <= study.PITCH[1]).all()
    assert np.hypot(*np.diff(xy, axis=0).T).max() <= 32.0 / study.FPS + 1e-9   # at most 32 m/s


def test_the_same_seed_gives_the_same_path():
    a = study.simulate(500, np.random.default_rng(3))
    b = study.simulate(500, np.random.default_rng(3))

    np.testing.assert_array_equal(a, b)


def test_a_window_hides_exactly_the_gap_and_keeps_the_context():
    windows = study.make_gaps(4, 1.0, np.random.default_rng(1), noise=0.25)

    assert len(windows) == 4
    for truth, seen, gap in windows:
        assert gap == 25
        assert len(truth) == len(seen) == 2 * study.CONTEXT + 25
        hidden = np.isnan(seen[:, 0])
        assert hidden.sum() == 25
        assert hidden[study.CONTEXT : study.CONTEXT + 25].all()
        assert not np.isnan(truth).any()


def test_both_fills_are_exact_on_a_noise_free_straight_path():
    truth = straight(2 * study.CONTEXT + 20)
    seen = truth.copy()
    seen[study.CONTEXT : study.CONTEXT + 20] = np.nan
    window = (truth, seen, 20)

    assert study.rmse([window], study.fill_linear) == pytest.approx(0.0, abs=1e-9)
    # a small measurement noise setting is not needed: the data has none and the path is straight
    assert study.rmse([window], lambda z, g: study.fill_kalman(z, g, 1.0)) < 0.05


def test_kalman_beats_the_straight_line_on_a_curve_and_reports_more_doubt_in_the_middle():
    t = np.arange(2 * study.CONTEXT + 40) * study.DT
    truth = np.column_stack([10.0 + 8.0 * t, 20.0 + 3.0 * t**2])
    seen = truth.copy()
    seen[study.CONTEXT : study.CONTEXT + 40] = np.nan
    window = (truth, seen, 40)

    linear = study.rmse([window], study.fill_linear)
    kalman = study.rmse([window], lambda z, g: study.fill_kalman(z, g, 6.4))
    _, sigma = study.fill_kalman(seen, 40, 6.4)

    assert kalman < 0.3 * linear
    assert sigma[20] > sigma[0] and sigma[20] > sigma[-1]


def test_spearman_is_one_for_a_monotone_pair_and_minus_one_for_a_reversed_one():
    a = np.array([1.0, 5.0, 2.0, 9.0])

    assert study.spearman(a, a * 3 + 1) == pytest.approx(1.0)
    assert study.spearman(a, -a) == pytest.approx(-1.0)


def test_main_runs_and_prints_the_three_tables(capsys):
    study.main(["--gaps", "5"])

    out = capsys.readouterr().out
    assert "process_noise tuned on seed 1" in out
    assert "Is sigma honest?" in out
    assert "Error of the filled frames by sigma" in out
