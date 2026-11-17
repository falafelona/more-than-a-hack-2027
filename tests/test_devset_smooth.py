# ws:eval: the dev set can smooth the IDSSE ball like ours (degrade.ball_smooth_s). Made up only.

import numpy as np
import pytest
import two_halves as th

from falafelona import devset

DEGRADE = {"missing_share": 0.3, "gaps": [{"min_s": 0.5, "max_s": 8.0, "share": 1.0}]}


def test_no_smoothing_is_exactly_as_before():
    match = th.make_match()
    plain, report = devset.degrade_ball(match, DEGRADE, seed=0)
    zero, report0 = devset.degrade_ball(match, {**DEGRADE, "ball_smooth_s": 0.0}, seed=0)
    assert np.array_equal(plain.ball_xy, zero.ball_xy, equal_nan=True)
    assert report["ball_smooth_s"] == report0["ball_smooth_s"] == 0.0


def test_smoothing_keeps_the_gaps_and_rounds_the_track():
    match = th.make_match()
    plain, _ = devset.degrade_ball(match, DEGRADE, seed=0)
    smooth, report = devset.degrade_ball(match, {**DEGRADE, "ball_smooth_s": 0.25}, seed=0)
    assert report["ball_smooth_s"] == 0.25
    assert np.array_equal(np.isnan(plain.ball_xy), np.isnan(smooth.ball_xy))
    seen = ~np.isnan(plain.ball_xy[:, 0]) & (match.period > 0)
    assert np.abs(plain.ball_xy[seen] - smooth.ball_xy[seen]).max() > 0.1


def test_smoothing_rounds_a_sharp_turn():
    match = th.make_match()
    part = slice(300, 450)                               # a dribble, then a fast pass: no gap
    sharp = match.ball_xy[part]
    round_ = devset.smooth_ball(match.ball_xy, match.period, 0.25 * th.FPS)[part]
    assert np.isfinite(sharp).all() and np.isfinite(round_).all()
    assert np.abs(np.diff(round_, n=2, axis=0)).max() < np.abs(np.diff(sharp, n=2, axis=0)).max()


def test_a_seen_run_shorter_than_the_kernel_is_smoothed_too():
    xy = np.full((40, 2), np.nan)
    xy[10:15] = np.column_stack([np.arange(5.0), np.zeros(5)])
    out = devset.smooth_ball(xy, np.ones(40, dtype=np.int8), 6.0)
    assert np.isfinite(out[10:15]).all() and np.isnan(out[:10]).all()
    assert out[10, 0] > 0.0 and out[14, 0] < 4.0


def test_smoothing_stays_inside_a_half_and_leaves_ungraded_frames():
    xy = np.column_stack([np.arange(20.0), np.zeros(20)])
    period = np.array([0] * 4 + [1] * 8 + [2] * 8, dtype=np.int8)
    out = devset.smooth_ball(xy, period, 2.0)
    assert np.array_equal(out[:4], xy[:4])
    assert out[11, 0] < 11.0 and out[12, 0] > 12.0       # no pull across the half edge
    assert np.allclose(out[:, 1], 0.0)


def test_a_negative_smoothing_is_refused():
    with pytest.raises(ValueError, match="ball_smooth_s"):
        devset.degrade_ball(th.make_match(), {**DEGRADE, "ball_smooth_s": -0.1}, seed=0)