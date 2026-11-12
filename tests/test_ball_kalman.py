# ws:ball: tests for ball method `kalman` (task 2.5, team-board issue #35).
# Synthetic data only: every ball here is a made-up path.
#
# What the method does: a gap in the ball track is filled with the estimate of a Kalman filter
# and RTS smoother (constant velocity model), but only if the smoother is sure enough about the
# whole gap (its position std stays under max_sigma). A gap it is not sure about stays missing
# (NaN), so the dead-ball stage still treats it as "no ball". Observed frames are never changed.
#
# How it is checked:
#   - the plain 2x2 algebra in ball._smooth_run equals the same filter written with matrices
#     (`dense_smoother` below, the slow obvious way), on random data with random gaps
#   - a ball at steady speed is filled exactly; a ball on a curve is filled much better than
#     the straight line of method `interpolate`
#   - the sigma gate: short gaps are filled, long ones are not, and a gap is filled whole or not
#     at all
#   - the same edge rules as `interpolate` (data edges, half edges), the speed field, the
#     reject_first option, the speed guard, the config checks

from dataclasses import replace

import numpy as np
import pytest

from falafelona import ball, deadball, io, vendor
from falafelona.ball import INTERPOLATED, OBSERVED, REJECTED, UNKNOWN, DataCleaner, clean
from falafelona.config import load_config
from falafelona.io import Match

CONFIGS = vendor.REPO_ROOT / "configs"
PITCH = vendor.games.GAMES[vendor.games.GAME]["pitch"]
DT = 1 / 25

BASE = {
    "process_noise": 1.6,
    "measurement_noise": 0.3,
    "initial_speed_std": 15.0,
    "max_sigma": 2.0,
}


def cfg(**ball_settings):
    """A config for method `kalman`; the given settings replace the defaults in BASE."""
    return {"ball": {"method": "kalman", **BASE, **ball_settings}}


def walk(n, per_frame=0.3, start=(10.0, 30.0)):
    """A ball path of n frames: it moves per_frame metres along the pitch on every frame."""
    xy = np.zeros((n, 2))
    xy[:, 0] = start[0] + per_frame * np.arange(n)
    xy[:, 1] = start[1]
    return xy


def blank(xy, *gaps):
    """A copy of the path with the given (start, stop) ranges of frames set to NaN."""
    out = xy.copy()
    for lo, hi in gaps:
        out[lo:hi] = np.nan
    return out


def match_from(xy, period=None, fps=25.0, speed=5.0):
    """A Match around a ball path, with no players. `period` defaults to one half, all graded."""
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    period = np.ones(n, dtype=np.int8) if period is None else np.asarray(period, dtype=np.int8)
    speed = np.broadcast_to(np.asarray(speed, dtype=float), (n,)).copy()
    speed[np.isnan(xy[:, 0])] = np.nan
    return Match(
        n_frames=n,
        fps=fps,
        period=period,
        ball_xy=xy,
        ball_speed=speed,
        players_xy=np.full((n, 0, 2), np.nan),
        players_speed=np.full((n, 0), np.nan),
        slot_team=np.array([], dtype="<U1"),
        pitch=PITCH,
    )


def frames_with(track, source):
    """The frame numbers whose source is `source`."""
    return np.flatnonzero(track.source == source)


def sigma_of(xy, dt=DT, **settings):
    """The smoothed std of every frame of a path whose first and last frame are seen."""
    s = {**BASE, **settings}
    observed = ~np.isnan(xy[:, 0])
    _, sigma = ball._smooth_run(
        xy, observed, dt, s["process_noise"], s["measurement_noise"], s["initial_speed_std"]
    )
    return sigma


def dense_smoother(z, observed, dt, q, r, v0):
    """The same filter and smoother as ball._smooth_run, written with 2x2 matrices, per axis."""
    m = len(z)
    f = np.array([[1.0, dt], [0.0, 1.0]])
    qm = q * q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    h = np.array([[1.0, 0.0]])
    out = np.zeros((m, 2))
    sigma = None
    for axis in (0, 1):
        x, p = np.array([z[0, axis], 0.0]), np.diag([r * r, v0 * v0])
        xf, pf, xp, pp = [x], [p], [None], [None]
        for t in range(1, m):
            x_pred, p_pred = f @ x, f @ p @ f.T + qm
            xp.append(x_pred)
            pp.append(p_pred)
            if observed[t]:
                gain = p_pred @ h.T @ np.linalg.inv(h @ p_pred @ h.T + r * r)
                x = x_pred + gain @ (np.array([z[t, axis]]) - h @ x_pred)
                p = (np.eye(2) - gain @ h) @ p_pred
            else:
                x, p = x_pred, p_pred
            xf.append(x)
            pf.append(p)
        xs, ps = xf[:], pf[:]
        for t in range(m - 2, -1, -1):
            c = pf[t] @ f.T @ np.linalg.inv(pp[t + 1])
            xs[t] = xf[t] + c @ (xs[t + 1] - xp[t + 1])
            ps[t] = pf[t] + c @ (ps[t + 1] - pp[t + 1]) @ c.T
        out[:, axis] = [state[0] for state in xs]
        sigma = np.sqrt([cov[0, 0] for cov in ps])
    return out, sigma


# ---------------------------------------------------------------------------------------------
# The smoother itself
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_smoother_equals_the_same_filter_written_with_matrices(seed):
    """Random wandering ball, random gaps: positions and std agree with the matrix version."""
    rng = np.random.default_rng(seed)
    m = 300
    path = np.cumsum(rng.normal(0.0, 0.4, size=(m, 2)), axis=0) + [50.0, 30.0]
    observed = np.ones(m, dtype=bool)
    for start, length in ((20, 5), (60, 1), (100, 40), (180, 12), (250, 30)):
        observed[start : start + length] = False
    z = np.where(observed[:, None], path, np.nan)
    args = (DT, 6.0, 0.4, 12.0)

    fast_xy, fast_sigma = ball._smooth_run(z, observed, *args)
    slow_xy, slow_sigma = dense_smoother(z, observed, *args)

    np.testing.assert_allclose(fast_xy, slow_xy, atol=1e-6)
    np.testing.assert_allclose(fast_sigma, slow_sigma, atol=1e-6)


def test_the_std_is_small_where_the_ball_is_seen_and_peaks_inside_a_gap():
    xy = blank(walk(120), (50, 80))
    sigma = sigma_of(xy)

    seen = ~np.isnan(xy[:, 0])
    inside = sigma[10:40]                            # seen, away from the two ends of the data
    assert (sigma[seen] < BASE["measurement_noise"]).all()
    assert sigma[50:80].max() > 2 * inside.max()
    middle = 50 + int(np.argmax(sigma[50:80]))
    assert 60 <= middle <= 70                       # the least sure frame is near the middle
    assert sigma[50] < sigma[middle] and sigma[79] < sigma[middle]


def test_a_longer_gap_is_less_certain():
    short = sigma_of(blank(walk(200), (80, 90)))[80:90].max()
    long_ = sigma_of(blank(walk(200), (60, 130)))[60:130].max()

    assert long_ > short


def test_the_model_does_not_depend_on_how_finely_time_is_cut():
    """Check the model, not only the algebra: the noise is per second, not per frame.

    Take the same ball and cut each frame into 5 smaller time steps, with no ball seen on the
    in-between steps. If the noise is really continuous in time, the smoother gives the same
    answer on the original frames. (dense_smoother above shares the noise matrix with the code;
    this check does not, and a noise matrix that mixes up per frame and per second fails it.)
    """
    rng = np.random.default_rng(7)
    m, q, r, v0, cut = 80, 3.0, 0.4, 10.0, 5
    observed = np.ones(m, dtype=bool)
    observed[20:45] = False
    observed[60:63] = False
    z = np.cumsum(rng.normal(0.0, 0.3, size=(m, 2)), axis=0) + [20.0, 30.0]
    z = np.where(observed[:, None], z, np.nan)

    fine_len = (m - 1) * cut + 1
    fine_z = np.full((fine_len, 2), np.nan)
    fine_seen = np.zeros(fine_len, dtype=bool)
    fine_z[::cut], fine_seen[::cut] = z, observed

    coarse_xy, coarse_sigma = ball._smooth_run(z, observed, DT, q, r, v0)
    fine_xy, fine_sigma = ball._smooth_run(fine_z, fine_seen, DT / cut, q, r, v0)

    np.testing.assert_allclose(coarse_xy, fine_xy[::cut], atol=1e-6)
    np.testing.assert_allclose(coarse_sigma, fine_sigma[::cut], atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------------------------


def test_a_ball_at_steady_speed_is_filled_exactly():
    """A straight walk: the smoother finds the straight line (up to the small measurement noise)."""
    truth = walk(100)
    xy = blank(truth, (40, 50))
    track = clean(match_from(xy), cfg(measurement_noise=0.001, process_noise=0.1))

    assert (frames_with(track, INTERPOLATED) == np.arange(40, 50)).all()
    np.testing.assert_allclose(track.xy[40:50], truth[40:50], atol=2e-3)


def test_a_curved_path_is_filled_far_better_than_a_straight_line():
    """The ball accelerates sideways at 6 m/s^2 for the whole path: a parabola.

    A straight line between the two ends cuts the corner; the smoother uses the speed on both
    sides, so it follows the curve. This is what the method is for.
    """
    t = np.arange(120) * DT
    truth = np.column_stack([10.0 + 8.0 * t, 20.0 + 0.5 * 6.0 * t**2])
    gap = (50, 70)
    xy = blank(truth, gap)

    kalman = clean(match_from(xy), cfg(measurement_noise=0.05, process_noise=0.2, max_sigma=5.0))
    linear = clean(match_from(xy), {"ball": {"method": "interpolate", "max_gap": 1.0}})

    kalman_error = np.abs(kalman.xy[gap[0] : gap[1]] - truth[gap[0] : gap[1]]).max()
    linear_error = np.abs(linear.xy[gap[0] : gap[1]] - truth[gap[0] : gap[1]]).max()
    assert (kalman.source[gap[0] : gap[1]] == INTERPOLATED).all()
    assert linear_error > 0.4                        # the corner the straight line cuts
    assert kalman_error < 0.25 * linear_error


def test_observed_frames_are_never_changed():
    rng = np.random.default_rng(3)
    truth = walk(200) + rng.normal(0.0, 0.2, size=(200, 2))
    xy = blank(truth, (30, 36), (90, 100), (150, 153))
    match = match_from(xy)
    seen = ~np.isnan(xy[:, 0])

    track = clean(match, cfg())

    np.testing.assert_array_equal(track.xy[seen], xy[seen])
    assert (track.source[seen] == OBSERVED).all()
    assert (frames_with(track, INTERPOLATED).size) > 0


def test_the_input_match_is_not_changed():
    xy = blank(walk(100), (40, 46))
    match = match_from(xy)
    before_xy, before_speed = match.ball_xy.copy(), match.ball_speed.copy()

    clean(match, cfg())

    np.testing.assert_array_equal(match.ball_xy, before_xy)
    np.testing.assert_array_equal(match.ball_speed, before_speed)


def test_the_same_input_gives_the_same_output():
    match = match_from(blank(walk(150), (30, 40), (90, 120)))

    a, b = clean(match, cfg()), clean(match, cfg())

    np.testing.assert_array_equal(a.xy, b.xy)
    np.testing.assert_array_equal(a.source, b.source)


def test_no_gaps_gives_the_raw_track():
    match = match_from(walk(60))

    track = clean(match, cfg())

    np.testing.assert_array_equal(track.xy, match.ball_xy)
    assert (track.source == OBSERVED).all()


# ---------------------------------------------------------------------------------------------
# The sigma gate
# ---------------------------------------------------------------------------------------------


def test_short_gaps_are_filled_and_long_gaps_are_not():
    xy = blank(walk(400), (50, 55), (200, 320))
    track = clean(match_from(xy), cfg(max_sigma=1.0))

    assert (track.source[50:55] == INTERPOLATED).all()
    assert (track.source[200:320] == UNKNOWN).all()
    assert np.isnan(track.xy[200:320]).all()


def test_a_gap_is_filled_whole_or_not_at_all():
    """The ends of a gap are sure, the middle is not: nothing is filled, not even the ends."""
    xy = blank(walk(200), (60, 100))
    sigma = sigma_of(xy)
    assert sigma[60] < sigma[79]                     # the end is surer than the middle
    between = (sigma[60:100].max() + sigma[60]) / 2  # sure at the end, not in the middle

    track = clean(match_from(xy), cfg(max_sigma=float(between)))
    assert (track.source[60:100] == UNKNOWN).all()

    track = clean(match_from(xy), cfg(max_sigma=float(sigma[60:100].max()) * 1.01))
    assert (track.source[60:100] == INTERPOLATED).all()


def test_a_one_frame_gap_is_gated_too():
    xy = blank(walk(60), (30, 31))
    sigma = sigma_of(xy)[30]

    assert clean(match_from(xy), cfg(max_sigma=sigma * 1.01)).source[30] == INTERPOLATED
    assert clean(match_from(xy), cfg(max_sigma=sigma * 0.99)).source[30] == UNKNOWN


def test_the_frame_rate_sets_the_time_step_of_the_model():
    """The same 40 frames are a shorter time at 50 fps, so the ball is surer about them."""
    xy = blank(walk(150), (50, 90))
    fast, slow = sigma_of(xy, dt=1 / 50)[50:90].max(), sigma_of(xy, dt=1 / 25)[50:90].max()
    assert fast < slow
    limit = float(np.sqrt(fast * slow))

    at_50 = clean(match_from(xy, fps=50.0), cfg(max_sigma=limit))
    at_25 = clean(match_from(xy, fps=25.0), cfg(max_sigma=limit))

    assert (at_50.source[50:90] == INTERPOLATED).all()
    assert (at_25.source[50:90] == UNKNOWN).all()


@pytest.mark.parametrize("fps", [10.0, 50.0])
def test_the_same_gap_in_seconds_is_about_as_uncertain_at_any_frame_rate(fps):
    """A 2 s gap: the std peak is about the same at 10, 25 and 50 fps (it was not before)."""
    def peak(rate):
        n = int(rate * 6)
        xy = blank(walk(n), (int(rate * 2), int(rate * 4)))
        return sigma_of(xy, dt=1 / rate)[int(rate * 2) : int(rate * 4)].max()

    assert peak(fps) == pytest.approx(peak(25.0), rel=0.15)


def test_a_gap_just_before_the_last_seen_frame_uses_that_frame():
    """The ball is seen again at frame 59 and then never: that last sighting still counts.

    The path is a curve, so a fill that ignored frame 59 (running on from the left) would miss.
    """
    t = np.arange(100) * DT
    truth = np.column_stack([10.0 + 8.0 * t, 20.0 + 3.0 * t**2])
    xy = blank(truth, (45, 59))
    xy[60:] = np.nan

    track = clean(match_from(xy), cfg(measurement_noise=0.05, process_noise=1.6, max_sigma=5.0))

    assert (track.source[45:59] == INTERPOLATED).all()
    np.testing.assert_allclose(track.xy[45:59], truth[45:59], atol=0.2)   # without it: 1.3 m off


def test_a_larger_process_noise_makes_the_model_less_sure_so_fewer_gaps_are_filled():
    xy = blank(walk(300), (50, 70), (150, 250))      # a 0.8 s gap and a 4 s gap
    calm = clean(match_from(xy), cfg(process_noise=0.2, max_sigma=1.5))
    wild = clean(match_from(xy), cfg(process_noise=4.0, max_sigma=1.5))

    assert len(frames_with(calm, INTERPOLATED)) == 120   # both gaps
    assert len(frames_with(wild, INTERPOLATED)) == 20    # only the short one


# ---------------------------------------------------------------------------------------------
# Edges: same rules as `interpolate`
# ---------------------------------------------------------------------------------------------


def test_a_gap_at_the_start_or_end_of_the_data_is_not_filled():
    xy = blank(walk(100), (0, 5), (95, 100))

    track = clean(match_from(xy), cfg())

    assert (track.source[:5] == UNKNOWN).all() and (track.source[95:] == UNKNOWN).all()


def test_a_gap_across_the_edge_of_a_half_is_not_filled():
    xy = blank(walk(100), (48, 52))
    period = np.array([1] * 50 + [2] * 50)

    track = clean(match_from(xy, period=period), cfg())

    assert (track.source[48:52] == UNKNOWN).all()


def test_frames_outside_the_halves_are_not_filled_and_not_used():
    period = np.array([1] * 40 + [0] * 20 + [2] * 40)
    xy = walk(100)
    xy[40:60] = np.nan                               # the ungraded stretch has no ball
    xy[70:75] = np.nan                               # a real gap in half 2
    track = clean(match_from(xy, period=period), cfg())

    assert (track.source[40:60] == UNKNOWN).all()
    assert (track.source[70:75] == INTERPOLATED).all()


def test_a_half_with_fewer_than_two_seen_frames_is_skipped():
    xy = np.full((60, 2), np.nan)
    xy[10] = (20.0, 20.0)                            # one seen frame in the whole half

    track = clean(match_from(xy), cfg())

    assert (frames_with(track, INTERPOLATED).size) == 0
    assert np.isnan(track.xy[np.arange(60) != 10]).all()


def test_the_two_halves_are_smoothed_separately():
    """Half 2 starts far from where half 1 ended: no gap, no filter state carried over."""
    first = blank(walk(60), (20, 24))
    second = blank(walk(60, start=(90.0, 10.0)), (20, 24))
    xy = np.vstack([first, second])
    period = np.array([1] * 60 + [2] * 60)

    track = clean(match_from(xy, period=period), cfg(measurement_noise=0.001, process_noise=0.1))

    np.testing.assert_allclose(track.xy[80:84], walk(60, start=(90.0, 10.0))[20:24], atol=2e-3)
    np.testing.assert_allclose(track.xy[20:24], walk(60)[20:24], atol=2e-3)


# ---------------------------------------------------------------------------------------------
# Speed field
# ---------------------------------------------------------------------------------------------


def test_speed_is_filled_with_a_straight_line_between_the_ends():
    xy = blank(walk(60), (20, 24))
    speed = np.full(60, 4.0)
    speed[24:] = 8.0                                 # speed field jumps across the gap
    match = replace(match_from(xy), ball_speed=np.where(np.isnan(xy[:, 0]), np.nan, speed))

    track = clean(match, cfg())

    np.testing.assert_allclose(track.speed[20:24], [4.8, 5.6, 6.4, 7.2])


def test_speed_stays_missing_when_an_end_has_no_speed():
    xy = blank(walk(60), (20, 24))
    match = match_from(xy)
    match.ball_speed[19] = np.nan

    track = clean(match, cfg())

    assert (frames_with(track, INTERPOLATED) == np.arange(20, 24)).all()
    assert np.isnan(track.speed[20:24]).all()


# ---------------------------------------------------------------------------------------------
# reject_first and the speed guard
# ---------------------------------------------------------------------------------------------


def test_reject_first_turns_false_detections_into_gaps_that_are_then_filled():
    truth = walk(100)
    xy = truth.copy()
    xy[40:43] = (90.0, 5.0)                          # three frames of a false detection
    reject = {"max_speed": 40.0, "speed_slack": 1.0, "max_false_run": 1.0}

    track = clean(match_from(xy), cfg(reject_first=True, **reject))
    plain = clean(match_from(xy), cfg(**reject))

    assert (track.source[40:43] == INTERPOLATED).all()   # rejected, then filled
    np.testing.assert_allclose(track.xy[40:43], truth[40:43], atol=0.2)
    assert (plain.source[40:43] == OBSERVED).all()       # without reject_first the fake stays
    assert not np.isin(track.source, (REJECTED,)).any()  # every rejected frame was refilled


def test_reject_first_gaps_are_judged_one_by_one():
    """A short rejected gap is filled and a long real gap beside it is not."""
    xy = walk(400)
    xy[200:204] = (90.0, 5.0)
    xy[210:330] = np.nan                             # the gap after the false detection: too long
    reject = {"max_speed": 40.0, "speed_slack": 1.0, "max_false_run": 1.0}

    track = clean(match_from(xy), cfg(reject_first=True, max_sigma=1.0, **reject))

    assert (track.source[200:204] == INTERPOLATED).all()  # a short gap of its own: filled
    assert (track.source[210:330] == UNKNOWN).all()


def test_the_speed_guard_limit_is_inclusive_and_the_slack_is_added():
    """walk(.., 1 m per frame) is 25 m/s: a 5-frame gap crosses 5 m in 0.2 s."""
    xy = blank(walk(60, per_frame=1.0), (30, 34))      
    gap = (30, 34)
    ends = np.hypot(*(xy[gap[1]] - xy[gap[0] - 1]))      # distance between the two seen ends
    seconds = (gap[1] - (gap[0] - 1)) / 25.0
    speed = ends / seconds
    assert speed == pytest.approx(25.0)

    exact = clean(match_from(xy), cfg(max_speed=float(ends / seconds), speed_slack=0.0))
    below = clean(match_from(xy), cfg(max_speed=speed - 0.1, speed_slack=0.0))
    slack = clean(match_from(xy), cfg(max_speed=speed - 0.1, speed_slack=0.5))

    assert (exact.source[30:34] == INTERPOLATED).all()   # equal to the limit: allowed
    assert (below.source[30:34] == UNKNOWN).all()
    assert (slack.source[30:34] == INTERPOLATED).all()   # 0.1 m/s * 0.16 s < 0.5 m


def test_the_speed_guard_refuses_a_gap_the_ball_could_not_have_crossed():
    xy = walk(100)
    xy[50:] += (40.0, 0.0)                           # the ball is 40 m further on after the gap
    xy = blank(xy, (48, 52))
    guard = {"max_speed": 40.0, "speed_slack": 1.0}

    free = clean(match_from(xy), cfg())
    guarded = clean(match_from(xy), cfg(**guard))

    assert (free.source[48:52] == INTERPOLATED).all()    # the model alone does not mind
    assert (guarded.source[48:52] == UNKNOWN).all()      # 40 m in 0.2 s is 200 m/s


# ---------------------------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------------------------


def test_datacleaner_hands_the_filled_ball_to_the_next_stage():
    xy = blank(walk(100), (30, 36))
    match = match_from(xy)

    cleaned = DataCleaner(match, cfg()).return_cleaned_data()

    assert cleaned is not match
    assert not np.isnan(cleaned.ball_xy[30:36]).any()
    assert np.isnan(match.ball_xy[30:36]).all()      # the raw Match is untouched
    assert cleaned.players_xy is match.players_xy


def test_a_gap_that_stays_missing_is_still_dead_ball_for_the_next_stage():
    xy = blank(walk(400), (50, 55), (200, 320))
    cleaned = DataCleaner(match_from(xy), cfg(max_sigma=1.0)).return_cleaned_data()

    score = deadball.detect(cleaned, None, {"deadball": {"method": "ball_missing"}})

    assert (score[50:55] == 0).all()
    assert (score[200:320] == 1).all()


def test_the_method_is_registered():
    assert "kalman" in ball.METHODS


def test_period_runs_gives_the_graded_stretches():
    period = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])

    assert ball._period_runs(period) == [(2, 5), (6, 8)]


@pytest.mark.parametrize(
    "ball_settings, text",
    [
        ({"process_noise": None}, "process_noise"),
        ({"measurement_noise": None}, "measurement_noise"),
        ({"initial_speed_std": None}, "initial_speed_std"),
        ({"max_sigma": None}, "max_sigma"),
        ({"process_noise": 0.0}, "process_noise"),
        ({"process_noise": -1.0}, "process_noise"),
        ({"measurement_noise": 0.0}, "measurement_noise"),
        ({"initial_speed_std": float("nan")}, "initial_speed_std"),
        ({"max_sigma": float("inf")}, "max_sigma"),
        ({"max_speed": 40.0}, "speed_slack"),
        ({"max_speed": 40.0, "speed_slack": -1.0}, "speed_slack"),
        ({"max_speed": 0.0, "speed_slack": 0.0}, "max_speed"),
        ({"reject_first": "yes"}, "reject_first"),
        ({"reject_first": 1}, "reject_first"),
        ({"reject_first": True}, "does nothing"),    # no reject keys given
        ({"reject_first": True, "max_speed": 40.0, "speed_slack": 0.0}, "max_false_run"),
    ],
)
def test_a_wrong_or_incomplete_config_fails_loudly(ball_settings, text):
    section = {"method": "kalman", **BASE}
    for key, value in ball_settings.items():
        if value is None:
            del section[key]
        else:
            section[key] = value

    with pytest.raises(ValueError, match=text):
        clean(match_from(walk(20)), {"ball": section})


def test_the_experiment_config_loads_and_runs(bundle):
    """configs/exp/ball-kalman.yaml is the real file: it must name this method and work."""
    config = load_config(CONFIGS / "exp" / "ball-kalman.yaml")
    match = io.load_match(n_frames=bundle["n_frames"], use_cache=False)
    gap_start, gap_stop = bundle["ball_gap"]         # one 40-frame (1.6 s) gap in the fake match

    track = clean(match, config)

    assert config["ball"]["method"] == "kalman"
    assert track.xy.shape == match.ball_xy.shape
    # the fake match has no graded half, so nothing may be filled
    assert (track.source[gap_start:gap_stop] == UNKNOWN).all()
    assert (track.source[:gap_start] == OBSERVED).all()


@pytest.mark.parametrize(
    "name, longest",
    [
        ("ball-kalman.yaml", 47),                    # 1.9 s at 25 fps, as the yaml comments say
        ("ball-kalman-short.yaml", 12),              # 0.5 s: the same limit as ball-interpolate.yaml
    ],
)
def test_the_experiment_configs_fill_the_gap_length_their_comments_promise(name, longest):
    config = load_config(CONFIGS / "exp" / name)
    xy = blank(walk(400), (50, 50 + longest), (200, 200 + longest + 1))

    track = clean(match_from(xy), config)

    assert (track.source[50 : 50 + longest] == INTERPOLATED).all()
    assert (track.source[200 : 200 + longest + 1] == UNKNOWN).all()


def test_the_short_config_differs_from_the_long_one_only_in_max_sigma():
    long_ = load_config(CONFIGS / "exp" / "ball-kalman.yaml")
    short = load_config(CONFIGS / "exp" / "ball-kalman-short.yaml")

    assert {k: v for k, v in long_["ball"].items() if k != "max_sigma"} == {
        k: v for k, v in short["ball"].items() if k != "max_sigma"
    }
    assert short["ball"]["max_sigma"] < long_["ball"]["max_sigma"]
    for section in ("deadball", "possession", "output", "seed"):
        assert long_[section] == short[section]


def test_baseline_and_final_configs_still_use_the_untouched_ball():
    """This task must not change the baseline: both shipped configs keep `ball: method: none`."""
    for name in ("baseline.yaml", "final.yaml"):
        assert load_config(CONFIGS / name)["ball"]["method"] == "none"
