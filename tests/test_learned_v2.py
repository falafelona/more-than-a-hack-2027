# ws:dead: the review-round changes of the learned dead-ball score: the ball speed at a gap's edge,
# the per-gap-kind and no-length variants, and the fit script's --drop. Made-up numbers only.

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_learned_dead import FPS, inputs

from falafelona import learned_dead as ld
from falafelona import vendor
from falafelona.config import load_config

EXP = vendor.REPO_ROOT / "configs" / "exp"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_deadball.py"
spec = importlib.util.spec_from_file_location("fit_deadball", SCRIPT)
fit_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_script)


def with_speed(inp, known_speed, unknown_next_to_gap=2):
    """A copy of inp whose seen ball moves at known_speed, but is unknown for a few frames next
    to each gap: the real speed needs seen positions on both sides of the frame."""
    speed = np.where(np.isnan(inp["ball_xy"][:, 0]), np.nan, float(known_speed))
    for first, stop in inp["gap_runs"]:
        speed[max(first - unknown_next_to_gap, 0):first] = np.nan
        speed[stop:stop + unknown_next_to_gap] = np.nan
    return {**inp, "ball_speed": speed}


def test_the_ball_speed_at_a_gap_edge_is_read_a_few_frames_in_not_on_the_edge():
    inp = with_speed(inputs(1000, gaps=[(300, 500)]), 8.0)      # speed unknown on 298-299, 500-501

    _, gap_x, _ = ld.frame_features(inp)

    f = ld.GAP_FEATURES
    assert gap_x[400, f.index("last_ball_speed")] == pytest.approx(np.log1p(8.0))
    assert gap_x[400, f.index("next_ball_speed")] == pytest.approx(np.log1p(8.0))


def test_no_known_speed_within_the_look_distance_leaves_the_feature_at_zero():
    inp = with_speed(inputs(1000, gaps=[(300, 500)]), 8.0,
                     unknown_next_to_gap=round(ld.SPEED_LOOK_S * FPS) + 5)

    _, gap_x, _ = ld.frame_features(inp)

    f = ld.GAP_FEATURES
    assert gap_x[400, f.index("last_ball_speed")] == 0.0
    assert gap_x[400, f.index("next_ball_speed")] == 0.0


def test_the_speed_search_does_not_cross_into_another_half():
    period = np.array([1] * 500 + [2] * 500, dtype=np.int8)
    speed = np.full(1000, np.nan)
    speed[500:] = 20.0                          # known only in the other half

    assert np.isnan(ld._speed_near(speed, 498, 1, 12, period))
    assert ld._speed_near(speed, 502, -1, 12, period) == 20.0
    far = np.full(1000, np.nan)
    far[480] = 7.0                                                # known, 18 frames from 498
    assert np.isnan(ld._speed_near(far, 498, -1, 12, period))
    assert ld._speed_near(far, 498, -1, 20, period) == 7.0


def test_the_model_version_was_bumped_so_old_model_files_are_refused():
    assert ld.VERSION == 3 and set(ld.READABLE) == {2, 3}     # v2 files still load (v3 gap)


@pytest.mark.parametrize("tag", ["random", "stoppage", "nolen"])
def test_the_variant_configs_differ_from_the_candidate_only_in_the_model_path(tag):
    base = load_config(EXP / "poss-learned.yaml")
    cfg = load_config(EXP / f"poss-learned-{tag}.yaml")
    model = cfg["deadball"]["learned"].pop("model")
    assert model == f"models/deadball_{tag}_lomo_{{match_id}}.json"
    base["deadball"]["learned"].pop("model")
    assert cfg == base


def test_drop_zeroes_the_named_column_in_every_match_and_nothing_else():
    rng = np.random.default_rng(0)
    rows = {m: {"gap": (rng.normal(size=(20, len(ld.GAP_FEATURES))) + 1.0, np.zeros(20)),
                "visible": (rng.normal(size=(20, len(ld.VISIBLE_FEATURES))) + 1.0, np.zeros(20))}
            for m in ("A", "B")}
    before = copy.deepcopy(rows)

    fit_script.drop_features(rows, ["gap:log2_gap_s"])

    column = ld.GAP_FEATURES.index("log2_gap_s")
    for m in rows:
        assert (rows[m]["gap"][0][:, column] == 0.0).all()
        keep = [i for i in range(len(ld.GAP_FEATURES)) if i != column]
        assert np.array_equal(rows[m]["gap"][0][:, keep], before[m]["gap"][0][:, keep])
        assert np.array_equal(rows[m]["visible"][0], before[m]["visible"][0])


def test_drop_refuses_a_misspelt_feature():
    with pytest.raises(SystemExit):
        fit_script.drop_features({"A": {}}, ["gap:log2_gap"])
