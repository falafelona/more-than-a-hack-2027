# possession method `switch_cost` (task 4.3): Viterbi decoder with a switch cost.
# Synthetic arrays only, no match data.
import numpy as np
import pytest

from falafelona import possession
from falafelona.config import load_config
from falafelona.possession import Possession, decode_switch_cost, no_evidence_within, viterbi

HOME, AWAY, NOBODY = Possession.HOME, Possession.AWAY, Possession.NOBODY


def runs(*parts):
    """runs((HOME, 100), (AWAY, 3)) -> int8 codes."""
    return np.concatenate([np.full(n, code) for code, n in parts]).astype(np.int8)


def brute_force(costs, switch_cost):
    """Cheapest path by trying every sequence (tiny inputs only)."""
    import itertools
    n, k = costs.shape
    best = None
    for path in itertools.product(range(k), repeat=n):
        total = sum(costs[t, s] for t, s in enumerate(path))
        total += switch_cost * sum(a != b for a, b in itertools.pairwise(path))
        if best is None or total < best[0] - 1e-12:
            best = (total, path)
    return best[0]


def path_cost(costs, path, switch_cost):
    return costs[np.arange(len(path)), path].sum() + switch_cost * np.sum(path[1:] != path[:-1])


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_viterbi_is_optimal(k, seed):
    rng = np.random.default_rng(seed)
    costs = rng.uniform(0, 1, size=(8, k))
    for switch_cost in (0.0, 0.3, 1.5):
        path = viterbi(costs, switch_cost)
        assert path_cost(costs, path, switch_cost) == pytest.approx(brute_force(costs, switch_cost))


def test_zero_switch_cost_is_per_frame_choice():
    costs = np.random.default_rng(0).uniform(size=(50, 3))
    np.testing.assert_array_equal(viterbi(costs, 0.0), costs.argmin(axis=1))


def test_viterbi_ties_stay_put_and_empty_input():
    assert viterbi(np.zeros((5, 2)), 1.0).tolist() == [0] * 5
    assert viterbi(np.zeros((0, 2)), 1.0).shape == (0,)


def test_flicker_is_removed():
    codes = runs((HOME, 100), (AWAY, 3), (HOME, 100))
    assert set(decode_switch_cost(codes, switch_cost=10, dead_margin=50)) == {"H"}


def test_real_switch_is_kept_on_time():
    codes = runs((HOME, 100), (AWAY, 100))
    labels = decode_switch_cost(codes, switch_cost=25, dead_margin=50)
    assert "".join(labels) == "H" * 100 + "A" * 100


def test_switch_needs_enough_evidence():
    # 20 frames of A evidence. Flipping there and back costs 2 * switch_cost and saves 20:
    # worth it with switch_cost 5 (10 < 20), not with 25 (50 > 20).
    codes = runs((HOME, 100), (AWAY, 20), (HOME, 100))
    assert "A" in decode_switch_cost(codes, switch_cost=5, dead_margin=50)
    assert "A" not in decode_switch_cost(codes, switch_cost=25, dead_margin=50)


def test_d_matches_the_baseline_vote():
    rng = np.random.default_rng(1)
    # long and short evidence gaps, like a ball that blinks out and a real stoppage
    codes = rng.choice([HOME, AWAY, NOBODY], size=3000, p=[0.3, 0.3, 0.4]).astype(np.int8)
    codes[500:800] = NOBODY
    codes[1500:1560] = NOBODY
    baseline = Possession(None).smooth(codes, 50)
    ours = decode_switch_cost(codes, switch_cost=25, dead_margin=50)
    np.testing.assert_array_equal(ours == "D", baseline == "D")
    np.testing.assert_array_equal(no_evidence_within(codes, 50), baseline == "D")


def test_switch_cost_flickers_less_than_the_vote():
    rng = np.random.default_rng(2)
    truth = runs((HOME, 400), (AWAY, 400), (HOME, 400), (AWAY, 400))
    noisy = truth.copy()
    flip = rng.random(len(truth)) < 0.3
    noisy[flip] = np.where(truth[flip] == HOME, AWAY, HOME)
    ours = decode_switch_cost(noisy, switch_cost=25, dead_margin=50)
    assert int(np.sum(ours[1:] != ours[:-1])) == 3   # exactly the three real switches


def test_predict_dispatch_and_config():
    cfg = load_config("configs/exp/poss-switch-cost.yaml")
    assert cfg["possession"]["method"] in possession.METHODS
    with pytest.raises(ValueError, match="available: nearest_majority, switch_cost"):
        possession.predict(None, None, np.zeros(3), {"possession": {"method": "nope"}})


@pytest.mark.parametrize("key,value", [
    ("switch_cost", -1), ("switch_cost", "x"), ("dead_margin", 2.5), ("dead_margin", True),
])
def test_bad_settings_are_refused(key, value):
    section = {"method": "switch_cost", "switch_cost": 25, "dead_margin": 50,
               "dead_threshold": 0.5, key: value}
    with pytest.raises(ValueError, match=key):
        possession._switch_cost(None, np.zeros(3), section)


def test_runs_through_the_pipeline(bundle, tmp_path):
    from falafelona import pipeline
    cfg = load_config("configs/exp/poss-switch-cost.yaml")
    result = pipeline.run(cfg, tmp_path / "out.csv", n_frames=bundle["n_frames"])
    assert result["rows"] == bundle["n_frames"]
    assert set(result["counts"]) >= {"H", "A", "D"}
