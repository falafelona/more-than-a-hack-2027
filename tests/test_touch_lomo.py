"""Leave-one-match-out check of the touch sweep (scripts/touch_lomo.py, devset_eval --scores-json).

Made-up numbers and synthetic IDSSE files only (tests/synthetic_idsse.py): never real data.
"""

import importlib.util
import json
from pathlib import Path

import synthetic_idsse as syn
import yaml

from falafelona import vendor

ROOT = vendor.REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lomo = _load("touch_lomo")
cli = _load("devset_eval")
MATCHES = [f"M{i}" for i in range(7)]


def _run(gain_by_config):
    """{config: {match: spellF1}} with reference 0.3 and the given gain per config and match."""
    out = {"poss-place-late": {m: 0.3 for m in MATCHES}}
    for name, gains in gain_by_config.items():
        out[name] = {m: 0.3 + g for m, g in zip(MATCHES, gains, strict=True)}
    return out


def test_a_config_good_only_on_one_match_is_never_chosen_for_that_match():
    # touch-spiky wins in-sample only through M0; on M0 held out, touch-steady must be chosen.
    run = _run({"touch-spiky": [0.9, 0, 0, 0, 0, 0, 0], "touch-steady": [0.05] * 7})
    rows = lomo.lomo(lomo.gains({"s0": run}, "poss-place-late", "touch-"), ["s0"])

    chosen = {held: best for held, best, _ in rows}
    assert chosen["M0"] == "touch-steady"
    assert all(chosen[m] == "touch-spiky" for m in MATCHES[1:])  # 0.9/6 > 0.05 on the rest
    gain_m0 = next(per["s0"] for held, _, per in rows if held == "M0")
    assert abs(gain_m0 - 0.05) < 1e-12


def test_report_goes_only_when_every_run_passes(tmp_path):
    good = _run({"touch-a": [0.06] * 7, "touch-b": [0.01] * 7})
    weak = _run({"touch-a": [0.06] * 5 + [-0.01] * 2, "touch-b": [0.01] * 7})
    paths = []
    for name, run in (("s0-random", good), ("s1-random", good), ("s1-stop", weak)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(run))
        paths.append(f"{name}={path}")

    go = lomo.report(lomo.load_runs(paths[:2]), "poss-place-late", "touch-", ["s0-random"],
                     ["touch-a"])
    nogo = lomo.report(lomo.load_runs(paths), "poss-place-late", "touch-", ["s0-random"],
                       ["touch-a"])

    assert go.endswith("verdict: Go (LOMO mean >= +0.04 and better on >= 6/7 in every run)")
    assert "verdict: No-Go" in nogo
    assert "s1-stop: mean +0.040, better on 5/7" in nogo


def test_the_sweep_configs_hold_the_best_one_and_differ_only_in_touch():
    base = yaml.safe_load((ROOT / "configs/exp/poss-place-late.yaml").read_text())
    best = yaml.safe_load((ROOT / "configs/exp/poss-touch-best.yaml").read_text())
    paths = sorted((ROOT / "configs/exp/touch-lomo").glob("touch-*.yaml"))
    assert len(paths) == 17
    blocks = []
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        blocks.append(cfg["possession"].pop("touch"))
        assert cfg == base, path.name
    assert len({tuple(sorted(b.items())) for b in blocks}) == 17          # no duplicates
    assert yaml.safe_load(
        (ROOT / "configs/exp/touch-lomo/touch-r4-off0.12-margin0.1-hold0.8.yaml").read_text()
    ) == best


def test_devset_eval_writes_per_match_scores_only(tmp_path, monkeypatch, capsys):
    syn.write_idsse(tmp_path, "J03WTEST")
    monkeypatch.setenv("IDSSE_DATA_DIR", str(tmp_path))
    dev_cfg = tmp_path / "devset.yaml"
    dev_cfg.write_text(yaml.safe_dump({
        "seed": 0, "matches": ["J03WTEST"],
        "degrade": {"missing_share": 0.2, "gaps": [{"min_s": 0.5, "max_s": 2.0, "share": 1.0}]},
    }))
    out = tmp_path / "scores.json"

    cli.main(["--devset", str(dev_cfg), "--scores-json", str(out), "--config",
              str(ROOT / "configs/exp/poss-place-late.yaml"),
              str(ROOT / "configs/exp/touch-lomo/touch-r2-wide-cost2.yaml")])

    scores = json.loads(out.read_text())
    assert set(scores) == {"poss-place-late", "touch-r2-wide-cost2"}
    for per in scores.values():
        assert set(per) == {"J03WTEST"}
        assert all(isinstance(v, float) and 0 <= v <= 1 for v in per.values())
    assert "Dev set" in capsys.readouterr().out
    assert Path(out).stat().st_size < 1000
