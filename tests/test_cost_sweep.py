"""Decoder cost sweep (scripts/cost_sweep.py): config generation and the analysis.

Made-up numbers and synthetic IDSSE files only (tests/synthetic_idsse.py): never real data.
"""

import importlib.util
import json

import pytest
import synthetic_idsse as syn
import yaml

from falafelona import vendor
from falafelona.config import load_config

ROOT = vendor.REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cs = _load("cost_sweep")
cli = _load("devset_eval")
MATCHES = [f"M{i}" for i in range(7)]
REF = cs.REFERENCE


def _without_costs(cfg):
    for key in cs.KEYS:
        cfg["possession"].pop(key)
    return cfg


def test_committed_stage1_configs_differ_from_the_dev_config_only_in_the_costs():
    base = yaml.safe_load(cs.BASE.read_text())
    paths = sorted(cs.OUT_DIR.glob("cost-*.yaml"))
    assert len(paths) == 30
    points = set()
    for path in paths:
        load_config(path)                                   # the strict loader accepts it
        cfg = yaml.safe_load(path.read_text())
        points.add(tuple(cfg["possession"][k] for k in cs.KEYS))
        assert path.stem == cs.config_name(cfg["possession"])
        assert _without_costs(cfg) == _without_costs(yaml.safe_load(cs.BASE.read_text())), path
    assert len(points) == 30
    shipped = yaml.safe_load((cs.OUT_DIR / "cost-sw3.5-td6-fd6-dw2.yaml").read_text())
    assert shipped == base                                  # the sweep holds the submitted point
    # the dev config keeps the leave-one-match-out dead model, never the one fitted on all 7
    assert "{match_id}" in base["deadball"]["learned"]["model"]


def test_generate_unties_dead_costs_only_when_asked(tmp_path):
    tied = cs.grid_points([3.5, 7], [6, 9], None, [2])
    untied = cs.grid_points([3.5], [6, 9], [6, 13], [1.5, 2])
    assert len(tied) == 4 and all(p["to_dead_cost"] == p["from_dead_cost"] for p in tied)
    assert len(untied) == 8
    paths = cs.generate(untied, tmp_path)
    assert len({p.name for p in paths}) == 8
    cfg = yaml.safe_load((tmp_path / "cost-sw3.5-td9-fd13-dw1.5.yaml").read_text())
    assert [cfg["possession"][k] for k in cs.KEYS] == [3.5, 9, 13, 1.5]
    cs.generate(untied, tmp_path)                           # same content again: fine
    paths[0].write_text("changed by hand\n")
    with pytest.raises(SystemExit):
        cs.generate(untied, tmp_path)                       # never overwrite a different file


def _params(points):
    return {cs.config_name(p): p for p in points}


def _run(gain_of):
    """{config: {match: spellF1}}: reference 0.45, each config gets gain_of(params) per match."""
    out = {REF: {m: 0.45 for m in MATCHES}}
    for name, p in gain_of.items():
        out[name] = {m: 0.45 + g for m, g in zip(MATCHES, p, strict=True)}
    return out


def test_best_on_the_edge_is_flagged_and_the_next_grid_extends_past_it():
    params = _params(cs.grid_points([3.5, 7, 14], [6, 13], None, [2]))
    # gain grows with switch_cost: the best (14) is the top of the grid
    gains = {n: [p["switch_cost"] / 100] * 7 for n, p in params.items()}
    text = cs.analyse({"s0-random": _run(gains)}, REF, ["s0-random"], params)
    assert "GRID EDGE" in text and "switch_cost (high)" in text
    assert "--switch 14 28 56" in text
    assert cs.next_values("to_dead_cost", {"to_dead_cost": 13.0}, params) == [13.0, 28.17, 61.03]


def test_best_inside_the_grid_is_refined_and_not_flagged():
    params = _params(cs.grid_points([3.5, 7, 14], [6, 9, 13], None, [1.5, 2, 3]))
    best = {"switch_cost": 7.0, "to_dead_cost": 9.0, "from_dead_cost": 9.0, "dead_weight": 2.0}
    gains = {n: [0.05 if p == best else 0.01] * 7 for n, p in params.items()}
    text = cs.analyse({"s0-random": _run(gains)}, REF, ["s0-random"], params)
    assert "Grid edges: none" in text
    assert "--switch 5.25 7 10.5" in text and "--to-dead 7.5 9 11" in text
    assert text.splitlines()[3].startswith("  cost-sw7-td9-fd9-dw2: +0.050")


def test_verdict_uses_the_lomo_rule_and_s1_only_checks(tmp_path):
    params = _params(cs.grid_points([3.5, 7], [6], None, [2]))
    good = _run({"cost-sw7-td6-fd6-dw2": [0.06] * 7, "cost-sw3.5-td6-fd6-dw2": [0.0] * 7})
    weak = _run({"cost-sw7-td6-fd6-dw2": [0.06] * 5 + [-0.01] * 2,
                 "cost-sw3.5-td6-fd6-dw2": [0.0] * 7})
    go = cs.analyse({"s0-random": good, "s1-random": good}, REF, ["s0-random"], params)
    nogo = cs.analyse({"s0-random": good, "s1-random": weak}, REF, ["s0-random"], params)
    assert "verdict: Go" in go
    assert "verdict: No-Go" in nogo
    assert "Leave-one-match-out over 2 cost configs" in go


def test_stage_files_of_one_run_merge_and_mismatched_runs_are_refused(tmp_path):
    a = {REF: {m: 0.45 for m in MATCHES}, "cost-sw7-td6-fd6-dw2": {m: 0.5 for m in MATCHES}}
    b = {REF: {m: 0.45 for m in MATCHES}, "cost-sw10-td6-fd6-dw2": {m: 0.52 for m in MATCHES}}
    c = {REF: {m: 0.40 for m in MATCHES}}                    # other seed: reference differs
    for name, run in (("a", a), ("b", b), ("c", c)):
        (tmp_path / f"{name}.json").write_text(json.dumps(run))
    runs = cs.load_runs([f"s0-random={tmp_path / 'a.json'},{tmp_path / 'b.json'}"])
    assert set(runs["s0-random"]) == {REF, "cost-sw7-td6-fd6-dw2", "cost-sw10-td6-fd6-dw2"}
    with pytest.raises(SystemExit):
        cs.load_runs([f"s0-random={tmp_path / 'a.json'},{tmp_path / 'c.json'}"])


def test_end_to_end_on_a_synthetic_idsse_match(tmp_path, monkeypatch, capsys):
    syn.write_idsse(tmp_path, "J03WTEST")
    syn.write_idsse(tmp_path, "J03WTWO")
    monkeypatch.setenv("IDSSE_DATA_DIR", str(tmp_path))
    dev_cfg = tmp_path / "devset.yaml"
    dev_cfg.write_text(yaml.safe_dump({
        "seed": 0, "matches": ["J03WTEST", "J03WTWO"],
        "degrade": {"missing_share": 0.2, "gaps": [{"min_s": 0.5, "max_s": 2.0, "share": 1.0}]},
    }))
    sweep = tmp_path / "sweep"
    cs.generate(cs.grid_points([3.5, 7], [6], None, [2]), sweep)
    # the dev config's learned model path holds {match_id}; use the plain model for the test match
    for path in [*sweep.glob("cost-*.yaml")]:
        path.write_text(path.read_text().replace("deadball_stoppage_lomo_{match_id}",
                                                 "deadball_stoppage_all"))
    ref = tmp_path / f"{REF}.yaml"
    ref.write_text(cs.BASE.read_text().replace("deadball_stoppage_lomo_{match_id}",
                                               "deadball_stoppage_all"))
    out = tmp_path / "s0-random.json"
    cli.main(["--devset", str(dev_cfg), "--scores-json", str(out), "--config", str(ref),
              *map(str, sorted(sweep.glob("cost-*.yaml")))])
    capsys.readouterr()
    assert cs.main(["analyse", "--run", f"s0-random={out}", "--configs", str(sweep)]) == 0
    text = capsys.readouterr().out
    assert "Decoder cost sweep: 2 configs" in text and "verdict:" in text
    assert "held out J03WTEST" in text and "held out J03WTWO" in text
    scores = json.loads(out.read_text())
    assert scores["cost-sw3.5-td6-fd6-dw2"] == scores[REF]   # the shipped point reproduces it
