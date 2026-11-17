"""Dead-ball oracle on the dev set (devset.oracle_dead_score, run_devset(oracles=...), --oracle).

Synthetic data only (tests/synthetic_idsse.py): never real match or IDSSE files.
"""

import importlib.util

import numpy as np
import pytest
import synthetic_idsse as syn
import yaml

from falafelona import devset, vendor
from falafelona.config import load_config

spec = importlib.util.spec_from_file_location(
    "devset_eval", vendor.REPO_ROOT / "scripts" / "devset_eval.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

GAPS = {   # long gaps, so the dead/alive call inside them matters
    "missing_share": 0.3,
    "gaps": [
        {"min_s": 0.0, "max_s": 0.5, "share": 0.1},
        {"min_s": 0.5, "max_s": 2.0, "share": 0.3},
        {"min_s": 2.0, "max_s": 8.0, "share": 0.6},
    ],
}
DEV_CFG = {"seed": 0, "matches": ["J03WTEST"], "degrade": GAPS}
FINAL = load_config(vendor.REPO_ROOT / "configs" / "final.yaml")


@pytest.fixture
def dev(tmp_path):
    syn.write_idsse(tmp_path, "J03WTEST")
    return devset.load_idsse("J03WTEST", root=tmp_path)


# --- the score itself ---------------------------------------------------------------------------

TRUTH = np.array(list("HHDDAAXDHA"))
MISSING = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=bool)
OURS = np.full(10, 0.3)


@pytest.mark.parametrize("mode, changed", [
    ("dead", ~np.isin(TRUTH, ["X"])),
    ("dead_gaps", MISSING & (TRUTH != "X")),
    ("dead_seen", ~MISSING & (TRUTH != "X")),
])
def test_each_oracle_changes_only_its_own_frames(mode, changed):
    out = devset.oracle_dead_score(OURS, TRUTH, MISSING, mode)

    expected = np.where(TRUTH == "D", 1.0, 0.0)
    assert np.array_equal(out[changed], expected[changed])
    assert np.array_equal(out[~changed], OURS[~changed])     # incl. truth X: ours
    assert np.array_equal(OURS, np.full(10, 0.3))            # input left alone


def test_an_unknown_oracle_or_wrong_lengths_are_refused(dev):
    with pytest.raises(ValueError, match="unknown oracle"):
        devset.oracle_dead_score(OURS, TRUTH, MISSING, "dead_all")
    with pytest.raises(ValueError, match="one value per frame"):
        devset.oracle_dead_score(OURS[:5], TRUTH, MISSING, "dead")
    with pytest.raises(ValueError, match="unknown oracle"):
        devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=("nope",))


# --- on the dev set -----------------------------------------------------------------------------


def test_no_oracle_leaves_the_result_exactly_as_before(dev):
    before = devset.run_devset({"final": FINAL}, DEV_CFG, [dev])
    with_oracle = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=("dead",))

    assert "oracle_of" not in before
    assert list(before["scores"]) == ["final"]
    assert with_oracle["scores"]["final"] == before["scores"]["final"]
    assert with_oracle["matches"] == before["matches"]
    assert with_oracle["oracle_of"] == {"final + oracle dead": "final"}


def test_the_dead_oracle_makes_the_dead_alive_call_right(dev):
    result = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=devset.ORACLES)
    ours = result["scores"]["final"]["J03WTEST"]
    oracle = result["scores"]["final + oracle dead"]["J03WTEST"]

    assert oracle["alive_dead_f1"] >= 0.97
    assert oracle["alive_dead_f1"] >= ours["alive_dead_f1"]
    for mode in devset.ORACLES:
        assert f"final + oracle {mode}" in result["scores"]


def test_each_mode_hands_the_decoder_the_truth_on_its_frames_only(dev, monkeypatch):
    seen = []
    real = devset.predict_labels

    def spy(match, cfg, prepared=None):
        seen.append((np.isnan(prepared[0].ball_xy[:, 0]), prepared[2].copy()))
        return real(match, cfg, prepared)

    monkeypatch.setattr(devset, "predict_labels", spy)
    devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=devset.DEAD_ORACLES)

    (missing, ours), *oracles = seen
    graded = np.isin(dev.truth, ("H", "A", "D"))
    truth = (dev.truth == "D").astype(float)
    masks = {"dead": graded, "dead_gaps": graded & missing, "dead_seen": graded & ~missing}
    assert missing.any() and (~missing).any()
    for mode, (_missing, score) in zip(devset.DEAD_ORACLES, oracles, strict=True):
        assert np.array_equal(score[masks[mode]], truth[masks[mode]])
        assert np.array_equal(score[~masks[mode]], ours[~masks[mode]])


def test_the_report_gives_each_oracle_against_its_own_config(dev):
    configs = {"final": FINAL, "other": {**FINAL, "seed": 1}}
    result = devset.run_devset(configs, DEV_CFG, [dev], oracles=("dead", "dead_gaps"))
    text = devset.format_report(result)

    assert "paired spellF1 difference against final" in text
    assert "  other: " in text
    head, oracle_part = text.split("dead oracle: paired spellF1 difference")
    assert "oracle" not in head.split("paired spellF1 difference against final")[1]
    for name in ("final + oracle dead", "final + oracle dead_gaps",
                 "other + oracle dead", "other + oracle dead_gaps"):
        assert f"  {name}: " in oracle_part
    assert "possessions" in oracle_part and "truth" in oracle_part


def test_main_oracle_flag(tmp_path, monkeypatch, capsys):
    data = tmp_path / "idsse"
    syn.write_idsse(data, "J03WTEST")
    cfg = yaml.safe_load((vendor.REPO_ROOT / "configs" / "devset.yaml").read_text())
    cfg["matches"] = ["J03WTEST"]
    path = tmp_path / "devset.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("IDSSE_DATA_DIR", str(data))
    final = str(vendor.REPO_ROOT / "configs" / "final.yaml")

    assert cli.main(["--devset", str(path), "--config", final]) == 0
    assert "oracle" not in capsys.readouterr().out
    assert cli.main(["--devset", str(path), "--config", final,
                     "--oracle", "dead", "--oracle", "dead_seen"]) == 0
    out = capsys.readouterr().out
    assert "final + oracle dead:" in out and "final + oracle dead_seen:" in out
    with pytest.raises(SystemExit):
        cli.main(["--devset", str(path), "--config", final, "--oracle", "everything"])


# --- why possessions are missed (spell_diagnosis, --diagnose) -----------------------------------


def seq(*parts):
    return np.array([label for label, n in parts for _ in range(n)])


def diagnose(truth, labels, missing=None):
    missing = np.zeros(len(truth), dtype=bool) if missing is None else missing
    return devset.spell_diagnosis(truth, labels, missing, fps=1.0)


def test_a_short_opponent_spell_run_over_counts_as_merged_and_other_team():
    truth = seq(("H", 10), ("A", 2), ("H", 10), ("D", 5), ("A", 10))
    ours = seq(("H", 22), ("D", 5), ("A", 10))

    d = diagnose(truth, ours)

    assert sum(d["truth_by_len"]) == 4 and sum(d["missed_by_len"]) == 3
    assert d["merged"] == 2 and d["merged_over_other_team"] == 2 and d["merged_over_dead"] == 0
    assert d["other_team"] == 1 and d["called_dead"] == 0 and d["cut_short"] == 0
    assert d["missed_by_len"][1] == 1                  # the 2 s spell is in the 2-5 s bin
    assert sum(d["fake_by_len"]) == 1 and d["same_team"] == 1


def test_a_missed_stoppage_merges_two_spells_over_dead():
    d = diagnose(seq(("H", 10), ("D", 4), ("H", 10)), seq(("H", 24)))

    assert d["merged"] == 2 and d["merged_over_dead"] == 2 and d["merged_over_other_team"] == 0


def test_called_dead_cut_short_and_fake_causes():
    d = diagnose(seq(("A", 10)), seq(("D", 10)))
    assert d["called_dead"] == 1 and sum(d["fake_by_len"]) == 0

    d = diagnose(seq(("H", 20)), seq(("H", 8), ("A", 5), ("H", 7)))
    assert d["cut_short"] == 1
    assert sum(d["fake_by_len"]) == 3
    assert d["over_other_team"] == 1 and d["same_team"] == 2


def test_matched_spells_are_not_counted_and_seen_share_is_split():
    truth = seq(("H", 10), ("D", 5), ("A", 10))
    missing = np.zeros(len(truth), dtype=bool)
    missing[15:25] = True                              # the A spell happens with no ball

    d = diagnose(truth, truth, missing)
    assert sum(d["missed_by_len"]) == 0 and sum(d["fake_by_len"]) == 0
    assert d["truth_seen"] == 1 and d["truth_unseen"] == 1

    d = diagnose(truth, seq(("H", 10), ("D", 15)), missing)
    assert d["missed_unseen"] == 1 and d["missed_seen"] == 0 and d["called_dead"] == 1


def test_truth_x_is_never_a_possession_and_is_never_merged_over():
    truth = seq(("H", 5), ("X", 3), ("H", 5))
    d = diagnose(truth, seq(("H", 13)))

    assert sum(d["truth_by_len"]) == 2 and sum(d["missed_by_len"]) == 0


def test_diagnose_reports_counts_per_config_and_changes_no_score(dev):
    plain = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=("dead",))
    result = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=("dead",), diagnose=True)
    text = devset.format_report(result)

    assert result["scores"] == plain["scores"]
    assert "diagnosis" not in plain and "why possessions are missed" not in devset.format_report(
        plain)
    assert set(result["diagnosis"]) == {"final", "final + oracle dead"}
    for name in ("final", "final + oracle dead"):
        d = result["diagnosis"][name]["J03WTEST"]
        s = result["scores"][name]["J03WTEST"]
        assert sum(d["truth_by_len"]) == s["truth_possessions"]
        assert sum(d["ours_by_len"]) == s["our_possessions"]
        assert sum(d["missed_by_len"]) == s["spell_missed"]
        assert sum(d["fake_by_len"]) == s["spell_spurious"]
        assert sum(d[c] for c in devset.MISSED_CAUSES) == s["spell_missed"]
        assert sum(d[c] for c in devset.FAKE_CAUSES) == s["spell_spurious"]
        assert f"  {name}: missed" in text


def test_the_switch_oracle_only_cheapens_switches_in_gaps_near_a_true_change(dev, monkeypatch):
    seen = {}
    real = devset.predict_labels

    def spy(match, cfg, prepared=None):
        if "_oracle_team_switch" in cfg["possession"]:
            seen["costs"] = cfg["possession"]["_oracle_team_switch"].copy()
            seen["missing"] = np.isnan(prepared[0].ball_xy[:, 0])
        return real(match, cfg, prepared)

    monkeypatch.setattr(devset, "predict_labels", spy)
    result = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=("switch",))

    base = float(FINAL["possession"]["switch_cost"])
    costs, missing = seen["costs"], seen["missing"]
    cheap = costs < base
    assert cheap.any() and not cheap[~missing].any()
    assert set(np.unique(costs)) <= {base, devset.ORACLE_SWITCH_COST}
    assert "_oracle_team_switch" not in FINAL["possession"]           # the config is not changed
    assert "final + oracle switch" in result["scores"]


def test_the_team_oracle_adds_truth_evidence_only_inside_gaps(dev, monkeypatch):
    seen = {}
    real = devset.predict_labels

    def spy(match, cfg, prepared=None):
        if "_oracle_team_extra" in cfg["possession"]:
            seen["extra"] = cfg["possession"]["_oracle_team_extra"].copy()
            seen["missing"] = np.isnan(prepared[0].ball_xy[:, 0])
        return real(match, cfg, prepared)

    monkeypatch.setattr(devset, "predict_labels", spy)
    result = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=("team_gaps",))

    extra, missing = seen["extra"], seen["missing"]
    assert extra.shape == (len(dev.truth), 2) and not extra[~missing].any()
    assert (extra[missing & (dev.truth == "H"), 1] == devset.ORACLE_TEAM_COST).all()
    assert (extra[missing & (dev.truth == "A"), 0] == devset.ORACLE_TEAM_COST).all()
    assert not extra[dev.truth == "D"].any()
    a = result["scores"]["final + oracle team_gaps"]["J03WTEST"]
    assert a["team_acc"] >= result["scores"]["final"]["J03WTEST"]["team_acc"]


def test_team_changes_marks_the_first_frame_of_a_new_team_only():
    truth = np.array(list("HHDDAAXAHH"))
    period = np.ones(10, dtype=np.int8)
    assert np.flatnonzero(devset.team_changes(truth, period)).tolist() == [4, 8]
    period[4:] = 2                                          # a new half is not a change
    assert np.flatnonzero(devset.team_changes(truth, period)).tolist() == [8]


@pytest.mark.parametrize("mode", ["touch_team", "touch_all"])
def test_the_touch_oracles_give_touches_the_true_team(dev, monkeypatch, mode):
    seen = {}
    real = devset.predict_labels

    def spy(match, cfg, prepared=None):
        if "_oracle_touch_events" in cfg["possession"]:
            seen["events"] = cfg["possession"]["_oracle_touch_events"]
            seen["fps"] = match.fps
        return real(match, cfg, prepared)

    monkeypatch.setattr(devset, "predict_labels", spy)
    result = devset.run_devset({"final": FINAL}, DEV_CFG, [dev], oracles=(mode,))

    touches, teams = seen["events"]
    after = np.minimum(touches + round(devset.ORACLE_TOUCH_AFTER_S * seen["fps"]),
                       len(dev.truth) - 1)
    truth = dev.truth[after]
    assert len(touches) == len(teams) and len(touches) > 0
    assert (teams[truth == "H"] == 0).all() and (teams[truth == "A"] == 1).all()
    assert (teams[~np.isin(truth, ["H", "A"])] == -1).all()
    assert f"final + oracle {mode}" in result["scores"]