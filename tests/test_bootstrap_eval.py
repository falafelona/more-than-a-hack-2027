# ws:eval: tests for scripts/bootstrap_eval.py (tasks 5.1, 5.2, 5.3). Synthetic data only.
#
# Every file here is made up inside a temporary folder: no match data, no public answers file.
# Safe for anyone, for CI and for AI tools (CLAUDE.md).

import importlib.util
from pathlib import Path
from types import SimpleNamespace as Tag

import numpy as np
import pytest

# scripts/ is not a package, so load the script as a module
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_eval.py"
spec = importlib.util.spec_from_file_location("bootstrap_eval", SCRIPT)
be = importlib.util.module_from_spec(spec)
spec.loader.exec_module(be)


def write_csv(path, labels):
    rows = "\n".join(f"{i},{label}" for i, label in enumerate(labels))
    path.write_text("frame,label\n" + rows + "\n")
    return path


def test_possession_lengths():
    assert be.possession_lengths(list("HHDDHH")) == [2, 2]
    assert be.possession_lengths(list("HHHAAD")) == [3, 2]
    assert be.possession_lengths(list("DDXX")) == []


def test_match_stats():
    labels = np.array(["H"] * 50 + ["A"] * 10 + ["D"] * 40)  # 2 s of H, 0.4 s of A, 1.6 s of D
    stats = be.match_stats(labels)
    assert stats["possessions"] == 2
    assert stats["flicker_under_1s"] == 1  # the 10-frame A
    assert stats["share_H"] == 0.5 and stats["share_D"] == 0.4 and stats["share_X"] == 0.0


def test_public_check_perfect_copy(tmp_path):
    truth = np.array(["H"] * 100 + ["D"] * 50 + ["A"] * 100)
    gt = write_csv(tmp_path / "gt.csv", truth)
    assert be.public_check(truth, gt) == {"spellF1": 1.0, "macroF1": 1.0}


def test_main_runs_and_rejects_bad_files(tmp_path, capsys):
    truth = ["H"] * 100 + ["D"] * 50 + ["A"] * 100
    gt = write_csv(tmp_path / "gt.csv", truth)
    sub = write_csv(tmp_path / "sub.csv", truth)
    be.main([str(sub), "--gt", str(gt)])
    assert "spellF1" in capsys.readouterr().out

    short = write_csv(tmp_path / "short.csv", truth[:-1])  # one row missing
    with pytest.raises(SystemExit):
        be.main([str(short), "--gt", str(gt)])


def test_possessions_per_minute():
    labels = np.array(list("HHHHAAAADDDD") * 125)  # 1500 frames = 1 minute, 250 possessions
    graded = np.ones(len(labels), dtype=np.int8)
    assert be.possessions_per_minute(labels, graded) == 250.0
    assert be.possessions_per_minute(labels, np.zeros(len(labels), dtype=np.int8)) is None


def test_tag_agreement():
    labels = np.array(["H"] * 8 + ["A"] * 2 + ["D"] * 5)
    tags = [
        Tag(code="BUILD UP", team="H", start=0, end=10),            # 8 of 10 frames agree
        Tag(code="DEFENDING IN MIDDLE THIRD", team="A", start=0, end=10),  # skipped: defending
        Tag(code="PROGRESSION", team=None, start=0, end=10),        # skipped: no team
        Tag(code="Build Up", team="H", start=10, end=15),           # only D frames: skipped
    ]
    assert be.tag_agreement(labels, tags) == 0.8
    assert be.tag_agreement(labels, []) is None