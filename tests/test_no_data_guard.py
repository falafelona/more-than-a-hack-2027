# scripts/check_no_data.py blocks what it should: data file names/types, files > 5 MB,
# notebooks with outputs. And it lets normal code files through.
# Only file NAMES are checked here; no data-shaped file is created.
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_no_data", ROOT / "scripts/check_no_data.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


@pytest.mark.parametrize("path", [
    "BAR-ATH_tactical_atd.txt",
    "somewhere/BAR-ATH_tactical_atd.xml",
    "BAR-ATH_tactical_smart_tagging.xml",
    "BAR-ATH_tactical_homography.csv",
    "BAR-ATH_possession_gt_public.csv",
    "match.mp4",
    "features.npz",
    "model.pkl",
    "Falafelona.pdf",
    "outputs/Falafelona.csv",
    "data/tactical/anything.txt",
    "cache/match.bin",
    "labels.csv",
])
def test_blocks_data(path):
    assert guard.reason_refused(path) is not None


@pytest.mark.parametrize("path", [
    "src/falafelona/io.py",
    "configs/baseline.yaml",
    "README.md",
    "outputs/.gitkeep",
    "experiments/runs.csv",
])
def test_allows_code(path):
    assert guard.reason_refused(path) is None


def test_blocks_big_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("big.txt").write_bytes(b"0" * (guard.MAX_BYTES + 1))
    assert guard.reason_refused("big.txt") == "larger than 5 MB"


def test_blocks_notebook_with_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = {"cell_type": "code", "source": "1", "outputs": [{"text": "1"}]}
    Path("nb.ipynb").write_text(json.dumps({"cells": [cell]}))
    assert guard.reason_refused("nb.ipynb") is not None
    cell["outputs"] = []
    Path("nb.ipynb").write_text(json.dumps({"cells": [cell]}))
    assert guard.reason_refused("nb.ipynb") is None


def test_main_exit_codes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # never scan the real repo's data folders from a test
    assert guard.main(["src/falafelona/io.py"]) == 0
    assert guard.main(["BAR-ATH_tactical_atd.txt"]) == 1
